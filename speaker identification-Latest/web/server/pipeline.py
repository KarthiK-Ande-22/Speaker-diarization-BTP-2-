"""Runs the pipeline for one recording, stage by stage, for the website.

Each stage runs in its own Python process, so the memory used by pyannote or Whisper is given back
before the next stage starts. Progress is kept in data/<id>/job.json and each stage's output in
data/<id>/logs/<stage>.log.

    python web/server/pipeline.py STAGE RECORDING_ID     # run one stage (the website does this)
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
for step in ("00_clean_audio", "01_diarization", "02_speech_features", "03_transcription", "04_medical_terms",
             "05_video", "06_decision"):
    sys.path.insert(0, str(ROOT / step))

# key, title shown on the website, the file that shows the stage is done, the stages it needs
STAGES = [
    {"key": "clean", "title": "Clean audio", "output": "audio_clean_16k.wav", "needs": []},
    {"key": "diarization", "title": "Who spoke when", "output": "turns.csv", "needs": []},
    {"key": "speech", "title": "Speech features", "output": "speaker_features.csv", "needs": ["diarization"]},
    {"key": "transcription", "title": "Transcript", "output": "transcript.csv", "needs": ["diarization"]},
    {"key": "medical", "title": "Medical terms", "output": "medical_features.csv", "needs": ["transcription"]},
    {"key": "video", "title": "Video features", "output": "video_features.csv", "needs": ["diarization"]},
    {"key": "decision", "title": "Role decision (Sarvam)", "output": "decision.json", "needs": []},
]
FEATURE_STAGES = ("speech", "medical", "video")

# The model choices the website offers. The first option of each is the default: our own pipeline, on this
# computer. "short" is how the recordings list names the choice ("" = not mentioned).
MODELS = {
    "clean": {"title": "Clean audio", "help": "remove background noise before finding speakers and writing the transcript",
              "options": [{"key": "off", "label": "Off", "short": ""},
                          {"key": "isolator", "label": "ElevenLabs Voice Isolator", "short": "cleaned audio", "cloud": True}]},
    "diarization": {"title": "Who spoke when", "help": "the model that finds the two speakers",
                    "options": [{"key": "pyannote", "label": "pyannote community-1", "short": "pyannote"},
                                {"key": "scribe", "label": "ElevenLabs Scribe", "short": "Scribe", "cloud": True}]},
    "transcription": {"title": "Transcript", "help": "the model that writes down what was said",
                      "options": [{"key": "whisper", "label": "Whisper medium", "short": "Whisper"},
                                  {"key": "scribe", "label": "ElevenLabs Scribe v2 Medical", "short": "Scribe Medical", "cloud": True}]},
    "medical": {"title": "Medical terms", "help": "the model that finds medicines, tests, conditions and symptoms",
                "options": [{"key": "ner", "label": "d4data biomedical NER", "short": ""},
                            {"key": "scribe", "label": "ElevenLabs entity detection", "short": "ElevenLabs terms", "cloud": True}]},
}
DEFAULT_MODELS = {part: info["options"][0]["key"] for part, info in MODELS.items()}
# Saved model results a new run of the same file can reuse when it cleans the audio the same way
REUSABLE = ["audio_clean_16k.wav", "pyannote.rttm", "pyannote_exclusive.rttm", "whisper.json", "elevenlabs_scribe.json"]


def now():
    return datetime.now().isoformat(timespec="seconds")


def has_video(media):
    """True if the file has a real picture stream (an mp3's cover image does not count)."""
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:stream_disposition=attached_pic",
                             "-of", "json", str(media)], capture_output=True, text=True)
    streams = json.loads(result.stdout or "{}").get("streams", [])
    return any(s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic") for s in streams)


def job_models(job):
    """The job's model choices (recordings made before a choice existed used its default)."""
    chosen = job.get("models") or {}
    return {part: chosen.get(part, default) for part, default in DEFAULT_MODELS.items()}


def recording_id(name, media, start, end, language, models, keyterms):
    """The same file and settings give the same id, so a finished analysis is found again instead of redone.
    (With the default models the id is the same as before the model choices existed.)"""
    digest = hashlib.sha1()
    with open(media, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    digest.update(f"{start}|{end}|{language}".encode())
    if models != DEFAULT_MODELS or keyterms:
        digest.update(json.dumps({"models": models, "keyterms": keyterms}, sort_keys=True).encode())
    stem = re.sub(r"[^a-z0-9]+", "_", Path(name).stem.lower()).strip("_")[:40] or "recording"
    return f"{stem}_{digest.hexdigest()[:6]}"


def reuse_results(source_id, new_id):
    """Copy what the slow models already made for another run of the same file: the audio always, the rest
    only if both runs clean the audio the same way (they were made from that audio). Each step then uses the
    saved result instead of running its model again (Scribe's answer is also checked against the request)."""
    source, new = DATA / source_id, DATA / new_id
    same_audio = job_models(load_job(source_id))["clean"] == job_models(load_job(new_id))["clean"]
    for name in ["audio_16k.wav"] + (REUSABLE if same_audio else []):
        if (source / name).exists():
            shutil.copy2(source / name, new / name)


def save_job(job):
    path = DATA / job["id"] / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2))


def load_job(rec_id):
    """job.json, or (for a recording made with the command-line scripts) settings from recordings.csv
    and stage states from the files that exist."""
    path, folder = DATA / rec_id / "job.json", DATA / rec_id
    if path.exists():
        job = json.loads(path.read_text())
    elif folder.is_dir():
        from diarize import read_recordings
        row = next((r for r in read_recordings() if r["recording_id"] == rec_id), None)
        media = str((ROOT / row["media_path"]).resolve()) if row else ""
        job = {"id": rec_id, "name": Path(media).name if media else rec_id, "media": media,
               "start": row["start"] if row else None, "end": row["end"] if row else None,
               "language": (row.get("language") or None) if row else None,
               "created": datetime.fromtimestamp(folder.stat().st_mtime).isoformat(timespec="seconds"),
               "has_video": has_video(media) if media and Path(media).exists() else (folder / "video_features.csv").exists(),
               "stages": {s["key"]: {"status": "done"} for s in STAGES if (folder / s["output"]).exists()}}
    else:
        return None
    if job_models(job)["clean"] == "off":                    # cleaning is off: the stage is skipped, not waiting
        job["stages"].setdefault("clean", {"status": "skipped", "note": "off"})
    for stage in STAGES:
        job["stages"].setdefault(stage["key"], {"status": "waiting"})
    return job


def last_line(path):
    lines = [l for l in path.read_text(errors="replace").splitlines() if l.strip()] if path.exists() else []
    return lines[-1][:300] if lines else ""


def run_job(rec_id, only_missing=False):
    """Run every stage in order (or only the ones not done yet). A stage whose needs failed is skipped."""
    job = load_job(rec_id)
    models = job_models(job)
    folder = DATA / rec_id
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        key = stage["key"]
        if only_missing and job["stages"].get(key, {}).get("status") == "done" and (folder / stage["output"]).exists():
            continue                                         # already done: keep its result
        status = lambda k: job["stages"].get(k, {}).get("status")
        needs = stage["needs"] + (["clean"] if key == "diarization" and models["clean"] != "off" else [])
        reason = None
        if key == "clean" and models["clean"] == "off":
            reason = "off"
        elif any(status(n) != "done" for n in needs):
            reason = "needs " + ", ".join(n for n in needs if status(n) != "done")
        elif key == "video" and not job["has_video"]:
            reason = "this file has no picture"
        elif key == "decision" and not any(status(k) == "done" for k in FEATURE_STAGES):
            reason = "no features were extracted"
        if reason:
            job["stages"][key] = {"status": "skipped", "note": reason}
            save_job(job)
            continue

        job["stages"][key] = {"status": "running", "started": now()}
        save_job(job)
        began, log = time.time(), folder / "logs" / f"{key}.log"
        with open(log, "w") as out:
            code = subprocess.run([sys.executable, "-u", __file__, key, rec_id], stdout=out,
                                  stderr=subprocess.STDOUT, cwd=ROOT).returncode
        done = code == 0 and (folder / stage["output"]).exists()
        job["stages"][key] = {"status": "done" if done else ("skipped" if code == 0 else "failed"),
                              "started": job["stages"][key]["started"], "seconds": round(time.time() - began, 1),
                              **({} if done else {"note": last_line(log)})}
        save_job(job)


def run_stage(key, rec_id):
    """One stage, using the step scripts' own functions and the models chosen for this recording."""
    job = load_job(rec_id)
    models, language, keyterms = job_models(job), job["language"], job.get("keyterms") or []
    if key == "clean":
        import clean_audio
        clean_audio.clean(job["media"], rec_id, job["start"], job["end"])
    elif key == "diarization":
        import diarize
        diarize.diarize(job["media"], rec_id, job["start"], job["end"], model=models["diarization"],
                        language=language, keyterms=keyterms)
    elif key == "speech":
        import speech_features
        speech_features.extract(rec_id)
    elif key == "transcription":
        import transcribe
        transcribe.transcribe(rec_id, language, model=models["transcription"], keyterms=keyterms)
    elif key == "medical":
        import medical_terms
        medical_terms.extract(rec_id, model=models["medical"], language=language, keyterms=keyterms)
    elif key == "video":
        import video_features
        video_features.extract({"recording_id": rec_id, "media_path": job["media"], "start": job["start"], "end": job["end"]})
    elif key == "decision":
        import llm_decision
        llm_decision.decide_recording(rec_id)


if __name__ == "__main__":
    run_stage(sys.argv[1], sys.argv[2])
