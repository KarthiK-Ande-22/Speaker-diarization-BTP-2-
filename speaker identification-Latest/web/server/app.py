"""Website backend: upload a recording, run the pipeline, and serve every stage's results.

Run (from the "speaker identification-Latest" folder, after building the website once, see web/README.md):
    python web/server/app.py          # then open http://127.0.0.1:5000

By default everything runs on this computer, except the last stage, which sends the extracted features
(numbers and the list of medical terms, not the audio) to Sarvam. For four stages the user can choose
ElevenLabs models instead (pipeline.MODELS); those send the audio to ElevenLabs.
"""

import csv
import json
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline  # noqa: E402
from elevenlabs_api import api_key  # noqa: E402  (pipeline.py put the step folders on the path)
from llm_decision import DESCRIPTIONS  # noqa: E402

ROOT, DATA = pipeline.ROOT, pipeline.DATA
UPLOADS = ROOT / "uploads"
FRONTEND = ROOT / "web" / "frontend" / "dist"
FILES = {"audio": "audio_16k.wav", "clean": "audio_clean_16k.wav", "people": "video_people.jpg"}
EXTRA_MEANINGS = {"face_link_confidence": "how sure the software is that the faces were matched to the right voices (0.5 = no idea, 1 = certain)"}

app = Flask(__name__, static_folder=None)
app.json.sort_keys = False                               # keep the order of the stages and model choices
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3        # 2 GB uploads
running = {"id": None}
lock = threading.Lock()


# ---------------------------------------------------------------- helpers

def valid(rec_id):
    if not re.fullmatch(r"[a-z0-9_]+", rec_id or "") or not (DATA / rec_id).is_dir():
        abort(404)
    return rec_id


def read_csv(path):
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def number(text):
    return None if text in ("", None) else float(text)


def features(folder, name, group):
    rows = read_csv(folder / name)
    meanings = DESCRIPTIONS.get(group, {}) | EXTRA_MEANINGS
    return [{"feature": r["feature"], "meaning": meanings.get(r["feature"], r["feature"].replace("_", " ")),
             "A": number(r["A"]), "B": number(r["B"]), "diff": number(r["A_minus_B"]),
             "expected": r["expected_higher"], "suggests": r["doctor_by_this_feature"]} for r in rows or []]


def read_settings(models, keyterms_text):
    """Check the chosen models (defaults for anything not chosen) and the key terms.
    Returns (models, keyterms, error message or None)."""
    chosen = {}
    for part, info in pipeline.MODELS.items():
        key = (models or {}).get(part) or pipeline.DEFAULT_MODELS[part]
        option = next((o for o in info["options"] if o["key"] == key), None)
        if option is None:
            return None, None, f"Unknown choice for {info['title']}: {key}"
        if option.get("cloud") and not api_key():
            return None, None, "ElevenLabs is not set up: add ELEVENLABS_API_KEY to the .env file and restart the server."
        chosen[part] = key
    terms = []
    if "scribe" in chosen.values():                       # key terms only matter for Scribe
        terms = list(dict.fromkeys(t.strip() for t in re.split(r"[,;\n]", keyterms_text or "") if t.strip()))
        bad = next((t for t in terms if len(t) > 50 or len(t.split()) > 5 or re.search(r"[<>{}\[\]\\]", t)), None)
        if bad:
            return None, None, f"A key term can have at most 50 characters and 5 words, without <>{{}}[]\\: {bad}"
        if len(terms) > 100:
            return None, None, "At most 100 key terms."
    return chosen, terms, None


def models_used(job):
    """[{part, title, label, cloud}] for the recording page."""
    models = pipeline.job_models(job)
    out = []
    for part, info in pipeline.MODELS.items():
        option = next(o for o in info["options"] if o["key"] == models[part])
        out.append({"part": part, "title": info["title"], "label": option["label"], "cloud": bool(option.get("cloud"))})
    return out


def models_text(job):
    """Short summary for the recordings list, e.g. "pyannote + Whisper"."""
    models = pipeline.job_models(job)
    shorts = [next(o for o in info["options"] if o["key"] == models[part])["short"] for part, info in pipeline.MODELS.items()]
    return " + ".join(s for s in shorts if s)


def job_view(rec_id):
    """The job with its stages in order; a stage left "running" by a stopped server is shown as stopped."""
    job = pipeline.load_job(rec_id)
    stages = []
    for stage in pipeline.STAGES:
        state = dict(job["stages"].get(stage["key"], {"status": "waiting"}))
        if state["status"] == "running" and running["id"] != rec_id:
            state.update(status="failed", note="stopped (the server was restarted)")
        stages.append({"key": stage["key"], "title": stage["title"], **state})
    return {key: job.get(key) for key in ("id", "name", "created", "start", "end", "language", "has_video")} | {
        "models": pipeline.job_models(job), "keyterms": job.get("keyterms") or [], "models_used": models_used(job),
        "stages": stages, "running": running["id"] == rec_id}


def results(rec_id):
    folder = DATA / rec_id
    out = {}
    if (folder / FILES["clean"]).exists():
        import soundfile as sf
        out["clean"] = {"original_s": round(sf.info(str(folder / FILES["audio"])).duration, 2),
                        "cleaned_s": round(sf.info(str(folder / FILES["clean"])).duration, 2)}
    turns = read_csv(folder / "turns.csv")
    if turns and (folder / "info.json").exists():
        info = json.loads((folder / "info.json").read_text())
        out["diarization"] = {"model": info.get("model"), "audio_s": info["audio_s"], "talk_s": info.get("talk_s", {}), "turns": info.get("turns", {}),
                              "both_talking_s": info.get("both_talking_s", 0), "voice_recheck": info.get("voice_recheck"),
                              "audio": (folder / FILES["audio"]).exists(),
                              "segments": [{"speaker": t["speaker"], "start": float(t["start"]), "end": float(t["end"])} for t in turns]}
    if (folder / "speaker_features.csv").exists():
        out["speech"] = {"features": features(folder, "speaker_features.csv", "speech")}
    transcript = read_csv(folder / "transcript.csv")
    if transcript:
        out["transcript"] = [{"turn": int(t["turn"]), "speaker": t["speaker"], "start": float(t["start"]), "end": float(t["end"]),
                              "text": t["text"]} for t in transcript if t["text"]]
    if (folder / "medical_features.csv").exists():
        out["medical"] = {"features": features(folder, "medical_features.csv", "medical terms"),
                          "terms": [{"speaker": t["speaker"], "term": t["term"], "type": t["type"], "side": t["side"],
                                     "question": t["question"] == "1"} for t in read_csv(folder / "medical_terms.csv") or []]}
    if (folder / "video_features.csv").exists():
        info = json.loads((folder / "video_info.json").read_text())
        out["video"] = {"features": features(folder, "video_features.csv", "video"),
                        "link": {name: value["confidence"] for name, value in info["link"].items()},
                        "faces": info.get("faces_found"), "image": (folder / FILES["people"]).exists()}
    if (folder / "decision.json").exists():
        out["decision"] = json.loads((folder / "decision.json").read_text())
    return out


def start(rec_id, only_missing):
    """Run the pipeline in the background; one recording at a time (the laptop has no GPU)."""
    def work():
        try:
            pipeline.run_job(rec_id, only_missing)
        finally:
            running["id"] = None
    with lock:
        if running["id"]:
            return False
        running["id"] = rec_id
    threading.Thread(target=work, daemon=True).start()
    return True


def busy():
    name = pipeline.load_job(running["id"])["name"] if running["id"] else ""
    return jsonify(error=f"Another recording is being analysed ({name}). Please wait until it finishes."), 409


# ---------------------------------------------------------------- API

@app.get("/api/status")
def status():
    """What is running now (for the bar at the bottom of every page)."""
    rec_id = running["id"]
    if not rec_id:
        return jsonify(running=None)
    job = job_view(rec_id)
    number, stage = next(((i + 1, s) for i, s in enumerate(job["stages"]) if s["status"] == "running"), (None, None))
    return jsonify(running={"id": rec_id, "name": job["name"], "stage": stage and stage["key"], "title": stage and stage["title"],
                            "started": stage and stage.get("started"), "number": number, "total": len(job["stages"])})


@app.get("/api/models")
def model_choices():
    """The model choices for each stage (the first is the default) and whether ElevenLabs is set up."""
    return jsonify(models=pipeline.MODELS, defaults=pipeline.DEFAULT_MODELS, elevenlabs=bool(api_key()))


@app.get("/api/recordings")
def list_recordings():
    items = []
    for folder in sorted(DATA.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True):
        if not folder.is_dir() or not re.fullmatch(r"[a-z0-9_]+", folder.name):
            continue
        job = job_view(folder.name)
        states = [s["status"] for s in job["stages"]]
        decision = json.loads((folder / "decision.json").read_text()).get("final") if (folder / "decision.json").exists() else None
        info = json.loads((folder / "info.json").read_text()) if (folder / "info.json").exists() else {}
        items.append({"id": job["id"], "name": job["name"], "created": job["created"],
                      "status": "running" if job["running"] else "failed" if "failed" in states
                      else "done" if all(s in ("done", "skipped") for s in states) else "partial",
                      "stages_done": states.count("done"), "stages": len(states),
                      "doctor": decision["decision"] if decision else None,
                      "confidence": decision["confidence"] if decision else None,
                      "audio_s": info.get("audio_s"), "has_video": job["has_video"],
                      "models": models_text(pipeline.load_job(folder.name)),
                      "image": (folder / FILES["people"]).exists()})
    return jsonify(items)


@app.post("/api/recordings")
def upload():
    if running["id"]:
        return busy()
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="Choose a video or audio file first."), 400
    form = request.form
    start_s = float(form["start"]) if form.get("start") else None
    end_s = float(form["end"]) if form.get("end") else None
    language = form.get("language") or None
    models, keyterms, problem = read_settings(json.loads(form.get("models") or "{}"), form.get("keyterms"))
    if problem:
        return jsonify(error=problem), 400

    UPLOADS.mkdir(exist_ok=True)
    temporary = UPLOADS / f"upload_{datetime.now():%Y%m%d%H%M%S%f}"
    file.save(temporary)
    rec_id = pipeline.recording_id(file.filename, temporary, start_s, end_s, language, models, keyterms)
    safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(file.filename).name)
    media = UPLOADS / rec_id / safe_name
    media.parent.mkdir(exist_ok=True)
    temporary.replace(media)

    if not (DATA / rec_id / "job.json").exists():
        pipeline.save_job({"id": rec_id, "name": file.filename, "media": str(media), "start": start_s, "end": end_s,
                           "language": language, "models": models, "keyterms": keyterms, "created": pipeline.now(),
                           "has_video": pipeline.has_video(media), "stages": {}})
    if not start(rec_id, only_missing=True):
        return busy()
    return jsonify(id=rec_id)


@app.get("/api/recordings/<rec_id>")
def recording(rec_id):
    valid(rec_id)
    return jsonify(job=job_view(rec_id), results=results(rec_id))


@app.post("/api/recordings/<rec_id>/run")
def run_missing(rec_id):
    valid(rec_id)
    job = pipeline.load_job(rec_id)
    if not job.get("media") or not Path(job["media"]).exists():
        if not (DATA / rec_id / "turns.csv").exists():
            return jsonify(error="The original file of this recording is missing."), 400
    pipeline.save_job(job)                                   # recordings made on the command line get a job.json
    return jsonify(started=True) if start(rec_id, only_missing=True) else busy()


@app.post("/api/recordings/<rec_id>/rerun")
def rerun(rec_id):
    """Analyse the same file again with other models, as a new recording. What the slow models already made
    for this file is copied over (pipeline.reuse_results), so only what changed runs again."""
    valid(rec_id)
    if running["id"]:
        return busy()
    source = pipeline.load_job(rec_id)
    if not source.get("media") or not Path(source["media"]).exists():
        return jsonify(error="The original file of this recording is missing."), 400
    body = request.get_json(silent=True) or {}
    models, keyterms, problem = read_settings(body.get("models"), body.get("keyterms"))
    if problem:
        return jsonify(error=problem), 400
    new_id = pipeline.recording_id(source["name"], source["media"], source["start"], source["end"], source["language"],
                                   models, keyterms)
    if not (DATA / new_id / "job.json").exists():
        media = UPLOADS / new_id / Path(source["media"]).name
        media.parent.mkdir(parents=True, exist_ok=True)
        if not media.exists():
            os.link(source["media"], media)                # the same file on disk, no copy
        pipeline.save_job({**{k: source.get(k) for k in ("name", "start", "end", "language", "has_video")},
                           "id": new_id, "media": str(media), "models": models, "keyterms": keyterms,
                           "created": pipeline.now(), "made_from": rec_id, "stages": {}})
        pipeline.reuse_results(rec_id, new_id)
    if not start(new_id, only_missing=True):
        return busy()
    return jsonify(id=new_id)


@app.get("/api/recordings/<rec_id>/log/<stage>")
def log(rec_id, stage):
    valid(rec_id)
    if stage not in {s["key"] for s in pipeline.STAGES}:
        abort(404)
    path = DATA / rec_id / "logs" / f"{stage}.log"
    lines = path.read_text(errors="replace").splitlines() if path.exists() else ["(no log for this stage)"]
    return "\n".join(line.split("\r")[-1] for line in lines[-200:]), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/api/recordings/<rec_id>/file/<kind>")
def recording_file(rec_id, kind):
    valid(rec_id)
    path = DATA / rec_id / FILES.get(kind, "-")
    if not path.exists():
        abort(404)
    return send_file(path, conditional=True)


# ---------------------------------------------------------------- the website itself

@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def website(path):
    if not (FRONTEND / "index.html").exists():
        return "The website is not built yet: run `npm install && npm run build` in web/frontend (see web/README.md).", 500
    if path and (FRONTEND / path).is_file():
        return send_from_directory(FRONTEND, path)
    page = send_from_directory(FRONTEND, "index.html")
    page.headers["Cache-Control"] = "no-cache"      # always check for a new build (its scripts have new names)
    return page


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, threaded=True)
