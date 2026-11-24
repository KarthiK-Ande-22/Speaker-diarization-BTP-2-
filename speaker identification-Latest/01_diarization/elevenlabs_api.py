"""ElevenLabs cloud models. All optional: by default the pipeline runs only on this computer.

    scribe(folder, language, keyterms)  Scribe v2 Medical: every word with its time, which of (at most) 2 speakers
                                        said it, and the medical terms in its transcript (entity detection,
                                        category "phi": drugs, conditions, procedures and tests, ...). One request
                                        per recording: the answer is saved as elevenlabs_scribe.json and reused by
                                        Step 1 (who spoke when), Step 3 (transcript) and Step 4 (medical terms).
    isolate(wav, out)                   Voice Isolator: the same audio with background noise removed (Step 0).

Needs ELEVENLABS_API_KEY in the .env file of the "speaker identification-Latest" folder. The audio is sent to
ElevenLabs. API: https://elevenlabs.io/docs/api-reference/speech-to-text/convert and .../audio-isolation/convert
"""

import json
import math
import os
import re
import subprocess
import time

import requests

from diarize import ROOT, SAMPLE_RATE, analysis_audio

API = "https://api.elevenlabs.io/v1"
SCRIBE_MODEL = "scribe_v2_medical"
MAX_SPEAKERS = 2
ENTITIES = "phi"        # entity detection: the medical category (no extra price is listed for it)


def api_key():
    """ELEVENLABS_API_KEY from the environment or the .env file ("" if it is missing)."""
    if not os.environ.get("ELEVENLABS_API_KEY") and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("ELEVENLABS_API_KEY="):
                os.environ["ELEVENLABS_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
    return os.environ.get("ELEVENLABS_API_KEY", "")


def post(endpoint, files, data=None):
    key = api_key()
    if not key:
        raise SystemExit("ELEVENLABS_API_KEY is missing: put it in the .env file of this folder")
    response = requests.post(f"{API}/{endpoint}", headers={"xi-api-key": key}, files=files, data=data or {}, timeout=1800)
    if not response.ok:
        raise SystemExit(f"ElevenLabs {endpoint} failed ({response.status_code}): {response.text[:400]}")
    return response


def scribe(folder, language=None, keyterms=()):
    """Scribe's answer for this recording: asked once, then read from elevenlabs_scribe.json (asked again only
    if the audio, language or key terms are different)."""
    wav = analysis_audio(folder)
    request = {"model": SCRIBE_MODEL, "audio": wav.name, "language": language or None, "keyterms": sorted(keyterms),
               "entities": ENTITIES}
    saved = folder / "elevenlabs_scribe.json"
    if saved.exists():
        answer = json.loads(saved.read_text())
        if answer.get("request") == request:
            print("Using the saved ElevenLabs Scribe result", flush=True)
            return answer
    print(f"Sending {wav.name} to ElevenLabs Scribe ({SCRIBE_MODEL}, at most {MAX_SPEAKERS} speakers, "
          f"{len(keyterms)} key terms) ...", flush=True)
    data = {"model_id": SCRIBE_MODEL, "diarize": "true", "num_speakers": str(MAX_SPEAKERS),
            "tag_audio_events": "false", "timestamps_granularity": "word", "entity_detection": ENTITIES}
    if language:
        data["language_code"] = language
    if keyterms:
        data["keyterms"] = list(keyterms)                       # sent as one form field per term
    began = time.time()
    with open(wav, "rb") as f:
        answer = post("speech-to-text", {"file": (wav.name, f, "audio/wav")}, data).json()
    answer.update(request=request, runtime_s=round(time.time() - began, 1))
    saved.write_text(json.dumps(answer, indent=1, ensure_ascii=False))
    items = answer.get("words", [])
    print(f"  {sum(w['type'] == 'word' for w in items)} words, {len(answer.get('entities') or [])} medical terms, "
          f"speakers {sorted({w['speaker_id'] for w in items if w.get('speaker_id')})}, "
          f"language {answer.get('language_code')}, {answer['runtime_s']} s", flush=True)
    return answer


def spoken_words(answer):
    """The words (not spaces or sounds): text, start, end, speaker_id, probability (from Scribe's log-probability)."""
    return [{"text": w["text"].strip(), "start": round(w["start"], 3), "end": round(w["end"], 3),
             "speaker_id": w.get("speaker_id"), "probability": round(math.exp(w.get("logprob") or 0.0), 3)}
            for w in answer.get("words", []) if w["type"] == "word" and w.get("start") is not None and w["text"].strip()]


def medical_entities(answer):
    """The medical terms Scribe found in its transcript, each with its time and the sentence it is in:
    term, entity_type, start, end, sentence. (Scribe gives character positions in its transcript; the
    time comes from the words at those positions.)"""
    text, position, spans = answer.get("text", ""), 0, []
    for w in answer.get("words", []):                     # where each word is in the transcript text
        at = text.find(w["text"], position) if w["text"] else -1
        if at >= 0:
            position = at + len(w["text"])
            if w["type"] == "word" and w.get("start") is not None:
                spans.append((at, position, w))
    found = []
    for e in answer.get("entities") or []:
        words = [w for a, b, w in spans if a < e["end_char"] and b > e["start_char"]]
        if not words:
            continue
        begin = max((m.end() for m in re.finditer(r"[.?!]\s+", text[:e["start_char"]])), default=0)
        end = re.search(r"[.?!](\s|$)", text[e["end_char"]:])
        sentence = text[begin:e["end_char"] + (end.end() if end else len(text))].strip()
        found.append({"term": e["text"], "entity_type": e["entity_type"], "start": words[0]["start"],
                      "end": words[-1]["end"], "sentence": sentence})
    return found


def isolate(wav, out):
    """Send the audio to the Voice Isolator and save its answer as 16 kHz mono WAV (needs ffmpeg)."""
    print(f"Sending {wav.name} to the ElevenLabs Voice Isolator ...", flush=True)
    began = time.time()
    with open(wav, "rb") as f:
        response = post("audio-isolation", {"audio": (wav.name, f, "audio/wav")})
    download = out.with_suffix(".download")
    download.write_bytes(response.content)
    print(f"  received {len(response.content) / 1e6:.1f} MB ({response.headers.get('content-type')}) "
          f"in {time.time() - began:.0f} s", flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(download), "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-c:a", "pcm_s16le", str(out)], check=True)
    download.unlink()
