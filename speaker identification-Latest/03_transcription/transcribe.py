"""Step 3 - Transcription: what each speaker said.

Input : data/<id>/audio_16k.wav, diarization_exclusive.rttm, turns.csv   (made by Step 1)
Output: data/<id>/whisper.json     Whisper's result with the time of every word (reused next time)
        data/<id>/words.csv        one row per word: start, end, Whisper's confidence, speaker, turn
        data/<id>/transcript.csv   one row per turn: turn, speaker, start, end, number of words, text
        data/<id>/transcript.txt   the same, easy to read

Whisper medium (openai-whisper, ~/.cache/whisper/medium.pt) runs once on the whole recording with word
times. Each word is given to the speaker who is talking at the middle of the word (pyannote's
one-speaker-at-a-time output), and then to that speaker's turn.

Run (from the "speaker identification-Latest" folder):
    python 03_transcription/transcribe.py           # every recording in recordings.csv
Whisper's result is saved and reused; --redo runs Whisper again.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_diarization"))
from diarize import DATA, read_recordings, read_rttm  # noqa: E402

WHISPER_MODEL = "medium"


def run_whisper(wav, language):
    """Whole recording, with word times. Returns a list of words: text, start, end, probability."""
    import whisper

    model = whisper.load_model(WHISPER_MODEL, device="cpu")
    result = model.transcribe(str(wav), language=language or None, word_timestamps=True,
                              fp16=False, verbose=False)
    return {"model": WHISPER_MODEL, "language": result.get("language"),
            "words": [{"text": w["word"].strip(), "start": round(w["start"], 3), "end": round(w["end"], 3),
                       "probability": round(w["probability"], 3)}
                      for segment in result["segments"] for w in segment.get("words", []) if w["word"].strip()]}


def nearest(items, moment):
    """The item (with start/end) that contains the moment, or else the closest one."""
    return min(items, key=lambda x: 0.0 if x["start"] <= moment <= x["end"]
               else min(abs(moment - x["start"]), abs(moment - x["end"])))


def assign_words(words, exclusive, turns):
    """Speaker = who talks at the middle of the word; turn = that speaker's turn at (or nearest to) it."""
    segments = [{"speaker": s, "start": a, "end": b} for s, a, b in exclusive]
    rows = []
    for w in words:
        middle = (w["start"] + w["end"]) / 2
        speaker = nearest(segments, middle)["speaker"]
        turn = nearest([t for t in turns if t["speaker"] == speaker], middle)["turn"]
        rows.append({"word": w["text"], "start": w["start"], "end": w["end"],
                     "probability": w["probability"], "speaker": speaker, "turn": turn})
    return rows


def transcribe(rec_id, language=None, redo=False):
    folder = DATA / rec_id
    raw = folder / "whisper.json"
    if redo or not raw.exists():
        print(f"Transcribing {rec_id} with Whisper {WHISPER_MODEL} on CPU (several minutes) ...", flush=True)
        began = time.time()
        result = run_whisper(folder / "audio_16k.wav", language)
        result["runtime_s"] = round(time.time() - began, 1)
        raw.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    else:
        result = json.loads(raw.read_text())
        print(f"Using the saved Whisper result for {rec_id} (--redo to run Whisper again)")

    with open(folder / "turns.csv") as f:
        turns = [{"turn": int(r["turn"]), "speaker": r["speaker"], "start": float(r["start"]), "end": float(r["end"])}
                 for r in csv.DictReader(f)]
    words = assign_words(result["words"], read_rttm(folder / "diarization_exclusive.rttm"), turns)

    with open(folder / "words.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(words[0]))
        writer.writeheader()
        writer.writerows(words)

    lines = []
    for t in turns:
        text = " ".join(w["word"] for w in words if w["turn"] == t["turn"])
        lines.append({"turn": t["turn"], "speaker": t["speaker"], "start": t["start"], "end": t["end"],
                      "words": sum(w["turn"] == t["turn"] for w in words), "text": text})
    with open(folder / "transcript.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(lines[0]))
        writer.writeheader()
        writer.writerows(lines)
    (folder / "transcript.txt").write_text("\n\n".join(
        f"[{t['turn']:>2}] {t['speaker']}  {t['start']:6.1f}-{t['end']:6.1f} s\n{t['text']}" for t in lines if t["text"]) + "\n")

    count = {s: sum(w["speaker"] == s for w in words) for s in sorted({w["speaker"] for w in words})}
    print(f"\n=== {rec_id}: {len(words)} words ({', '.join(f'{s}: {n}' for s, n in count.items())}), "
          f"language {result.get('language')}, Whisper took {result.get('runtime_s', '?')} s ===")
    print(f"Empty turns (no words): {[t['turn'] for t in lines if not t['text']]}")
    print(f"Saved in {folder.relative_to(ROOT)}/ (transcript.txt is the readable version)")


def main():
    parser = argparse.ArgumentParser(description="Step 3: transcription (what each speaker said)")
    parser.add_argument("--redo", action="store_true", help="run Whisper again even if a result is saved")
    args = parser.parse_args()
    for row in read_recordings():
        transcribe(row["recording_id"], row.get("language"), args.redo)


if __name__ == "__main__":
    main()
