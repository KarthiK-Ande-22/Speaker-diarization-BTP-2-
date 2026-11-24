"""Step 1 - Speaker diarization: who spoke when.

Two models to choose from (--model):
    pyannote   pyannote community-1 on this computer, then a voice re-check (the default)
    scribe     ElevenLabs Scribe in the cloud: one speaker per word (needs ELEVENLABS_API_KEY; the audio is
               sent to ElevenLabs). It never marks two people talking at once.

Input : a video or audio file (optionally only the part between --start and --end seconds). If Step 0 made
        a cleaned copy of the audio (audio_clean_16k.wav), the model listens to that one.
Output: data/<recording_id>/
    audio_16k.wav               the analysed audio (16 kHz mono). All times below are relative to it.
    pyannote.rttm               pyannote's own segments, overlapping speech kept (saved and reused)
    pyannote_exclusive.rttm     pyannote's one-speaker-at-a-time version
    voice_recheck.csv           every piece of speech whose voice was re-checked, and where it went
    elevenlabs_scribe.json      (scribe) Scribe's answer, also used by Steps 2 and 3 (saved and reused)
    diarization.rttm            the final segments (Speaker_A / Speaker_B); used by every later step
    diarization_exclusive.rttm  the same, but only one speaker at any moment (used for speech-to-text)
    turns.csv                   turns: one speaker's speech until the other person speaks (pauses
                                inside a turn are kept in it), and the seconds of each turn in which
                                the other person also talks
    info.json                   source file, cut window, model, talk time per speaker

Speaker A = the person who speaks first, Speaker B = the other one.

Voice re-check (after pyannote): pyannote can miss a short reply inside the other person's long turn,
especially with similar voices. Every segment is cut at its pauses; each piece of at least 1 s gets a
voice fingerprint (pyannote's own embedding model) and is moved to the other speaker if it is closer
to that speaker's average voice than to its own speaker's (two rounds).

Run (from the "speaker identification-Latest" folder):
    python 01_diarization/diarize.py "path/to/video.mp4" --start 11.2 --end 315
    python 01_diarization/diarize.py "path/to/video.mp4" --model scribe --language en
    python 01_diarization/diarize.py --all        # every recording listed in recordings.csv
pyannote's result is saved and reused (turns.csv is rebuilt from it each time); --redo runs pyannote again.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECORDINGS = ROOT / "recordings.csv"

MODEL = "pyannote/speaker-diarization-community-1"
NUM_SPEAKERS = 2
SAMPLE_RATE = 16_000
MIN_SEGMENT_S = 0.2   # shorter pyannote segments are fragments, not speech, and are dropped
# Voice re-check
RECHECK_MIN_PIECE_S = 1.0       # shorter pieces are too short for a reliable voice fingerprint
RECHECK_MIN_SILENCE_S = 0.2     # segments are cut into pieces at silences at least this long
SILENCE_BELOW_PEAK_DB = 25      # silence = 25 dB quieter than the recording's 99th-percentile loudness
# ElevenLabs Scribe
SCRIBE_JOIN_GAP_S = 0.5         # a speaker's words closer together than this form one segment

os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")   # no usage statistics sent to pyannote
warnings.filterwarnings("ignore", category=UserWarning)


def recording_id(path):
    """'My Visit (1).mp4' -> 'my_visit_1'."""
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(path).stem).strip("_").lower()[:60] or "recording"


def media_path(media):
    """Absolute path of the media file (relative paths are relative to this folder)."""
    media = Path(media).expanduser()
    return media if media.is_absolute() else (ROOT / media).resolve()


def analysis_audio(folder):
    """The audio the models listen to: the cleaned copy made by Step 0 if there is one, else the original."""
    cleaned = folder / "audio_clean_16k.wav"
    return cleaned if cleaned.exists() else folder / "audio_16k.wav"


def extract_audio(media, out, start=None, end=None):
    """Cut [start, end] from the file and save it as 16 kHz mono WAV (needs ffmpeg)."""
    cut = ["-ss", str(start)] if start else []
    length = ["-t", str(end - (start or 0))] if end else []
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cut, "-i", str(media), *length,
                    "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out)], check=True)


def run_pyannote(wav):
    """Returns (overlap-kept segments, exclusive segments), each a list of (label, start, end)."""
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline

    audio, sample_rate = sf.read(wav, dtype="float32")
    pipeline = Pipeline.from_pretrained(MODEL)   # uses the token saved by `huggingface-cli login`
    printed = {}

    def progress(step, artifact=None, file=None, total=None, completed=None):
        """pyannote calls this while it works: print each step once, and every 10 % of the long ones."""
        if total:
            tenth = min(10, int(10 * (completed or 0) / total))
            if printed.get(step) != tenth:
                printed[step] = tenth
                print(f"  {step}: {10 * tenth} %", flush=True)
        elif step not in printed:
            printed[step] = None
            print(f"  {step} ...", flush=True)

    output = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sample_rate},
                      num_speakers=NUM_SPEAKERS, hook=progress)

    def as_list(annotation):
        return [(label, round(seg.start, 3), round(seg.end, 3))
                for seg, _, label in annotation.itertracks(yield_label=True)]

    return as_list(output.speaker_diarization), as_list(output.exclusive_speaker_diarization)


def scribe_segments(words):
    """ElevenLabs Scribe words -> segments (speaker_id, start, end): a speaker's words are joined while the same
    speaker goes on with gaps shorter than SCRIBE_JOIN_GAP_S."""
    segments = []
    for w in words:
        if not w["speaker_id"]:
            continue
        if segments and segments[-1][0] == w["speaker_id"] and w["start"] - segments[-1][2] < SCRIBE_JOIN_GAP_S:
            segments[-1] = (w["speaker_id"], segments[-1][1], max(segments[-1][2], w["end"]))
        else:
            segments.append((w["speaker_id"], w["start"], w["end"]))
    return segments


def name_speakers(segments, *others):
    """Rename pyannote labels to A, B, ... in the order the speakers first talk."""
    order = list(dict.fromkeys(label for label, _, _ in sorted(segments, key=lambda s: s[1])))
    names = {label: "ABCDEFGH"[i] for i, label in enumerate(order)}
    return [[(names[label], start, end) for label, start, end in segs] for segs in (segments, *others)]


def write_rttm(segments, path, uri):
    with open(path, "w") as f:
        for speaker, start, end in sorted(segments, key=lambda s: s[1]):
            f.write(f"SPEAKER {uri} 1 {start:.3f} {end - start:.3f} <NA> <NA> Speaker_{speaker} <NA> <NA>\n")


def read_rttm(path):
    segments = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "SPEAKER":
            start, length = float(fields[3]), float(fields[4])
            segments.append((fields[7].removeprefix("Speaker_"), start, round(start + length, 3)))
    return sorted(segments, key=lambda s: s[1])


def overlap(start, end, segments):
    """Seconds of [start, end] covered by the given segments."""
    return sum(max(0.0, min(end, e) - max(start, s)) for _, s, e in segments)


def make_turns(segments):
    """A turn is one speaker's speech until the other person speaks (Heldner & Edlund 2010): a speaker's
    segments are joined across pauses unless the other person talks for at least MIN_SEGMENT_S in the
    pause. Segments shorter than MIN_SEGMENT_S are dropped first. Adds how many seconds of each turn
    the other person also talks."""
    segments = [s for s in segments if s[2] - s[1] >= MIN_SEGMENT_S]
    turns = []
    for speaker, start, end in sorted(segments, key=lambda s: s[1]):
        last = next((t for t in reversed(turns) if t["speaker"] == speaker), None)
        other = [s for s in segments if s[0] != speaker]
        other_spoke = last is not None and overlap(last["end"], start, other) >= MIN_SEGMENT_S
        if last is not None and not other_spoke:
            last["end"] = max(last["end"], end)
        else:
            turns.append({"speaker": speaker, "start": start, "end": end})
    turns.sort(key=lambda t: (t["start"], t["end"]))
    for number, turn in enumerate(turns, start=1):
        others = [s for s in segments if s[0] != turn["speaker"]]
        turn.update(turn=number, duration=round(turn["end"] - turn["start"], 3),
                    overlap_s=round(overlap(turn["start"], turn["end"], others), 3))
    return turns


def write_turns(turns, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["turn", "speaker", "start", "end", "duration", "overlap_s"])
        writer.writeheader()
        writer.writerows(turns)


def summary(info, turns):
    """Talk time per speaker, both-at-once time, and the first turns (to check by listening)."""
    lines = [f"\n=== {info['recording_id']}: {info['audio_s']:.1f} s of audio ==="]
    for speaker, talk in info["talk_s"].items():
        lines.append(f"  Speaker {speaker}: {info['turns'][speaker]:3d} turns, {talk:6.1f} s talking "
                     f"({100 * talk / sum(info['talk_s'].values()):.0f} %)")
    lines.append(f"  Both talking at once: {info['both_talking_s']:.1f} s")
    lines.append("  First turns: " + " | ".join(f"{t['speaker']} {t['start']:.1f}-{t['end']:.1f}" for t in turns[:8]))
    return "\n".join(lines)


def speech_pieces(audio, sample_rate, segments):
    """Cut every segment at silences of at least RECHECK_MIN_SILENCE_S. Returns pieces of at least
    RECHECK_MIN_PIECE_S that do not overlap the other speaker: (speaker, start, end)."""
    import numpy as np

    hop = sample_rate // 100                                       # 10 ms frames
    frames = audio[: len(audio) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)
    silent = db < np.percentile(db, 99) - SILENCE_BELOW_PEAK_DB
    pieces = []
    for speaker, start, end in segments:
        first, last = int(start * 100), min(int(end * 100), len(silent))
        run, cut_points = 0, [first]
        for i in range(first, last):
            run = run + 1 if silent[i] else 0
            if run == int(RECHECK_MIN_SILENCE_S * 100):            # a long enough silence: cut here
                cut_points.append(i)
        cut_points.append(last)
        for a, b in zip(cut_points, cut_points[1:]):
            loud = np.flatnonzero(~silent[a:b])
            if len(loud) == 0:
                continue
            p_start, p_end = (a + loud[0]) / 100, (a + loud[-1] + 1) / 100
            others = [s for s in segments if s[0] != speaker]
            if p_end - p_start >= RECHECK_MIN_PIECE_S and overlap(p_start, p_end, others) == 0:
                pieces.append((speaker, round(p_start, 2), round(p_end, 2)))
    return pieces


def recheck_voices(wav, segments):
    """Move pieces whose voice is closer to the other speaker. Returns (moves, all checked pieces)."""
    import numpy as np
    import soundfile as sf
    import torch
    from pyannote.audio import Inference, Model

    audio, sample_rate = sf.read(str(wav), dtype="float32")
    pieces = speech_pieces(audio, sample_rate, segments)
    model = Model.from_pretrained(MODEL, subfolder="embedding")    # pyannote's own voice model
    inference = Inference(model, window="whole")
    prints = []
    for _, a, b in pieces:
        e = inference({"waveform": torch.from_numpy(audio[int(a * sample_rate):int(b * sample_rate)]).unsqueeze(0),
                       "sample_rate": sample_rate})
        prints.append(e / np.linalg.norm(e))
    prints = np.array(prints)
    lengths = np.array([b - a for _, a, b in pieces])
    original = np.array([s for s, _, _ in pieces])
    labels = original.copy()
    speakers = sorted(set(original))
    for _ in range(2):                                             # two rounds: average voices, then reassign
        centres = np.array([(prints[labels == s] * lengths[labels == s, None]).sum(0) for s in speakers])
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        similarity = prints @ centres.T
        labels = np.array(speakers)[similarity.argmax(1)]
    checked = [{"start": a, "end": b, "pyannote": s, **{f"voice_similarity_{k}": round(float(similarity[i, j]), 3)
                                                         for j, k in enumerate(speakers)}, "after_check": labels[i]}
               for i, (s, a, b) in enumerate(pieces)]
    moves = [(c["start"], c["end"], c["pyannote"], c["after_check"]) for c in checked if c["after_check"] != c["pyannote"]]
    return moves, checked


def apply_moves(segments, moves):
    """Give each moved piece's time to the new speaker, cutting it out of the old speaker's segments."""
    for a, b, old, new in moves:
        kept = []
        for speaker, s, e in segments:
            if speaker == old and s < b and e > a:
                kept += [(speaker, s, a)] if s < a else []
                kept += [(speaker, b, e)] if e > b else []
            else:
                kept.append((speaker, s, e))
        segments = kept + [(new, a, b)]
    return sorted(segments, key=lambda s: s[1])


def diarize(media, rec_id=None, start=None, end=None, redo=False, model="pyannote", language=None, keyterms=()):
    """model: "pyannote" (on this computer) or "scribe" (ElevenLabs, in the cloud; language and keyterms are
    passed to it)."""
    media = media_path(media)
    rec_id = rec_id or recording_id(media)
    out = DATA / rec_id
    out.mkdir(parents=True, exist_ok=True)
    wav = out / "audio_16k.wav"
    if redo or not wav.exists():
        extract_audio(media, wav, start, end)
    import soundfile as sf
    info = {"recording_id": rec_id, "source": str(media), "start_in_source_s": start or 0.0, "end_in_source_s": end,
            "audio_s": round(sf.info(str(wav)).duration, 3), "cleaned_audio": analysis_audio(out) != wav}

    if model == "pyannote":
        raw, raw_exclusive = out / "pyannote.rttm", out / "pyannote_exclusive.rttm"
        if redo or not raw.exists():
            print(f"Diarizing {rec_id} ({info['audio_s']:.0f} s of audio) with {MODEL} on CPU ...", flush=True)
            began = time.time()
            segments, exclusive = name_speakers(*run_pyannote(str(analysis_audio(out))))
            write_rttm(segments, raw, rec_id)
            write_rttm(exclusive, raw_exclusive, rec_id)
            info["runtime_s"] = round(time.time() - began, 1)
        else:
            print(f"Using the saved pyannote result for {rec_id} (--redo to run pyannote again)")
        info["model"] = MODEL

        # Voice re-check, then the final segments used by every later step
        segments = [s for s in read_rttm(raw) if s[2] - s[1] >= MIN_SEGMENT_S]
        moves, checked = recheck_voices(analysis_audio(out), segments)
        segments = apply_moves(segments, moves)
        exclusive = apply_moves(read_rttm(raw_exclusive), moves)
        with open(out / "voice_recheck.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(checked[0]))
            writer.writeheader()
            writer.writerows(checked)
        info["voice_recheck"] = {"pieces_checked": len(checked), "pieces_moved": len(moves),
                                 "seconds_moved": round(sum(b - a for a, b, _, _ in moves), 2),
                                 "moved": [f"{a:.2f}-{b:.2f} {old}->{new}" for a, b, old, new in moves]}
        print(f"  Voice re-check: {len(checked)} pieces checked, {len(moves)} moved "
              f"({info['voice_recheck']['seconds_moved']} s): " + ", ".join(info["voice_recheck"]["moved"]))
    elif model == "scribe":
        from elevenlabs_api import SCRIBE_MODEL, scribe, spoken_words
        answer = scribe(out, language, keyterms)
        segments = name_speakers(scribe_segments(spoken_words(answer)))[0]
        exclusive = segments                        # one speaker per word: never two at once
        info.update(model=f"ElevenLabs {SCRIBE_MODEL}", runtime_s=answer.get("runtime_s"))
        (out / "voice_recheck.csv").unlink(missing_ok=True)
    else:
        raise SystemExit(f"Unknown model {model!r}: use pyannote or scribe")
    write_rttm(segments, out / "diarization.rttm", rec_id)
    write_rttm(exclusive, out / "diarization_exclusive.rttm", rec_id)

    turns = make_turns(segments)
    write_turns(turns, out / "turns.csv")
    speakers = sorted({t["speaker"] for t in turns})
    segments = [s for s in segments if s[2] - s[1] >= MIN_SEGMENT_S]
    seg_a = [s for s in segments if s[0] == "A"]
    info.update({
        "min_segment_s": MIN_SEGMENT_S, "speakers_found": len(speakers),
        "turns": {s: sum(t["speaker"] == s for t in turns) for s in speakers},
        "talk_s": {s: round(sum(t["duration"] for t in turns if t["speaker"] == s), 2) for s in speakers},
        "both_talking_s": round(sum(overlap(s, e, [x for x in segments if x[0] != "A"]) for _, s, e in seg_a), 2),
    })
    (out / "info.json").write_text(json.dumps(info, indent=2))

    print(summary(info, turns) + f"\n  Saved in {out.relative_to(ROOT)}/")
    if len(speakers) != NUM_SPEAKERS:
        print(f"  WARNING: {len(speakers)} speakers found, expected {NUM_SPEAKERS}")
    return out


def read_recordings():
    """Rows of recordings.csv; lines starting with # are skipped."""
    with open(RECORDINGS) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    number = lambda text: float(text) if text and text.strip() else None
    return [{**r, "start": number(r["start"]), "end": number(r["end"])} for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Step 1: speaker diarization (who spoke when)")
    parser.add_argument("media", nargs="?", help="video or audio file")
    parser.add_argument("--start", type=float, help="analyse from this second")
    parser.add_argument("--end", type=float, help="analyse up to this second")
    parser.add_argument("--id", help="name for the output folder (default: from the file name)")
    parser.add_argument("--all", action="store_true", help="every recording in recordings.csv")
    parser.add_argument("--redo", action="store_true", help="run pyannote again even if a result is saved")
    parser.add_argument("--model", choices=["pyannote", "scribe"], default="pyannote",
                        help="pyannote (this computer, default) or scribe (ElevenLabs cloud)")
    parser.add_argument("--language", help="language code for scribe, e.g. en or te (default: detect)")
    parser.add_argument("--keyterms", default="", help="scribe: comma-separated words to spell correctly")
    args = parser.parse_args()
    keyterms = [t.strip() for t in args.keyterms.split(",") if t.strip()]

    if args.all:
        for row in read_recordings():
            diarize(row["media_path"], row["recording_id"], row["start"], row["end"], args.redo, args.model,
                    args.language or row.get("language"), keyterms)
        return
    media = args.media or input("Path of the video/audio file: ").strip().strip("'\"")
    diarize(media, args.id, args.start, args.end, args.redo, args.model, args.language, keyterms)


if __name__ == "__main__":
    main()
