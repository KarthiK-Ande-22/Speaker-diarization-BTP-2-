"""Step 0 (optional, off by default) - Clean audio: remove background noise with the ElevenLabs Voice Isolator.

Input : a video or audio file (optionally only the part between --start and --end seconds)
Output: data/<recording_id>/
    audio_16k.wav         the original audio (16 kHz mono), the same file Step 1 makes
    audio_clean_16k.wav   the same audio with background noise removed

When audio_clean_16k.wav exists, the models of Step 1 (who spoke when) and Step 3 (transcript) listen to it.
Step 2 (pitch, pauses, speed) and the website's player keep the original audio, because noise removal can
change pitch and loudness. The cleaned audio must be as long as the original (otherwise the times of the
later steps would not match the original audio) and is not used if it is not.

The audio is sent to ElevenLabs (needs ELEVENLABS_API_KEY in .env).

Run (from the "speaker identification-Latest" folder):
    python 00_clean_audio/clean_audio.py "path/to/video.mp4" --start 11.2 --end 315 --id my_name
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_diarization"))
from diarize import DATA, extract_audio, media_path, recording_id  # noqa: E402
from elevenlabs_api import isolate  # noqa: E402

MAX_LENGTH_CHANGE_S = 0.1   # the cleaned audio may be at most this much longer or shorter than the original


def clean(media, rec_id=None, start=None, end=None, redo=False):
    import soundfile as sf

    media = media_path(media)
    rec_id = rec_id or recording_id(media)
    out = DATA / rec_id
    out.mkdir(parents=True, exist_ok=True)
    wav, cleaned = out / "audio_16k.wav", out / "audio_clean_16k.wav"
    if redo or not wav.exists():
        extract_audio(media, wav, start, end)
    if cleaned.exists() and not redo:
        print(f"Using the saved cleaned audio of {rec_id} (--redo to clean again)")
        return cleaned

    isolate(wav, cleaned)
    before, after = sf.info(str(wav)).duration, sf.info(str(cleaned)).duration
    print(f"  original {before:.2f} s, cleaned {after:.2f} s")
    if abs(after - before) > MAX_LENGTH_CHANGE_S:
        cleaned.unlink()
        raise SystemExit(f"The cleaned audio is {after - before:+.2f} s longer than the original, so its times would "
                         "not match the original: it is not used")
    print(f"Saved {cleaned.relative_to(ROOT)}: Steps 1 and 3 will listen to it")
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Step 0: remove background noise (ElevenLabs Voice Isolator)")
    parser.add_argument("media", help="video or audio file")
    parser.add_argument("--start", type=float, help="analyse from this second")
    parser.add_argument("--end", type=float, help="analyse up to this second")
    parser.add_argument("--id", help="name for the output folder (default: from the file name)")
    parser.add_argument("--redo", action="store_true", help="clean again even if a cleaned file is saved")
    args = parser.parse_args()
    clean(args.media, args.id, args.start, args.end, args.redo)


if __name__ == "__main__":
    main()
