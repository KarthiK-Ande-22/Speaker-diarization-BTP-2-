# Speaker identification (latest): who is the Doctor, who is the Patient

The BTP pipeline. This is the only code in the `diarization` folder; the older code and results were deleted.
Each step is finished and checked before the next one starts. Every test uses only the MerloLab video
(`recordings.csv`).

| Step | Folder | What it does | Status |
|---|---|---|---|
| 0 | `00_clean_audio/` | optional, off by default: remove background noise (ElevenLabs Voice Isolator) | **coded** |
| 1 | `01_diarization/` | split the recording into Speaker A (speaks first) and Speaker B, keeping overlapping speech; voice re-check | **done** |
| 2 | `02_speech_features/` | the top 10 speech features (timing, pitch, speed, pauses) | **done** (see `feature_list.md`) |
| 3 | `03_transcription/` | speech to text with Whisper medium; every word given to A or B | **done** (MerloLab: 954 words, 7.8 min) |
| 4 | `04_medical_terms/` | medical terms per speaker with NER (d4data/biomedical-ner-all), features M1–M4 | **done** (see `feature_list.md`) |
| 5 | `05_video/` | who is on screen, linked to A/B; white coat, stethoscope, notes, hospital gown / bed (YuNet, SFace, CLIP) | **done** (see `feature_list.md`) |
| 6 | `06_decision/` | the LLM (Sarvam) reads the feature cards and says who is the doctor, with a confidence | **done** (`approach.md`, `llm_decision.py`) |
| 7 | `07_experiments/` | the LLM on all 7 combinations of speech / medical terms / video, plus a transcript comparison | **coded** (`llm_combinations.py`) |

## Model choices (local by default, ElevenLabs optional)

Every step runs on this computer by default. Four parts can use an ElevenLabs model instead (in the cloud: the
audio is sent to ElevenLabs; needs `ELEVENLABS_API_KEY` in `.env`). The code is in `01_diarization/elevenlabs_api.py`.

| Part | Default (this computer) | ElevenLabs choice | Command line |
|---|---|---|---|
| Step 0, clean audio | off | Voice Isolator | `python 00_clean_audio/clean_audio.py file.mp4` |
| Step 1, who spoke when | pyannote community-1 + voice re-check | Scribe (one speaker per word, at most 2 speakers) | `diarize.py ... --model scribe` |
| Step 3, transcript | Whisper medium | Scribe v2 Medical, optional key terms | `transcribe.py --model scribe --keyterms "a, b"` |
| Step 4, medical terms | d4data biomedical NER | Scribe's entity detection (medical category): drugs, conditions, injuries, procedures and tests, measurements; no symptoms, body parts or history | `medical_terms.py --model scribe` |

All Scribe choices share one request per recording (`elevenlabs_scribe.json`, model `scribe_v2_medical`, with entity
detection). Whichever model finds the speakers, Speaker A is whoever talks first, and the transcript's words and the
medical terms are always given to the speakers found in Step 1. On the website the choices are on the upload form, and a recording's page has "Run again with other
models", which makes a new recording from the same file and reuses the saved results that still apply.

## Website

Upload a recording and see every stage's output and the final decision in the browser:

```bash
python web/server/app.py        # then open http://127.0.0.1:5000  (build it once first: see web/README.md)
```

## Setup

```bash
cd "~/Documents/sem_7/BtP-2/diarization/speaker identification-Latest"
source .venv/bin/activate
```

* The `.venv` reuses the packages already installed for BtP-1, so it takes no extra disk space.
  On a new computer: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
* ffmpeg must be installed (`sudo apt install ffmpeg`).
* pyannote needs a Hugging Face token, saved once with `huggingface-cli login`.

## Step 1: diarization

```bash
python 01_diarization/diarize.py --all     # the recordings in recordings.csv (MerloLab)
python 01_diarization/diarize.py "file.mp4" --start 11.2 --end 315 --id my_name
```

* Model: `pyannote/speaker-diarization-community-1`, 2 speakers, on CPU. MerloLab (304 s) takes about 6 min.
* The audio is taken from the video itself (ffmpeg, 16 kHz mono). MerloLab uses 11.2–315 s, which
  cuts the narrator at the start and the end.
* pyannote's result is saved and reused. `turns.csv` is rebuilt from it in a second. `--redo` runs
  pyannote again.

Output in `data/<recording_id>/`:

| File | Contents |
|---|---|
| `audio_16k.wav` | the analysed audio; every time in the other files is measured from its start |
| `diarization.rttm` | pyannote's segments, overlapping speech kept (Speaker_A / Speaker_B) |
| `diarization_exclusive.rttm` | the same, with only one speaker at any moment (for speech-to-text later) |
| `turns.csv` | turn, speaker, start, end, duration, overlap_s (seconds the other person also talks) |
| `info.json` | source file, cut window, model, run time, turns and talk time per speaker |

Turns: a turn is one speaker's speech until the other person speaks. Silences inside it are pauses.
pyannote pieces shorter than 0.2 s are dropped first (it leaves a few fragments of 0.02–0.1 s), and the
other person must talk for at least 0.2 s in a pause to end a turn.

## Step 2: speech features

```bash
python 02_speech_features/speech_features.py
```

Computes the 10 features in `02_speech_features/feature_list.md` from `turns.csv` and `audio_16k.wav`
(Praat through parselmouth) and saves `turn_features.csv` and `speaker_features.csv` in `data/<recording_id>/`.

## Step 3: transcription

```bash
python 03_transcription/transcribe.py      # --redo runs Whisper again
```

Whisper medium (already downloaded in `~/.cache/whisper/medium.pt`) runs once on the whole recording
on CPU, with the time of every word. Each word goes to the speaker talking at its middle, then to that
speaker's turn. Output in `data/<recording_id>/`: `whisper.json` (Whisper's result, reused next time),
`words.csv`, `transcript.csv` and `transcript.txt` (readable).

## Step 4: medical terms (NER)

```bash
python 04_medical_terms/medical_terms.py
```

Runs `d4data/biomedical-ner-all` on every sentence (offline), keeps terms that contain a noun (NLTK tagger), and computes the 4 features in
`04_medical_terms/feature_list.md`. Output: `medical_terms.csv` (every term with its speaker, type and
model confidence) and `medical_features.csv` (M1–M4 for A, B and A − B).

## Step 5: video

```bash
python 05_video/video_features.py
```

Finds faces 10 times per second (YuNet), groups them into 2 people (SFace + k-means), links each person to
Speaker A or B (who is on screen while each one talks, plus mouth movement), and describes each person with
CLIP (white coat, stethoscope, notes, hospital gown / bed). Output: `video_faces.csv`, `video_people.jpg` (check the grouping by
eye), `video_info.json`, `video_features.csv`. Details in `05_video/feature_list.md`.

## Voice re-check (part of Step 1)

After pyannote, every segment is cut at its pauses. Each piece of at least 1 s gets a voice fingerprint (pyannote's own
embedding model) and is moved to the other speaker if it sounds closer to that speaker's average voice. pyannote's own
result stays in `pyannote.rttm`; the corrected result is `diarization.rttm`, and every checked piece is listed in
`voice_recheck.csv`. MerloLab: 71 pieces checked, 3 moved (all correct: patient replies inside the doctor's last turn).

## Step 6 and 7: LLM decision and experiments

```bash
python 07_experiments/llm_combinations.py     # needs SARVAM_API_KEY in .env (this folder); --redo asks again
```

Builds a feature card per group, asks Sarvam who is the doctor for every combination of groups (both speaker orders,
3 answers each), and writes `results/llm_sarvam/summary.md`. Every question and answer is kept in
`results/llm_sarvam/calls_<recording>.jsonl`. The features (and, for the transcript comparison, the transcript) are sent
to Sarvam. Details in `06_decision/approach.md`.
