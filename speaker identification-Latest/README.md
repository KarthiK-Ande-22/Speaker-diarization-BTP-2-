# Speaker identification: who is the Doctor, who is the Patient

BTP-2 pipeline. A recorded consultation (video or audio) is split into two speakers, Speaker A (whoever talks
first) and Speaker B. Features are extracted for each speaker, and an LLM (Sarvam) decides from the features
which speaker is the doctor, with a confidence. Every test so far uses only the MerloLab video
(`recordings.csv`).

| Step | Folder | What it does |
|---|---|---|
| 0 | `00_clean_audio/` | optional, off by default: remove background noise (ElevenLabs Voice Isolator) |
| 1 | `01_diarization/` | who spoke when: Speaker A and B, overlapping speech kept, voice re-check |
| 2 | `02_speech_features/` | 10 speech features per speaker (timing, pitch, speed, pauses) |
| 3 | `03_transcription/` | speech to text; every word given to A or B |
| 4 | `04_medical_terms/` | medical terms per speaker and 4 features (M1–M4) |
| 5 | `05_video/` | faces linked to A/B; white coat, stethoscope, notes, hospital gown / bed |
| 6 | `06_decision/` | the LLM reads the features and says who is the doctor |
| 7 | `07_experiments/` | the LLM on every combination of feature groups |
| | `web/` | website: upload a recording, run every step, see all outputs |

## Setup

```bash
cd ~/Documents/sem_7/BtP-2/diarization/"speaker identification-Latest"
source .venv/bin/activate
```

On a new computer:

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger_eng      # Step 4's noun check
sudo apt install ffmpeg                                       # audio extraction
huggingface-cli login                                         # Step 1 (pyannote)
```

* pyannote needs a Hugging Face token, and access to `pyannote/speaker-diarization-community-1` accepted on
  its model page.
* Steps 4 and 5 run offline. The first time, download their models (`d4data/biomedical-ner-all`, CLIP) by running
  them once with `HF_HUB_OFFLINE=0`. Whisper downloads its model by itself on the first run.
* `.env` in this folder (never committed): `SARVAM_API_KEY=...` for Steps 6–7, and optionally
  `ELEVENLABS_API_KEY=...` for the ElevenLabs models.
* Recordings are not in git. `recordings.csv` gives each recording's file (relative to this folder), the part to
  analyse (start, end in seconds), the language and which speaker is the doctor. The command-line scripts
  process every recording listed there.
* Outputs go to `data/<recording_id>/` and `results/` (not in git; running the steps makes them again).

## Model choices (this computer by default, ElevenLabs optional)

| Part | Default (this computer) | ElevenLabs choice (cloud) | Command line |
|---|---|---|---|
| Step 0, clean audio | off | Voice Isolator | `python 00_clean_audio/clean_audio.py file.mp4` |
| Step 1, who spoke when | pyannote community-1 + voice re-check | Scribe (one speaker per word, at most 2 speakers) | `diarize.py ... --model scribe` |
| Step 3, transcript | Whisper medium | Scribe v2 Medical, optional key terms | `transcribe.py --model scribe --keyterms "a, b"` |
| Step 4, medical terms | d4data biomedical NER | Scribe's entity detection: drugs, conditions, injuries, procedures and tests, measurements (no symptoms, body parts or history) | `medical_terms.py --model scribe` |

The ElevenLabs choices send the audio to ElevenLabs and need `ELEVENLABS_API_KEY`. All Scribe choices share one
request per recording (model `scribe_v2_medical`, saved as `elevenlabs_scribe.json`; code in
`01_diarization/elevenlabs_api.py`). Whichever model finds the speakers, Speaker A is whoever talks first, and
the words and medical terms are always given to the speakers found in Step 1. Scribe never marks two people
talking at once, so the interruption feature is 0 with it. With a third voice (MerloLab's narrator) Scribe may
split "narrator vs everyone else": cut such parts with start/end.

## Website

```bash
cd web/frontend && npm install && npm run build && cd ../..   # once, and after changing the website (needs Node.js)
python web/server/app.py                                      # then open http://127.0.0.1:5000
```

Upload a video or audio file (optionally a start/end, the language and the models of each part). Every step
runs in order, one recording at a time, and each step's output and log are shown as it finishes. A recording's
page also has "Run again with other models": a new recording from the same file, reusing the saved results
that still apply. Uploaded files go to `uploads/`, outputs to `data/<id>/`.

## Step 1: who spoke when

```bash
python 01_diarization/diarize.py --all     # every recording in recordings.csv
python 01_diarization/diarize.py "file.mp4" --start 11.2 --end 315 --id my_name
```

* pyannote community-1, 2 speakers, on CPU: about 7 min for MerloLab (304 s). Its result is saved and reused;
  `--redo` runs it again.
* The audio is taken from the file with ffmpeg (16 kHz mono). MerloLab uses 11.2–315 s, which cuts the
  narrator at the start and the end.
* Turns: a turn is one speaker's speech until the other person speaks; silences inside it are pauses. Pieces
  shorter than 0.2 s are dropped, and the other person must talk for at least 0.2 s to end a turn.
* Voice re-check: after pyannote, every segment is cut at its pauses. Each piece of at least 1 s gets a voice
  fingerprint (pyannote's embedding model) and is moved to the other speaker if it sounds closer to that
  speaker's average voice (two rounds). MerloLab: 71 pieces checked, 3 moved, all correctly.

Output in `data/<recording_id>/`:

| File | Contents |
|---|---|
| `audio_16k.wav` | the analysed audio; every time in the other files is measured from its start |
| `pyannote.rttm`, `pyannote_exclusive.rttm` | pyannote's own result, with and without overlapping speech |
| `voice_recheck.csv` | every piece the voice re-check looked at, and where it went |
| `elevenlabs_scribe.json` | Scribe's answer (only with an ElevenLabs choice) |
| `diarization.rttm` | the final segments (Speaker_A / Speaker_B), used by every later step |
| `diarization_exclusive.rttm` | the same, only one speaker at any moment (for the transcript and the video link) |
| `turns.csv` | turn, speaker, start, end, duration, overlap_s (seconds the other person also talks) |
| `info.json` | source file, cut, model, run time, turns and talk time per speaker |

## Step 2: speech features

```bash
python 02_speech_features/speech_features.py
```

From `turns.csv` and the original audio (Praat through parselmouth), for A and B: talk share, median turn
length, turn length variation, interruptions per minute, initiative share, response gap, pitch variation,
rising turn ends, articulation rate, pause rate. Output: `turn_features.csv`, `speaker_features.csv`.

## Step 3: transcript

```bash
python 03_transcription/transcribe.py      # --redo runs Whisper again; --model scribe for ElevenLabs
```

Whisper medium runs once on the whole recording on CPU, with the time of every word. It is slow (several minutes
for 5 min of audio) and needs about 4.5 GB of memory, so close other large programs first. Each word goes to
the speaker talking at its middle, then to that speaker's turn. Output: `whisper.json` (reused next time),
`words.csv`, `transcript.csv`, `transcript.txt`.

## Step 4: medical terms

```bash
python 04_medical_terms/medical_terms.py   # --model scribe for ElevenLabs entity detection
```

By default `d4data/biomedical-ner-all` runs on every sentence of the transcript, and a term is kept only if it
contains a noun (NLTK tagger). Medication, dosage, test and treatment terms are doctor-side; symptom, body part,
history and condition terms are patient-side. Features: M1 doctor-side terms per 100 words, M2 patient-side
terms said in statements per 100 words, M3 patient-side terms said in questions per 100 words, M4 share of the
doctor-side terms this speaker mentioned first. Output: `medical_terms.csv`, `medical_features.csv`.

## Step 5: video

```bash
python 05_video/video_features.py
```

Finds faces 10 times per second (YuNet), groups them into 2 people (SFace fingerprints + k-means), and links each
person to Speaker A or B (who is on screen while each one talks, plus mouth movement). CLIP then gives each person
the probability of a white coat or scrubs, a stethoscope, notes or a computer, and a hospital gown or bed, plus
how sure the face-to-voice link is. Models: `05_video/models/` (YuNet, SFace) and CLIP. Output:
`video_faces.csv`, `video_people.jpg` (check the grouping by eye), `video_info.json`, `video_features.csv`.

## Steps 6 and 7: LLM decision and experiments

```bash
python 07_experiments/llm_combinations.py     # needs SARVAM_API_KEY; --redo asks every question again
```

Each feature group becomes a card of measured values (no expected directions; the speakers are called X and Y).
Sarvam (`sarvam-105b-conversations`) is asked who is the doctor in both speaker orders, 3 times each; the
confidence is the average probability over the 6 answers. Step 7 asks this for all 7 combinations of speech,
medical terms and video, plus one where the LLM reads the whole transcript instead, and adds a "first speaker
is the doctor" baseline. Output in `results/llm_sarvam/`: `summary.csv`, `summary.md`, and every question and
answer in `calls_<recording>.jsonl` (answers are reused). The website uses Step 6 for each group alone and all
groups together. Only the features (and, for the transcript comparison, the transcript) are sent to Sarvam.
