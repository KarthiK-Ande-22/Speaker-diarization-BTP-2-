# Website

Upload a video or audio file of a consultation and watch the whole pipeline run: who spoke when,
speech features, transcript, medical terms, video features, and finally which
speaker is the doctor, with a confidence and the model's reasons.

```
web/
├── server/
│   ├── app.py        Flask: the API and the built website (http://127.0.0.1:5000)
│   └── pipeline.py   runs the 7 stages one after another, each in its own process; the model choices (MODELS)
└── frontend/         React + TypeScript (Vite)
    └── src/
        ├── App.tsx                 pages: Home, Recordings, How it works, one recording
        ├── api.ts, types.ts        talking to the server, and the shape of its data
        ├── stages.ts               icon and description of each of the 7 stages
        ├── styles.css              all colours and layout
        └── components/
            ├── Navbar.tsx          logo and menu (Home, Recordings, How it works, New analysis)
            ├── UploadCard.tsx      the home page upload area: file, optional start/end, language and models
            ├── ModelChoices.tsx    the model of each part (default: this computer; ElevenLabs optional) and key terms
            ├── RecordingList.tsx   recordings as a simple list
            ├── HowItWorks.tsx      the 7 stages explained
            ├── RecordingView.tsx   one recording: stages on the left (with the models used and "Run again with
            │                       other models"), the chosen stage on the right
            ├── StageNav.tsx        the 7 numbered stages with status and time (click one to open it)
            ├── StagePanel.tsx      the chosen stage: live log while it runs, its output when done
            ├── Sections.tsx        the output of each stage
            ├── FeatureTable.tsx    the feature tables
            ├── AudioPlayer.tsx     play/pause, progress and time; the timeline and transcript jump to a moment
            ├── BottomBar.tsx       always at the bottom: what is running now
            └── ui.tsx              small shared pieces (card, speaker tag, status badge and icon, timer)
```

Layout: the home page is the upload area (large drop zone, start/end, language, Start button), with the
recent analyses as a simple list below it. On a recording page,
the stages are listed on the left (with the decision and an audio player above them); click a stage to
see it on the right: its live log while it runs, and its output (with a Log tab) once it is done. The
address remembers the stage (`#/recording/<id>/<stage>`).

Look: near-white page, white panels with thin grey lines, black buttons. Speaker A is blue and Speaker B is
rust (a small square marker before the name); green and red only for status. Text in IBM Plex Sans; numbers,
times, logs and feature names in IBM Plex Mono.
* Home, Recordings and How it works: rounded cards, a large centred upload area, list rows with icons, and
  icon cards for the stages.
* Recording page: a plain tool layout — numbered stages on the left, tables and numbered lists on the right.
The font and icons are installed with npm (@fontsource/ibm-plex-sans, @fontsource/ibm-plex-mono, lucide-react),
so nothing is loaded from the internet. The server sends the page with `Cache-Control: no-cache`, so a new
build shows up on the next reload.

## Run it

From the `speaker identification-Latest` folder:

```bash
source .venv/bin/activate
pip install flask                      # once
cd web/frontend && npm install && npm run build && cd ../..   # once, and after changing the website
python web/server/app.py               # then open http://127.0.0.1:5000
```

* Uploaded files go to `uploads/<id>/`. Each recording's outputs go to `data/<id>/`, the same place the
  command-line scripts use, so recordings made either way appear in the list.
* Uploading the same file with the same settings again reuses the saved results.
* Models: by default every stage runs on this computer (Clean audio is off). On the upload form, or later with "Run
  again with other models" on a recording's page, the user can choose ElevenLabs models for Clean audio (Voice
  Isolator), Who spoke when (Scribe), Transcript (Scribe v2 Medical, with optional key terms) and Medical terms
  (Scribe's entity detection).
  These send the audio to ElevenLabs and need `ELEVENLABS_API_KEY` in `.env`; without it they are greyed out. Each
  set of choices is its own recording, so the same file analysed with different models appears side by side in the
  list. "Run again" copies what the slow models already made (audio, pyannote, Whisper, Scribe) when it still applies.
* One recording is analysed at a time. The slow stages (finding the speakers, the transcript) each
  take a little longer than the recording itself on this laptop's CPU.
* The last stage needs `SARVAM_API_KEY` in `.env`. It sends the extracted features (numbers and the
  list of medical terms, not the recording) to Sarvam. It asks about each feature group alone and all
  groups together, in both speaker orders, 3 times each.

While changing the website: run `python web/server/app.py` and, in a second terminal,
`cd web/frontend && npm run dev` (it opens on http://localhost:5173 and sends API calls to Flask).
