# Step 2 — Speech features for Doctor / Patient identification

Diarization gives two speakers, **A** (speaks first) and **B**. This step answers one question:
*using only how they speak (not what they say), which one sounds like the doctor?*

Text, medical terms and image come in later steps. Here we only describe **what can be measured from
speech**, why it should help, and how reliable each cue is. Coding comes after we agree on this list.

---

## 1. Four rules for every speech feature

1. **Compare A with B in the same recording.** We always have exactly one doctor and one patient, so
   the question is "which of the two is more doctor-like?", not "is this person a doctor?". We use the
   difference **A − B** for each feature. Microphone, room noise and language then affect both speakers
   equally and cancel out. We never use fixed thresholds such as "faster than 4 syllables/s means doctor".
2. **Measure each speaker against their own voice.** Pitch is measured in semitones from that speaker's
   own median pitch, with a pitch range set for each speaker. A man and a woman can then be compared fairly.
3. **Do not use identity traits.** Absolute pitch (mostly tells gender), estimated age and accent say
   who the person is, not their role. A female doctor with a male patient would break any rule built on
   them, and they would bias the system.
4. **Speech features do not depend on the language.** Pitch, timing and voice quality are measured the
   same way in English, Telugu and code-mixed speech, without a transcript. This is the main reason to
   use speech at all, because speech-to-text for Telugu is still weak (see the earlier Whisper results).

---

## 2. What can be measured from speech: nine groups

The **Expected** column says which role should score higher. **Evidence** says where the expectation
comes from: *literature* (published study), *our data* (the two earlier recordings, see Section 3),
or *hypothesis* (reasonable, not yet tested).

### Group 1 — Turn-taking and timing (from diarization timestamps only)

Role-recognition research on broadcast and meeting recordings found that **who talks when, and for how
long, is one of the strongest speech cues for roles**. Salamin & Vinciarelli reached over 85 % role
accuracy on about 50 h of broadcast material using turn-taking and prosody.

| Feature | How it is measured | Expected | Evidence |
|---|---|---|---|
| Talk share | speaker's speaking time ÷ total speaking time | unclear: doctor in explanation, patient in history-taking | literature (RIAS "verbal dominance") |
| Turns per minute | number of turns ÷ minutes | about equal (turns alternate) | hypothesis |
| Median turn length | median of turn durations (s) | patient (longer answers) | hypothesis |
| Turn length variation | standard deviation ÷ mean of turn durations | doctor (short questions *and* long explanations) | hypothesis |
| Long turns (monologues) | share of turns longer than 10 s | depends on the phase of the consultation | hypothesis |
| Interruptions | times the speaker starts talking while the other is still talking, per minute | **doctor** | literature: clinicians interrupted patients a median 11 s after the patient began (Singh Ospina et al., 2018/2019) |
| Backchannels | very short turns (< 1 s: "mm", "okay", "haan", "avunu") said inside the other's speech | unclear | hypothesis |
| Response gap | median silence before the speaker takes the turn | patient (thinks or recalls before answering) | hypothesis |
| Who speaks last | speaker of the final turn (closing advice, follow-up) | doctor | hypothesis |
| Initiative | exchanges a speaker starts (a short turn of 5 s or less, answered by a turn at least twice as long) | **doctor** | hypothesis |

Note: "who speaks first" is **not** a feature here, because A is defined as the first speaker. It stays
as a baseline experiment ("A is always the doctor").

Interruptions and backchannels need the **overlap-aware** diarization output. Step 1 must keep it, not
only the cleaned "one speaker at a time" version.

### Group 2 — Intonation and pitch (prosody)

| Feature | How it is measured | Expected | Evidence |
|---|---|---|---|
| Rising ends | share of turns whose pitch rises over the last ~0.5 s | doctor, if questions rise | **our data contradicted this** (patient higher in 2 of 2 recordings) |
| Pitch variation | standard deviation of pitch in semitones from the speaker's median | patient (emotion, worry, pain) | our data agree (2 of 2); literature is mixed, because depression and fatigue *reduce* variation |
| Pitch range | 95th − 5th percentile of pitch, in semitones | similar to pitch variation | hypothesis |
| Mean pitch | Hz | **do not use** (mostly tells gender) | — |

Why "rising ends" is weaker than expected: questions starting with a question word (what, how long,
ఎప్పటి నుంచి) usually end with a fall. Doctors in motivational-interviewing style use reflective
statements that also fall. Unsure patients often rise ("I think it was Monday?"). Telugu yes/no questions
use the *-ā* particle, and beginner course material says it rises, but we found no acoustic study of Telugu
question intonation. So we measure it and let the data decide the direction.

### Group 3 — Speaking speed and fluency

| Feature | How it is measured | Expected | Evidence |
|---|---|---|---|
| Articulation rate | syllables ÷ time spent speaking (pauses removed) | **doctor** (routine, practised phrasing) | our data agree (2 of 2) |
| Speech rate | syllables ÷ total turn time (pauses included) | doctor | hypothesis |
| Pause rate | silent pauses ≥ 0.25 s inside the speaker's own turns, per minute | patient | our data agree (1 of 1 usable recording) |
| Mean pause length | seconds | patient | our data: about equal |
| Hesitation sounds | "um", "aa", "ante", repeated words, restarts | patient (recalling symptoms) | hypothesis; mostly needs a word-for-word transcript, so it belongs to the text step |

Syllables are counted **without a transcript**: each syllable is a peak in loudness surrounded by dips,
and it must be voiced (de Jong & Wempe, 2009). The method was checked on Dutch. Telugu has a regular
syllable rhythm and should work, but we should hand-count syllables on a few Telugu turns to confirm.

### Group 4 — Loudness

| Feature | How it is measured | Expected | Evidence |
|---|---|---|---|
| Mean loudness | dB | **do not use** (depends on how close each person is to the microphone) | — |
| Loudness variation | standard deviation of dB within speech | unclear (emphasis while explaining) | hypothesis |

### Group 5 — Voice quality

These say whether a voice is rough, breathy or strained.

| Feature | How it is measured | Expected | Evidence |
|---|---|---|---|
| Jitter | cycle-to-cycle change in pitch period | patient, *only if the illness affects the voice* | **our data contradicted this** (doctor higher in 2 of 2) |
| Shimmer | cycle-to-cycle change in loudness | patient (same condition) | hypothesis |
| HNR (harmonics-to-noise ratio) | how clear versus noisy the voice is, in dB | doctor higher | hypothesis |
| CPPS (smoothed cepstral peak prominence) | clarity of voicing; more reliable than jitter or shimmer in normal conversation | doctor higher | hypothesis |

These help when the patient has a cold, cough, asthma, a throat problem, fever or fatigue. For other
complaints they mostly measure noise. They are also very sensitive to recording quality, so we use only
voiced parts of turns longer than 1 s.

### Group 6 — Non-speech sounds

| Sound | Expected | How it is detected |
|---|---|---|
| Cough, throat clearing, sniff, heavy breathing | patient | a pretrained sound classifier (Audio Spectrogram Transformer trained on AudioSet, which has "Cough", "Sniff", "Breathing" and "Sigh" classes) run over 1 s windows; each detected sound goes to the speaker who is active at that moment |
| Sigh, crying | patient | same classifier |
| Laughter | either speaker (rapport) | same classifier |

This is a strong cue when present (for example, the asthma check-up recording) and costs little.
However, the model is about 350 MB, which may download slowly on this network. Coughs are also
sometimes left out of the diarization turns, so they may need to be matched to the nearest turn.

### Group 7 — Emotion and arousal (optional)

A pretrained speech-emotion model gives arousal (calm to excited) and valence (negative to positive)
for each turn. **Expected:** the patient sounds more anxious or negative and the doctor calmer. Ambady
et al. (2002) also showed that listeners can rate how dominant a surgeon's voice sounds from short clips
with the words filtered out. So role-related qualities are audible in the voice alone.
**Limits:** these models are large and slow on our CPU, and they were trained mostly on English. Treat
this group as an optional later experiment.

### Group 8 — Accommodation: who adapts to whom (experimental, novel for the BTP)

During a conversation, people shift their pitch, loudness and speed toward their partner, and **the
lower-status person adapts more**. Gregory & Webster (1996) showed this in Larry King's interviews: King
adapted to high-status guests, and low-status guests adapted to King. Prosodic entrainment has also been
studied in psychiatrist–patient and therapist–client conversations. **Expected:** the patient adapts to
the doctor more than the doctor adapts to the patient.

**How it is measured:** for each pair of consecutive turns (X then Y), check whether Y's pitch, loudness
and speed move closer to X's previous turn. Compare how much B adapts to A with how much A adapts to B.
We found no study of this in ordinary doctor–patient consultations, so it would be a new contribution.
It also needs many turns, and short recordings will give noisy values.

### Group 9 — Standard feature sets and learned embeddings (for a trained classifier later)

| Option | What it is | When to use |
|---|---|---|
| **eGeMAPS** (openSMILE) | the standard set of 88 voice measures for voice and emotion research (Eyben et al., 2016); covers Groups 2–5 in one call | as a comparison baseline; a trained classifier on it needs about 20 or more labelled recordings |
| Speaker embeddings (ECAPA, WavLM) | numbers a neural network computes from a voice | mostly capture *who* is speaking, not the role; need a lot of labelled data, so skip for now |

---

## 3. What the earlier system measured (2 recordings, before the restart)

| Feature | Expected before | MerloLab (English) D / P | clip_2 (Telugu) D / P | Agrees? |
|---|---|---|---|---|
| Rising ends | doctor higher | 0.07 / 0.14 | 0.20 / 0.25 | ✗ ✗ |
| Articulation rate (syllables/s) | doctor higher | 5.06 / 4.42 | 4.52 / 3.94 | ✓ ✓ |
| Pauses per minute | patient higher | 10.2 / 16.8 | 0 / 0 | ✓ – |
| Pitch variation (semitones) | patient higher | 5.33 / 5.63 | 3.08 / 4.95 | ✓ ✓ |
| Jitter | patient higher | 0.020 / 0.018 | 0.025 / 0.020 | ✗ ✗ |

Two recordings cannot prove anything, but they already show that **the direction of a cue should not be
fixed in advance**. We measure every feature and let the experiments decide which ones help.

---

## 4. How the features will be measured (plan for the coding step)

* **Units:** a *turn* is a diarization segment, with segments of the same speaker that are less than
  0.3 s apart merged into one. Regions where both people talk at once are left out of pitch, voice and
  speed measures, but used for interruption counts.
* **Minimum amounts:** pitch-slope and voice measures use only turns of at least 1 s. A speaker needs at
  least about 20 s of speech in total, otherwise the feature is marked "not enough speech".
* **Pitch:** Praat through `parselmouth`, in two passes. The first pass finds the speaker's typical pitch
  and sets the pitch range to floor = 0.75 × 25th percentile and ceiling = 1.5 × 75th percentile
  (De Looze & Hirst). The second pass measures the pitch. Values are converted to semitones from the
  speaker's median.
* **Syllables and pauses:** the de Jong & Wempe peak-picking method, written in Python with `parselmouth`.
  A pause is a silence of at least 0.25 s inside the speaker's own turn.
* **Voice quality:** Praat jitter (local), shimmer (local), HNR and CPPS, on voiced parts only.
* **eGeMAPS:** `opensmile` (already installed).
* **Score for each feature:** A − B. Its sign is a vote for A or B being the doctor, and features are
  combined in the decision step.
* **Already installed in the venv:** parselmouth 0.4.7, opensmile 2.6.0, librosa 0.11, pyannote.audio
  4.0.3, speechbrain 1.0.3, transformers 4.57. Only the sound classifier for Group 6 needs a download.

---

## 5. Recommendation

**Version 1, to code first: the top 10 features** (all cheap and needing no transcript; exact
definitions in `feature_list.md`):

| Group | Features |
|---|---|
| 1 Turn-taking | talk share, median turn length, turn length variation, interruptions per minute, initiative, response gap |
| 2 Prosody | pitch variation, rising ends |
| 3 Speed and fluency | articulation rate, pause rate |

**Added once version 1 works:** turns per minute, long-turn share, backchannels, speaks last, pitch
range, mean pause length, loudness variation, voice quality (Group 5), then non-speech sounds
(Group 6), accommodation (Group 8), eGeMAPS (Group 9) and emotion (Group 7) if the CPU allows.

**Speech-only experiments, to combine later with text, medical terms and image:**

| ID | Features used |
|---|---|
| S0 | baseline: "A (first speaker) is the doctor" |
| S1 | conversation timing only (features 1–6) |
| S2 | pitch and speed only (features 7–10) |
| S3 | all 10 speech features |

---

## 6. Honest limitations

* **Speech alone will probably not be the strongest module.** In meeting recordings, *formal* roles
  (such as doctor and patient) were recognised best from the words used, while prosody worked best for
  *social* roles (Sapru & Valente, 2012). Recent doctor–patient systems also rely mainly on text. Speech
  is still valuable because it works when the transcript is poor (Telugu), and it adds evidence that the
  text does not have.
* **The phase of the consultation changes behaviour.** The patient talks more during history-taking and
  the doctor talks more during explanation and advice. Features computed separately for the first and
  second half of a consultation may help.
* **Data size.** With 2 labelled recordings, accuracy can only be 0 %, 50 % or 100 %. We need at least
  about 20 labelled consultations (English and Telugu) before accuracy numbers mean anything, and
  before any classifier can be trained.

---

## Sources

* Salamin & Vinciarelli — automatic role recognition from turn-taking and prosody (IEEE Trans. Multimedia, DOI 10.1109/TMM.2011.2173927); Salamin, PhD thesis, Glasgow 2013 — https://theses.gla.ac.uk/4367/
* Favre, Salamin, Dines, Vinciarelli — role recognition using social affiliation networks — https://publications.idiap.ch/attachments/reports/2008/sfavre-idiap-rr-08-64.pdf
* Sapru & Valente, "Automatic Speaker Role Labeling in AMI Meetings: Recognition of Formal and Social Roles", ICASSP 2012 — https://infoscience.epfl.ch/record/192681
* Laskowski, Ostendorf & Schultz, "Modeling Vocal Interaction for Text-Independent Participant Characterization in Multi-Party Conversation", SIGdial 2008 — https://aclanthology.org/W08-0124/
* Singh Ospina et al., "Eliciting the Patient's Agenda", J Gen Intern Med 34:36–40 (online 2018) — https://rd.springer.com/content/pdf/10.1007/s11606-018-4540-5.pdf
* Roter — RIAS and physician verbal dominance — https://www.bumc.bu.edu/healthliteracyconference/files/2009/10/roter-slides.pdf
* Ambady et al., "Surgeons' tone of voice: a clue to malpractice history", Surgery 2002 — https://pubmed.ncbi.nlm.nih.gov/12110787/
* Gregory & Webster, J Pers Soc Psychol 70(6):1231–1240, 1996 (vocal accommodation and status, Larry King interviews) — https://typeset.io/pdf/a-nonverbal-signal-in-voices-of-interview-partners-5e89ka20i0.pdf
* Prosodic entrainment and therapist empathy (counselling) — https://arxiv.org/pdf/2310.14181 ; entrainment in disordered speech — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4532918/
* de Jong & Wempe, "Praat script to detect syllable nuclei and measure speech rate automatically", Behav Res Methods 41(2), 2009 — https://dx.doi.org/10.3758/BRM.41.2.385
* Eyben et al., "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS)", IEEE TAFFC 2016 — https://research.utwente.nl/en/publications/the-geneva-minimalistic-acoustic-parameter-set-gemaps-for-voice-r/
* AudioSet classes in the AST model (Cough, Sniff, Breathing, Sigh, Laughter) — https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/blob/main/config.json
* Voice-based pain detection review (Bioengineering 2023) — https://mdpi.com/2306-5354/10/4/500
* Speaker role identification in clinical conversations (medRxiv 2025) — https://www.medrxiv.org/content/10.1101/2025.08.14.25332837.full.pdf ; Interspeech 2025 hybrid role diarization — https://www.isca-archive.org/interspeech_2025/kim25k_interspeech.html ; ASR-synchronized speaker-role diarization — https://arxiv.org/abs/2507.17765v3
* Telugu yes/no questions with *-ā* (beginner course material, not an acoustic study) — https://sublearn.com/ko/learn/te/grammar/chapter/asking-questions ; South Asian final rises (Khan 2024) — https://www.reed.edu/linguistics/khan/assets/Khan2024-UW.pdf
