# Speech features: version 1 (top 10)

This is the specification the code will follow. The reasons behind each feature are in
`speech_features_research.md`. The first version uses only the 10 features most likely to separate the
doctor from the patient. The others are listed at the end and will be added once this works.

## Shared preparation (done once per recording)

1. Convert the recording to 16 kHz mono WAV (ffmpeg).
2. Diarization (Step 1) gives segments: speaker (A/B), start, end. Overlapping speech is kept.
3. **Turns** (made by Step 1, in `data/<id>/turns.csv`): a turn is one speaker's speech until the
   other person speaks. Silences inside it are pauses, not turn ends (Heldner & Edlund 2010).
   pyannote pieces shorter than 0.2 s are dropped first. The other person must talk for at least
   0.2 s in a pause to end the turn.
4. **Overlap:** times when A and B talk at the same moment. They are used only for interruptions, and
   cut out of the pitch and speed measurements.
5. Pitch and speed are measured with Praat through `parselmouth`, on each turn's audio.
6. **Output:** each feature gives a value for A, a value for B, and the difference **A − B**.

## The 10 features

### Conversation timing (6 features, from turn times)

For features 5 and 6, turns that lie completely inside the other person's turn are skipped
(short replies said while the other talks). Gaps are measured from the sound: from the last moment
the previous speaker is louder than their silence level to the first moment the next speaker is. Times
alone can't be used, because pyannote often gives the silence between two speakers to one of them.

| # | Feature | How it is computed | Expected |
|---|---|---|---|
| 1 | `talk_share` | speaker's total turn time ÷ (A's total + B's total) | doctor usually talks more overall (can flip during long history-taking) |
| 2 | `median_turn_s` | median of the speaker's turn durations, in seconds | patient (longer answers; the doctor's many short questions pull the median down) |
| 3 | `turn_length_cv` | standard deviation ÷ mean of the speaker's turn durations | doctor (short questions *and* long explanations) |
| 4 | `interruptions_per_min` | the speaker starts a turn of at least 1 s while the other person is still talking and keeps talking for at least 0.2 s more; count ÷ minutes | doctor (published: clinicians interrupt after a median of 11 s) |
| 5 | `initiative_share` | an "exchange start" is a turn of 5 s or less that the other person answers within 2 s with a turn at least twice as long; this speaker's exchange starts ÷ both speakers' exchange starts | doctor (asks, the patient answers) |
| 6 | `response_gap_s` | median gap (measured from the sound) between the other person stopping and this speaker starting, at each speaker change; negative means the speaker started before the other stopped | patient (thinks or recalls before answering) |

### Pitch (2 features)

Pitch is tracked every 10 ms in two passes:

* **Pass 1:** track with a 60–600 Hz range and find the 25th and 75th percentiles.
* **Pass 2:** track again with floor = 0.75 × 25th percentile and ceiling = 1.5 × 75th percentile.

Each voiced frame is converted to semitones: 12 × log2(pitch ÷ speaker's median pitch).

| # | Feature | How it is computed | Expected |
|---|---|---|---|
| 7 | `pitch_variation_st` | standard deviation of all the speaker's semitone values | patient (emotion, worry, pain) |
| 8 | `rising_end_share` | for each turn of at least 1 s, fit a straight line to the semitone values in the last 0.5 s (at least 10 voiced frames needed). The turn counts as "rising" if the line climbs 2 semitones or more. Rising turns ÷ turns measured | **decide from data**: doctors ask questions, but in our 2 earlier recordings the patient rose more |

### Speed and fluency (2 features; de Jong & Wempe method)

* **Loudness curve:** Praat intensity every 10 ms.
* **Silence level:** that speaker's 99th-percentile loudness − 25 dB.
* **Syllable:** a loudness peak above the silence level, at least 2 dB above the dip before it, at a
  moment when the pitch track is voiced.
* **Pause:** a stretch below the silence level lasting at least 0.25 s, inside one of the speaker's turns.

| # | Feature | How it is computed | Expected |
|---|---|---|---|
| 9 | `articulation_rate` | syllables ÷ (turn time − pause time), in syllables per second | doctor (fluent, practised phrasing) |
| 10 | `pause_rate_per_min` | number of pauses ÷ minutes of turn time | patient (hesitation, recalling symptoms) |

## Added later, once version 1 works

| Feature | Why it waits |
|---|---|
| `turns_per_min` | turns alternate, so both speakers get nearly the same value |
| `long_turn_share` | mostly repeats features 2 and 3 |
| `backchannels_per_min` | direction unclear; diarization often misses very short sounds |
| `speaks_last` | a single yes/no value; recordings are often cut before the end |
| `pitch_range_st` | mostly repeats feature 7 |
| `mean_pause_s` | was about equal for doctor and patient in our data |
| `loudness_variation_db` | direction unclear; depends on the microphone |
| `jitter_local`, `shimmer_local`, `hnr_db`, `cpps_db` | fine-grained voice-quality measures: useful only when the illness affects the voice, very sensitive to noise, and jitter went the wrong way in our data |
| Body sounds, laughter, who adapts to whom, eGeMAPS, emotion | version 2 (see `speech_features_research.md`) |

## Saved outputs

* `turn_features.csv`: one row per turn: speaker, start, end, overlap, first and last loud moment,
  median pitch, end rise in semitones, rising yes/no, syllables, pauses, speaking time, syllables per second.
* `speaker_features.csv`: one row per feature, with columns A, B, A − B, the expected direction, and
  which speaker that direction says is the doctor.

Run: `python 02_speech_features/speech_features.py` (under a second for MerloLab).

## Check of the syllable counter (MerloLab)

The detected syllables of each turn were compared with syllables counted from the old Whisper
transcript of the same video, over 32 turns. Correlation 0.99; the detector finds 75 % of the
syllables. This undercount is known for this method and is the same for both speakers, so A − B
is not affected.
