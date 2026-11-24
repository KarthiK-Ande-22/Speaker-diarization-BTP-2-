# Video features: version 1

This is the specification `video_features.py` follows. All models are local and need no download:
YuNet and SFace in `06_video/models/`, CLIP in the Hugging Face cache.

## How it works

1. **Faces:** YuNet face detector on 10 frames per second, with frames enlarged from 320 to 640 px wide
   and a detector confidence of at least 0.6.
2. **People:** SFace face fingerprints, grouped into 2 people with k-means (only faces of at least
   30 px are used to find the groups). Two faces in one frame are always two different people.
   P1 = the person seen first.
3. **Linking people to Speakers A and B** (needs Step 1's one-speaker-at-a-time diarization):
   * **On-screen vote:** when only one person is visible, they get a vote for whoever is talking at
     that moment. Video editors usually show the person who is speaking.
   * **Mouth vote:** when both are visible, the person whose mouth area changes more since 0.1 s
     earlier (minus the change around the eyes, which cancels head movement) gets the vote.
   * The votes are divided by each speaker's number of frames. The link with more votes wins.
     Link confidence = the share of votes that agree, from 0.5 (no idea) to 1.
4. **Looks:** CLIP (zero-shot) once per second on a picture of each person: the face plus the body
   below it, stopping halfway to another face in the same frame. Each check is the probability of the
   "yes" descriptions against the "no" description, averaged over all pictures of that person. The
   descriptions were written before running and not tuned.

| Check | "yes" descriptions | "no" description |
|---|---|---|
| medical clothing | a person wearing a white doctor's coat; … wearing medical scrubs | … wearing everyday casual clothes |
| stethoscope | a person with a stethoscope around the neck | … with nothing around the neck |
| notes or computer | a person holding a clipboard or notes; … using a computer | … with empty hands |

## The 4 features (per speaker)

| # | Feature | Expected higher |
|---|---|---|
| V1 | `medical_clothing_prob` | doctor |
| V2 | `stethoscope_prob` | doctor |
| V3 | `notes_or_computer_prob` | doctor |
| V4 | `face_link_confidence` (the same for A and B: how much to trust V1–V3) | — |

## Saved outputs

* `video_faces.csv`: every face: time, person, box, detector score, fingerprint similarity, mouth and
  eye movement, who is talking, and the CLIP score of each picture.
* `video_people.jpg`: 8 pictures of each person, labelled with the speaker they were linked to. Look at
  it to check the grouping.
* `video_info.json`: votes and confidence of each linking signal, faces per person, and CLIP averages.
* `video_features.csv`: V1–V4 with columns A, B, A − B, the expected direction, and which speaker that
  direction says is the doctor.

Run: `python 06_video/video_features.py` (about 3.5 minutes for MerloLab, 1.6 GB memory).

## MerloLab result (A = doctor)

* **Faces:** 4,993 faces in 3,035 frames. The contact sheet shows the grouping is correct: P1 is always
  the doctor (white coat, stethoscope) and P2 is always the patient (purple top).
* **Link:** P1 = A and P2 = B, which is correct.
  * On-screen votes: confidence 0.98 (P1 on screen alone: 493 frames while A talked, 0 while B talked).
  * Mouth votes: 0.51, which is chance. At 240p the faces in two-person shots are too small to see lips move.
  * Both combined: 0.73. The mouth votes dilute the combined confidence.

| Feature | A (P1) | B (P2) | Doctor by this feature |
|---|---|---|---|
| V1 medical clothing | 0.997 | 0.663 | A |
| V2 stethoscope | 0.965 | 0.211 | A |
| V3 notes or computer | 0.967 | 0.912 | A (small difference) |
| V4 link confidence | 0.727 | 0.727 | — |

**Why the patient's absolute values are high:**
* In close-ups, the patient's plain purple V-neck top looks like medical scrubs to CLIP (0.87, against
  0.59 in wide shots).
* "Empty hands" is a weak "no" description, so the notes check gives about 0.9 to everyone.

So the absolute probabilities are not reliable; the difference between A and B is what carries
information. The stethoscope check separates the two best.

## Known limits

* At 240p the mouth vote doesn't work. It should help with sharper video.
* Audio-only recordings have no video features.
* It assumes exactly 2 people. A third person (for example a nurse) would be forced into one of the two groups.
* In real Indian outpatient clinics, doctors often don't wear a white coat or stethoscope, so V1–V3
  may say little there.
