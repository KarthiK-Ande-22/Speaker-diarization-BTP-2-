"""Step 5 - Video features: what each speaker looks like (definitions in feature_list.md in this folder).

Input : the video (recordings.csv) and data/<id>/diarization_exclusive.rttm   (made by Step 1)
Output: data/<id>/video_faces.csv      every face found (10 frames per second): time, person, box,
                                       mouth and eye movement, who is talking
        data/<id>/video_people.jpg     sample pictures of each person, to check the grouping by eye
        data/<id>/video_info.json      how the two people were linked to A and B, and how sure that is
        data/<id>/video_features.csv   V1-V4: value for A, value for B, A - B

1. Faces   : YuNet face detector on 10 frames per second (frames enlarged to 640 px wide).
2. People  : SFace face fingerprints grouped into 2 people (k-means). Two faces in one frame are
             always two different people.
3. Link    : when one person is on screen, they get a vote for whoever is talking at that moment;
             when both are on screen, the one whose mouth moves more (minus eye movement, which cancels
             head movement) gets the vote. Votes are divided by each speaker's number of frames.
4. Looks   : CLIP (zero-shot) on a picture of each person (face and body below it), once per second.
All models are local: YuNet and SFace in 05_video/models/, CLIP in the Hugging Face cache.

Run (from the "speaker identification-Latest" folder):
    python 05_video/video_features.py        # every recording in recordings.csv
"""

import csv
import json
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")     # CLIP is already downloaded; never wait on the network
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_diarization"))
from diarize import DATA, read_recordings, read_rttm  # noqa: E402

MODELS = Path(__file__).resolve().parent / "models"
FACE_DETECTOR = MODELS / "face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER = MODELS / "face_recognition_sface_2021dec.onnx"
CLIP_MODEL = "openai/clip-vit-base-patch32"

WIDTH = 640            # frames are enlarged to this width before looking for faces
FACE_FPS = 10          # frames per second searched for faces
MIN_FACE_SCORE = 0.6   # face detector confidence
MIN_FACE_PX = 30       # smaller faces (in the enlarged frame) are too blurry for fingerprints and CLIP
PATCH = (32, 16)       # mouth and eye areas are resized to this size before comparing frames

# CLIP descriptions: probability of the "yes" descriptions against the "no" descriptions.
# Written before running on any recording; not tuned.
PROMPTS = {
    "medical_clothing": (["a photo of a person wearing a white doctor's coat", "a photo of a person wearing medical scrubs"],
                         ["a photo of a person wearing everyday casual clothes"]),
    "stethoscope": (["a photo of a person with a stethoscope around the neck"],
                    ["a photo of a person with nothing around the neck"]),
    "notes_or_computer": (["a photo of a person holding a clipboard or notes", "a photo of a person using a computer"],
                          ["a photo of a person with empty hands"]),
    "patient_gown_or_bed": (["a photo of a person wearing a hospital gown", "a photo of a patient lying on a hospital bed"],
                            ["a photo of a person in everyday clothes sitting on a chair"]),
}
EXPECTED = {"medical_clothing": "doctor", "stethoscope": "doctor", "notes_or_computer": "doctor",
            "patient_gown_or_bed": "patient"}
SPEAKERS = ("A", "B")


# ---------------------------------------------------------------- 1. faces

def frames(video, start, end):
    """(time from the start of the analysed audio, frame number, enlarged frame), FACE_FPS per second."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / FACE_FPS))
    first, last = round((start or 0) * fps), round(end * fps) if end else None
    index = 0
    while last is None or index < last:
        ok, frame = capture.read()
        if not ok:
            break
        if index >= first and (index - first) % step == 0:
            height = round(frame.shape[0] * WIDTH / frame.shape[1])
            yield (index - first) / fps, (index - first) // step, cv2.resize(frame, (WIDTH, height), interpolation=cv2.INTER_CUBIC)
        index += 1
    capture.release()


def area(gray, center, half_w, half_h):
    """A small grey picture around `center` (mouth or eyes), resized to PATCH; None at the frame edge."""
    x0, x1 = int(center[0] - half_w), int(center[0] + half_w)
    y0, y1 = int(center[1] - half_h), int(center[1] + half_h)
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0] or x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return cv2.resize(gray[y0:y1, x0:x1], PATCH).astype(np.float32)


def body_crop(frame, x, y, w, h, left=0, right=None):
    """The face plus the body below it (where a coat, stethoscope or clipboard would be), as RGB.
    left/right: the picture stops halfway to another face in the same frame, so it shows one person."""
    right = frame.shape[1] if right is None else right
    x0, x1 = max(left, int(x + w / 2 - 1.6 * w)), min(right, int(x + w / 2 + 1.6 * w))
    y0, y1 = max(0, int(y - 0.3 * h)), min(frame.shape[0], int(y + 5 * h))
    return cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)


def find_faces(video, start, end):
    detector = recognizer = None
    faces = []
    for time, number, frame in frames(video, start, end):
        if number % (FACE_FPS * 30) == 0:
            print(f"  {time:.0f} s of video searched, {len(faces)} faces so far", flush=True)
        if detector is None:
            detector = cv2.FaceDetectorYN.create(str(FACE_DETECTOR), "", (frame.shape[1], frame.shape[0]), MIN_FACE_SCORE)
            recognizer = cv2.FaceRecognizerSF.create(str(FACE_RECOGNIZER), "")
        _, found = detector.detect(frame)
        found = found if found is not None else []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        centres = [float(f[0] + f[2] / 2) for f in found]
        for f, centre in zip(found, centres):
            x, y, w, h = (float(v) for v in f[:4])
            left = max([int((centre + c) / 2) for c in centres if c < centre], default=0)
            right = min([int((centre + c) / 2) for c in centres if c > centre], default=frame.shape[1])
            marks = f[4:14].reshape(5, 2)       # right eye, left eye, nose, right mouth corner, left mouth corner
            mouth_w = np.linalg.norm(marks[4] - marks[3])
            eyes_w = np.linalg.norm(marks[1] - marks[0])
            fingerprint = recognizer.feature(recognizer.alignCrop(frame, f)).flatten()
            faces.append({
                "time": round(time, 2), "frame": number, "x": x, "y": y, "w": w, "h": h, "score": float(f[14]),
                "fingerprint": fingerprint / np.linalg.norm(fingerprint),
                "mouth": area(gray, marks[3:5].mean(0), 0.8 * mouth_w, 0.6 * mouth_w),
                "eyes": area(gray, marks[0:2].mean(0), 0.9 * eyes_w, 0.35 * eyes_w),
                "crop": body_crop(frame, x, y, w, h, left, right) if number % FACE_FPS == 0 and w >= MIN_FACE_PX else None,
            })
    return faces


# ---------------------------------------------------------------- 2. people

def group_people(faces):
    """Two people with k-means on face fingerprints; two faces in one frame must be different people."""
    from sklearn.cluster import KMeans

    prints = np.array([f["fingerprint"] for f in faces])
    big = [i for i, f in enumerate(faces) if f["w"] >= MIN_FACE_PX]
    centres = KMeans(n_clusters=2, n_init=10, random_state=0).fit(prints[big]).cluster_centers_
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    similarity = prints @ centres.T
    person = similarity.argmax(1)
    by_frame = {}
    for i, f in enumerate(faces):
        by_frame.setdefault(f["frame"], []).append(i)
    for idx in by_frame.values():
        idx.sort(key=lambda i: -faces[i]["w"])
        for i in idx[2:]:                         # more than two faces: keep the two largest
            person[i] = -1
        if len(idx) >= 2 and person[idx[0]] == person[idx[1]]:
            margin = [similarity[i, person[i]] - similarity[i, 1 - person[i]] for i in idx[:2]]
            unsure = idx[int(np.argmin(margin))]
            person[unsure] = 1 - person[unsure]
    first_seen = [min((f["time"] for f, p in zip(faces, person) if p == k), default=1e9) for k in (0, 1)]
    order = np.argsort(first_seen)                # P1 = the person seen first
    for i, f in enumerate(faces):
        f["person"] = "" if person[i] < 0 else f"P{int(np.where(order == person[i])[0][0]) + 1}"
        f["similarity"] = round(float(similarity[i].max()), 3)
    return faces


# ---------------------------------------------------------------- 3. link people to A / B

def movement(faces):
    """Mouth movement minus eye movement between a face and the same person's face 0.1 s earlier."""
    previous = {}
    for f in sorted(faces, key=lambda f: f["frame"]):
        before = previous.get(f["person"])
        f["mouth_motion"] = f["eye_motion"] = f["relative_motion"] = None
        if (before is not None and before["frame"] == f["frame"] - 1 and f["mouth"] is not None
                and before["mouth"] is not None and f["eyes"] is not None and before["eyes"] is not None
                and abs(f["x"] - before["x"]) < 0.5 * f["w"] and 0.75 < f["w"] / before["w"] < 1.33):  # same shot
            f["mouth_motion"] = float(np.abs(f["mouth"] - before["mouth"]).mean() / 255)
            f["eye_motion"] = float(np.abs(f["eyes"] - before["eyes"]).mean() / 255)
            f["relative_motion"] = f["mouth_motion"] - f["eye_motion"]
        if f["person"]:
            previous[f["person"]] = f


def link_people(faces, exclusive):
    """Votes person x speaker from being on screen alone, and from mouth movement when both are on screen."""
    def talking(t):
        return next((s for s, a, b in exclusive if a <= t < b), None)

    votes = {"on_screen": np.zeros((2, 2)), "mouth": np.zeros((2, 2))}   # rows P1, P2; columns A, B
    by_frame = {}
    for f in faces:
        f["talking"] = talking(f["time"]) or ""
        if f["person"]:
            by_frame.setdefault(f["frame"], []).append(f)
    for shown in by_frame.values():
        speaker = shown[0]["talking"]
        if speaker not in SPEAKERS:
            continue
        column = SPEAKERS.index(speaker)
        if len(shown) == 1:
            votes["on_screen"][int(shown[0]["person"][1]) - 1, column] += 1
        elif all(f["relative_motion"] is not None for f in shown) and shown[0]["relative_motion"] != shown[1]["relative_motion"]:
            mover = max(shown, key=lambda f: f["relative_motion"])
            votes["mouth"][int(mover["person"][1]) - 1, column] += 1
    votes["both"] = votes["on_screen"] + votes["mouth"]

    def decide(v):
        """P1->A/P2->B or P1->B/P2->A, and the share of (per-speaker normalised) votes that agree."""
        rate = v / np.maximum(v.sum(axis=0, keepdims=True), 1)
        straight, swapped = rate[0, 0] + rate[1, 1], rate[0, 1] + rate[1, 0]
        total = straight + swapped
        mapping = {"P1": "A", "P2": "B"} if straight >= swapped else {"P1": "B", "P2": "A"}
        return mapping, (max(straight, swapped) / total if total else None)

    result = {name: {"votes P1,P2 x A,B": v.astype(int).tolist(), "mapping": decide(v)[0],
                     "confidence": None if decide(v)[1] is None else round(decide(v)[1], 3)} for name, v in votes.items()}
    return result["both"]["mapping"], result["both"]["confidence"], result


# ---------------------------------------------------------------- 4. looks (CLIP)

def describe(faces):
    """Average CLIP probability of each description, per person."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    texts = [t for yes, no in PROMPTS.values() for t in yes + no]
    with torch.no_grad():
        text = model.get_text_features(**processor(text=texts, return_tensors="pt", padding=True))
        text = text / text.norm(dim=-1, keepdim=True)
        scores = {}
        for person in ("P1", "P2"):
            shown = [f for f in faces if f["person"] == person and f["crop"] is not None and f["crop"].size]
            if not shown:
                continue
            for i in range(0, len(shown), 32):
                batch = shown[i:i + 32]
                image = model.get_image_features(**processor(images=[f["crop"] for f in batch], return_tensors="pt"))
                logits = model.logit_scale.exp() * (image / image.norm(dim=-1, keepdim=True)) @ text.T
                position = 0
                for name, (yes, no) in PROMPTS.items():
                    p = logits[:, position:position + len(yes) + len(no)].softmax(dim=-1)[:, :len(yes)].sum(dim=-1)
                    for f, value in zip(batch, p.tolist()):
                        f[f"clip_{name}"] = value          # saved per picture in video_faces.csv
                    position += len(yes) + len(no)
            scores[person] = {name: float(np.mean([f[f"clip_{name}"] for f in shown])) for name in PROMPTS} | {"pictures": len(shown)}
    return scores


def contact_sheet(faces, mapping, path):
    """8 pictures of each person in a row, labelled with the speaker they were linked to."""
    rows = []
    for person in ("P1", "P2"):
        crops = [f["crop"] for f in faces if f["person"] == person and f["crop"] is not None and f["crop"].size]
        if not crops:
            continue
        picks = [crops[int(i)] for i in np.linspace(0, len(crops) - 1, min(8, len(crops)))]
        tiles = [cv2.resize(cv2.cvtColor(c, cv2.COLOR_RGB2BGR), (110, 160)) for c in picks]
        row = np.hstack(tiles + [np.zeros((160, 110, 3), np.uint8)] * (8 - len(tiles)))
        cv2.putText(row, f"{person} = Speaker {mapping.get(person, '?')}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(row)
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))


# ---------------------------------------------------------------- run and save

def extract(row):
    rec_id = row["recording_id"]
    folder = DATA / rec_id
    video = (ROOT / row["media_path"]).resolve()
    print(f"1/4 Finding faces ({FACE_FPS} frames per second) ...", flush=True)
    faces = find_faces(video, row["start"], row["end"])
    if not faces:
        print(f"{rec_id}: no faces found (audio only, or no one visible); video features not available")
        return
    print(f"2/4 Grouping {len(faces)} faces into 2 people ...", flush=True)
    group_people(faces)
    movement(faces)
    print("3/4 Linking each person to Speaker A or B ...", flush=True)
    mapping, confidence, link = link_people(faces, read_rttm(folder / "diarization_exclusive.rttm"))
    print("4/4 Describing each person with CLIP ...", flush=True)
    looks = describe(faces)
    contact_sheet(faces, mapping, folder / "video_people.jpg")

    with open(folder / "video_faces.csv", "w", newline="") as f:
        fields = ["time", "frame", "person", "x", "y", "w", "h", "score", "similarity",
                  "mouth_motion", "eye_motion", "relative_motion", "talking"] + [f"clip_{name}" for name in PROMPTS]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{k: (round(v, 4) if isinstance(v, float) else ("" if v is None else v)) for k, v in face.items()}
                          for face in faces])

    speaker_of = {p: mapping[p] for p in mapping}
    person_of = {s: p for p, s in speaker_of.items()}
    table = []
    for name in PROMPTS:
        a, b = (looks.get(person_of[s], {}).get(name) for s in SPEAKERS)
        diff = a - b if a is not None and b is not None else None
        higher = None if not diff else ("A" if diff > 0 else "B")
        doctor = "-" if higher is None else (higher if EXPECTED[name] == "doctor" else ("B" if higher == "A" else "A"))
        table.append({"feature": f"{name}_prob", "A": a, "B": b, "A_minus_B": diff,
                      "expected_higher": EXPECTED[name], "doctor_by_this_feature": doctor})
    table.append({"feature": "face_link_confidence", "A": confidence, "B": confidence, "A_minus_B": 0,
                  "expected_higher": "-", "doctor_by_this_feature": "-"})
    with open(folder / "video_features.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows([{k: round(v, 4) if isinstance(v, float) else ("" if v is None else v) for k, v in r.items()}
                          for r in table])

    counts = {p: sum(f["person"] == p for f in faces) for p in ("P1", "P2")}
    people_in_frame = {}
    for f in faces:
        if f["person"]:
            people_in_frame[f["frame"]] = people_in_frame.get(f["frame"], 0) + 1
    info = {"frames_searched_per_second": FACE_FPS, "faces_found": len(faces), "faces_per_person": counts,
            "frames_with_one_person": sum(n == 1 for n in people_in_frame.values()),
            "frames_with_both": sum(n == 2 for n in people_in_frame.values()),
            "link": link, "person_to_speaker": mapping, "clip": looks}
    (folder / "video_info.json").write_text(json.dumps(info, indent=2))

    print(f"\n=== {rec_id}: video ===")
    print(f"Faces found: {len(faces)} in {len({f['frame'] for f in faces})} frames; per person {counts}")
    for name, r in link.items():
        print(f"Link from {name:9s}: votes (rows P1,P2; columns A,B) {r['votes P1,P2 x A,B']} -> {r['mapping']}, confidence {r['confidence']}")
    for person, values in looks.items():
        print(f"{person} (= Speaker {mapping[person]}): " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                                                    for k, v in values.items()))
    show = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"\n{'feature':26s}{'A':>8s}{'B':>8s}{'A-B':>8s}   expected higher   doctor by this feature")
    for r in table:
        print(f"{r['feature']:26s}{show(r['A']):>8s}{show(r['B']):>8s}{show(r['A_minus_B']):>8s}   "
              f"{r['expected_higher']:17s} {r['doctor_by_this_feature']}")
    print(f"\nSaved {folder.relative_to(ROOT)}/video_features.csv, video_faces.csv, video_info.json, video_people.jpg")


def main():
    for row in read_recordings():
        extract(row)


if __name__ == "__main__":
    main()
