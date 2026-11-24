"""Step 4 - Medical terms with NER (definitions in feature_list.md in this folder).

Input : data/<id>/transcript.csv            (made by Step 3)
Output: data/<id>/medical_terms.csv          every medical term found: turn, speaker, sentence, term,
                                             type, side, said in a question?, model confidence
        data/<id>/medical_features.csv       the 4 features M1-M4: value for A, value for B, A - B

Model: d4data/biomedical-ner-all (DistilBERT, English, 43 entity types), run on each sentence, offline.
A term is kept only if one of its words is a noun in its sentence (NLTK part-of-speech tagger).

Run (from the "speaker identification-Latest" folder):
    python 04_medical_terms/medical_terms.py        # every recording in recordings.csv
"""

import csv
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")     # the model is already downloaded; never wait on the network
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_diarization"))
from diarize import DATA, read_recordings  # noqa: E402

NER_MODEL = "d4data/biomedical-ner-all"
MIN_SCORE = 0.5        # terms the model is less sure about are dropped
MIN_LETTERS = 3        # "1", "10", "Ok" and word pieces are not terms

# Entity types kept, grouped by who usually says them
DOCTOR_SIDE = {"Medication": "medication", "Dosage": "dosage", "Administration": "how to take",
               "Diagnostic_procedure": "test", "Therapeutic_procedure": "treatment"}
PATIENT_SIDE = {"Sign_symptom": "symptom", "Biological_structure": "body part", "History": "history",
                "Family_history": "family history", "Disease_disorder": "condition"}
STOPWORDS = set("""i me my we our you your he him his she her it its they them their this that these those
a an the and or but so to of in on at for with from by as is are was were be been being am have has had do
does did didn don doesn not no yes just really all some any very much more most other thing things""".split())

# Which role should have the HIGHER value
EXPECTED = {
    "treatment_test_per_100w": "doctor",              # M1
    "symptom_history_statements_per_100w": "patient",  # M2
    "symptom_questions_per_100w": "doctor",           # M3
    "treatment_test_first_share": "doctor",           # M4
}
SPEAKERS = ("A", "B")
OTHER = {"A": "B", "B": "A"}


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s.strip()]


def is_term(text):
    words = re.findall(r"[a-z]+", text.lower())
    return sum(len(w) for w in words) >= MIN_LETTERS and any(w not in STOPWORDS for w in words)


def contains_noun(term, sentence):
    """Medical entities are noun phrases. True if one of the term's words is a noun in this sentence
    ("fall" in "where would you fall" is a verb, "successful" is an adjective). Uses NLTK's tagger."""
    import nltk

    tokens = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", sentence)
    tags = [tag for _, tag in nltk.pos_tag(tokens)]
    words, lower = re.findall(r"[a-z]+|\d+", term.lower()), [t.lower() for t in tokens]
    for i in range(len(lower) - len(words) + 1):
        if lower[i:i + len(words)] == words:
            return any(tag.startswith("NN") for tag in tags[i:i + len(words)])
    return True     # the term's words were not found one by one in the sentence: keep it


def concept(text):
    """Simple name for "the same thing": lowercase, letters only, no plural s ("Medications" -> "medication")."""
    words = re.findall(r"[a-z]+", text.lower())
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words)


def find_terms(turns):
    from transformers import pipeline

    ner = pipeline("ner", model=NER_MODEL, aggregation_strategy="first")
    terms = []
    for t in turns:
        for number, sentence in enumerate(sentences(t["text"]), start=1):
            for e in ner(sentence):
                kind = e["entity_group"]
                side = "doctor" if kind in DOCTOR_SIDE else "patient" if kind in PATIENT_SIDE else None
                if side and e["score"] >= MIN_SCORE and is_term(e["word"]) and contains_noun(e["word"], sentence):
                    terms.append({"turn": t["turn"], "speaker": t["speaker"], "sentence": number,
                                  "term": e["word"], "type": {**DOCTOR_SIDE, **PATIENT_SIDE}[kind], "side": side,
                                  "question": int(sentence.endswith("?")), "score": round(float(e["score"]), 3),
                                  "sentence_text": sentence})
    return terms


def features(terms, words):
    first_by = {}                                   # doctor-side concept -> who said it first
    for term in terms:                              # terms are in time order (turn, then sentence)
        if term["side"] == "doctor":
            first_by.setdefault(concept(term["term"]), term["speaker"])
    result = {}
    for s in SPEAKERS:
        own = [x for x in terms if x["speaker"] == s]
        per_100w = lambda n: 100 * n / words[s] if words[s] else None
        result[s] = {
            "treatment_test_per_100w": per_100w(sum(x["side"] == "doctor" for x in own)),
            "symptom_history_statements_per_100w": per_100w(sum(x["side"] == "patient" and not x["question"] for x in own)),
            "symptom_questions_per_100w": per_100w(sum(x["side"] == "patient" and x["question"] for x in own)),
            "treatment_test_first_share": (sum(v == s for v in first_by.values()) / len(first_by)) if first_by else None,
        }
    return result, first_by


def points_to(feature, diff):
    """Which speaker this feature says is the doctor, using the expected direction."""
    if diff is None or diff == 0:
        return "-"
    higher = "A" if diff > 0 else "B"
    return higher if EXPECTED[feature] == "doctor" else OTHER[higher]


def extract(rec_id):
    folder = DATA / rec_id
    with open(folder / "transcript.csv") as f:
        turns = [{**r, "turn": int(r["turn"]), "words": int(r["words"])} for r in csv.DictReader(f)]
    words = {s: sum(t["words"] for t in turns if t["speaker"] == s) for s in SPEAKERS}

    terms = find_terms(turns)
    values, first_by = features(terms, words)

    with open(folder / "medical_terms.csv", "w", newline="") as f:
        fields = ["turn", "speaker", "sentence", "term", "type", "side", "question", "score", "sentence_text"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(terms)

    table = []
    for feature in EXPECTED:
        a, b = values["A"][feature], values["B"][feature]
        diff = a - b if a is not None and b is not None else None
        table.append({"feature": feature, "A": a, "B": b, "A_minus_B": diff,
                      "expected_higher": EXPECTED[feature], "doctor_by_this_feature": points_to(feature, diff)})
    with open(folder / "medical_features.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows([{k: round(v, 4) if isinstance(v, float) else ("" if v is None else v) for k, v in r.items()}
                          for r in table])

    print(f"\n=== {rec_id}: medical terms (words: A {words['A']}, B {words['B']}) ===")
    for s in SPEAKERS:
        print(f"\nSpeaker {s}:")
        for side in ("doctor", "patient"):
            found = [f"{x['term']} ({x['type']}{', asked' if x['question'] else ''})"
                     for x in terms if x["speaker"] == s and x["side"] == side]
            print(f"  {side}-side terms ({len(found)}): {'; '.join(found) or '-'}")
    print(f"\nTreatment/test concepts and who said them first: {first_by or '-'}")
    show = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"\n{'feature':38s}{'A':>8s}{'B':>8s}{'A-B':>8s}   expected higher   doctor by this feature")
    for r in table:
        print(f"{r['feature']:38s}{show(r['A']):>8s}{show(r['B']):>8s}{show(r['A_minus_B']):>8s}   "
              f"{r['expected_higher']:17s} {r['doctor_by_this_feature']}")
    print(f"\nSaved {folder.relative_to(ROOT)}/medical_terms.csv and medical_features.csv")


def main():
    for row in read_recordings():
        extract(row["recording_id"])


if __name__ == "__main__":
    main()
