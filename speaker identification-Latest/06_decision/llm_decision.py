"""Step 6 - Decision by an LLM: give it the extracted features, ask who is the doctor.

Feature cards are built from the files of Steps 2, 4 and 5 (and, for one comparison, Step 3's
transcript). The LLM sees only measured values and what they mean, never our expected directions or
our own votes. The speakers are called X and Y; each question is asked in both orders (X = A, then
X = B), so a preference for "the first one listed" cancels out.

Used by 07_experiments/llm_combinations.py and the website. Needs SARVAM_API_KEY in the .env file of this folder;
the feature cards (and, for the transcript comparison, the transcript) are sent to Sarvam.
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_diarization"))
from diarize import DATA  # noqa: E402

SARVAM_MODEL = "sarvam-105b-conversations"
TEMPERATURE = 0.7          # > 0 so repeated questions can disagree; agreement is part of the confidence
MAX_TOKENS = 4000

GROUPS = {"S": "speech", "M": "medical terms", "V": "video"}

# What each feature means, in plain words (the LLM gets these, not our expected directions)
DESCRIPTIONS = {
    "speech": {
        "talk_share": "share of all talking time",
        "median_turn_s": "median length of a turn, in seconds (a turn lasts until the other person speaks)",
        "turn_length_cv": "how much turn lengths vary (standard deviation / mean)",
        "interruptions_per_min": "times per minute this person started talking (for at least 1 s) while the other was still talking",
        "initiative_share": "share of short-turn-then-longer-reply exchanges that this person started",
        "response_gap_s": "median silence before this person starts talking after the other stops, in seconds",
        "pitch_variation_st": "how much the voice pitch goes up and down, in semitones",
        "rising_end_share": "share of turns whose pitch rises at the end",
        "articulation_rate": "speaking speed in syllables per second, pauses removed",
        "pause_rate_per_min": "silent pauses of at least 0.25 s inside the person's own turns, per minute",
    },
    "medical terms": {
        "treatment_test_per_100w": "medication, dosage, test and treatment terms per 100 words",
        "symptom_history_statements_per_100w": "symptom, body part, history and condition terms said in statements, per 100 words",
        "symptom_questions_per_100w": "symptom, body part, history and condition terms said inside questions, per 100 words",
        "treatment_test_first_share": "share of the different medication/test/treatment terms this person mentioned first",
    },
    "video": {
        "medical_clothing_prob": "image model's probability that the person wears a white doctor's coat or medical scrubs (absolute values are not reliable; compare the two people)",
        "stethoscope_prob": "image model's probability that the person has a stethoscope around the neck",
        "notes_or_computer_prob": "image model's probability that the person holds a clipboard or notes or uses a computer",
        "patient_gown_or_bed_prob": "image model's probability that the person wears a hospital gown or lies on a hospital bed",
    },
}
FILES = {"speech": "speaker_features.csv", "medical terms": "medical_features.csv", "video": "video_features.csv"}

SYSTEM = """You are helping with a research experiment on recorded medical consultations.
Each consultation has exactly two people: one is the doctor and the other is the patient.
Software measured how each person speaks, what they say and how they look. The measurements can contain errors.
The speakers are called X and Y. These labels are random: X is not necessarily the person who spoke first.
Decide which speaker is the doctor, using only the information given.
Answer with JSON only, no other text:
{"doctor": "X" or "Y", "confidence": a number from 50 to 100 (how sure you are), "reasons": ["short reason", "..."]}"""


def load_env():
    """Read KEY=value lines from the .env file of this folder (the key never leaves this machine except to Sarvam)."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def read_table(rec_id, group):
    with open(DATA / rec_id / FILES[group]) as f:
        return {r["feature"]: (r["A"], r["B"]) for r in csv.DictReader(f)}


def number(text):
    return "not available" if text in ("", None) else f"{float(text):.3f}".rstrip("0").rstrip(".")


def card(rec_id, group, x):
    """One group's measurements as plain text. x = the real speaker ("A" or "B") called X."""
    y = "B" if x == "A" else "A"
    values = read_table(rec_id, group)
    pick = lambda pair, who: pair[0] if who == "A" else pair[1]
    lines = [f"## {group.capitalize()}"]
    for feature, meaning in DESCRIPTIONS[group].items():
        if feature in values:
            lines.append(f"- {meaning}: X {number(pick(values[feature], x))}, Y {number(pick(values[feature], y))}")
    if group == "medical terms":
        with open(DATA / rec_id / "medical_terms.csv") as f:
            terms = list(csv.DictReader(f))
        for label, who in (("X", x), ("Y", y)):
            said = [f"{t['term']} ({t['type']}{', in a question' if t['question'] == '1' else ''})" for t in terms if t["speaker"] == who]
            lines.append(f"- medical terms said by {label}: {'; '.join(said) or 'none'}")
    if group == "video":
        info = json.loads((DATA / rec_id / "video_info.json").read_text())
        link = info["link"]
        lines.append("- how sure the software is that the faces were matched to the right voices (0.5 = no idea, 1 = certain): "
                     f"from who is on screen while each person talks {link['on_screen']['confidence']}, "
                     f"from mouth movement {link['mouth']['confidence']}, combined {link['both']['confidence']}")
    return "\n".join(lines)


def transcript_card(rec_id, x):
    with open(DATA / rec_id / "transcript.csv") as f:
        turns = [t for t in csv.DictReader(f) if t["text"]]
    name = {x: "X", ("B" if x == "A" else "A"): "Y"}
    return "## Transcript (automatic speech-to-text)\n" + "\n".join(f"{name[t['speaker']]}: {t['text']}" for t in turns)


def prompt(rec_id, letters, x):
    """letters: group letters such as "SMT"; "transcript" for the transcript comparison."""
    if letters == "transcript":
        body = transcript_card(rec_id, x)
    else:
        body = "\n\n".join(card(rec_id, GROUPS[g], x) for g in letters)
    return f"Measurements for speakers X and Y:\n\n{body}\n\nWhich speaker is the doctor? Answer with the JSON only."


def ask_sarvam(user_text):
    """One question to Sarvam. Returns (answer dict or None, raw text, seconds)."""
    from sarvamai import SarvamAI

    load_env()
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise SystemExit("SARVAM_API_KEY is missing: put it in the .env file of this folder")
    began = time.time()
    response = SarvamAI(api_subscription_key=key).chat.completions(
        model=SARVAM_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}])
    raw = response.choices[0].message.content or ""
    return parse(raw), raw, round(time.time() - began, 1)


def ask_combination(rec_id, letters, repeats=3):
    """Ask in both orders (X = A, then X = B), `repeats` times each. Each call records the chosen doctor as A or B."""
    calls = []
    for x in ("A", "B"):
        y = "B" if x == "A" else "A"
        for repeat in range(1, repeats + 1):
            answer, raw, seconds = None, "", 0.0
            for _ in range(3):                                   # retry if the answer can't be read
                answer, raw, seconds = ask_sarvam(prompt(rec_id, letters, x))
                if answer:
                    break
            calls.append({"combination": letters, "x_is": x, "repeat": repeat, "model": SARVAM_MODEL, "answer": answer,
                          "doctor": None if answer is None else (x if answer["doctor"] == "X" else y),
                          "seconds": seconds, "raw": raw})
    return calls


def real_names(text, x):
    """Turn the LLM's X / Y back into Speaker A / B (x = the real speaker that was called X)."""
    y = "B" if x == "A" else "A"
    return re.sub(r"\b([XY])\b", lambda m: f"Speaker {x if m.group(1) == 'X' else y}", text)


def combine(calls):
    """One result from several answers: average P(A is doctor), agreement, order check, and the reasons of the
    most confident answer that agrees with the decision. None if no answer could be read."""
    answers = [c for c in calls if c["doctor"]]
    if not answers:
        return None
    p_a = [c["answer"]["confidence"] / 100 if c["doctor"] == "A" else 1 - c["answer"]["confidence"] / 100 for c in answers]
    mean = sum(p_a) / len(p_a)
    decision = "A" if mean > 0.5 else "B" if mean < 0.5 else "-"
    majority = lambda x: max("AB", key=lambda s: sum(c["doctor"] == s for c in answers if c["x_is"] == x))
    agreeing = sorted((c for c in answers if c["doctor"] == decision), key=lambda c: -c["answer"]["confidence"])
    return {"decision": decision, "confidence": round(max(mean, 1 - mean), 3),
            "agree": sum(c["doctor"] == decision for c in answers), "answers": len(answers),
            "agreement": f"{sum(c['doctor'] == decision for c in answers)}/{len(answers)}",
            "same_in_both_orders": majority("A") == majority("B"),
            "mean_stated_confidence": round(sum(c["answer"]["confidence"] for c in answers) / len(answers), 1),
            "reasons": [real_names(r, agreeing[0]["x_is"]) for r in agreeing[0]["answer"].get("reasons", [])] if agreeing else []}


def decide_recording(rec_id, repeats=3):
    """For the website: each available feature group alone, then all of them together. Saves decision.json
    (results) and decision_calls.jsonl (every question and answer) in data/<rec_id>/."""
    letters = [g for g in GROUPS if (DATA / rec_id / FILES[GROUPS[g]]).exists()]
    if not letters:
        raise SystemExit("No feature files found: run the feature steps first")
    combos = letters + (["".join(letters)] if len(letters) > 1 else [])
    results, all_calls = [], []
    for combo in combos:
        calls = ask_combination(rec_id, combo, repeats)
        all_calls += calls
        summary = combine(calls)
        if summary:
            results.append({"letters": combo, "groups": [GROUPS[g] for g in combo], **summary})
            print(f"{' + '.join(GROUPS[g] for g in combo):45s} doctor {summary['decision']}  confidence {summary['confidence']:.2f}  "
                  f"agreement {summary['agreement']}", flush=True)
    folder = DATA / rec_id
    (folder / "decision_calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in all_calls))
    final = next((r for r in results if r["letters"] == combos[-1]), None)
    decision = {"model": SARVAM_MODEL, "temperature": TEMPERATURE, "answers_per_order": repeats,
                "combinations": results, "final": final}
    (folder / "decision.json").write_text(json.dumps(decision, indent=2))
    if final:
        print(f"\nDoctor: Speaker {final['decision']} (confidence {final['confidence']:.2f}, {final['agreement']} answers agree)")
    return decision


def parse(raw):
    """The last JSON object with a "doctor" field (thinking text and code fences are skipped)."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    for candidate in reversed(re.findall(r"\{[^{}]*\"doctor\"[^{}]*\}", text, flags=re.S)):
        try:
            answer = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if str(answer.get("doctor", "")).strip().upper() in ("X", "Y"):
            answer["doctor"] = answer["doctor"].strip().upper()
            answer["confidence"] = float(answer.get("confidence", 50))
            return answer
    return None
