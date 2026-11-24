"""Step 5 - Text and context features (definitions in feature_list.md in this folder).

Input : data/<id>/transcript.csv          (made by Step 3)
Output: data/<id>/text_sentences.csv       every sentence with its speaker and yes/no labels
                                           (question, instruction, advice, empathy, uncertainty,
                                           opening, closing) and its I / you / we word counts
        data/<id>/text_features.csv        the 10 features: value for A, value for B, A - B

English phrase patterns plus NLTK's part-of-speech tagger (to find sentences that start with a
command verb). The patterns were written from general doctor-patient communication guides, not from
the test recording. Telugu patterns come later.

Run (from the "speaker identification-Latest" folder):
    python 05_text_context/text_context.py        # every recording in recordings.csv
"""

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_diarization"))
from diarize import DATA, read_recordings  # noqa: E402

AUX = r"(?:do|does|did|are|is|was|were|am|can|could|will|would|have|has|had|should|may)"
SUBJECT = r"(?:you|your|it|he|she|they|we|i|there|that|this|the|anyone|anything)"
LEAD = r"(?:(?:so|and|okay|ok|well|now|then|but|alright|right),?\s+)*"   # "So, how ...", "And do you ..."

# A question: ends with "?", or starts like one (when Whisper left out the "?")
QUESTION_START = re.compile(rf"^{LEAD}(?:what|when|where|why|which|who|whom|whose|how|{AUX}\s+{SUBJECT})\b")

# An instruction: a phrase below, or a sentence that starts with a command verb ("Take ...", "Come back ...")
INSTRUCTION = [r"\byou (?:need|have|must|'ll need|will need|got) to\b", r"\byou must\b", r"\bmake sure\b",
               r"^(?:please\s+)?don't\s+\w+", r"\bplease\s+(?!\w+\?)\w+", r"\bremember to\b",
               r"\bi (?:want|'d like|would like) you to\b", r"\bit's (?:very )?important (?:that )?you\b"]
NOT_COMMANDS = {"thank", "let", "see", "look", "mind", "say", "mean", "guess", "like", "bless"}
LEAD_WORDS = {"so", "okay", "ok", "now", "then", "and", "just", "please", "well", "alright", "also", "but"}

# Advice and plans
ADVICE = [r"\byou (?:should|could try|might want to|may want to|ought to)\b",
          r"\bi(?:'d| would)? (?:recommend|suggest|advise)\b",
          r"\bit(?:'s| is| would be| might be) (?:important|better|best|a good idea|helpful|good) to\b",
          r"\bwe (?:can|could|will|should|'ll|are going to|'re going to|might)\b", r"\blet's\b",
          r"\bi(?:'ll| will| can| could|'m going to| am going to) (?:prescribe|give you|refer|order|start you|send you|write|check|arrange)\b",
          r"\bthere (?:are|is) (?:a lot of |some |many |several |different )?(?:options?|treatments?|medications?|things you can)\b"]

# Empathy and reassurance (NURSE statements: Naming, Understanding, Respecting, Supporting, Exploring)
EMPATHY = [r"\b(?:it|that) sounds like\b", r"^sounds like\b", r"\byou (?:seem|sound) (?:worried|upset|frustrated|scared|anxious|stressed)\b",
           r"\bi (?:can )?understand\b", r"\bi can (?:imagine|see)\b", r"\b(?:that|it|this) must (?:be|have been)\b",
           r"\byou(?:'ve| have) (?:done|been doing) (?:a )?(?:great|good|well|everything)\b", r"\byou(?:'re| are) doing (?:the best|a good|great|well|everything)\b",
           r"\bi(?:'m| am) (?:here|going to be here) (?:for|to help)\b", r"\b(?:we'll|we will) (?:work on|get through|figure) (?:this|it|that)\b",
           r"\btell me (?:a little )?more\b", r"\bdon't worry\b", r"\b(?:nothing|not much) to worry\b",
           r"\b(?:that's|that is|it's|it is) (?:very )?(?:normal|common|understandable)\b", r"\bi(?:'m| am) sorry (?:to hear|that|you)\b"]

# Uncertainty (hedges)
UNCERTAINTY = [r"\bi don't know\b", r"\bi(?:'m| am) not (?:sure|certain)\b", r"\bnot sure\b", r"\bmaybe\b",
               r"\bi guess\b", r"\bi don't think\b", r"\bi(?:'m| am) confused\b", r"\bi have no idea\b"]

# Opening and closing moves of a consultation
OPENING = [r"\bwhat (?:brings|brought) you\b", r"\bhow can i help\b", r"\bwhat can i do for you\b",
           r"\bwhat seems to be\b", r"\bwhat's (?:going on|the problem|wrong|bothering you)\b",
           r"\bhow (?:are|have) you (?:been )?(?:doing|feeling)\b", r"\bi'm (?:dr|doctor)\b", r"\bthis is (?:dr|doctor)\b"]
CLOSING = [r"\bcome back\b", r"\bfollow[- ]?up\b", r"\b(?:another|next|an|a follow-up) appointment\b",
           r"\bsee you (?:in|next|again|soon)\b", r"\bany (?:other |more )?questions\b", r"\b(?:is there )?anything else\b",
           r"\b(?:call|contact) (?:us|me|the office|the clinic)\b"]

I_WORDS = re.compile(r"\b(?:i|me|my|mine|myself)\b")
YOU_WORDS = re.compile(r"\b(?:you|your|yours|yourself|yourselves)\b")
WE_WORDS = re.compile(r"\b(?:we|us|our|ours|ourselves|let's)\b")

# Which role should have the HIGHER value
EXPECTED = {
    "question_share": "doctor", "instructions_per_100w": "doctor", "advice_plan_per_100w": "doctor",
    "empathy_reassurance_per_100w": "doctor", "uncertainty_per_100w": "patient",
    "i_per_100w": "patient", "you_per_100w": "doctor", "we_per_100w": "doctor",
    "qa_initiative_share": "doctor", "opening_closing_moves": "doctor",
}
SPEAKERS = ("A", "B")
OTHER = {"A": "B", "B": "A"}


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s.strip()]


def matches(patterns, lower):
    return any(re.search(p, lower) for p in patterns)


def starts_with_command(sentence):
    """True if the first real word (after "so", "okay", "please" ...) is a base-form verb: "Take ...", "Come back ..."."""
    import nltk

    # Split contractions ("I'm" -> "I", "'m") and write the pronoun as "I": the tagger reads "i'm" and
    # lowercase "i" as verbs. Other words are lowercased, because a capitalised "Come" is read as a name.
    tokens = ["I" if w.lower() == "i" else w.lower()
              for w in re.findall(r"[A-Za-z]+|'[A-Za-z]+", sentence.replace("’", "'"))]
    i = 0
    while i < len(tokens) and tokens[i].lower() in LEAD_WORDS:
        i += 1
    if i >= len(tokens) or tokens[i].lower() in NOT_COMMANDS:
        return False
    return nltk.pos_tag(tokens)[i][1] == "VB"


def label_sentence(sentence):
    lower = sentence.lower().replace("’", "'")
    question = sentence.endswith("?") or bool(QUESTION_START.search(lower))
    return {
        "question": int(question),
        "instruction": int(not question and (matches(INSTRUCTION, lower) or starts_with_command(sentence))),
        "advice": int(matches(ADVICE, lower)),
        "empathy": int(matches(EMPATHY, lower)),
        "uncertainty": int(matches(UNCERTAINTY, lower)),
        "opening": int(matches(OPENING, lower)),
        "closing": int(matches(CLOSING, lower)),
        "i_words": len(I_WORDS.findall(lower)),
        "you_words": len(YOU_WORDS.findall(lower)),
        "we_words": len(WE_WORDS.findall(lower)),
    }


def qa_exchanges(turns, rows):
    """Who starts question-answer exchanges: a turn with a question, followed by the other person's turn.
    Turns that lie completely inside the other person's turn (short replies) are skipped."""
    asks = {t["turn"]: any(r["question"] for r in rows if r["turn"] == t["turn"]) for t in turns}
    spoken = [t for t in turns if t["text"]]
    floor = [t for t in spoken if not any(o["speaker"] != t["speaker"] and o["start"] <= t["start"] and t["end"] <= o["end"]
                                          for o in spoken)]
    starts = {s: 0 for s in SPEAKERS}
    for prev, cur in zip(floor, floor[1:]):
        if cur["speaker"] != prev["speaker"] and asks[prev["turn"]]:
            starts[prev["speaker"]] += 1
    return starts


def extract(rec_id):
    folder = DATA / rec_id
    with open(folder / "transcript.csv") as f:
        turns = [{**r, "turn": int(r["turn"]), "words": int(r["words"]), "start": float(r["start"]), "end": float(r["end"])}
                 for r in csv.DictReader(f)]
    words = {s: sum(t["words"] for t in turns if t["speaker"] == s) for s in SPEAKERS}

    rows = []
    for t in turns:
        for number, sentence in enumerate(sentences(t["text"]), start=1):
            rows.append({"turn": t["turn"], "speaker": t["speaker"], "sentence": number, "text": sentence,
                         **label_sentence(sentence)})
    exchanges = qa_exchanges(turns, rows)

    count = lambda s, key: sum(r[key] for r in rows if r["speaker"] == s)
    per_100w = lambda s, key: 100 * count(s, key) / words[s] if words[s] else None
    share = lambda s, key: count(s, key) / (count("A", key) + count("B", key)) if count("A", key) + count("B", key) else None
    values = {s: {
        "question_share": share(s, "question"),
        "instructions_per_100w": per_100w(s, "instruction"),
        "advice_plan_per_100w": per_100w(s, "advice"),
        "empathy_reassurance_per_100w": per_100w(s, "empathy"),
        "uncertainty_per_100w": per_100w(s, "uncertainty"),
        "i_per_100w": per_100w(s, "i_words"),
        "you_per_100w": per_100w(s, "you_words"),
        "we_per_100w": per_100w(s, "we_words"),
        "qa_initiative_share": exchanges[s] / sum(exchanges.values()) if sum(exchanges.values()) else None,
        "opening_closing_moves": count(s, "opening") + count(s, "closing"),
    } for s in SPEAKERS}

    with open(folder / "text_sentences.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = []
    for feature in EXPECTED:
        a, b = values["A"][feature], values["B"][feature]
        diff = a - b if a is not None and b is not None else None
        higher = None if not diff else ("A" if diff > 0 else "B")
        doctor = "-" if higher is None else (higher if EXPECTED[feature] == "doctor" else OTHER[higher])
        table.append({"feature": feature, "A": a, "B": b, "A_minus_B": diff,
                      "expected_higher": EXPECTED[feature], "doctor_by_this_feature": doctor})
    with open(folder / "text_features.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows([{k: round(v, 4) if isinstance(v, float) else ("" if v is None else v) for k, v in r.items()}
                          for r in table])

    print(f"\n=== {rec_id}: text and context ({len(rows)} sentences; words A {words['A']}, B {words['B']}) ===")
    for key in ("question", "instruction", "advice", "empathy", "uncertainty", "opening", "closing"):
        print(f"\n{key}: A {count('A', key)}, B {count('B', key)}")
        for r in rows:
            if r[key]:
                print(f"   {r['speaker']} t{r['turn']:<3} {r['text'][:100]}")
    print(f"\nQuestion-answer exchanges started: {exchanges}")
    show = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"\n{'feature':30s}{'A':>8s}{'B':>8s}{'A-B':>8s}   expected higher   doctor by this feature")
    for r in table:
        print(f"{r['feature']:30s}{show(r['A']):>8s}{show(r['B']):>8s}{show(r['A_minus_B']):>8s}   "
              f"{r['expected_higher']:17s} {r['doctor_by_this_feature']}")
    print(f"\nSaved {folder.relative_to(ROOT)}/text_sentences.csv and text_features.csv")


def main():
    for row in read_recordings():
        extract(row["recording_id"])


if __name__ == "__main__":
    main()
