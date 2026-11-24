"""Step 8 - Experiments: which combination of feature groups lets the LLM find the doctor?

Groups: S = speech, M = medical terms, T = text and context, V = video -> 15 combinations, plus one
comparison where the LLM reads the whole transcript instead of features. Each combination is asked
in both orders (X = A, then X = B), REPEATS times each, so 2 x REPEATS answers per combination.

Per combination:
    P(A is doctor) per answer = confidence/100 if it chose A, else 1 - confidence/100; averaged
    decision     = A if the average > 0.5, B if < 0.5
    confidence   = max(average, 1 - average)
    agreement    = share of answers that chose the decided speaker
    order check  = the majority answer is the same in both orders (no "first listed" effect)

Output in results/llm_sarvam/:
    calls_<recording>.jsonl   every question and answer (reused: questions already answered are not asked again)
    summary.csv, summary.md   one row per recording and combination

Run (from the "speaker identification-Latest" folder):
    python 08_experiments/llm_combinations.py           # --redo asks every question again
"""

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_diarization"))
sys.path.insert(0, str(ROOT / "07_decision"))
from diarize import read_recordings  # noqa: E402
import llm_decision as llm  # noqa: E402

REPEATS = 3
OUT = ROOT / "results" / "llm_sarvam"
COMBINATIONS = ["".join(c) for n in range(1, 5) for c in combinations("SMTV", n)] + ["transcript"]


def run_calls(rec_id, redo):
    """Ask every combination, order and repeat; keep earlier answers unless --redo."""
    path = OUT / f"calls_{rec_id}.jsonl"
    done = {} if redo or not path.exists() else {
        (c["combination"], c["x_is"], c["repeat"]): c for c in map(json.loads, path.read_text().splitlines())}
    with open(path, "w") as f:
        for combination in COMBINATIONS:
            for x in ("A", "B"):
                for repeat in range(1, REPEATS + 1):
                    call = done.get((combination, x, repeat))
                    if call is None or call["answer"] is None:
                        answer, raw = None, ""
                        for _ in range(3):                           # retry if the answer can't be read
                            answer, raw, seconds = llm.ask_sarvam(llm.prompt(rec_id, combination, x))
                            if answer:
                                break
                        y = "B" if x == "A" else "A"
                        call = {"combination": combination, "x_is": x, "repeat": repeat, "model": llm.SARVAM_MODEL,
                                "answer": answer, "doctor": None if answer is None else (x if answer["doctor"] == "X" else y),
                                "seconds": seconds, "raw": raw}
                        print(f"  {combination:10s} X={x} #{repeat}: doctor {call['doctor']} "
                              f"({answer['confidence'] if answer else '-'})", flush=True)
                    f.write(json.dumps(call) + "\n")
                    done[(combination, x, repeat)] = call
    return list(done.values())


def summarise(rec_id, truth, calls):
    rows = []
    for combination in COMBINATIONS:
        answers = [c for c in calls if c["combination"] == combination and c["doctor"]]
        if not answers:
            continue
        p_a = [c["answer"]["confidence"] / 100 if c["doctor"] == "A" else 1 - c["answer"]["confidence"] / 100 for c in answers]
        mean = sum(p_a) / len(p_a)
        decision = "A" if mean > 0.5 else "B" if mean < 0.5 else "-"
        majority = lambda x: max("AB", key=lambda s: sum(c["doctor"] == s for c in answers if c["x_is"] == x))
        rows.append({
            "recording": rec_id, "combination": combination,
            "groups": "transcript only" if combination == "transcript" else " + ".join(llm.GROUPS[g] for g in combination),
            "decision": decision, "correct": int(decision == truth), "confidence": round(max(mean, 1 - mean), 3),
            "agreement": f"{sum(c['doctor'] == decision for c in answers)}/{len(answers)}",
            "same_in_both_orders": int(majority("A") == majority("B")),
            "mean_stated_confidence": round(sum(c["answer"]["confidence"] for c in answers) / len(answers), 1),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="LLM role decision for every combination of feature groups")
    parser.add_argument("--redo", action="store_true", help="ask every question again")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in read_recordings():
        print(f"\n=== {rec['recording_id']} (true doctor: {rec['doctor']}) with {llm.SARVAM_MODEL} ===")
        rows += summarise(rec["recording_id"], rec["doctor"], run_calls(rec["recording_id"], args.redo))
        rows.append({"recording": rec["recording_id"], "combination": "baseline", "groups": "first speaker is the doctor (no LLM)",
                     "decision": "A", "correct": int(rec["doctor"] == "A"), "confidence": "", "agreement": "",
                     "same_in_both_orders": "", "mean_stated_confidence": ""})

    with open(OUT / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = "| Recording | Groups given to the LLM | Doctor chosen | Correct | Confidence | Agreement | Same in both orders | Mean stated confidence |"
    lines = [f"# LLM role decision ({llm.SARVAM_MODEL}, {REPEATS} answers per order, temperature {llm.TEMPERATURE})", "",
             header, "|" + "---|" * 8]
    for r in rows:
        lines.append(f"| {r['recording']} | {r['groups']} | {r['decision']} | {'yes' if r['correct'] else 'no'} | {r['confidence']} | "
                     f"{r['agreement']} | {'' if r['same_in_both_orders'] == '' else ('yes' if r['same_in_both_orders'] else 'no')} | "
                     f"{r['mean_stated_confidence']} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
