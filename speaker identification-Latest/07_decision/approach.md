# Decision: who is the doctor

## Main method: an LLM reads the extracted features (`llm_decision.py`)

* **Feature cards:** one card per group: speech (S), medical terms (M), text and context (T) and
  video (V). Each line is a feature's plain-language meaning plus the value for each speaker. The
  medical card also lists the terms each speaker said. The video card also gives how sure the
  face-to-voice link is (from who is on screen, from mouth movement, and combined).
* **No hints:** the LLM never sees our expected directions or our own "doctor by this feature" votes.
* **Neutral labels:** the speakers are called X and Y. Every question is asked in both orders (X = A,
  then X = B), so a preference for the first-listed speaker cancels out.
* **Answer:** JSON with `doctor` (X or Y), `confidence` (50–100) and `reasons`.
* **Repeats:** 3 answers per order (temperature 0.7), so 6 per combination.
* **Combined result per combination:**
  * P(A is doctor) per answer = confidence/100 if it chose A, else 1 − confidence/100; averaged.
  * Decision: A if the average is above 0.5, otherwise B. Confidence = max(P, 1 − P).
  * Also reported: agreement (how many of the 6 answers agree) and whether the majority is the same
    in both orders.
* **LLM:** Sarvam `sarvam-105b-conversations` first; Grok later (needs an xAI API key).
* **Experiments** (`08_experiments/llm_combinations.py`): all 15 combinations of S, M, T and V, plus
  two comparisons: the LLM reading the whole transcript, and "the first speaker is the doctor".

## Possible later comparison: rule-based score (not coded)

1. Each feature, resampled over turns 1,000 times, gives *p* = the share of draws in which it says A
   is the doctor, using its expected direction.
2. Average *p* within each group, then across groups. Confidence = max(P, 1 − P).
3. With about 20 or more labelled recordings: learn feature weights by logistic regression, calibrate,
   and test leaving one recording out.

## Known limits

* An LLM's stated confidence is not calibrated. Agreement across repeats and orders is reported next
  to it for that reason.
* The LLM can misread numbers. In a test it said "Y has fewer pauses (18.071 vs 11.114)" when Y had
  more. Its reasons are saved with every answer in `results/llm_sarvam/calls_<recording>.jsonl`.
* With one labelled recording, each combination is simply right or wrong. Ranking the combinations
  needs more recordings.
