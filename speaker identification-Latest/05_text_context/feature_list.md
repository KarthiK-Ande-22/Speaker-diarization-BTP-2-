# Text and context features: version 1

This is the specification `text_context.py` follows.

## How sentences are labelled

1. **Input:** Step 3's `transcript.csv`. Each turn's text is split into sentences at `.`, `?` and `!`.
2. **Labels:** every sentence gets yes/no labels from English phrase patterns. The patterns sit at the
   top of `text_context.py`. They were written from general doctor–patient communication guides (for
   example the NURSE empathy statements), **before** the script was run on MerloLab, and were not
   changed to fit it. Only bugs were fixed.
3. **Commands:** a sentence that starts with a command verb ("Take…", "Come back…") counts as an
   instruction. NLTK's part-of-speech tagger finds the verb, after skipping "so", "okay", "please" and
   similar. Contractions are split and the pronoun is written "I", because the tagger reads "i'm" as a verb.

| Label | Examples of what counts |
|---|---|
| question | ends with "?", or starts with what/when/where/why/how, or with "do you", "is it", "have you"… |
| instruction | starts with a command verb; "you need to", "you have to", "make sure", "don't …", "please …", "remember to" |
| advice / plan | "you should", "I recommend", "it's important to", "we can/could/will", "let's", "I'll prescribe/refer/order", "there are options" |
| empathy / reassurance | "it sounds like", "I understand", "I can imagine", "that must be", "you're doing the best", "tell me more", "don't worry", "that's normal", "I'm sorry to hear" |
| uncertainty | "I don't know", "I'm not sure", "maybe", "I guess", "I don't think", "I have no idea" |
| opening | "what brings you", "how can I help", "what seems to be", "how are you feeling", "I'm Dr" |
| closing | "come back", "follow-up", "another appointment", "see you in", "any questions", "anything else", "call us" |

## The 10 features (per speaker)

| # | Feature | How it is computed | Expected higher |
|---|---|---|---|
| T1 | `question_share` | the speaker's questions ÷ all questions | doctor |
| T2 | `instructions_per_100w` | instructions ÷ words × 100 | doctor |
| T3 | `advice_plan_per_100w` | advice/plan sentences ÷ words × 100 | doctor |
| T4 | `empathy_reassurance_per_100w` | empathy sentences ÷ words × 100 | doctor |
| T5 | `uncertainty_per_100w` | uncertainty sentences ÷ words × 100 | patient |
| T6 | `i_per_100w` | I, me, my, mine, myself ÷ words × 100 | patient (lower status talks about the self; Kacewicz et al. 2014) |
| T7 | `you_per_100w` | you, your, yours, yourself ÷ words × 100 | doctor |
| T8 | `we_per_100w` | we, us, our, ours, let's ÷ words × 100 | doctor |
| C1 | `qa_initiative_share` | a turn with a question answered by the other person's next turn (short replies said inside the other's turn are skipped); this speaker's count ÷ all | doctor |
| C2 | `opening_closing_moves` | number of opening and closing sentences | doctor |

## Saved outputs

* `text_sentences.csv`: every sentence with turn, speaker, text, the 7 labels, and its I / you / we counts.
* `text_features.csv`: the 10 features with columns A, B, A − B, the expected direction, and which
  speaker that direction says is the doctor.

Run: `python 05_text_context/text_context.py` (a few seconds).

## MerloLab result (A = doctor)

| Feature | A | B | Doctor by this feature |
|---|---|---|---|
| T1 question share | 1.00 (7 questions) | 0 | A |
| T2 instructions per 100 words | 0 | 0.23 | B (only "don't know.", a sentence cut by a transcript error) |
| T3 advice / plan | 0.57 | 0 | A |
| T4 empathy / reassurance | 0.76 | 0 | A |
| T5 uncertainty | 0.38 | 0.23 | B (one of A's two is the patient's line inside A's last turn) |
| T6 "I" words | 2.85 | 11.45 | A |
| T7 "you" words | 8.56 | 0.23 | A |
| T8 "we" words | 0.76 | 0.47 | A |
| C1 question-answer initiative | 1.00 (6 exchanges) | 0 | A |
| C2 opening / closing moves | 1 ("set up another appointment") | 0 | A |

8 of 10 point to A. The 2 that point to B come from transcript and diarization errors, not from the
patterns.

## Known limits

* English only. Telugu needs its own patterns, for example yes/no questions ending in "-ā" and command
  forms such as "తీసుకోండి".
* Patterns miss phrasings they don't list. In MerloLab, "I **just** don't know" was not counted as
  uncertainty, because the pattern is "I don't know". It was left unchanged on purpose, so the
  patterns are not fitted to the test recording.
* The tagger misses some command verbs. In a general test it got 12 of 13 sentences right; it missed
  "Avoid oily food."
* MerloLab is a motivational-interviewing demo with no opening (the recording starts mid-visit) and no
  medicine instructions, so T2 and the opening half of C2 have little to find.
* Errors in diarization and in giving words to speakers carry over. For example, the patient's line
  "I'm not sure if I have the time" is in A's last turn.
