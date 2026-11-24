# Medical terms: version 1 (NER)

This is the specification `medical_terms.py` follows.

## How terms are found

1. **Transcript:** Step 3 transcribes the recording with Whisper medium. Each word goes to the speaker
   talking at its middle, then to that speaker's turn.
2. **Sentences:** each turn's text is split into sentences at `.`, `?` and `!`. A sentence ending in `?`
   is a question.
3. **NER:** the model `d4data/biomedical-ner-all` runs on each sentence. It is DistilBERT, trained on
   medical case reports, and recognises 43 entity types. It was already downloaded, and it runs offline
   and in seconds.
4. **Kept:** model confidence ≥ 0.5, at least 3 letters, not only common words such as "it" or "didn",
   and **at least one noun** in the sentence, because medical entities are noun phrases. NLTK's
   part-of-speech tagger checks this. Without the noun rule the model also tagged adjectives and verbs
   such as "concerned", "successful", "hesitant", and "fall" (in "where would you fall on that scale"),
   often with confidence above 0.9, so a higher confidence threshold would not remove them.
   Running sentence by sentence instead of on whole turns also removes noise: numbers such as
   "1 to 10" were read as lab values when whole turns were used.
5. **Entity types kept, grouped by who usually says them:**

| Side | Entity types |
|---|---|
| Doctor-side (treatment and tests) | Medication, Dosage, Administration, Diagnostic_procedure, Therapeutic_procedure |
| Patient-side (the problem) | Sign_symptom, Biological_structure (body part), History, Family_history, Disease_disorder |

Not kept: Activity (for example "smoking", "quit"), lab values, times, ages and other general types.

## The 4 features (per speaker)

Counts are per 100 of the speaker's words, so the person who talks more is not favoured.

| # | Feature | How it is computed | Expected higher |
|---|---|---|---|
| M1 | `treatment_test_per_100w` | doctor-side terms ÷ words × 100 | doctor |
| M2 | `symptom_history_statements_per_100w` | patient-side terms said in statements (not questions) ÷ words × 100 | patient |
| M3 | `symptom_questions_per_100w` | patient-side terms said inside questions ("any pain in your chest?") ÷ words × 100 | doctor |
| M4 | `treatment_test_first_share` | of the different treatment and test concepts, the share this speaker mentioned first ("medications" and "medication" count as one) | doctor |

## Saved outputs

* `medical_terms.csv`: every term with turn, speaker, sentence, type, side, question yes/no, model
  confidence and the sentence it was found in.
* `medical_features.csv`: M1–M4 with columns A, B, A − B, the expected direction, and which speaker that
  direction says is the doctor.

## MerloLab result (A = doctor)

| Feature | A | B | Doctor by this feature |
|---|---|---|---|
| M1 treatment and test terms per 100 words | 0.57 (antibiotics, medications, risks*) | 0 | A |
| M2 symptom and history statements per 100 words | 0.57 | 1.64 (asthma, ear infections, going into labor early, …) | A |
| M3 symptom terms in questions per 100 words | 0 | 0 | tie (this doctor asks open questions, not symptom questions) |
| M4 treatment and test concepts mentioned first | 1.0 | 0 | A |

\* "risks" ("the risks were so scary") is a false positive. Without it, M1 and M4 still point to A.
13 terms remain after the filters. The clear false positives are "risks" and "successful quitting in
the past". The noun rule also dropped "ear" (in "ear infections"), once for each speaker.

## Known limits

* The model was trained on written case reports, not conversations, so it misses some everyday
  words and sometimes tags non-medical words.
* English only. For Telugu (later): Sarvam's code-mixed transcripts keep English medical words in
  English letters, so NER works on those. Telugu-script words can first be translated with NLLB, which
  is already downloaded.
* Terms go to whoever diarization says spoke them, so diarization errors carry over (see the known
  error at the end of MerloLab).
