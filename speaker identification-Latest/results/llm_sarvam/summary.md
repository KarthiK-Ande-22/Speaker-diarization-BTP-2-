# LLM role decision (sarvam-105b-conversations, 3 answers per order, temperature 0.7)

| Recording | Groups given to the LLM | Doctor chosen | Correct | Confidence | Agreement | Same in both orders | Mean stated confidence |
|---|---|---|---|---|---|---|---|
| merlolab_motivational_interviewing | speech | B | no | 0.663 | 5/6 | yes | 70.3 |
| merlolab_motivational_interviewing | medical terms | A | yes | 0.717 | 6/6 | yes | 71.7 |
| merlolab_motivational_interviewing | text and context | A | yes | 0.937 | 6/6 | yes | 93.7 |
| merlolab_motivational_interviewing | video | A | yes | 0.952 | 6/6 | yes | 95.2 |
| merlolab_motivational_interviewing | speech + medical terms | A | yes | 0.812 | 6/6 | yes | 81.2 |
| merlolab_motivational_interviewing | speech + text and context | A | yes | 0.95 | 6/6 | yes | 95.0 |
| merlolab_motivational_interviewing | speech + video | A | yes | 0.95 | 6/6 | yes | 95.0 |
| merlolab_motivational_interviewing | medical terms + text and context | A | yes | 0.75 | 4/6 | no | 85.0 |
| merlolab_motivational_interviewing | medical terms + video | A | yes | 0.933 | 6/6 | yes | 93.3 |
| merlolab_motivational_interviewing | text and context + video | A | yes | 0.955 | 6/6 | yes | 95.5 |
| merlolab_motivational_interviewing | speech + medical terms + text and context | A | yes | 0.913 | 6/6 | yes | 91.3 |
| merlolab_motivational_interviewing | speech + medical terms + video | A | yes | 0.955 | 6/6 | yes | 95.5 |
| merlolab_motivational_interviewing | speech + text and context + video | A | yes | 0.968 | 6/6 | yes | 96.8 |
| merlolab_motivational_interviewing | medical terms + text and context + video | A | yes | 0.947 | 6/6 | yes | 94.7 |
| merlolab_motivational_interviewing | speech + medical terms + text and context + video | A | yes | 0.967 | 6/6 | yes | 96.7 |
| merlolab_motivational_interviewing | transcript only | A | yes | 0.95 | 6/6 | yes | 95.0 |
| merlolab_motivational_interviewing | first speaker is the doctor (no LLM) | A | yes |  |  |  |  |
