# Two-annotator protocol for R and D

The completed pilot used **one** annotator. This file is the protocol for the second annotator and for Cohen’s κ.

## Labels (same as the paper)

| Code | Meaning |
|------|---------|
| R1 | The extracted term is consistent with the transcript context. |
| R0 | It is not (wrong language, not in the lecture, or not a domain term). |
| D1 | The explanation is correct relative to TDK (Turkish) or OED (English). |
| D0 | It is not. |

Labels are **binary**. Do not add a third “partial” class. If unsure, mark 0 and put a note in `comment`.

## What you are *not* scoring

- Whether the term is useful for a beginner (that is the B1+ ratio, English only, automatic).
- Whether GPT over-extracted. R1% is precision-like over emitted terms; there is no gold term list.
- Style, length, or JSON formatting.

## Procedure

1. Work from `labels_todo.csv` (generated from an LLM JSONL). Do **not** look at the other annotator’s file.
2. Read the transcript span around the term (the `context` column if present, otherwise the source transcript).
3. Look the term up in TDK (`https://sozluk.gov.tr`) or OED (`https://www.oed.com`).
4. Fill `R` and `D` as `1`/`0` (or `R1`/`R0`, `D1`/`D0`).
5. Save as `labels_a2.csv`. Annotator 1’s completed sheet is `labels_a1.csv`.

Recommended subset if time is limited: **all terms from one Turkish and one English video**, plus a random 20% of the rest. Report κ on the double-labeled subset only.

## After both files exist

```text
python iaa.py --a1 annotation/labels_a1.csv --a2 annotation/labels_a2.csv
```

Report:

- raw agreement and Cohen’s κ for R
- raw agreement and Cohen’s κ for D
- n of double-labeled items
- how disagreements were adjudicated (if at all)

Do **not** write a κ value into the paper until this command has been run on real labels.

## Disagreement

A third pass (the two annotators together, or a third person) may resolve conflicts for the **final** R1/D1 counts used in Table V. κ is always computed on the **independent** first-pass labels, not on the adjudicated sheet.
