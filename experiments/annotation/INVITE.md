# Message to send annotator 2

Copy, fill the name, send to Berat / Batur / Yekta (or another domain-knowledgeable person).

---

Subject: 20–40 min labeling pass for the ASR+LLM paper (R/D)

Hi —,

We need a second independent annotator for the bitirme / conference paper. The pilot R1%/D1% scores were labeled by one person. A reviewer will ask for agreement.

Task:
- For each extracted term, mark R (term fits the lecture context) and D (explanation matches TDK or OED).
- Binary only: 1 or 0. Protocol: `experiments/annotation/PROTOCOL.md`.
- Work from `labels_todo.csv`. Do not look at anyone else’s sheet.
- Time: a subset is enough (one TR video + one EN video, or ~40 items).

Save as `labels_a2.csv` and send it back. We will run:

    python iaa.py --a1 annotation/labels_a1.csv --a2 annotation/labels_a2.csv

That writes Cohen’s kappa. Disagreements can be discussed later; kappa uses the independent first pass only.

Thanks
---
