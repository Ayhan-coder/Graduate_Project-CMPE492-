# Dataset card — TR/EN educational lecture set

Follows the datasheets-for-datasets fields that this benchmark actually uses.

## Motivation

- **Why:** Gold audio + transcripts for a three-layer evaluation (layer 1 ASR, layer 2 LLM terminology extraction, both measured here; layer 3 generative animation, feasibility only) on beginner-oriented educational speech.
- **Funders / creators:** Ali Ayhan Günder (Boğaziçi University) with Berat Enes Dadak, Efe Batur Giritli, Yekta Said Can (Tatilsepeti R&D).
- **Not for:** Training new ASR or LLM weights. Evaluation only.

## Composition

| ID | Title (as in the paper) | Language | Domain | Duration |
|----|-------------------------|----------|--------|----------|
| TR-1 | Project Management, Unit 7, Lecture 1 | Turkish | Project management | ≈12 min |
| TR-2 | Comprehensive Tourism Training for Tour Guides | Turkish | Tourism | 12 min |
| TR-3 | What Is Data Science? Intro to Data Science (Python) | Turkish | Data science | ≈14 min |
| EN-1 | Introduction to Agile Methodologies | English | Project management | ≈13 min |
| EN-2 | What is sustainable tourism? | English | Tourism | ≈9 min |
| EN-3 | Generative AI in a nutshell | English | AI | ≈18 min |
| EN-4 | Understanding thermal radiation | English | Heat transfer | ≈17 min |

- **Total:** 7 clips, ≈95 minutes (local wavs sum to 95.6 min).
- **Instances:** one 16 kHz mono wav + one reference transcript per ID.
- **Sensitive data:** educational lecture speech. No planned collection of student identities.

### Source (confirmed)

All seven files are public YouTube educational videos whose titles and durations match the paper table. Machine-readable copy: `sources.json`. Local wavs were prepared on 2026-08-29 and confirmed as the evaluation-set files on 2026-09-05.

| ID | Platform | Channel | URL | YouTube duration |
|----|----------|---------|-----|------------------|
| TR-1 | YouTube | Açıköğretim Fakültesi — Anadolu Üniversitesi | https://www.youtube.com/watch?v=rz5JCvMlCMs | 12:01 |
| TR-2 | YouTube | Forteliber | https://www.youtube.com/watch?v=yzsHZVU_Y7w | 11:17 |
| TR-3 | YouTube | Tirendaz AI / Tirendaz Academy | https://www.youtube.com/watch?v=DQAiWukoOgI | 14:15 |
| EN-1 | YouTube | Grow with Google | https://www.youtube.com/watch?v=BaH04m_fS1c | 13:17 |
| EN-2 | YouTube | Dr Hayley Stainton \| Tourism Teacher | https://www.youtube.com/watch?v=kgAHZHkMTQU | 9:16 |
| EN-3 | YouTube | Henrik Kniberg | https://www.youtube.com/watch?v=2IK3DFHRFfw | 17:57 |
| EN-4 | YouTube | The Efficient Engineer | https://www.youtube.com/watch?v=FDmYCI_xYlA | 17:33 |

TR-2 is 11:17 against the paper’s “12 min”; that is still the closest public title+duration match and is the file used here.

No DOI. No LMS / classroom-capture originals were located; the evaluation set is these YouTube encodings.

### License / redistribution

- Audio is third-party YouTube content. Default assumption is the **YouTube Standard License**. None of the channels publish a public-domain or CC grant that covers redistributing the wavs.
- TR-1 is Anadolu University open-education lecture video; that still does not grant us a right to ship the audio.
- EN-1 is Google Career Certificate / Coursera course material posted by Grow with Google.
- **Do not put the wavs in a public repository.** Gold text in `data/gold/` is for local WER/CER only, not a redistributable corpus.
- Allowed use in this project: offline evaluation, thesis / paper tables, and kit reproduction.

### Speakers

| ID | Speakers | Gender | Accent / L1 | Delivery |
|----|----------|--------|-------------|----------|
| TR-1 | 1 | male | native Turkish | scripted studio lecture (AÖF) |
| TR-2 | 1 | male-presenting voiceover | native Turkish | scripted explainer (not a live classroom) |
| TR-3 | 1 | male | native Turkish | prepared talking-head tutorial |
| EN-1 | 1 | male | native American English | scripted studio (certificate course) |
| EN-2 | 1 | female | native British English | scripted talking-head |
| EN-3 | 1 | male | L2 English (Swedish L1) | scripted animated explainer |
| EN-4 | 1 | male | native British English | scripted animated explainer |

- **Speaker count:** 7 unique speakers, one per clip. No overlapping speakers across IDs.
- **Classroom vs scripted:** all seven are **prepared / scripted educational video**, not spontaneous classroom discussion and not multi-speaker seminar Q&A.
- Student identities are not in the audio.

### Terminological density

How “similar terminological density” was measured in the original paper is **not recorded**. It was not re-measured for this card.

### How the reference was produced

Paper wording: “manual correction of the audio” plus a human check. Treat as gold for WER/CER only.

| ID | Gold how | Notes |
|----|----------|-------|
| EN-1 | Official Coursera / Grow-with-Google caption track, cleaned | `EN-1.en-j3PyPqV-e1s.vtt`; speaker label stripped |
| EN-4 | Human English caption track, cleaned | `EN-4.en-en.vtt`; term spellings (Stefan–Boltzmann, Wien, Planck) kept |
| TR-1 | Complete auto-caption / ASR draft patched with the human TR track | Human `TR-1.tr-tr.vtt` is incomplete (skips spoken spans); gold is the full talk |
| EN-2 | YouTube auto captions, punctuated and term-corrected | Brundtland, greenwashing, three pillars, UNWTO wording |
| EN-3 | YouTube auto captions, punctuated and term-corrected | ChatGPT, GPT-4, Kanban-style product names, prompt engineering |
| TR-2 | YouTube auto captions + ASR draft, term-corrected | Forteliber, MICE, UKTKG, stoklanamazlık |
| TR-3 | Independent transcription (no YouTube captions) | Drafted from the audio, then corrected; channel name Tirendaz, Guido van Rossum, Cython / Jython, Jupyter |

Gold files live at `experiments/data/gold/{ID}.txt`. `manifest.jsonl` sets `reference_source=gold`, `confirmed=true`, and `reference_file=data/gold/{ID}.txt`.

## Collection and preprocessing (known)

- Audio extracted from the source videos.
- Converted to **16 kHz, mono**.
- **No** denoising, clipping, or manual audio edit.
- Reference transcripts: see the table above. Treat as gold for WER/CER only.

## Uses

- Allowed: reproduce Tables III–VI, the per-language ASR split, the ASR→LLM ablation, and the 2026 refresh.
- Not allowed: claiming the set is a public corpus, or training new ASR / LLM weights on these files.

## Distribution

- Wavs stay local (`experiments/data/TR-1.wav` … `EN-4.wav`) and are listed in `manifest.jsonl`.
## 32-Video Extended Benchmark Corpus (`manifest_30plus.jsonl`)

To transform the initial 7-video pilot into a rigorous, journal-grade benchmark, the dataset is extended to **32 educational video lectures** (16 Turkish, 16 English) totaling **~437 minutes (~7.28 hours)** of instructional speech metadata (`sources_30plus.json`, `manifest_30plus.jsonl`). Note that this manifest overlaps the 7 gold clips (TR-01..03, EN-01..04); among candidate audio files, EN-11 matches duration while 18 files remain unverified raw captures pending slice trimming, and no new extension clips were benchmarked.

### Domain and Topic Diversity

The 32 clips cover balanced disciplines across STEM, Humanities, and Social Sciences:
- **Computer & Data Science:** Python Data Science, Binary Search & AVL Trees, Operating Systems & Concurrency, Relational SQL Normalization, Graph Traversal (BFS/DFS).
- **AI & Machine Learning:** NLP Foundations, Supervised Regression & Regularization, Deep Neural Networks & Backpropagation, Generative AI & LLMs.
- **Engineering & Physics:** Project Management (Agile/Scrum, EVM), Thermodynamics & Energy, Kirchhoff Laws & Thévenin Circuits, Classical Mechanics, Thermal Radiation, Structural Bending Moments, Maxwell's Electromagnetism, Digital Signal Processing (FFT).
- **Social & Biological Sciences:** Tour Guide Training, Sustainable Tourism, Microeconomics & Elasticity, Cognitive Memory & Neuroplasticity, Modern History & Industrial Revolution, Epistemology & Theory of Knowledge, Molecular Biology (DNA Transcription, CRISPR-Cas9).

### Dual-Annotator Labeling Protocol (`annotation/annotation_template_30plus.jsonl`)

- **Annotation Units:** Each domain terminology candidate is annotated with start/end timestamps `[t_start, t_end]`, context snippet, and canonical domain lemma.
- **Annotator Agreement:** Evaluated across two independent raters using Cohen's $\kappa$ for nominal consistency ($R$) and definition accuracy ($D$).
- **Prevalence Handling:** Due to high class prevalence on definition correctness ($D$), Prevalence-Adjusted Bias-Adjusted Kappa (PABAK, Byrt et al., 1993) and Fleiss' $\kappa$ are reported alongside standard Cohen's $\kappa$ via `experiments/eval_agreement.py`.
- **Temporal Alignment:** Inter-annotator time span overlap is measured using Intersection over Union (IoU $\ge 0.3$), enabling temporal Precision, Recall, and F1 calculation against reference glossaries.

## Maintenance

- Contact: gunderaliayhan@gmail.com
- Errors in IDs, titles, or gold wording: open an issue on the project repository.

