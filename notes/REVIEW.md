# Review of the original paper

Source: `paper/original/ASR-LLM_adaptive_education_paper.pdf`  
Title: *A Two-Layer Evaluation of Speech-to-Text and LLM-Based Terminology Extraction Systems for Multilingual Education*  
Authors: Cakil Yilmaz (Koç University); Berat Enes Dadak, Efe Batur Giritli, Yekta Said Can (Tatilsepeti)  
Length: 6 pages, IEEE two-column, produced with pdfTeX

## What the paper actually is

A **benchmark**, not a new architecture and not an adaptive tutor. Two layers:

1. **ASR** on ~95 minutes of educational video audio (3 Turkish + 4 English lectures): Whisper Medium, Whisper Large-v3, Wav2Vec 2.0, SpeechBrain. Metrics: raw/normalized WER and CER, latency, RTF. All runs are off-the-shelf / zero-shot on CPU.
2. **LLM** on the *best* transcripts only: GPT-4.1 vs GPT-4o-mini. Task: extract domain terms and write CEFR-B1 explanations. Metrics: R1% (term–context consistency), D1% (explanation correctness vs TDK / OED), plus B1+ ratio and BLEU on English only.

Headline numbers (kept unchanged in the revision):

| ASR (normalized) | WER | CER | Mean latency | RTF |
|---|---:|---:|---:|---:|
| Whisper Medium | 0.021 | 0.015 | 136.5 s | 0.156 |
| Whisper Large-v3 | 0.129 | 0.112 | 653.4 s | 0.703 |
| Wav2Vec2 | 0.548 | 0.196 | 46.1 s | 0.051 |
| SpeechBrain | 0.382 | 0.273 | 1795.1 s | 2.005 |

| LLM | Lang | Terms | R1% | D1% | B1+ | BLEU | Mean latency (TR) |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-4.1 | EN | 105 | 96.2 | 99.0 | 0.074 | 0.61 | 17.79 s |
| GPT-4o-mini | EN | 56 | 91.1 | 96.4 | 0.191 | 0.57 | 12.53 s |
| GPT-4.1 | TR | 78 | 96.2 | 98.7 | — | — |  |
| GPT-4o-mini | TR | 36 | 91.7 | 94.4 | — | — |  |

Those results are usable. The writing and the *claims* around them are what need work.

## High-severity issues

### 1. The central claim is not measured

The abstract, introduction, related-work close, results, and conclusion all say that transcription quality **directly affects** semantic-processing quality, and that the paper **jointly** evaluates the ASR+LLM pipeline.

What was actually done:

- Layer 1: ASR systems compared against reference transcripts.
- Layer 2: LLMs run on “the most suitable transcriptions” (Whisper), not on the worse ASR outputs.

Future work item (b) later admits the missing experiment: feed ASR outputs of different quality to the same LLM. Until that ablation exists, the paper must not claim a measured dependency. It can only **motivate** one.

Contribution 3 (“we report the effect of ASR–LLM component choices on the semantic outputs”) overstates the same point.

### 2. Introduction and Related Work repeat each other

Bloom’s 2-sigma result, ITS / adaptive hypermedia, knowledge tracing, VanLehn, and “LLMs as a flexible substrate” appear almost verbatim in both sections. On a 6-page budget that is wasted space and reads like two drafts concatenated.

### 3. Broken typesetting in the PDF

The compiled PDF visibly splits English ligatures: “fi nd ings”, “eff i ciency”, “conf i guration”, “suff i cient”. That is a font/encoding problem in the original TeX setup, not an extraction artifact. It looks unprofessional in print.

Other production problems:

- Figure 1 caption: “Eraser online tool was used for creating the figure.”
- Figure 2 y-axis is Turkish (`Hata Oranı`) in an English paper.
- Table I wraps `ENG-1` into `E 1 NG-`.
- Table II is cramped and, in the PDF, nearly unreadable (Latency / BLEU cells collide).
- Table VI repeats min/max in both the table and the caption.

### 4. Citation errors

| Ref | Problem |
|-----|---------|
| [20] OpenAI API | Cited for the *LMS plugin / term-card player* scenario. That scenario is hypothetical and has nothing to do with the OpenAI API reference. |
| [9] Jurafsky & Martin | In the bibliography, never cited in the body. |
| [15] | A *book review* of Manning, Raghavan & Schütze is cited as if it were the IR textbook. Cite the book. |
| [17] | Authors listed as “T. Tran, H. Nguyen, and M. Nguyen”. The survey this is pointing at is Tran, Martinc, Caporusso, Doucet, Pollak (ACM Computing Surveys). |
| [1] LiTeWi | Volume only; issue and pages missing (JASIST 67(2):380–399, 2016). |

### 5. Multilingual claim is stronger than the tables

The text says tests were run separately for Turkish and English. Table III reports only **pooled** ASR error rates. Given that Wav2Vec2 / SpeechBrain used English-oriented checkpoints, a TR vs EN split is the one table a reviewer will ask for. It is not in the paper.

B1+ and BLEU are English-only (justified), so the two languages are not evaluated symmetrically. That is already in Limitations; it should also be visible in the abstract so readers are not surprised.

### 6. Fairness of the ASR comparison

Wav2Vec2 raw WER 1.017 and SpeechBrain raw WER 1.246 are not architectural failures; they are the expected result of English-oriented checkpoints on Turkish audio. The paper *does* say this, which is good. The abstract and the first paragraph of Results still read as “Whisper wins, the others fail,” which a reviewer can attack. Frame every ASR ranking as **out-of-the-box multilingual suitability**, not as a ranking of architectures.

### 7. Whisper Medium beating Large-v3

Normalized WER 0.021 vs 0.129 is large, not “slightly lower.” The language-detection / long-form decoding explanation is plausible, but “slightly” understates a 6× gap. Either report decoding settings that explain it, or call it a configuration-sensitive result and avoid ranking Medium above Large-v3 in general.

### 8. Semantic evaluation is thin

- Single annotator, no IAA.
- R and D are binary labels against dictionary entries. A term can be context-correct and still useless for a beginner; a definition can match OED and still be too hard (B1+ is trying to catch this, but only for English).
- No gold term list, so there is no recall. GPT-4.1 extracting 105 EN terms vs 56 for mini looks like better coverage, but it could also be over-extraction. R1% only scores precision-like context consistency of the terms that were emitted.
- BLEU against dictionary definitions is a weak measure of explanation quality (reference mismatch, not meaning).

### 9. Filename vs content

The file is named `ASR-LLM_adaptive_education_paper.pdf`. There is no adaptive component: no learner model, no knowledge tracing, no personalization. Adaptation is future work. Keep the current title; do not put “adaptive” back in the title unless that module is built.

## Medium issues

- Author line uses “1st / 2nd / 3rd / 4th” and “Koc” without the cedilla. Standard IEEE author blocks are cleaner; the university is **Koç**.
- Industry emails are written as `berat.dadak, batur.giritli, yekta.can @tatilsepeti.com`.
- Dataset construction is underspecified: source platforms, license, speaker count, accent, spontaneous vs read, how “similar terminological density” was measured, how reference transcripts were produced (one transcriber? guidelines?).
- “No manual cleaning of audio” is good; say whether videos were already studio-quality or classroom-noisy. That changes how to read WER 0.021.
- Prompt is held fixed (good) but lives only in the appendix. A one-sentence description of output schema in the method is enough; keep the full template.
- Reproducibility paragraph is a single run-on sentence. Hardware model, batching, chunking of long audio, and Whisper `language=` / `condition_on_previous_text` flags are missing — and those flags are exactly what would explain Medium vs Large-v3.
- Reference transcripts: “manually correcting the audio recordings and verified by a human” is ambiguous (correcting ASR output vs independent transcription).
- No statistical tests, no confidence intervals, n = 7 videos. Already noted in Limitations; do not use language like “clearly” / “substantially” without that caveat nearby.

## What is already working

- Honest positioning as a benchmark rather than a new model (abstract, first page).
- Off-the-shelf / zero-shot design is a coherent choice for a low-resource deployment story.
- CER alongside WER is the right extra metric for Turkish morphology.
- Raw vs normalized WER/CER is the right pair.
- Limitations section exists and is specific.
- Future work (a)–(e) is a real research agenda, not filler.
- Prompt template in the appendix is the right reproducibility artifact.

## Implication for the revision

This pass fixes writing, claims, structure, tables, figures, and references so the paper says only what the experiments support. It does **not** add the missing TR/EN ASR split, the cross-layer ablation, IAA, or more data — those need you.
