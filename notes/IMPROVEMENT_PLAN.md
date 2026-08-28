# Improvement plan

## Pass 1 (done in `paper/main.tex`)

These changes do not invent results.

1. **Claims match the design.** Abstract, contributions, results, and conclusion no longer say the paper *measured* an ASR→LLM dependency. They say the two layers were evaluated, LLMs were run on the best transcripts, and a cross-layer ablation is future work.
2. **Introduction shortened; Related Work no longer repeats Bloom / ITS / KT.** Space goes to a clearer gap statement.
3. **Research questions** stated explicitly (useful for a bitirme jury as well as a conference reviewer).
4. **Table I** rebuilt (EN-1…EN-4, no broken `ENG-` wrapping). **Table II** rebuilt as a readable protocol table. **Table VI** no longer repeats min/max in the caption.
5. **Figures rebuilt in TikZ / pgfplots.** Fig. 1 is a two-layer pipeline, not an Eraser export. Fig. 2 y-axis is “Normalized error rate,” not “Hata Oranı.”
6. **Citations.** Jurafsky cited at the WER definition. OpenAI API cited only for the API. Manning IR book replaces the book-review cite. LiTeWi and the ATE survey completed / corrected.
7. **University name** Koç; author block is standard IEEE; industry emails use a proper `\texttt{\{a,b,c\}@…}` form.
8. **Whisper Medium vs Large-v3** described as a large, configuration-sensitive gap (6×), not “slightly lower.”
9. **ASR ranking** framed as out-of-the-box multilingual suitability.
10. **Limitations** now include the missing cross-layer ablation and the missing per-language ASR table as scope, not only as future work.

## Pass 2 (kit + paper protocol — 2026-08-28)

Scripts, dataset card, IAA protocol, hardware appendix, and empty paper tables are in the repo. **No new WER/R1/κ cells were filled.** See `notes/PASS2_STATUS.md`.

## Pass 2 — still needs your input or new runs

Do these next, in this order, if the paper is going to a conference or a strict jury.

| # | Item | Why | What I need from you |
|---|------|-----|----------------------|
| A | **Per-language ASR table** (TR vs EN WER/CER for each model) | The multilingual claim is otherwise unsupported. Almost certainly already computable from existing logs. | Raw per-video or per-language scores, or the evaluation scripts/logs. |
| B | **Cross-layer ablation** | Feed Whisper, Wav2Vec2, and SpeechBrain transcripts of the *same* videos into *one* LLM (GPT-4o-mini is enough). Report R1%, D1%, term count vs ASR WER. This is the paper’s actual contribution if you can add it. | Access to the transcripts + API, or the existing JSON outputs. |
| C | **Decoding / hardware appendix** | CPU model, Whisper `language`, `condition_on_previous_text`, chunking, beam size. Explains Medium vs Large-v3. | The run scripts or a short note of the flags you used. |
| D | **Second annotator + IAA** (Cohen’s κ or agreement %) on a subset of R/D labels | Single-annotator semantic scores will be challenged. | A second person and the labeled spreadsheet. |
| E | **Dataset card** | Source (YouTube / course LMS / recorded lecture), license, speaker count, prepared vs spontaneous. | A short paragraph I can drop into §III-B. |
| F | **Venue / length** | 6-page IEEE conference vs a longer Turkish bitirme raporu. Different templates, different depth. | Tell me the target (conference name, or YÖK/department report). |
| G | **Shared repository** | Appendix already promises one for the prompt. | GitHub URL, or say “create a private repo skeleton.” |

## Pass 3 — only if the project grows

- Language-matched Wav2Vec2 / SpeechBrain checkpoints (Turkish XLS-R or fine-tuned).
- Prompt / CoT / RAG ablation on the term-extraction layer.
- A real adaptive loop (show only terms the learner missed). That would justify putting “adaptive” in the title; it is not in scope until A–E are done.

## What I will not do

- Invent per-language WER, IAA, or ablation numbers.
- Turn this into an “adaptive education system” paper without that system.
- Expand past ~6 IEEE pages unless you say the deliverable is a long thesis.
