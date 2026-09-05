# What changed in this revision (relative to the original PDF)

Pass 3 (2026-09-04) updated the refresh model set to the newest TR+EN-compatible checkpoints. No new WER/R1/κ cells. See `notes/HF_MODEL_REFRESH.md`.

- ASR add: `nvidia/nemotron-3.5-asr-streaming-0.6b` (official `tr-TR` + `en-US`).
- Refresh LLMs are free only: `Qwen/Qwen3.5-9B` (capacity) and `Qwen/Qwen3.5-4B` (latency / default ablation). Paid GPT-5.6 Sol / Luna / GPT-6 Astra are rejected. Pilot `gpt-4.1` / `gpt-4o-mini` stay in the registry only so the already-filled pilot table remains comparable.
- Open LLM replace: `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-4B` instead of Qwen3-8B / Qwen2.5-3B-Instruct.
- Kit: Nemotron backend; Qwen3.5 text-only with thinking off; `transformers>=5.13`.

Pass 2 (2026-08-28) added the evaluation kit and empty protocol tables.
See `notes/PASS2_STATUS.md`. No new experimental numbers.

# Pass 1 (claims / structure / production)

Experimental numbers are unchanged. Only claims, structure, production, and references moved.

## Claims

- Removed “strong dependency between transcription quality and semantic-processing performance” as a *finding*. The LLMs were only run on Whisper transcripts.
- Contribution 3 no longer says we measured the effect of ASR choice on semantic outputs.
- RQ1–RQ3 now state the actual questions.
- Whisper Medium vs Large-v3 is a ~6× WER gap, not “slightly lower.”
- ASR ranking is framed as out-of-the-box multilingual suitability.

## Structure

- Bloom / ITS / knowledge tracing appear once (Introduction), not again in Related Work.
- Related Work is three blocks: ASR, term extraction, adaptive/LLM tutors + gap.
- Limitations now lists the missing per-language ASR table and the missing cross-layer ablation.

## Production

- Standard IEEE author block; Koç with cedilla; industry emails in `{a,b,c}@tatilsepeti.com` form.
- Table I no longer breaks `ENG-1` across cells; IDs are `EN-1`…`EN-4`.
- Table II is a readable yes/-- protocol table.
- Fig. 1 is a TikZ pipeline (no “Eraser online tool” caption).
- Fig. 2 y-axis is English (“Normalized error rate”).
- Table VI caption no longer repeats min/max.

## References

- Jurafsky cited at the WER definition (was unused).
- OpenAI API cited only for the API (was attached to a hypothetical LMS plugin).
- Manning, Raghavan & Schütze textbook replaces a book review of that textbook.
- LiTeWi: volume, issue, pages.
- ATE survey: correct authors (Tran, Martinc, Caporusso, Doucet, Pollak).
