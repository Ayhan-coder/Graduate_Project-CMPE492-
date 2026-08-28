# What changed in this revision (relative to the original PDF)

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
