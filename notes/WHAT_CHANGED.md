# What changed in this revision (relative to the original PDF)

# Pass 4 (2026-10-04 - Manimator Generative Video Layer Integration)

- Genişletilen 3 Katmanlı Mimari: Mevcut iki katmanlı sistem (ASR -> LLM Semantik), açık kaynaklı Manimator mimarisi uyarlanarak üçüncü bir dinamik video üretim katmanına (ASR -> LLM Semantik -> Manimator Video Katmanı) genişletildi.
- "Takıldığım Yeri Öğret" (On-Demand Remediation): Öğrenci ders videosunu izlerken anlamadığı kavramı işaretlediğinde, son 45 saniyelik transkript kesiti ve terimler yakalanarak yerel `Qwen/Qwen3.5-4B` modeline iletilir; model CEFR B1 düzeyinde benzetmeli (analogy) anlatım senaryosu ve yürütülebilir Python Manim animasyon kodu üretir.
- Nöral Seslendirme ve FFmpeg Senkronizasyonu: Microsoft Edge Neural TTS (`tr-TR-AhmetNeural` / `tr-TR-EmelNeural`) ile doğal Türkçe seslendirme yapılarak senkronize altyazılı 30-90 saniyelik MP4 video derlenir.
- Paper2Video / PDF Figure Walkthrough Modu: Ders notu veya makale PDF'leri yüklenerek metin ve şekiller üzerinden figür rehberli narrated video üretimi eklendi.
- Telifli Bulut API Bağımsızlığı: Orijinal Manimator'un ücretli OpenAI/Claude/Gemini API bağımlılığı kaldırılarak sistem tamamen açık ağırlıklı yerel Qwen 3.5 üzerinde çalışacak şekilde uyarlandı.
- Kendi Kendini Onaran (Self-Healing) Kod Yürütme: Manim kodundaki sözdizimi ve LaTeX hatalarını tespit edip otomatik düzelten onarıcı mekanizma eklendi.
- Uyarlanabilir Öğrenci Profili Entegrasyonu: İzlenen remediye videoları `learner.py` profiline işlenerek öğrencinin eksik olduğu kavramlar pekiştirme havuzuna kaydedildi.
- Proje Raporu Güncellendi: `notes/Bitirme_Projesi_Raporu.md` ve derlenen `notes/Bitirme_Projesi_Raporu.pdf` dosyalarına yeni mimari ve stüdyo yetenekleri eklendi.

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
- Clarified $\kappa_D = 1.000$ vs $\kappa_R = 0.782$: on the 115-item overlap, D has 112 ones and 3 zeros ($P_e \approx 0.95$), so perfect kappa reflects extreme class prevalence rather than strong annotator discernment; $\kappa_R = 0.782$ (8–12 negatives, 4 disagreements) is the number that actually demonstrates inter-annotator reliability.
- Clarified Gold transcript provenance: references are human-verified post-edits of caption tracks and ASR drafts (not scratch-built independent transcriptions); explicitly noted that post-editing auto-captions slightly aligns reference style toward Whisper-like captions.
- Reframed Turkish vs. English ASR comparison: with $n=7$ clips (3 TR + 4 EN), unweighted means, and no confidence intervals or significance tests, we stay humble and explicitly avoid claiming "Turkish is harder" as a general linguistic or architectural result.
- Aligned empty cells across documents: removed any implication that Nemotron 3.5 ASR was benchmarked; explicitly noted that Nemotron 3.5, gold GPU Wav2Vec 2.0 / SpeechBrain, and Qwen3-ASR 1.7B on EN-4 remain unfilled table cells (---) matching the paper.
- Reaffirmed absence of semantic recall: explicitly documented that R1% is a precision-like score over emitted terms and GPT-4.1's higher term counts (105 vs 56) are an extraction volume indication rather than proven recall (flagged as a mandatory reminder for oral defense).
- Reframed GUI as an interactive local studio: clarified that the FastAPI interface is a local proof-of-concept demo (with CORS *, in-memory job store, temp WAVs, and no auth) and must not be presented as a deployed production product.

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
