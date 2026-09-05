# Pass 2 status

Started 2026-08-28. Continued 2026-08-29. No experimental number was invented for the IEEE tables.

| Item | In the repo | In the paper | Blocked on |
|------|-------------|--------------|------------|
| Per-language ASR table | `report.py` aggregates `run_asr.py` JSONL by `lang` | Table `tab:asr-lang` (`---`) | Confirmed original wavs + **gold** transcripts (captions are drafts) |
| Cross-layer ablation | `run_ablation.py` (default GPT-4o-mini) | Table `tab:ablation` (`---`) | ASR JSONL + API key + R/D labels |
| Decoding / hardware appendix | `HARDWARE.md`, `check_env.py`, sidecar `.meta.json` | Appendix B | Original pilot CPU model (still unknown) |
| Second annotator + IAA | `annotation/PROTOCOL.md`, `iaa.py`, example CSV | Appendix C, Method III-E | A second person and real labels |
| Dataset card | `data/DATASET_CARD.md` | §III-B points at the card | Source, license, speakers; author confirm of YouTube IDs |
| Shared GitHub repo | Local git + MIT + `.gitignore` | “accompanying repository” | GitHub remote / `gh` |
| 2026 model scores | `--all-selected`, Table `tab:asr-2026` | Empty cells | Confirmed wavs + gold + CUDA wheel for Qwen3 |

## Hunt (2026-08-29)

- Still no original lecture wav/mp4/gold on this PC. `OPENAI_API_KEY` not set. `gh` not installed.
- Restored original Table I titles from the starting PDF (they had been shortened): TR-2 *Comprehensive Tourism Training for Tour Guides*; EN-1 *Introduction to Agile Methodologies*; TR-3 *What Is Data Science? Intro to Data Science (Python)*.
- All **seven** 16 kHz mono candidate wavs are now local. **Not confirmed as the pilot files.** Caption drafts are not gold.

| ID | Confidence | Candidate | Duration |
|----|------------|-----------|----------|
| TR-1 | high | AÖF *PROJE YÖNETİMİ - Ünite 7 Konu Anlatımı 1* `rz5JCvMlCMs` | 12:00 |
| TR-2 | medium | Forteliber *Turist Rehberleri için Kapsamlı Turizm Eğitimi* `yzsHZVU_Y7w` | 11:17 vs 12 min |
| TR-3 | medium | Tirendaz AI *VERİ BİLİMİ NEDİR* `DQAiWukoOgI` | 14:15 |
| EN-1 | high | Grow with Google *Introduction to Agile Methodologies* `BaH04m_fS1c` | 13:17 |
| EN-2 | high | Hayley Stainton `kgAHZHkMTQU` | 9:16 |
| EN-3 | high | Kniberg `2IK3DFHRFfw` | 17:57 |
| EN-4 | high | Efficient Engineer `FDmYCI_xYlA` | 17:33 |

- EN-1 is the first unique title+duration hit (Google Project Management Certificate). TR-2 is a Turkish title translation of the paper title, 43 s short of the stated 12 min.

## Kit fix (2026-08-29)

- Yesterday’s `outputs/asr-20260828-215142.jsonl` is empty because `run_asr.py` passed the openai-whisper flag `condition_on_previous_text` into `transformers.generate`. HF rejects it (`ValueError: model_kwargs are not used: condition_on_previous_text`).
- The kit now sends `condition_on_prev_tokens` (HF name), caches each loaded model, writes an error row instead of dying with an empty file, and accepts `--ids`.
- Smoke test on 30 s of TR-1 (`data/smoke/`) completed: Whisper Medium on **CPU** produced a real Turkish hypothesis matching the lecture opening. WER vs a one-sentence caption stub is not a paper number (reference is too short). CPU RTF on that clip was ~5.8 — the transformers pipeline on CPU is not the paper’s 0.156, so a CUDA wheel is required before a full 95-minute run.

## Hardware

Kit development machine (not the pilot host): Intel Core i7-14650HX, 32 GB RAM, RTX 5060 Laptop (driver 616.56). PyTorch is now **2.12.0+cu130** (`cuda=True`, device name `NVIDIA GeForce RTX 5060 Laptop GPU`). Qwen3 can use this wheel.

Do **not** paste candidate WER/CER into Tables `tab:asr-lang`, `tab:asr-2026`, or `tab:ablation` until (a) an author sets `confirmed: true` and (b) the original gold transcripts replace `reference_source=caption_draft`. `report.py --write-tex` now refuses unless those two conditions hold.

## Full-set run (2026-08-29, later)

Whisper Medium started on all seven candidate wavs on the RTX 5060 (`outputs/asr-20260829-142108.*`). GPU was in use (~7 GB). That first GPU job used the experimental `pipeline(..., chunk_length_s=)` path and stalled on TR-1 (no row written). Switched to sequential `generate()` with official halt thresholds.

30 s GPU smoke test (`asr-20260829-144819.jsonl`): real Turkish hypothesis, RTF **0.72** on the RTX 5060 (`2.12.0+cu130`). `max_new_tokens` had to be 440, not 448, or Whisper overflows `max_target_positions`. The 7-clip sequential run (`asr-20260829-144852.jsonl`) started TR-1 and was killed with the session; the file is empty.

TR-3 still has no YouTube captions (`reference_source=missing`). A second hunt (29 Aug evening) found no better 12:00 TR-2 match than Forteliber 11:17, and no better TR-3 title+duration hit than Tirendaz 14:15.

## GPU run (2026-08-30 / finished 2026-09-01)

`outputs/asr-20260830-151020.jsonl` is the first complete Whisper Medium file (all seven clips). Sidecar says `cuda=True`, `NVIDIA GeForce RTX 5060 Laptop GPU`, `2.12.0+cu130`, sequential `generate()`.

| Clip | Status | Caption-draft WER_n | RTF | Notes |
|------|--------|--------------------:|----:|-------|
| TR-1 | done | 0.038 | 0.194 | caption_draft, unconfirmed |
| TR-2 | done | 0.032 | 0.380 | caption_draft, unconfirmed |
| TR-3 | done | — | 0.314 | `reference_source=missing`; hypothesis only |
| EN-1 | done | 0.021 | 0.488 | caption_draft, unconfirmed |
| EN-2 | done | 0.012 | 0.359 | caption_draft, unconfirmed |
| EN-3 | done | 0.031 | 0.330 | caption_draft, unconfirmed |
| EN-4 | done 1 Sep | 0.038 | 0.107 | caption_draft, unconfirmed |

Language means (caption-draft, unweighted; TR-3 excluded from WER): TR WER_n **0.035** (n=2), EN WER_n **0.025** (n=4). Mean GPU RTF ≈ 0.31.

These caption-draft scores are kit diagnostics. They are **not** IEEE table cells. `report.py --write-tex` still refuses.

TR-3 (`DQAiWukoOgI`) still has no YouTube captions (manual or auto). Do not invent a reference.

## GPU run (2026-09-01 / finished 2026-09-02)

`outputs/asr-20260901-large-v3.jsonl` is the first complete Whisper Large-v3 file (all seven clips). Same host and flags as the Medium run (`cuda=True`, sequential `generate()`, language forced). Resumed 2 Sep from TR-1 (written 1 Sep).

| Clip | Status | Caption-draft WER_n | RTF | Notes |
|------|--------|--------------------:|----:|-------|
| TR-1 | done 1 Sep | 0.017 | 0.199 | caption_draft, unconfirmed |
| TR-2 | done 2 Sep | 0.023 | 0.168 | caption_draft, unconfirmed |
| TR-3 | done 2 Sep | — | 0.087 | `reference_source=missing`; hypothesis only |
| EN-1 | done 2 Sep | 0.023 | 0.071 | caption_draft, unconfirmed |
| EN-2 | done 2 Sep | 0.013 | 0.435 | caption_draft, unconfirmed |
| EN-3 | done 2 Sep | 0.024 | 0.300 | caption_draft, unconfirmed |
| EN-4 | done 2 Sep | 0.041 | 0.059 | caption_draft, unconfirmed |

Language means (caption-draft, unweighted; TR-3 excluded from WER): TR WER_n **0.020** (n=2), EN WER_n **0.025** (n=4). Mean GPU RTF ≈ 0.19.

On these *candidate* files, Large-v3 is not the 6×-worse system from the CPU pilot (pooled WER_n 0.129). TR is better than Medium (0.020 vs 0.035); EN is tied (0.025). That is a kit diagnostic about decoding/hardware, not a paper result.

These caption-draft scores are kit diagnostics. They are **not** IEEE table cells. `report.py --write-tex` still refuses.

28 unit tests pass (2 Sep).

Next kit run (when the dGPU is up): Whisper Large-v3 Turbo (`openai/whisper-large-v3-turbo`) on the same seven wavs. Then Qwen3-ASR if VRAM allows. Still no gold / no `confirmed: true`, so tables stay empty.

## Evening checkpoint (2026-08-29)

- CUDA wheel is still `2.12.0+cu130`, but `torch.cuda.is_available()` is **False**. Windows reports the RTX 5060 as `CM_PROB_PHANTOM` (Code 45, “device not connected”); `nvlddmkm` is Stopped. This is after S0 modern standby. `nvidia-smi` / `pnputil /scan-devices` need admin and cannot wake it from this session.
- `run_asr.py` now accepts `--output` / `--resume` so a killed job can skip completed clip+model pairs.
- 29 Aug evening: Table I titles restored to the original PDF wording; Method III-F aligned with sequential `generate()`; Turkish I/İ mapping in `metrics.normalize`; `report.py` / `run_ablation.py` skip error rows. **25** unit tests pass.
- Do not start the 95-minute set on the CPU wheel (RTF ~5.8 → ~9 hours). Wait for the dGPU (plug in AC, or reboot).

Do **not** paste candidate WER/CER into Tables `tab:asr-lang`, `tab:asr-2026`, or `tab:ablation` until (a) an author sets `confirmed: true` and (b) the original gold transcripts replace `reference_source=caption_draft`. `report.py --write-tex` now refuses unless those two conditions hold.
