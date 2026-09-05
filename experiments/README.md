# Experiment kit

Reproduce the pilot tables, the **per-language ASR split**, the **ASR→LLM ablation**, the **2026 Hugging Face refresh** (4 Sep 2026: Nemotron 3.5 ASR, Qwen3.5-9B/4B; no paid APIs), and **Cohen’s κ**. Do not type numbers into the paper by hand — run the scripts and let `report.py` build the tables.

Configs for the 2026 models are in `hf_snapshots/`. Full weights are not downloaded.

## Prepare audio (if you do not already have the wavs)

```text
pip install yt-dlp
python prepare_audio.py --list
# only after you confirm the YouTube IDs in data/sources.json
python prepare_audio.py --only TR-1 TR-2 TR-3 EN-1 EN-2 EN-3 EN-4
# or, to try the public matches without confirming they were the pilot files:
python prepare_audio.py --allow-candidates
```

ffmpeg is already on this machine. Gold transcripts are in `data/gold/{ID}.txt` with `reference_source=gold` and `confirmed=true` in `data/manifest.jsonl`. `prepare_audio.py` still writes caption drafts if you re-download; do not overwrite the gold files.

## What you must drop in

1. Seven 16 kHz mono wavs: `data/TR-1.wav` … `data/EN-4.wav` (already local)
2. Gold transcripts: `data/gold/{ID}.txt` (already local; see `data/DATASET_CARD.md`)
3. Optional: `OPENAI_API_KEY` only if you must reproduce the already-filled pilot GPT-4.1 / GPT-4o-mini table. New semantic and ablation runs use free Qwen3.5 checkpoints and do not need a key.
4. Optional: CUDA build of PyTorch for Qwen3-ASR / Nemotron 3.5 / Qwen3.5

See `data/DATASET_CARD.md` and `HARDWARE.md`.

## Commands

```text
pip install -r requirements.txt
python check_env.py
python fetch_models.py --list

# Pilot four (Whisper Medium/Large-v3, Wav2Vec2, SpeechBrain)
python run_asr.py --manifest data/manifest.jsonl --all-pilot

# 2026 refresh
python fetch_models.py --weights --only openai/whisper-large-v3-turbo
python run_asr.py --manifest data/manifest.jsonl --all-selected

# Resume a killed run (skip completed clip+model pairs)
python run_asr.py --manifest data/manifest.jsonl --model openai/whisper-medium --output outputs/asr-TIMESTAMP.jsonl --resume

# Per-language + 2026 tables (no invented cells)
# --write-tex is refused until every clip is confirmed=true and reference_source=gold
python report.py --asr outputs/asr-TIMESTAMP.jsonl
python report.py --asr outputs/asr-medium.jsonl outputs/asr-large-v3.jsonl
python report.py --asr outputs/asr-TIMESTAMP.jsonl --write-tex

# Cross-layer ablation: same LLM on every ASR hypothesis
python run_ablation.py --asr-outputs outputs/asr-TIMESTAMP.jsonl --model Qwen/Qwen3.5-4B
python report.py --asr outputs/asr-TIMESTAMP.jsonl --ablation outputs/ablation-TIMESTAMP.jsonl --write-tex

# Second annotator
python export_labels.py --input outputs/ablation-TIMESTAMP.jsonl --output annotation/labels_todo.csv
python iaa.py --a1 annotation/labels_a1.csv --a2 annotation/labels_a2.csv
python report.py --ablation outputs/ablation-TIMESTAMP.jsonl --labels annotation/labels_a1.csv
```

Whisper decoding defaults (paper Appendix B): `task=transcribe`, language forced from the manifest, sequential long-form `generate()` (30 s windows), `condition_on_prev_tokens=False` (HF name; openai-whisper: `condition_on_previous_text`), plus the official compression-ratio / logprob halt. Use `--language-detect` or `--condition-on-prev` only when you intend to.

## Compatibility rule

A 2026 model is selected only if it officially supports Turkish **and** English, or it is a documented Turkish Whisper fine-tune used only on TR-1..TR-3.

Rejected despite leaderboard rank: NVIDIA Parakeet TDT 0.6B v3, Canary 1B v2, Canary-Qwen 2.5B, Cohere Transcribe 03-2026, Fun-ASR-MLT-Nano. NVIDIA Nemotron 3.5 ASR is the NVIDIA model that *does* list Turkish and is in the selected set. Aya Expanse 8B lists Turkish but is gated + NC. Qwen3.8-27B and GPT-6 Astra were inspected and left out (size / one-day-old premium).

## Tests (no GPU, no wavs)

```text
python -m unittest tests.test_eval_kit
```
