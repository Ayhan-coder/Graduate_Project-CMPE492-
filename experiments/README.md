# Experiment kit

Reproduce the pilot tables, the **per-language ASR split**, the **ASR→LLM ablation**, the **2026 Hugging Face refresh**, and **Cohen’s κ**. Do not type numbers into the paper by hand — run the scripts and let `report.py` build the tables.

Configs for the 2026 models are in `hf_snapshots/`. Full weights are not downloaded.

## What you must drop in

1. Seven 16 kHz mono wavs: `data/TR-1.wav` … `data/EN-4.wav`
2. Gold transcripts in `data/manifest.jsonl` (copy `data/manifest.example.jsonl`)
3. Optional: `OPENAI_API_KEY` for GPT-4o-mini / GPT-4.1
4. Optional: CUDA build of PyTorch for Qwen3-ASR / Qwen3-8B

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

# Per-language + 2026 tables (no invented cells)
python report.py --asr outputs/asr-TIMESTAMP.jsonl --write-tex

# Cross-layer ablation: same LLM on every ASR hypothesis
python run_ablation.py --asr-outputs outputs/asr-TIMESTAMP.jsonl --model gpt-4o-mini
python report.py --asr outputs/asr-TIMESTAMP.jsonl --ablation outputs/ablation-TIMESTAMP.jsonl --write-tex

# Second annotator
python export_labels.py --input outputs/ablation-TIMESTAMP.jsonl --output annotation/labels_todo.csv
python iaa.py --a1 annotation/labels_a1.csv --a2 annotation/labels_a2.csv
python report.py --ablation outputs/ablation-TIMESTAMP.jsonl --labels annotation/labels_a1.csv
```

Whisper decoding defaults (paper Appendix B): `task=transcribe`, language forced from the manifest, 30 s chunks, `condition_on_previous_text=False`. Use `--language-detect` or `--condition-on-prev` only when you intend to.

## Compatibility rule

A 2026 model is selected only if it officially supports Turkish **and** English, or it is a documented Turkish Whisper fine-tune used only on TR-1..TR-3.

Rejected despite leaderboard rank: NVIDIA Parakeet TDT 0.6B v3, Canary 1B v2, Canary-Qwen 2.5B, Cohere Transcribe 03-2026. Aya Expanse 8B lists Turkish but is gated + NC.

## Tests (no GPU, no wavs)

```text
python -m unittest tests.test_eval_kit
```
