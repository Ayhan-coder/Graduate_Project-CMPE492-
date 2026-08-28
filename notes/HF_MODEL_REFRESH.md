# Hugging Face model refresh (2026-08-28)

Live Hub query + config snapshot. No weights were pulled. No new WER/R1 numbers were written into the paper.

## Kept (compatible with TR+EN educational speech)

| Role | Hub ID | Revision (prefix) | Why |
|------|--------|-------------------|-----|
| Pilot baseline | `openai/whisper-medium` | `abdf7c39` | Original best WER |
| Pilot multilingual | `openai/whisper-large-v3` | `06f233fe` | Original multilingual ref |
| 2026 speed | `openai/whisper-large-v3-turbo` | `41f01f3f` | Same 99 langs incl. `tr`; 809M; MIT; 7.0M downloads |
| 2026 SOTA multilingual | `Qwen/Qwen3-ASR-1.7B-hf` | `bcd2b5b7` | Official `tr`+`en`; Apache-2.0; transformers-native; updated 2026-07-22 |
| 2026 efficient multilingual | `Qwen/Qwen3-ASR-0.6B-hf` | `7f1569a4` | Same language set, 0.6B |
| Language-matched TR | `selimc/whisper-large-v3-turbo-turkish` | `e914cebb` | CV17 fine-tune; **TR subset only** |
| Open LLM capacity | `Qwen/Qwen3-8B` | `b968826d` | Ungated, Apache-2.0, 13.6M downloads |
| Open LLM latency | `Qwen/Qwen2.5-3B-Instruct` | `aa8e7253` | Ungated counterpart to GPT-4o-mini |

GPT-4.1 and GPT-4o-mini stay as the API pair so the pilot semantic table remains comparable.

## Rejected

| Hub ID | Reason |
|--------|--------|
| `nvidia/parakeet-tdt-0.6b-v3` | 25 European languages, no Turkish |
| `nvidia/canary-1b-v2` | same European set, no Turkish |
| `nvidia/canary-qwen-2.5b` | English only |
| `CohereLabs/cohere-transcribe-03-2026` | 14 langs, no Turkish |
| `facebook/seamless-m4t-v2-large` | has TR, but NC license and 2024 translation stack |
| `CohereLabs/aya-expanse-8b` | has TR, but gated + CC-BY-NC-4.0 |

This rejection list is itself a paper point: the 2026 English Open ASR leaders are still the wrong models for a Turkish–English educational benchmark.

## What is on disk

- `experiments/models.json` — registry with pinned revisions
- `experiments/hf_snapshots/` — `config.json` / tokenizer / README for each selected model
- `experiments/hf_models_snapshot.json` — Hub metadata at fetch time
- `experiments/run_asr.py`, `run_llm.py` — eval entry points

## What is still needed to fill the new tables

1. The 7 lecture wavs + gold transcripts as `data/manifest.jsonl`
2. A GPU (Qwen3-ASR-1.7B and Qwen3-8B are not practical on CPU)
3. `python fetch_models.py --weights` for the models you want to run first (start with Whisper Turbo + Qwen2.5-3B-Instruct)
