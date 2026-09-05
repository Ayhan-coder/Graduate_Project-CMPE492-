# Hugging Face model refresh (2026-09-04)

Live Hub query + config snapshot. No weights were pulled for the new IDs. No new WER/R1 numbers were written into the paper.

The 28 August 2026 snapshot is superseded. Pilot IDs are unchanged so the completed semantic table stays comparable.

## Kept (compatible with TR+EN educational speech)

| Role | Hub / API ID | Revision (prefix) | Why |
|------|--------------|-------------------|-----|
| Pilot baseline | `openai/whisper-medium` | `abdf7c39` | Original best WER |
| Pilot multilingual | `openai/whisper-large-v3` | `06f233fe` | Original multilingual ref; still latest official Whisper accuracy ckpt |
| 2026 speed | `openai/whisper-large-v3-turbo` | `41f01f3f` | Same 99 langs incl. `tr`; still latest official Whisper speed ckpt |
| 2026 SOTA multilingual | `Qwen/Qwen3-ASR-1.7B-hf` | `bcd2b5b7` | Official `tr`+`en`; Apache-2.0; transformers-native |
| 2026 efficient multilingual | `Qwen/Qwen3-ASR-0.6B-hf` | `7f1569a4` | Same language set, 0.6B |
| 2026 NVIDIA TR+EN | `nvidia/nemotron-3.5-asr-streaming-0.6b` | `1c8deaec` | June 2026; `tr-TR` and `en-US` are transcription-ready; OpenMDW 1.1 |
| Language-matched TR | `selimc/whisper-large-v3-turbo-turkish` | `e914cebb` | CV17 fine-tune; **TR subset only** |
| Pilot API capacity | `gpt-4.1` | — | Historical only; already-filled table. No new paid calls |
| Pilot API latency | `gpt-4o-mini` | — | Historical only; already-filled table. No new paid calls |
| Open LLM capacity | `Qwen/Qwen3.5-9B` | `c2022362` | Free replacement for GPT-5.6 Sol / Qwen3-8B |
| Open LLM latency | `Qwen/Qwen3.5-4B` | `851bf6e8` | Free default ablation LLM; replaces Qwen2.5-3B and GPT-5.6 Luna |

## Rejected / superseded

| Hub ID | Reason |
|--------|--------|
| `nvidia/parakeet-tdt-0.6b-v3` | 25 European languages, no Turkish |
| `nvidia/canary-1b-v2` | same European set, no Turkish |
| `nvidia/canary-qwen-2.5b` | English only |
| `CohereLabs/cohere-transcribe-03-2026` | 14 langs, no Turkish |
| `FunAudioLLM/Fun-ASR-MLT-Nano-2512` | 31 langs, no Turkish |
| `facebook/seamless-m4t-v2-large` | has TR, but NC license and 2024 translation stack |
| `CohereLabs/aya-expanse-8b` | has TR, but gated + CC-BY-NC-4.0 |
| `Qwen/Qwen3-8B` | superseded by Qwen3.5-9B |
| `Qwen/Qwen2.5-3B-Instruct` | superseded by Qwen3.5-4B |
| `Qwen/Qwen3.8-27B` | newest Qwen dense open model, too large for the kit RTX 5060 |
| `gpt-5.6-sol` | paid OpenAI API; replaced by Qwen3.5-9B |
| `gpt-5.6-luna` | paid OpenAI API; new ablation uses Qwen3.5-4B |
| `gpt-6-astra` | paid OpenAI API |

Parakeet/Canary remaining rejected is itself a paper point: English Open ASR leaders are still the wrong models for a Turkish–English educational benchmark. Nemotron 3.5 is the NVIDIA exception that now lists Turkish.

## What is on disk

- `experiments/models.json` — registry with pinned revisions (fetched_at 2026-09-04)
- `experiments/hf_snapshots/` — `config.json` / tokenizer / README for each selected Hugging Face model
- `experiments/run_asr.py`, `run_llm.py` — eval entry points (Nemotron backend; Qwen3.5 text-only with thinking off)

## What is still needed to fill the new tables

1. The 7 lecture wavs + gold transcripts as `data/manifest.jsonl` (`confirmed=true`, `reference_source=gold`)
2. A GPU (Qwen3-ASR-1.7B, Nemotron 3.5, and Qwen3.5-9B are not practical on CPU)
3. `python fetch_models.py --weights` for the models you want to run first (start with Whisper Turbo + Nemotron 3.5 + Qwen3.5-4B)
4. No `OPENAI_API_KEY` is required for the 2026 refresh. Download Qwen3.5-4B first: `python fetch_models.py --weights --only Qwen/Qwen3.5-4B`
