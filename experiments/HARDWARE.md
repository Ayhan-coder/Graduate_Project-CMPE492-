# Decoding settings and hardware

Two different machines/settings must not be mixed.

## Pilot (published WER / RTF in the paper)

- Device: CPU (paper). **CPU model was not recorded.**
- Checkpoints: Whisper `medium` / `large-v3`; `facebook/wav2vec2-large-xlsr-53`; SpeechBrain “default greedy/beam” (Hub ID not pinned).
- Whisper `language`, `condition_on_previous_text`, beam size, and chunking: **not recorded.** Those flags are the most likely explanation of Medium 0.021 vs Large-v3 0.129 normalized WER.
- Treat the published pilot numbers as configuration-sensitive, not as a general ranking of Medium above Large-v3.

## Refresh kit (this repository)

Defaults in `run_asr.py`:

| Flag | Default | Why |
|------|---------|-----|
| `task` | `transcribe` | never translate |
| `language` | forced `tr` or `en` from the manifest | avoids Large-v3 language-detection drift |
| `--language-detect` | off | turn on only to reproduce auto-detect |
| `chunk_length_s` | 30 | Whisper receptive field; sequential `generate()` slides this window |
| `longform` | `sequential_generate` | not `pipeline(..., chunk_length_s=)` — that path stalled on TR-1 |
| `return_timestamps` | True | required for long-form Whisper |
| `condition_on_prev_tokens` | False (`--condition-on-prev` to enable) | HF name for openai-whisper `condition_on_previous_text`; known Large-v3 long-form failure mode |
| halt / fallback | compression 1.35, logprob −1.0, temperature sweep | stops the repetition loop that hung the first GPU run |
| `max_new_tokens` | 440 | Whisper decoder is 448 positions; language/task tokens already occupy a few |
| WER/CER normalize | lowercase + strip punct; Turkish `I`/`İ` → `ı`/`i` first | default Unicode `.lower()` is wrong for Turkish |
| device | CUDA if `torch.cuda.is_available()` else CPU | |

Every run writes `outputs/asr-TIMESTAMP.meta.json` with the flags and the host identity.

## Machine used to develop the kit (28 August–4 September 2026)

Recorded by `check_env.py` / WMI on the author laptop. **This is not claimed to be the pilot machine.**

| Item | Value |
|------|-------|
| CPU | Intel Core i7-14650HX (16 cores / 24 threads) |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 5060 Laptop |
| OS | Windows |
| Python | 3.12.9 |
| PyTorch present at kit time | 2.12.0+**cpu** (CUDA build not installed) |
| PyTorch after 29 Aug 2026 | `2.12.0+cu130` (CUDA build present) |
| GPU evening 29 Aug 2026 | RTX 5060 `CM_PROB_PHANTOM` after S0 sleep; `torch.cuda.is_available()=False` until the dGPU is reattached |
| GPU 30 Aug – 2 Sep 2026 | RTX 5060 OK; `cuda=True`; Whisper Medium complete (`outputs/asr-20260830-151020.jsonl`, 7/7) and Whisper Large-v3 complete (`outputs/asr-20260901-large-v3.jsonl`, 7/7). Caption-draft diagnostics only. |

Qwen3-ASR 1.7B, Nemotron 3.5 ASR, and Qwen3.5-9B need this CUDA wheel and the laptop GPU (or another GPU). Do not run them on a CPU-only wheel. After modern standby, check `python check_env.py` — if `cuda=False`, reboot or wake the RTX 5060 before a full run.

Nemotron 3.5 ASR (`transformers>=5.13`) decodes long-form lecture audio in non-overlapping 60 s windows with `language=tr-TR` or `en-US`. Qwen3.5 thinking mode is disabled for the JSON term-extraction prompt.

```text
python check_env.py
```
