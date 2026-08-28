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
| `chunk_length_s` | 30 | long-form lectures |
| `return_timestamps` | True | required for chunked Whisper |
| `condition_on_previous_text` | False (`--condition-on-prev` to enable) | known Large-v3 long-form failure mode |
| device | CUDA if `torch.cuda.is_available()` else CPU | |

Every run writes `outputs/asr-TIMESTAMP.meta.json` with the flags and the host identity.

## Machine used to develop the kit (28 August 2026)

Recorded by `check_env.py` / WMI on the author laptop. **This is not claimed to be the pilot machine.**

| Item | Value |
|------|-------|
| CPU | Intel Core i7-14650HX (16 cores / 24 threads) |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 5060 Laptop |
| OS | Windows |
| Python | 3.12.9 |
| PyTorch present at kit time | 2.12.0+**cpu** (CUDA build not installed) |

Qwen3-ASR 1.7B and Qwen3-8B need a CUDA build of PyTorch and the laptop GPU (or another GPU). Do not run them on the CPU-only wheel.

```text
python check_env.py
```
