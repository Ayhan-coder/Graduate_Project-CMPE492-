# Two-layer ASR + LLM evaluation (TR / EN educational speech)

Benchmark of off-the-shelf speech recognition and terminology-extraction models on Turkish and English lecture video. Not a new architecture and not an adaptive tutor.

| Path | What it is |
|------|------------|
| `paper/main.tex` | IEEE conference source (single file; compiles on Overleaf) |
| `paper/original/` | Unchanged starting PDF |
| `experiments/` | Evaluation kit: ASR, LLM, ablation, IAA, table export |
| `experiments/data/DATASET_CARD.md` | Dataset card (source / license / speakers still unknown) |
| `experiments/HARDWARE.md` | Decoding flags + kit machine |
| `experiments/annotation/PROTOCOL.md` | Second-annotator protocol |
| `notes/` | Review and change log |

## Compile the paper

LaTeX may not be installed locally. On [Overleaf](https://www.overleaf.com): new project → upload `paper/main.tex` → pdfLaTeX, TeX Live 2024+. Figures are TikZ / pgfplots (no image files).

## Reproduce numbers (do not invent cells)

Pending tables in the paper (per-language ASR, 2026 refresh, ASR→LLM ablation, Cohen’s κ) stay `---` until you run:

```text
cd experiments
pip install -r requirements.txt
python check_env.py
# put TR-1.wav … EN-4.wav and data/manifest.jsonl here first
python run_asr.py --manifest data/manifest.jsonl --all-pilot
python report.py --asr outputs/asr-TIMESTAMP.jsonl
python run_ablation.py --asr-outputs outputs/asr-TIMESTAMP.jsonl --model gpt-4o-mini
python iaa.py --a1 annotation/labels_a1.csv --a2 annotation/labels_a2.csv
```

Full command list: `experiments/README.md`.

## What is already filled vs empty

- **Filled:** pooled pilot WER/CER/RTF and GPT-4.1 / GPT-4o-mini semantic scores from the original study.
- **Protocol ready, cells empty:** Table per-language ASR, Table 2026 refresh, Table ablation, κ.
- **Blocked on you:** the seven wavs + gold transcripts, OpenAI key (for GPT ablation), second annotator, dataset-card source/license/speakers, GitHub remote.

## License

Evaluation code: MIT (`LICENSE`). Model weights keep their own licenses (see `experiments/models.json`). Audio is not shipped.
