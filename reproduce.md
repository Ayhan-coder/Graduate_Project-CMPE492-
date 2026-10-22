# EduTerm AI / Bitirme Projesi — Reproduction Guide (`reproduce.md`)

This document outlines the step-by-step instructions to reproduce the benchmarks, audits, engineering probes, and document compilations described in the project.

---

## 1. System Requirements & Environment

- **OS:** Windows 10/11 or Linux (Ubuntu 22.04+)
- **Python:** 3.10+
- **GPU:** NVIDIA GPU with $\ge 8$ GB VRAM (RTX 5060 Laptop tested locally)
- **LaTeX:** MiKTeX or TeX Live with `xelatex`, `beamer`, `beamerposter`, `pgfplots`
- **System Tools:** `ffmpeg`, `pdftoppm` (Poppler)

### Setup Python Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r experiments/requirements.txt
```

---

## 2. Dataset & Inventory Verification

The project manages an inventory of 40 educational and compliance videos (10.1 hours source metadata, of which 4.6 hours across 16/40 clips is duration-verified on disk; 20 TR + 20 EN across 14 domains). The measured ASR ground-truth benchmark is conducted on the 7-lecture human-verified gold test set (~95 minutes).

### Run Inventory Audit

```powershell
# Audits manifest entries, WPM rates, and gold transcript alignments
python experiments/data/audit_inventory.py

# Re-verifies corporate video records (KVKK/ISG) in the 40-video inventory
python experiments/data/append_corporate_videos.py
```

Generated inventory files:
- `experiments/data/data_inventory.json`
- `experiments/data/data_inventory.csv`

---

## 3. Acoustic ASR & SLM Benchmarking

### 3.1 Speech Recognition Evaluation (7 Gold Lectures)

Benchmark Whisper (Medium, Large-v3, Turbo) and evaluate WER/CER across the 7 verified test lectures:

```powershell
# 1. Run ASR transcription & WER/CER scoring against gold references:
python experiments/run_asr.py --manifest experiments/data/manifest.jsonl --all-selected

# 2. Aggregate language-level tables & 10,000-resample bootstrap 95% confidence intervals:
python experiments/report.py --asr experiments/outputs/asr.jsonl
```

Emitted artifact: `experiments/outputs/asr_by_lang.json` (contains language-level unweighted means, per-clip breakdown, and bootstrap `wer_ci`).

*(Note: To benchmark edge-device SRM and SLM inference latency and VRAM footprint separately, run `python experiments/benchmark_srm_slm.py`.)*

### 3.2 QLoRA Parameterized SLM Adapter Simulation (Dry-Run)

Simulate the domain-adapted LoRA fine-tuning architecture ($r=16, \alpha=32$) on Qwen2.5-0.5B:

```powershell
python experiments/finetune/finetune_qwen25.py --dry-run
```

Artifact emitted: `experiments/finetune/checkpoints/qwen25_lecture_adapter/adapter_config_simulated.json` (`status: DRY_RUN_PASSED`).

*(Note: To evaluate the empirical Whisper Small LoRA adapter on the 54-clip held-out validation set, run `python experiments/finetune/finetune_whisper_asr.py --eval-only`; outputs `experiments/finetune/checkpoints/srm_whisper_small_tr/eval_wer.json`.)*

---

## 4. Corporate Domain RAG & DAG Grounding (8-Clip Engineering Probe)

Runs the domain-filtered Hybrid BM25/TF-IDF retriever and NetworkX prerequisite DAG analyzer across the 8 long-form corporate compliance lectures (TR-17..20, EN-17..20) with single-pass Turkish suffix stripping, sliding window candidate matching (filler gap $\le 4$), and automated prerequisite gap detection:

```powershell
python experiments/rag/eval_corporate_rag.py
```

Validation unit tests:

```powershell
python -m unittest experiments/tests/test_domain_rag.py
```

Artifact emitted: `experiments/outputs/rag_corporate_results.json` (`status: ENGINEERING_PROBE_EVALUATED`, 60.7% candidate grounding precision, 17/28 candidates grounded: 7 exact canonical + 10 alias/subtype matches, with 5 pedagogical gap flags).

---

## 5. Recompiling Posters and Academic Reports

### 5.1 Recompiling Presentation Posters (A3 & A0)

```powershell
cd notes
xelatex -interaction=nonstopmode Bitirme_Projesi_A3_Posteri.tex
xelatex -interaction=nonstopmode Bitirme_Projesi_A0_Posteri.tex
pdftoppm -png -r 150 Bitirme_Projesi_A3_Posteri.pdf Bitirme_Projesi_A3_Posteri_Final
pdftoppm -png -r 150 Bitirme_Projesi_A0_Posteri.pdf Bitirme_Projesi_A0_Posteri_Final
cd ..
```

Output PDFs:
- `notes/Bitirme_Projesi_A3_Posteri.pdf`
- `notes/Bitirme_Projesi_A0_Posteri.pdf`

### 5.2 Recompiling Graduation Project Reports

The authoritative graduation thesis report is generated from `notes/Bitirme_Projesi_Raporu.md` via `md2tex.py` and compiled with LaTeX (`pdflatex` or `xelatex`):

```powershell
cd notes
python md2tex.py
pdflatex -interaction=nonstopmode Bitirme_Projesi_Raporu.tex
pdflatex -interaction=nonstopmode Bitirme_Projesi_Raporu.tex
python generate_revision_report.py
cd ..
```

*(Note: `python notes/generate_final_report_pdf.py` renders a legacy side-by-side ReportLab document to `Bitirme_Projesi_Raporu.legacy-reportlab.pdf` and will not overwrite the authoritative thesis deliverable unless `--allow-overwrite-report` is explicitly provided.)*

Output PDFs:
- `notes/Bitirme_Projesi_Raporu.pdf` (Authoritative thesis deliverable)
- `notes/What_Added_What_Improved.pdf` (Revision & expansion report)

---

## 6. Local Interactive Demonstration Studio

Launch the FastAPI web studio:

```powershell
uvicorn experiments.gui.server:app --host 127.0.0.1 --port 8000
```

Access the studio at `http://127.0.0.1:8000` and the A3 poster viewer at `http://127.0.0.1:8000/poster`.
