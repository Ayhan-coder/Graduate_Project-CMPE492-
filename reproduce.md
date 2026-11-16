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

### Scope Reconciliation, Provenance & Reproducibility Registries (`P1` / `P2`)

Reconcile all six evaluation scopes (`7 / 33 / 40 / 99 / 125 / 133`), audit `180` candidate human `A1` labels (`59` direct CSV exports vs. `121` browser-state recovered records; `55` pending EN items), and regenerate Slurm job / Hugging Face revision / checkpoint SHA-256 manifests:

```powershell
python experiments/data/build_audit_registries.py
```

Outputs:
- `experiments/outputs/scope_reconciliation_matrix.json`
- `experiments/outputs/annotation_provenance_180.json`
- `experiments/outputs/reproducibility_registry.json`

### 100+ Machine Corpus Extraction, Stratified Sampling & Human Queue Forms (`P1`)

Resume machine term extraction over the 99 fetched lectures (`manifest_100plus.jsonl`), build the deterministic ~180 stratified evaluation sample, and generate the single-annotator (`A1`) HTML forms:

```powershell
# 1. Resume/complete machine term candidates from experiments/outputs/machine_terms_cache_100plus.json
python -u experiments/data/build_item_sheet_32_machine.py `
  --manifest experiments/data/manifest_100plus.jsonl `
  --out experiments/annotation/labels_todo_100plus_machine.csv `
  --cache experiments/outputs/machine_terms_cache_100plus.json `
  --summary experiments/outputs/machine_labels_100plus_summary.json

# 2. Construct deterministic stratified ~180 sample & offline A1 HTML annotation form
python experiments/data/build_sample_100plus.py
python experiments/annotation/make_annotation_form.py --annotator a1 --items 100plus-sample

# 3. Render HTML form for the remaining 55 EN human-queue items from the 32-clip pool
python experiments/annotation/make_annotation_form.py --annotator a1 --items experiments/annotation/labels_todo_human_queue.csv
```

### Cross-Family Judge Audit (`P0.2`)

Strictly separate the `184` verified non-fallback `Qwen/Qwen2.5-7B-Instruct` judge verdicts from the `270` `"Parsing fallback"` (`R=1, D=1`) defaults under `max_new_tokens=65`, and recompute agreement against `mistralai/Mistral-7B-Instruct-v0.3` (`0/454` fallbacks) and single human annotator `A1`:

```powershell
python experiments/annotation/recompute_cross_family_clean.py
```

Outputs:
- `experiments/annotation/labels_panel_qwen25_7b_a40_clean184.csv`
- `experiments/annotation/labels_panel_qwen25_7b_a40_fallback270.csv`
- `experiments/outputs/cross_family_judge_report.json`

---

## 3. Acoustic ASR & SLM Benchmarking

### 3.1 Speech Recognition Evaluation (7 Gold Lectures & SRM Generated-Token Audit)

Benchmark Whisper (Medium, Large-v3, Turbo) and evaluate WER/CER across the 7 verified test lectures:

```powershell
# 1. Run ASR transcription & WER/CER scoring against gold references:
python experiments/run_asr.py --manifest experiments/data/manifest.jsonl --all-selected

# 2. Aggregate language-level tables & 10,000-resample bootstrap 95% confidence intervals:
python experiments/report.py --asr experiments/outputs/asr.jsonl

# 3. Measure SRM generated token counts (tiny / base / small / qwen3asr) on local RTX 5060:
python -u experiments/benchmark_srm_asr_local.py --models tiny base small qwen3asr
python experiments/analyze_rtf_paths.py
```

Emitted artifacts: `experiments/outputs/asr_by_lang.json`, `experiments/outputs/srm_asr_measured.json`, `experiments/outputs/rtf_path_audit.json`.

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

## 5. Recompiling Posters, IEEE Paper, and Thesis Reports

### 5.1 Recompiling Presentation Posters (A3 & A0 TR/EN)

```powershell
cd notes
xelatex -interaction=nonstopmode Bitirme_Projesi_A3_Posteri.tex
xelatex -interaction=nonstopmode Bitirme_Projesi_A0_Posteri.tex
xelatex -interaction=nonstopmode Bitirme_Projesi_A0_Posteri_EN.tex
pdftoppm -png -r 150 Bitirme_Projesi_A3_Posteri.pdf Bitirme_Projesi_A3_Posteri_Final
pdftoppm -png -r 150 Bitirme_Projesi_A0_Posteri.pdf Bitirme_Projesi_A0_Posteri_Final
cd ..
```

### 5.2 Recompiling CMPE 492 Final Report (ENG), Comprehensive Report (TR), & IEEE Paper

```powershell
cd notes
pdflatex -interaction=nonstopmode Boğaziçi_CMPE492_Final_Report.tex
pdflatex -interaction=nonstopmode Boğaziçi_CMPE492_Final_Report.tex
python md2tex.py
pdflatex -interaction=nonstopmode Bitirme_Projesi_Raporu.tex
pdflatex -interaction=nonstopmode Bitirme_Projesi_Raporu.tex
cd ..\paper
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
cd ..
```

### 5.3 Running Core Unit Tests

```powershell
python -m unittest experiments/tests/test_eval_kit.py experiments/tests/test_benchmark_srm_slm.py experiments/tests/test_domain_rag.py
```

Output PDFs (mirrored into `GUNCEL_TESLIM_PDFLERI/`):
- `notes/Boğaziçi_CMPE492_Final_Report.pdf` (`1_Bogazici_CMPE492_Final_Report_ENG.pdf`)
- `notes/Bitirme_Projesi_Raporu.pdf` (`2_Bitirme_Projesi_Kapsamli_Rapor_TR.pdf`)
- `notes/Bitirme_Projesi_A0_Posteri.pdf` (`3_Bitirme_Projesi_A0_Sunum_Posteri.pdf`)
- `notes/Bitirme_Projesi_A3_Posteri.pdf` (`4_Bitirme_Projesi_A3_Masaustu_Posteri.pdf`)
- `paper/main.pdf` (`5_Akademik_Bildiri_IEEE_Paper.pdf`)

---

## 6. Local Interactive Demonstration Studio

Launch the FastAPI web studio:

```powershell
uvicorn experiments.gui.server:app --host 127.0.0.1 --port 8000
```

Access the studio at `http://127.0.0.1:8000` and the A3 poster viewer at `http://127.0.0.1:8000/poster`.
