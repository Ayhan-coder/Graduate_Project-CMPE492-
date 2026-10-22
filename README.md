# Three-Layer Evaluation of ASR, LLM Terminology Extraction, and Generative Video-Animation Systems (TR / EN Educational Speech)

Comprehensive benchmark and open-weight evaluation framework for making multilingual (Turkish and English) educational and corporate compliance lectures accessible.

## Architecture Layers

1. **Layer 1 (Speech-to-Text / ASR):** Leading open-weight and commercial speech recognition models (Whisper Medium, Large-v3, Large-v3 Turbo, Turkish-matched Turbo, Qwen3-ASR 0.6B/1.7B, and NVIDIA Nemotron 3.5 ASR) benchmarked on GPU gold references across WER, CER, and RTF.
2. **Layer 2 (Semantic Analysis & Ontology Grounding):** LLM-based terminology extraction and CEFR-B1 simplified explanation generation evaluated across Ateşman (50–69) and Flesch (60–70) readability bands, Cohen's kappa inter-annotator agreement ($\kappa_R=0.782$, $\kappa_D=1.000$), closed 32-term gold lists, and hybrid RAG + NetworkX prerequisite DAG grounding.
3. **Layer 3 (Generative Explainer Video Animation):** On-demand personalized explainer video studio adapting open-source Manimator with Qwen 3.5 scenario generation, Edge-TTS neural voice synthesis, and FFmpeg synchronization.

| Directory / File | Description |
|---|---|
| `paper/main.tex` | IEEE conference paper source (compiles with pdfLaTeX on Overleaf / MiKTeX) |
| `notes/Bitirme_Projesi_Raporu.md` | Single source of truth for the comprehensive graduation thesis report |
| `notes/Bitirme_Projesi_Raporu.pdf` | 35-page compiled graduation thesis report |
| `experiments/` | Evaluation kit: ASR, LLM, ablation, fine-tuning, RAG, and table generation |
| `experiments/data/data_inventory.json` | 40-video balanced inventory (20 TR, 20 EN; STEM, Social Sciences, Corporate Compliance) |
| `experiments/data/DATASET_CARD.md` | Dataset documentation, sources, licensing, and annotation provenance |
| `experiments/finetune/` | LoRA fine-tuning scripts and evaluation checkpoints for Whisper Small TR |
| `experiments/rag/` | Hybrid BM25/TF-IDF RRF retriever, prerequisite DAG verifier, and grounding benchmarks |
| `gui/` | Interactive FastAPI local demo studio (`gui/app.py` on `localhost:7860`) |

## Quick Start & Reproduction

### 1. Environment Setup
```bash
pip install -r experiments/requirements.txt
python experiments/check_env.py
```

### 2. ASR Benchmarking & Corpus Pipeline
```bash
# Verify environment and hardware
python experiments/benchmark_srm_slm.py

# Long-form chunked transcription of corporate corpus (TR-17..20, EN-17..20)
python experiments/data/download_and_transcribe_corpus.py --corporate-only
```

### 3. Semantic Layer & RAG Grounding
```bash
# Evaluate cross-layer RAG grounding on corporate speech
python experiments/rag/eval_corporate_rag.py

# Run RAG ablation evaluation (synthetic baseline check)
python experiments/rag/domain_rag_extractor.py
```

### 4. Interactive Local Demo Studio
```bash
python run_app.py
# Opens browser at http://localhost:7860 (protected by local token)
```

## Scientific Honesty Guardrails

- **Held-Out Gold References:** ASR WER/CER scores are measured against human-corrected reference transcripts. Legacy English-only checkpoints without Turkish support (Wav2Vec 2.0 and SpeechBrain) remain documented as uncomputed (`---`) in multi-language comparison tables.
- **Precision vs. Recall Ceiling:** While LLM terminology precision is high, closed 32-term gold list recall remains bounded (<28.1% in TR, <56.2% in EN).
- **RAG Scoring Notice:** Domain RAG grounding is evaluated via surface-mention verification; synthetic queries derived from the knowledge base itself are non-citable as general accuracy numbers to avoid circularity (`paper/main.tex:198`).
- **Hardware Probes:** All reported VRAM usage and latencies reflect physical GPU measurements (`local_vram_probe.json` on NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB). Target profiles for unmeasured edge models are explicitly labeled as *Design Projections*.

## License

Evaluation code is released under the MIT License (`LICENSE`). Pre-trained model weights retain their respective upstream licenses. Audio clips are referenced from public educational YouTube channels under their respective licenses (CC0-1.0, CC-BY-SA, or Standard YouTube License).
