# Pass 2 status

Started 2026-08-28. No experimental number was invented.

| Item | In the repo | In the paper | Blocked on |
|------|-------------|--------------|------------|
| Per-language ASR table | `report.py` aggregates `run_asr.py` JSONL by `lang` | Table `tab:asr-lang` (`---`) | 7 wavs + gold `manifest.jsonl` |
| Cross-layer ablation | `run_ablation.py` (default GPT-4o-mini) | Table `tab:ablation` (`---`) | ASR JSONL + API key + R/D labels |
| Decoding / hardware appendix | `HARDWARE.md`, `check_env.py`, sidecar `.meta.json` | Appendix B | Original pilot CPU model (still unknown) |
| Second annotator + IAA | `annotation/PROTOCOL.md`, `iaa.py`, example CSV | Appendix C, Method III-E | A second person and real labels |
| Dataset card | `data/DATASET_CARD.md` | §III-B points at the card | Source, license, speakers |
| Shared GitHub repo | Local git + MIT + `.gitignore` | “accompanying repository” | GitHub remote / `gh` |
| 2026 model scores | `--all-selected`, Table `tab:asr-2026` | Empty cells | wavs + GPU wheel for Qwen3 |

Kit development machine (not the pilot host): Intel Core i7-14650HX, 32 GB RAM, RTX 5060 Laptop, PyTorch 2.12.0+cpu.
