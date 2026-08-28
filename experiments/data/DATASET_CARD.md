# Dataset card — TR/EN educational lecture set

Follows the datasheets-for-datasets fields that this benchmark actually uses.
**Do not invent source URLs, licenses, or speaker counts.** Fill the `unknown` fields after you confirm the original files.

## Motivation

- **Why:** Gold audio + transcripts for a two-layer ASR then LLM terminology benchmark on beginner-oriented educational speech.
- **Funders / creators:** Cakil Yilmaz (Koç University) with Berat Enes Dadak, Efe Batur Giritli, Yekta Said Can (Tatilsepeti R&D).
- **Not for:** Training new ASR or LLM weights. Evaluation only.

## Composition (known from the paper)

| ID | Title (as in the paper) | Language | Domain | Duration |
|----|-------------------------|----------|--------|----------|
| TR-1 | Project Management, Unit 7, Lecture 1 | Turkish | Project management | ≈12 min |
| TR-2 | Tourism training for tour guides | Turkish | Tourism | 12 min |
| TR-3 | Intro to data science (Python) | Turkish | Data science | ≈14 min |
| EN-1 | Introduction to agile methods | English | Project management | ≈13 min |
| EN-2 | What is sustainable tourism? | English | Tourism | ≈9 min |
| EN-3 | Generative AI in a nutshell | English | AI | ≈18 min |
| EN-4 | Understanding thermal radiation | English | Heat transfer | ≈17 min |

- **Total:** 7 clips, ≈95 minutes.
- **Instances:** one 16 kHz mono wav + one reference transcript per ID.
- **Sensitive data:** educational lecture speech. No planned collection of student identities.
- **Missing / unknown (fill these):**
  - Source platform (YouTube / LMS / classroom recording)
  - Canonical URL or DOI per clip
  - License / permission to redistribute audio
  - Speaker count, gender, accent, native vs L2
  - Prepared (scripted studio) vs spontaneous classroom
  - How “similar terminological density” was measured
  - How the reference was produced (independent transcription vs correction of an ASR draft)

## Collection and preprocessing (known)

- Audio extracted from the source videos.
- Converted to **16 kHz, mono**.
- **No** denoising, clipping, or manual audio edit.
- Reference transcripts: “manual correction of the audio” and a human check (wording from the paper). Treat as gold for WER/CER only.

## Candidate public matches (unverified)

Title + duration matches that a reader might assume. **Do not cite these in the paper until an author confirms the file is the one that was used.**

| ID | Candidate | Duration match |
|----|-----------|----------------|
| EN-3 | Henrik Kniberg, *Generative AI in a Nutshell*, YouTube `2IK3DFHRFfw` | 17:57 vs ≈18 min |
| EN-4 | The Efficient Engineer, *Understanding Thermal Radiation*, YouTube `FDmYCI_xYlA` | 17:33 vs ≈17 min |

TR-1, TR-2, TR-3, EN-1, EN-2: no unique public match confirmed.

## Uses

- Allowed: reproduce Tables III–VI, the per-language ASR split, the ASR→LLM ablation, and the 2026 refresh.
- Not allowed: claiming the set is a public corpus until licenses are recorded.

## Distribution

- Audio and gold transcripts are **not** in this repository (size + likely third-party copyright).
- Drop local files as `experiments/data/TR-1.wav` … `EN-4.wav` and list them in `manifest.jsonl` (see `manifest.example.jsonl`).
- Redistribution of wavs requires the license field above.

## Maintenance

- Contact: dogacakilyilmaz2005@gmail.com
- Errors in IDs or titles: open an issue on the project repository.
