# EduTerm Human Evaluation V2 — Academic Methodology

## 1. Purpose and Scope
This protocol establishes a blinded, counterbalanced human evaluation study complementing automated benchmarks in the Boğaziçi University CMPE 492 thesis (*EduTerm*). It evaluates:
- **Pedagogical Socratic Tutoring (Section 5):** `task_separated_hybrid` vs. `onpolicy_dpo_v2` (from `hpc_7b_bf16_3seed_nonlatin_report.json`, Job `35651`). Note that because multi-turn live simulations dynamically sample student turns per condition (`same_input_controlled = false`), these comparisons evaluate unscripted multi-turn trajectories rather than controlled same-input turns.
- **Pedagogical Safety & Anti-Cheating (Section 7):** `base_14b` vs. `dpo_v4_guardrail_sysguard` (from `hpc_guardrail_v4_ablation_report.json`, Job `36389`) under identical adversarial user prompts (`same_input_controlled = true`).
- **Missing Stimulus Modules (Sections 2, 4, 6):** Excluded from synthetic substitution because `cefr_multi_level_benchmark_report.json` uses `model_backend="mock"` and aggregate CEFR/OOD reports did not persist raw explanation strings.

## 2. Counterbalancing and Blinding
Two equivalent Google Forms (**V1** and **V2**) invert A/B display positions (`V2` swaps `A` and `B` relative to `V1`). Model checkpoint identities remain hidden from participants and are resolved post-hoc via `human_eval_manifest.json`.

## 3. Coding and Statistical Plan
- Unratable responses (`"Değerlendiremiyorum"`) are coded as `NA` (never coerced to midpoint `3`).
- Distinct `model_pair_id` comparisons are analyzed strictly separately (never pooled into one global win rate).
- Participant-level mean scores per model are compared via paired Wilcoxon signed-rank tests with rank-biserial effect sizes, 2,000-replicate bootstrap 95% confidence intervals, and Holm–Bonferroni correction across multiple tests.
- Objective comprehension items are not attributed to a single model if a participant reads both A and B prior to answering.
