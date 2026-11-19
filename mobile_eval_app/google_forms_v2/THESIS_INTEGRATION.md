# Thesis Integration Draft (§7 / §8 & §9.2)

## Blind Human Evaluation of Pedagogical Quality, Student Preferences, and Perceived Cognitive Load

### Research Questions
- **RQ1–RQ3 (Explanation Clarity & Cognitive Load):** Do EduTerm-aligned explanations improve perceived clarity and reduce mental effort compared with base `Qwen2.5-14B-Instruct`, and do perceived difficulty ratings align with automated readability metrics? *(Pending verbatim stimulus generation for Section 2/4).*
- **RQ4 (Socratic Tutoring Quality):** How do human learners rate `onpolicy_dpo_v2` relative to `task_separated_hybrid` across clarity, pedagogical tone, Socratic encouragement, logical linking, misconception handling, and guidance utility (aligned with Table 8.28 dimensions)?
- **RQ5 (Out-of-Domain Generalization):** Do pedagogical gains transfer to domains verified as absent from training corpora? *(Pending non-overlapping raw OOD stimulus extraction).*
- **RQ6 (Pedagogical Safety vs. Utility):** Does `dpo_v4_guardrail_sysguard` preserve perceived pedagogical usefulness and respectful tone while upholding academic integrity compared with `base_14b`?
- **RQ7–RQ8 (Human–LLM Agreement & Prior Knowledge):** To what extent do human preferences align with automated LLM-as-a-Judge verdicts (§8.15, §8.33), and how does domain prior knowledge moderate preferences?

### Participant Recruitment and Ethical Considerations
Participation is voluntary, anonymous, and preceded by explicit informed consent in Turkish. Declining consent branches directly to form submission without collecting demographic or evaluation responses. No names, student IDs, or email addresses are collected (`setCollectEmail(false)`).

### Stimulus Selection and Provenance
All verified stimuli are drawn verbatim without manual editing from archived HPC runs (`human_eval_manifest.json`): Jobs `35651` (`hpc_7b_bf16_3seed_nonlatin_report.json`) and `36389` (`hpc_guardrail_v4_ablation_report.json`). Mock artifacts (`cefr_multi_level_benchmark_report.json`) are strictly excluded.

### Counterbalanced Experimental Design
Two form variants (**V1** and **V2**) invert A/B ordering to mitigate position bias. Both models in each pair are rated independently before an overall preference item (`A`, `B`, `Equal`, `Neither`, `Undecided`).

### Evaluation Rubrics
Socratic tutoring items use 6 dimensions on a 1–5 Likert scale plus `"Değerlendiremiyorum"` (`NA`). Safety items use 5 dimensions (academic integrity, pedagogical utility, alternative help sufficiency, self-solution support, respectful tone). Mental effort uses a 1–9 scale.

### Statistical Analysis
Analyses are conducted separately per `model_pair_id` using participant-clustered means, paired Wilcoxon signed-rank tests, rank-biserial effect sizes, 95% bootstrap CIs, and Holm–Bonferroni multiplicity adjustment (`analyze_human_eval.py`).

### Results *(Empty Template — No Real Human Data Collected Yet)*

| Model Pair (`model_pair_id`) | $N$ Participants | Model A Median (IQR) | Model B Median (IQR) | Wilcoxon $p$ (Holm adj.) | Effect Size ($r_{rb}$) | Win / Tie / Neither |
|---|---|---|---|---|---|---|
| `task_separated_hybrid` vs. `onpolicy_dpo_v2` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `base_14b` vs. `dpo_v4_guardrail_sysguard` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

### Threats to Validity (Relation to §9.2)
- **Partially Addressed (§9.2):** Replaces single-annotator (`A1`) candidate auditing with blinded multi-participant Turkish evaluation across Socratic and anti-cheating dimensions.
- **Remaining Limitations:** (1) Multi-turn Socratic trajectories in `Job 35651` have distinct simulator-generated student turns per arm (`same_input_controlled = false`); (2) short single-session ratings do not establish causal long-term learning gains, eye-tracking fixation, or live BKT mastery; (3) Google Forms lacks native per-participant randomization, requiring balanced V1/V2 assignment.
