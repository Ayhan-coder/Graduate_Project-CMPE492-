"""Phase 0: Synthetic-vs-Real Hold-Out Fidelity Metrics.

Computes statistical fidelity between simulated student trajectories and
unseen real student interaction sequences (e.g., ASSISTments hold-out test split):
- Completion / mastery rate absolute error (percentage points) + bootstrap 95% CI
- Attempts-per-skill / interaction-count distribution (Wasserstein distance & KS test)
- Learning curve P(correct | attempt k = 1..K) RMSE
- Skill/concept difficulty rank correlation (Spearman rho)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def compute_learning_curve(sequences: Sequence[Sequence[int]], max_k: int = 12) -> List[float]:
    """Compute empirical P(correct at attempt k) for k=1..max_k across sequences."""
    counts = np.zeros(max_k, dtype=np.float64)
    corrects = np.zeros(max_k, dtype=np.float64)
    for seq in sequences:
        for idx, val in enumerate(seq[:max_k]):
            counts[idx] += 1.0
            corrects[idx] += float(val)
    curve = []
    for i in range(max_k):
        curve.append(float(corrects[i] / counts[i]) if counts[i] > 0 else 0.5)
    return curve


def bootstrap_rate_gap_ci(
    real_rates: np.ndarray,
    sim_rates: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute |sim_mean - real_mean| in percentage points with 95% bootstrap CI."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    nr, ns = len(real_rates), len(sim_rates)
    for b in range(n_boot):
        r_idx = rng.integers(0, nr, size=nr)
        s_idx = rng.integers(0, ns, size=ns)
        diffs[b] = (sim_rates[s_idx].mean() - real_rates[r_idx].mean()) * 100.0
    point = float((sim_rates.mean() - real_rates.mean()) * 100.0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "real_pct": round(float(real_rates.mean() * 100.0), 2),
        "sim_pct": round(float(sim_rates.mean() * 100.0), 2),
        "signed_gap_pp": round(point, 2),
        "abs_gap_pp": round(abs(point), 2),
        "ci95_low_pp": round(float(lo), 2),
        "ci95_high_pp": round(float(hi), 2),
    }


def evaluate_fidelity(
    real_sequences: Sequence[Sequence[int]],
    sim_sequences: Sequence[Sequence[int]],
    real_skill_error_rates: Dict[int, float] | None = None,
    sim_skill_error_rates: Dict[int, float] | None = None,
    max_k: int = 12,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare simulated student interaction sequences against real hold-out sequences."""
    real_lens = np.array([len(s) for s in real_sequences], dtype=np.float64)
    sim_lens = np.array([len(s) for s in sim_sequences], dtype=np.float64)

    # Accuracy / mastery proxy per sequence (final-state or overall correctness >= 0.70)
    real_acc = np.array([np.mean(s) for s in real_sequences if len(s) > 0], dtype=np.float64)
    sim_acc = np.array([np.mean(s) for s in sim_sequences if len(s) > 0], dtype=np.float64)

    real_mastery_flag = (real_acc >= 0.70).astype(np.float64)
    sim_mastery_flag = (sim_acc >= 0.70).astype(np.float64)

    mastery_gap = bootstrap_rate_gap_ci(real_mastery_flag, sim_mastery_flag, seed=seed)
    acc_gap = bootstrap_rate_gap_ci(real_acc, sim_acc, seed=seed)

    # Distribution distances on sequence length (attempts per student-skill)
    wass = float(stats.wasserstein_distance(real_lens, sim_lens))
    ks_stat, ks_p = stats.ks_2samp(real_lens, sim_lens)

    # Learning curve RMSE
    real_lc = np.array(compute_learning_curve(real_sequences, max_k=max_k), dtype=np.float64)
    sim_lc = np.array(compute_learning_curve(sim_sequences, max_k=max_k), dtype=np.float64)
    lc_rmse = float(np.sqrt(np.mean((real_lc - sim_lc) ** 2)))

    # Optional skill difficulty bottleneck rank correlation
    spearman_rho = None
    spearman_p = None
    if real_skill_error_rates and sim_skill_error_rates:
        common = sorted(set(real_skill_error_rates.keys()) & set(sim_skill_error_rates.keys()))
        if len(common) >= 5:
            r_vals = [real_skill_error_rates[k] for k in common]
            s_vals = [sim_skill_error_rates[k] for k in common]
            rho, pval = stats.spearmanr(r_vals, s_vals)
            if not np.isnan(rho):
                spearman_rho = round(float(rho), 4)
                spearman_p = round(float(pval), 6)

    return {
        "n_real_sequences": len(real_sequences),
        "n_sim_sequences": len(sim_sequences),
        "mastery_rate_gap": mastery_gap,
        "mean_accuracy_gap": acc_gap,
        "attempts_wasserstein": round(wass, 4),
        "attempts_ks_stat": round(float(ks_stat), 4),
        "attempts_ks_pvalue": round(float(ks_p), 6),
        "learning_curve_rmse": round(lc_rmse, 4),
        "real_learning_curve": [round(float(x), 4) for x in real_lc],
        "sim_learning_curve": [round(float(x), 4) for x in sim_lc],
        "skill_bottleneck_spearman_rho": spearman_rho,
        "skill_bottleneck_spearman_p": spearman_p,
    }

