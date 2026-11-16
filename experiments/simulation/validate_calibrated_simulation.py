"""Phase 4: Hold-Out Synthetic-vs-Real Fidelity Validation & Calibrated ISG DAG Simulation.

1. Loads `experiments/outputs/calibration.json` (fitted on ASSISTments `builder_train.csv`).
2. Evaluates synthetic-vs-real fidelity on the unseen hold-out test split
   (`builder_test.csv`, 856 real students) across an M0->M3 calibration ablation:
   - M0: Original hardcoded COHORTS (uncalibrated)
   - M1: Global EM-fitted BKT parameters
   - M2: Data-driven GMM 4 cohorts + survival-fitted patience
   - M3: Full calibrated model (+ empirical inter-practice forgetting lambda)
3. Runs the 12-node ISG Prerequisite DAG policies (Linear, Naive, DAG-Guided)
   using `spawn_calibrated_population()` across 5 seeds x 1,000 agents.

Outputs:
  experiments/outputs/calibrated_validation_report.json
  notes/figures/calibrated_fidelity_validation.png
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibrate_from_real_data import (
    DATA_DIR,
    extract_skill_sequences,
    load_piech_assistments,
)
from fidelity_metrics import evaluate_fidelity
from run_hybrid_student_simulation import HybridStudentSimulator
from run_student_journey_simulation import (
    COHORTS,
    StudentAgent,
    build_isg_curriculum_dag,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CalibratedValidation")

ROOT = Path(__file__).resolve().parents[2]
CAL_PATH = ROOT / "experiments" / "outputs" / "calibration.json"
OUT_JSON = ROOT / "experiments" / "outputs" / "calibrated_validation_report.json"
OUT_FIG = ROOT / "notes" / "figures" / "calibrated_fidelity_validation.png"


def simulate_kt_sequences_for_fidelity(
    mode: str,
    calibration: Dict[str, Any],
    real_lengths: List[int],
    seed: int = 42,
) -> List[List[int]]:
    """Simulate student-skill binary sequences matching hold-out length sample size."""
    rng = np.random.default_rng(seed)
    n_seqs = len(real_lengths)
    sim_seqs: List[List[int]] = []

    global_bkt = calibration["global_bkt_em"]
    cal_cohorts = calibration["cohorts"]
    cohort_names = list(cal_cohorts.keys())
    cal_props = np.array([cal_cohorts[c]["proportion"] for c in cohort_names], dtype=np.float64)
    cal_props /= cal_props.sum()

    hard_names = list(COHORTS.keys())
    hard_props = np.array([COHORTS[c].proportion for c in hard_names], dtype=np.float64)
    hard_props /= hard_props.sum()

    base_lam = float(calibration["forgetting_decay"]["fitted_lambda_decay"])

    for i in range(n_seqs):
        seq_len = int(real_lengths[i])
        if mode == "M0_hardcoded":
            cname = rng.choice(hard_names, p=hard_props)
            cfg = COHORTS[cname]
            p_l = float(np.clip(rng.normal(cfg.prior_mean, cfg.prior_std), 0.01, 0.95))
            p_t = float(np.clip(rng.normal(cfg.learn_rate_mean, cfg.learn_rate_std), 0.05, 0.90))
            p_s, p_g = cfg.slip, cfg.guess
            lam = 0.0
        elif mode == "M1_em_bkt_global":
            p_l = float(global_bkt["p_prior"])
            p_t = float(global_bkt["p_learn"])
            p_s = float(global_bkt["p_slip"])
            p_g = float(global_bkt["p_guess"])
            lam = 0.0
        else:
            cname = rng.choice(cohort_names, p=cal_props)
            cc = cal_cohorts[cname]
            p_l = float(np.clip(rng.normal(cc["prior_mean"], cc["prior_std"]), 0.02, 0.85))
            p_t = float(np.clip(rng.normal(cc["learn_rate_mean"], cc["learn_rate_std"]), 0.03, 0.75))
            p_s = float(cc["slip"])
            p_g = float(cc["guess"])
            lam = float(cc["decay_rate"]) if mode == "M3_calibrated_decay" else 0.0

        seq: List[int] = []
        known = bool(rng.random() < p_l)
        for step in range(seq_len):
            if lam > 0.0 and known and step > 0:
                # Empirical inter-practice forgetting probability per step
                if rng.random() < (1.0 - math.exp(-lam)):
                    known = False
            obs = int(rng.random() > p_s) if known else int(rng.random() < p_g)
            seq.append(obs)
            if not known and rng.random() < p_t:
                known = True
        sim_seqs.append(seq)
    return sim_seqs


def spawn_calibrated_population(
    dag,
    calibration: Dict[str, Any],
    num_agents: int = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Spawn N student agents using empirically fitted cohorts, BKT, survival, and decay."""
    rng = np.random.default_rng(seed)
    cal_cohorts = calibration["cohorts"]
    topo = list(nx_nodes(dag))

    # Allocate exact proportions
    wrapped: List[Dict[str, Any]] = []
    counts = {}
    remaining = num_agents
    cnames = list(cal_cohorts.keys())
    for idx, cname in enumerate(cnames):
        if idx == len(cnames) - 1:
            counts[cname] = remaining
        else:
            c = int(round(num_agents * cal_cohorts[cname]["proportion"]))
            counts[cname] = c
            remaining -= c

    for cname, count in counts.items():
        cc = cal_cohorts[cname]
        for i in range(count):
            p_l0 = float(np.clip(rng.normal(cc["prior_mean"], cc["prior_std"]), 0.02, 0.85))
            p_t = float(np.clip(rng.normal(cc["learn_rate_mean"], cc["learn_rate_std"]), 0.03, 0.75))
            agent = StudentAgent(
                agent_id=f"cal_{cname}_{i:04d}",
                cohort_key=cname,
                p_prior=p_l0,
                p_learn=p_t,
                p_slip=float(cc["slip"]),
                p_guess=float(cc["guess"]),
                patience=int(cc["patience"]),
                knowledge_states={n: p_l0 for n in topo},
            )
            wrapped.append(
                {
                    "agent": agent,
                    "engagement": 0.90 if cname in ("advanced", "standard") else 0.75,
                    "decay_rate": float(cc["decay_rate"]),
                    "last_practiced": {n: 0 for n in topo},
                    "misconceptions": {n: False for n in topo},
                }
            )
    return wrapped


def nx_nodes(dag):
    import networkx as nx
    return list(nx.topological_sort(dag))


def run_calibrated_isg_dag(calibration: Dict[str, Any], seeds: List[int], num_agents: int = 1000) -> Dict[str, Any]:
    """Run Linear, Naive, and DAG-Guided policies using calibrated population."""
    dag = build_isg_curriculum_dag()
    per_pol: Dict[str, List[Dict[str, Any]]] = {"linear": [], "naive": [], "dag": []}

    for seed in seeds:
        sim = HybridStudentSimulator(
            dag, num_agents=num_agents, seed=seed,
            use_ednet_bounds=False, use_memory_decay=True, use_misconceptions=False,
        )
        pop = spawn_calibrated_population(dag, calibration, num_agents=num_agents, seed=seed)
        for pol in ("linear", "naive", "dag"):
            per_pol[pol].append(sim.run_policy(pop, pol))

    summary: Dict[str, Any] = {}
    for pol, runs in per_pol.items():
        comp = [r["overall_completion_rate"] * 100.0 for r in runs]
        churn = [r["overall_churn_rate"] * 100.0 for r in runs]
        ret = [r["mean_foundational_retention"] * 100.0 for r in runs]
        summary[pol] = {
            "completion_mean_pct": round(float(np.mean(comp)), 2),
            "completion_std_pct": round(float(np.std(comp)), 2),
            "churn_mean_pct": round(float(np.mean(churn)), 2),
            "foundational_retention_pct": round(float(np.mean(ret)), 2),
        }
    return summary


def plot_validation_figure(
    fidelity_ablation: Dict[str, Any],
    calibration: Dict[str, Any],
    isg_summary: Dict[str, Any],
    n_train: int,
    n_test: int,
    out_path: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#cbd5e1"})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=220)
    fig.patch.set_facecolor("white")

    # (a) Learning Curve: Real Hold-Out vs M0 Hardcoded vs M3 Calibrated
    ax = axes[0, 0]
    real_lc = [x * 100.0 for x in fidelity_ablation["M3_calibrated_decay"]["real_learning_curve"]]
    m0_lc = [x * 100.0 for x in fidelity_ablation["M0_hardcoded"]["sim_learning_curve"]]
    m3_lc = [x * 100.0 for x in fidelity_ablation["M3_calibrated_decay"]["sim_learning_curve"]]
    ks = np.arange(1, len(real_lc) + 1)
    ax.plot(ks, real_lc, "k-o", linewidth=2.5, markersize=6, label=f"Real Hold-Out (ASSISTments, N={n_test})")
    ax.plot(ks, m0_lc, "--s", color="#dc2626", linewidth=2.0, markersize=5, label=f"M0: Hardcoded (RMSE={fidelity_ablation['M0_hardcoded']['learning_curve_rmse']:.3f})")
    ax.plot(ks, m3_lc, "-^", color="#059669", linewidth=2.2, markersize=6, label=f"M3: Calibrated (RMSE={fidelity_ablation['M3_calibrated_decay']['learning_curve_rmse']:.3f})")
    ax.set_xlabel("Attempt Index (k)", fontsize=10)
    ax.set_ylabel("Accuracy P(correct) (%)", fontsize=10)
    ax.set_ylim(30, 100)
    ax.set_title("(a) Hold-Out Learning Curve Fidelity (Unseen Students)", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.35)

    # (b) Hold-Out Fidelity Gap Reduction (M0 -> M3)
    ax = axes[0, 1]
    stages = ["M0_hardcoded", "M1_em_bkt_global", "M2_gmm_cohorts", "M3_calibrated_decay"]
    stage_labels = ["M0\nHardcoded", "M1\nEM BKT", "M2\n+GMM Cohorts", "M3\n+Decay Fit"]
    rmses = [fidelity_ablation[s]["learning_curve_rmse"] * 100.0 for s in stages]
    gaps = [fidelity_ablation[s]["mean_accuracy_gap"]["abs_gap_pp"] for s in stages]
    x = np.arange(len(stages))
    w = 0.36
    b1 = ax.bar(x - w / 2, rmses, w, label="Learning Curve RMSE (pp)", color="#2563eb", alpha=0.9)
    b2 = ax.bar(x + w / 2, gaps, w, label="|Mean Accuracy Gap| (pp)", color="#f59e0b", alpha=0.9)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=9)
    ax.set_ylabel("Hold-Out Error (percentage points, lower=better)", fontsize=9.5)
    ax.set_title("(b) Calibration Ablation on Hold-Out Fidelity Gap", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # (c) Data-Driven GMM Cohorts vs Old Hardcoded Proportions
    ax = axes[1, 0]
    cnames = ["advanced", "standard", "at_risk", "inattentive"]
    clabels = ["Advanced", "Standard", "At-Risk", "Inattentive"]
    old_p = [COHORTS[c].proportion * 100.0 for c in cnames]
    new_p = [calibration["cohorts"][c]["proportion"] * 100.0 for c in cnames]
    xc = np.arange(len(cnames))
    b1 = ax.bar(xc - w / 2, old_p, w, label="Old Hardcoded (%)", color="#94a3b8")
    b2 = ax.bar(xc + w / 2, new_p, w, label="GMM Fitted from ASSISTments (%)", color="#0d9488")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xc)
    ax.set_xticklabels(clabels, fontsize=9.5)
    ax.set_ylim(0, 58)
    ax.set_ylabel("Cohort Proportion (%)", fontsize=10)
    ax.set_title("(c) Derived GMM Student Cohorts vs. Assumed Proportions", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # (d) Calibrated 12-Node ISG DAG Policy Results
    ax = axes[1, 1]
    pols = ["linear", "naive", "dag"]
    plabels = ["Linear\nBaseline", "Naive Static\nReview", "DAG-Guided\nRemediation"]
    comp = [isg_summary[p]["completion_mean_pct"] for p in pols]
    std = [isg_summary[p]["completion_std_pct"] for p in pols]
    ret = [isg_summary[p]["foundational_retention_pct"] for p in pols]
    xp = np.arange(len(pols))
    b1 = ax.bar(xp - w / 2, comp, w, yerr=std, capsize=4, label="Completion (%)", color="#1d4ed8", alpha=0.9)
    b2 = ax.bar(xp + w / 2, ret, w, label="Foundational Retention (%)", color="#059669", alpha=0.88)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xp)
    ax.set_xticklabels(plabels, fontsize=9.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_title("(d) Calibrated ISG DAG Policy Performance (5 Seeds, N=1,000)", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        f"Empirical Calibration & Hold-Out Fidelity Validation (ASSISTments Train N={n_train:,} -> Test N={n_test:,})",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    calibration = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    test_csv = DATA_DIR / "builder_test.csv"
    test_students = load_piech_assistments(test_csv)
    real_seqs, _ = extract_skill_sequences(test_students, min_seq_len=3, max_seq_len=30)
    real_lens = [len(s) for s in real_seqs]
    logger.info("Loaded hold-out test split: %d real students, %d skill sequences.", len(test_students), len(real_seqs))

    fidelity_ablation = {}
    for mode in ("M0_hardcoded", "M1_em_bkt_global", "M2_gmm_cohorts", "M3_calibrated_decay"):
        sim_seqs = simulate_kt_sequences_for_fidelity(mode, calibration, real_lens, seed=42)
        fid = evaluate_fidelity(real_seqs, sim_seqs, max_k=12, seed=42)
        fidelity_ablation[mode] = fid
        logger.info(
            "  [%s] LC_RMSE=%.4f | AccGap=%.2f pp (95%% CI [%.2f, %.2f]) | MasteryGap=%.2f pp",
            mode,
            fid["learning_curve_rmse"],
            fid["mean_accuracy_gap"]["abs_gap_pp"],
            fid["mean_accuracy_gap"]["ci95_low_pp"],
            fid["mean_accuracy_gap"]["ci95_high_pp"],
            fid["mastery_rate_gap"]["abs_gap_pp"],
        )

    logger.info("Running calibrated 12-node ISG DAG simulation (5 seeds x 1,000 agents)...")
    isg_summary = run_calibrated_isg_dag(calibration, seeds=[42, 101, 202, 303, 404], num_agents=1000)
    for pol, vals in isg_summary.items():
        logger.info(
            "  [ISG %s] Completion: %.2f%% +/- %.2f%% | Retention: %.2f%%",
            pol.upper(), vals["completion_mean_pct"], vals["completion_std_pct"], vals["foundational_retention_pct"]
        )

    report = {
        "holdout_test_students": len(test_students),
        "holdout_test_sequences": len(real_seqs),
        "fidelity_ablation": fidelity_ablation,
        "calibrated_isg_dag_summary": isg_summary,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    n_train = int(calibration["provenance"]["n_train_students"])
    plot_validation_figure(fidelity_ablation, calibration, isg_summary, n_train, len(test_students), OUT_FIG)
    logger.info("Saved validation report -> %s", OUT_JSON)
    logger.info("Saved validation figure -> %s", OUT_FIG)


if __name__ == "__main__":
    main()

