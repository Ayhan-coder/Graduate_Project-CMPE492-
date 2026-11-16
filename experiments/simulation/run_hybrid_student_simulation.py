"""Hybrid 3-Source Empirical-Cognitive Student Journey Simulation.

Blends three peer-reviewed student modeling paradigms onto the 12-node İSG Prerequisite DAG:
1. ASSISTments & EdNet Empirical BKT Bounds + Lecture Engagement
   (Hawkins, Heffernan, & Baker, ITS 2014; Choi et al., AIED 2020)
2. Junyi Academy Prerequisite-Graph Dependency Coupling
   (Chang, Hsu, & Chen, EDM 2015)
3. Agent4Edu Ebbinghaus Memory Decay & Embracing Imperfection (Student_100) Misconception Persistence
   (Gao et al., AAAI 2025; Wu et al., ACL 2025)

Runs multi-seed evaluation (default: 5 seeds x 1,000 agents) and an ablation ladder,
saving JSON metrics to `experiments/outputs/hybrid_student_simulation_report.json`
and generating `notes/figures/hybrid_student_simulation.png`.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_student_journey_simulation import (
    COHORTS,
    StudentAgent,
    StudentJourneySimulator,
    build_isg_curriculum_dag,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HybridStudentSim")

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "experiments" / "outputs"
FIGURES_DIR = ROOT / "notes" / "figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cohort-specific empirical engagement (EdNet KT3/KT4) & cognitive decay/misconception profiles
HYBRID_COHORT_MODIFIERS = {
    "advanced": {
        "watch_engagement": 0.96,   # EdNet KT3: high lecture completion
        "decay_rate": 0.015,        # Agent4Edu: slow Ebbinghaus forgetting
        "misconception_prob": 0.10, # Student_100: low initial misconception rate
    },
    "standard": {
        "watch_engagement": 0.85,
        "decay_rate": 0.032,
        "misconception_prob": 0.25,
    },
    "at_risk": {
        "watch_engagement": 0.68,   # EdNet KT3/KT4: frequent lecture skipping
        "decay_rate": 0.055,        # Agent4Edu: rapid forgetting without reinforcement
        "misconception_prob": 0.45, # Student_100: persistent foundational misconceptions
    },
    "inattentive": {
        "watch_engagement": 0.72,
        "decay_rate": 0.042,
        "misconception_prob": 0.30,
    },
}


class HybridStudentSimulator(StudentJourneySimulator):
    """Extends StudentJourneySimulator with 3-source empirical/cognitive mechanisms."""

    def __init__(
        self,
        dag,
        num_agents: int = 1000,
        seed: int = 42,
        use_ednet_bounds: bool = True,
        use_memory_decay: bool = True,
        use_misconceptions: bool = True,
    ):
        super().__init__(dag, num_agents=num_agents, seed=seed)
        self.use_ednet_bounds = use_ednet_bounds
        self.use_memory_decay = use_memory_decay
        self.use_misconceptions = use_misconceptions

    def spawn_hybrid_population(self) -> List[Dict[str, Any]]:
        """Spawn agents with Hawkins (2014) BKT bounds + EdNet/Agent4Edu/Student_100 attributes."""
        base_agents = self.spawn_population()
        wrapped: List[Dict[str, Any]] = []
        for agent in base_agents:
            if self.use_ednet_bounds:
                # Hawkins et al. (2014) / Baker non-degenerate BKT bounds
                agent.p_prior = float(np.clip(agent.p_prior, 0.05, 0.70))
                agent.p_learn = float(np.clip(agent.p_learn, 0.10, 0.65))
                agent.p_slip = float(np.clip(agent.p_slip, 0.03, 0.30))
                agent.p_guess = float(np.clip(agent.p_guess, 0.10, 0.35))
                for node in agent.knowledge_states:
                    agent.knowledge_states[node] = agent.p_prior

            mod = HYBRID_COHORT_MODIFIERS[agent.cohort_key]
            engagement = (
                float(np.clip(np.random.normal(mod["watch_engagement"], 0.05), 0.45, 1.0))
                if self.use_ednet_bounds
                else 1.0
            )
            misconceptions = {}
            for node in self.topo_order:
                diff = self.dag.nodes[node]["difficulty"]
                p_misc = mod["misconception_prob"] * (0.6 + 0.8 * diff)
                misconceptions[node] = bool(self.use_misconceptions and (np.random.rand() < p_misc))

            wrapped.append(
                {
                    "agent": agent,
                    "engagement": engagement,
                    "decay_rate": mod["decay_rate"] if self.use_memory_decay else 0.0,
                    "last_practiced": {node: 0 for node in self.topo_order},
                    "misconceptions": misconceptions,
                }
            )
        return wrapped

    def _apply_memory_decay(self, state: Dict[str, Any], current_step: int) -> None:
        """Agent4Edu (AAAI 2025): Ebbinghaus forgetting on unpracticed prerequisite nodes."""
        if not self.use_memory_decay:
            return
        agent: StudentAgent = state["agent"]
        lam = state["decay_rate"]
        for node, last_t in state["last_practiced"].items():
            if last_t > 0:
                delta_t = current_step - last_t
                if delta_t > 0:
                    decayed = agent.knowledge_states[node] * math.exp(-lam * delta_t)
                    agent.knowledge_states[node] = float(max(0.05, decayed))

    def _effective_slip(self, state: Dict[str, Any], node: str) -> float:
        """Embracing Imperfection (ACL 2025): active misconceptions elevate slip."""
        agent: StudentAgent = state["agent"]
        if self.use_misconceptions and state["misconceptions"].get(node, False):
            return float(min(0.30, agent.p_slip + 0.14))
        return agent.p_slip

    def run_policy(self, population: List[Dict[str, Any]], policy: str) -> Dict[str, Any]:
        """Run 'linear', 'naive', or 'dag' policy under the blended hybrid mechanics."""
        pop = copy.deepcopy(population)
        node_retention = {node: 0 for node in self.topo_order}
        churn_reasons: Dict[str, int] = {}
        retention_snapshots: List[float] = []

        for state in pop:
            agent: StudentAgent = state["agent"]
            step_clock = 0

            if policy == "linear":
                for node in self.topo_order:
                    step_clock += 1
                    self._apply_memory_decay(state, step_clock)
                    agent.total_steps += 1
                    node_diff = self.dag.nodes[node]["difficulty"]
                    consecutive_fails = 0
                    mastered = False
                    while consecutive_fails < agent.patience:
                        agent.attempts[node] = agent.attempts.get(node, 0) + 1
                        p_known = agent.knowledge_states[node]
                        readiness = self.compute_prerequisite_readiness(agent, node)
                        eff_learn = agent.p_learn * state["engagement"] * readiness * (1.0 - node_diff * 0.3)
                        slip = self._effective_slip(state, node)
                        obs = self.simulate_assessment(p_known, slip, agent.p_guess)
                        p_next = self.update_bkt_posterior(p_known, obs, slip, agent.p_guess, eff_learn)
                        agent.knowledge_states[node] = p_next
                        state["last_practiced"][node] = step_clock
                        if obs or p_next >= 0.70:
                            mastered = True
                            break
                        consecutive_fails += 1
                        agent.failures[node] = agent.failures.get(node, 0) + 1

                    if mastered:
                        node_retention[node] += 1
                    else:
                        agent.churned = True
                        agent.churn_node = node
                        churn_reasons[node] = churn_reasons.get(node, 0) + 1
                        break

            elif policy == "naive":
                idx = 0
                while idx < len(self.topo_order):
                    node = self.topo_order[idx]
                    step_clock += 1
                    self._apply_memory_decay(state, step_clock)
                    agent.total_steps += 1
                    node_diff = self.dag.nodes[node]["difficulty"]
                    consecutive_fails = 0
                    mastered = False
                    while consecutive_fails < agent.patience + 1:
                        agent.attempts[node] = agent.attempts.get(node, 0) + 1
                        p_known = agent.knowledge_states[node]
                        readiness = self.compute_prerequisite_readiness(agent, node)
                        eff_learn = agent.p_learn * state["engagement"] * readiness * (1.0 - node_diff * 0.3)
                        slip = self._effective_slip(state, node)
                        obs = self.simulate_assessment(p_known, slip, agent.p_guess)
                        p_next = self.update_bkt_posterior(p_known, obs, slip, agent.p_guess, eff_learn)
                        agent.knowledge_states[node] = p_next
                        state["last_practiced"][node] = step_clock
                        if obs or p_next >= 0.70:
                            mastered = True
                            break
                        consecutive_fails += 1
                        agent.failures[node] = agent.failures.get(node, 0) + 1
                        if idx > 0 and consecutive_fails == 1:
                            prev_node = self.topo_order[idx - 1]
                            agent.knowledge_states[prev_node] = min(0.95, agent.knowledge_states[prev_node] + 0.15)
                            state["last_practiced"][prev_node] = step_clock
                            # Naive review rarely clears root misconception
                            if state["misconceptions"].get(prev_node, False) and np.random.rand() < 0.20:
                                state["misconceptions"][prev_node] = False
                            agent.remediations_received += 1

                    if mastered:
                        node_retention[node] += 1
                        idx += 1
                    else:
                        agent.churned = True
                        agent.churn_node = node
                        churn_reasons[node] = churn_reasons.get(node, 0) + 1
                        break

            elif policy == "dag":
                idx = 0
                max_remediations = 18
                while idx < len(self.topo_order):
                    node = self.topo_order[idx]
                    step_clock += 1
                    self._apply_memory_decay(state, step_clock)
                    agent.total_steps += 1
                    node_diff = self.dag.nodes[node]["difficulty"]
                    consecutive_fails = 0
                    mastered = False
                    while consecutive_fails < agent.patience + 2:
                        agent.attempts[node] = agent.attempts.get(node, 0) + 1
                        p_known = agent.knowledge_states[node]
                        readiness = self.compute_prerequisite_readiness(agent, node)
                        eff_learn = agent.p_learn * state["engagement"] * readiness * (1.0 - node_diff * 0.3)
                        slip = self._effective_slip(state, node)
                        obs = self.simulate_assessment(p_known, slip, agent.p_guess)
                        p_next = self.update_bkt_posterior(p_known, obs, slip, agent.p_guess, eff_learn)
                        agent.knowledge_states[node] = p_next
                        state["last_practiced"][node] = step_clock
                        if obs or p_next >= 0.70:
                            mastered = True
                            break
                        consecutive_fails += 1
                        agent.failures[node] = agent.failures.get(node, 0) + 1

                        # DAG-guided root-cause remediation + misconception resolution
                        if agent.remediations_received < max_remediations:
                            ancestors = list(self.dag.predecessors(node))
                            target = (
                                min(ancestors, key=lambda p: agent.knowledge_states.get(p, 0.0))
                                if ancestors
                                else node
                            )
                            agent.knowledge_states[target] = min(
                                0.96, agent.knowledge_states[target] + 0.32
                            )
                            state["last_practiced"][target] = step_clock
                            if state["misconceptions"].get(target, False) and np.random.rand() < 0.85:
                                state["misconceptions"][target] = False
                            if state["misconceptions"].get(node, False) and np.random.rand() < 0.70:
                                state["misconceptions"][node] = False
                            agent.remediations_received += 1

                    if mastered:
                        node_retention[node] += 1
                        idx += 1
                    else:
                        agent.churned = True
                        agent.churn_node = node
                        churn_reasons[node] = churn_reasons.get(node, 0) + 1
                        break

            if not agent.churned:
                agent.completed = True

            # Record mean mastery retention on early foundational nodes (C01..C04) at end of journey
            early_mean = float(np.mean([agent.knowledge_states[n] for n in self.topo_order[:4]]))
            retention_snapshots.append(early_mean)

        agents_only = [s["agent"] for s in pop]
        metrics = self._compile_metrics(f"hybrid_{policy}", agents_only, node_retention, churn_reasons)
        metrics["mean_foundational_retention"] = round(float(np.mean(retention_snapshots)), 4)
        return metrics


def aggregate_seed_runs(seeds: List[int], num_agents: int, **sim_kwargs) -> Dict[str, Any]:
    """Run all 3 policies across multiple seeds and compute mean/std."""
    dag = build_isg_curriculum_dag()
    per_policy: Dict[str, List[Dict[str, Any]]] = {"linear": [], "naive": [], "dag": []}

    for seed in seeds:
        sim = HybridStudentSimulator(dag, num_agents=num_agents, seed=seed, **sim_kwargs)
        pop = sim.spawn_hybrid_population()
        for pol in ("linear", "naive", "dag"):
            per_policy[pol].append(sim.run_policy(pop, pol))

    summary: Dict[str, Any] = {}
    for pol, runs in per_policy.items():
        comp = [r["overall_completion_rate"] * 100.0 for r in runs]
        churn = [r["overall_churn_rate"] * 100.0 for r in runs]
        ret = [r["mean_foundational_retention"] * 100.0 for r in runs]
        cohort_means = {}
        for ck in COHORTS:
            vals = [r["cohort_breakdown"][ck]["completion_rate"] * 100.0 for r in runs]
            cohort_means[ck] = {
                "mean_pct": round(float(np.mean(vals)), 2),
                "std_pct": round(float(np.std(vals)), 2),
            }
        summary[pol] = {
            "completion_mean_pct": round(float(np.mean(comp)), 2),
            "completion_std_pct": round(float(np.std(comp)), 2),
            "churn_mean_pct": round(float(np.mean(churn)), 2),
            "churn_std_pct": round(float(np.std(churn)), 2),
            "foundational_retention_pct": round(float(np.mean(ret)), 2),
            "cohorts": cohort_means,
        }
    return summary


def plot_hybrid_results(
    full_summary: Dict[str, Any],
    ablation: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """Generate 4-panel publication figure for the 3-source hybrid simulation."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.9,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=220)
    fig.patch.set_facecolor("white")

    # Panel (a): Overall Completion & Churn (5-seed mean +/- std)
    ax = axes[0, 0]
    policies = ["linear", "naive", "dag"]
    labels = ["Linear\nBaseline", "Naive Static\nReview", "DAG-Guided\nRemediation"]
    comp_means = [full_summary[p]["completion_mean_pct"] for p in policies]
    comp_stds = [full_summary[p]["completion_std_pct"] for p in policies]
    churn_means = [full_summary[p]["churn_mean_pct"] for p in policies]
    x = np.arange(len(policies))
    w = 0.36
    b1 = ax.bar(x - w / 2, comp_means, w, yerr=comp_stds, capsize=4, label="Completion (%)", color="#1d4ed8", alpha=0.9)
    b2 = ax.bar(x + w / 2, churn_means, w, label="Simulated Churn (%)", color="#dc2626", alpha=0.82)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1e3a8a")
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#991b1b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_title("(a) 3-Source Hybrid Completion vs. Churn (5 Seeds, N=1,000)", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel (b): Cohort Breakdown
    ax = axes[0, 1]
    cohort_keys = ["advanced", "standard", "at_risk", "inattentive"]
    cohort_labels = ["Advanced\n(20%)", "Standard\n(40%)", "At-Risk\n(30%)", "Inattentive\n(10%)"]
    xc = np.arange(len(cohort_keys))
    wc = 0.25
    colors = {"linear": "#94a3b8", "naive": "#f59e0b", "dag": "#059669"}
    names = {"linear": "Linear", "naive": "Naive Review", "dag": "DAG-Guided"}
    for i, pol in enumerate(policies):
        vals = [full_summary[pol]["cohorts"][ck]["mean_pct"] for ck in cohort_keys]
        bars = ax.bar(xc + (i - 1) * wc, vals, wc, label=names[pol], color=colors[pol], alpha=0.9)
        if pol == "dag":
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#065f46")
    ax.set_xticks(xc)
    ax.set_xticklabels(cohort_labels, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Cohort Completion (%)", fontsize=10)
    ax.set_title("(b) Cohort Resilience Under Decay & Misconceptions", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel (c): Ablation Across Blended Mechanisms
    ax = axes[1, 0]
    abl_keys = ["M0_assumed_bkt", "M1_assist_ednet", "M2_agent4edu_decay", "M3_full_hybrid"]
    abl_labels = [
        "M0: Assumed\nBKT",
        "M1: +ASSISTments\n& EdNet",
        "M2: +Agent4Edu\nMemory Decay",
        "M3: +Student_100\nMisconceptions",
    ]
    xa = np.arange(len(abl_keys))
    for pol, col, marker in [("linear", "#64748b", "o"), ("naive", "#d97706", "s"), ("dag", "#047857", "^")]:
        ys = [ablation[k][pol]["completion_mean_pct"] for k in abl_keys]
        ax.plot(xa, ys, marker=marker, linewidth=2.2, markersize=7, color=col, label=names[pol])
        for xi, yi in zip(xa, ys):
            ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=col, fontweight="bold")
    ax.set_xticks(xa)
    ax.set_xticklabels(abl_labels, fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Completion Rate (%)", fontsize=10)
    ax.set_title("(c) Mechanism Ablation Ladder (Realistic Difficulty Stress)", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel (d): Foundational Retention (C01-C04) After Ebbinghaus Decay
    ax = axes[1, 1]
    ret_vals = [full_summary[p]["foundational_retention_pct"] for p in policies]
    bars = ax.bar(labels, ret_vals, width=0.48, color=["#64748b", "#f59e0b", "#0d9488"], alpha=0.9)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Mean Foundational Mastery P(L) (%)", fontsize=10)
    ax.set_title("(d) End-of-Journey Foundational Retention (C01–C04)", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        "Hybrid 3-Source Student Simulation: ASSISTments/EdNet + Junyi DAG + Agent4Edu & Student_100",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-Source Blended Student Simulation.")
    parser.add_argument("--num-agents", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404])
    args = parser.parse_args()

    logger.info("Running 4-stage ablation ladder across %d seeds x %d agents...", len(args.seeds), args.num_agents)
    ablation = {
        "M0_assumed_bkt": aggregate_seed_runs(args.seeds, args.num_agents, use_ednet_bounds=False, use_memory_decay=False, use_misconceptions=False),
        "M1_assist_ednet": aggregate_seed_runs(args.seeds, args.num_agents, use_ednet_bounds=True, use_memory_decay=False, use_misconceptions=False),
        "M2_agent4edu_decay": aggregate_seed_runs(args.seeds, args.num_agents, use_ednet_bounds=True, use_memory_decay=True, use_misconceptions=False),
        "M3_full_hybrid": aggregate_seed_runs(args.seeds, args.num_agents, use_ednet_bounds=True, use_memory_decay=True, use_misconceptions=True),
    }

    full_summary = ablation["M3_full_hybrid"]
    out_json = OUTPUTS_DIR / "hybrid_student_simulation_report.json"
    payload = {
        "metadata": {
            "title": "Hybrid 3-Source Empirical-Cognitive Student Simulation",
            "sources": [
                "ASSISTments & EdNet Empirical BKT Bounds/Engagement (Hawkins et al., ITS 2014; Choi et al., AIED 2020)",
                "Junyi Academy Prerequisite Graph Coupling (Chang et al., EDM 2015)",
                "Agent4Edu Memory Decay & Student_100 Cognitive Imperfection (Gao et al., AAAI 2025; Wu et al., ACL 2025)",
            ],
            "num_agents_per_seed": args.num_agents,
            "seeds": args.seeds,
        },
        "full_hybrid_summary": full_summary,
        "ablation": ablation,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig_path = FIGURES_DIR / "hybrid_student_simulation.png"
    plot_hybrid_results(full_summary, ablation, fig_path)

    logger.info("Saved report -> %s", out_json)
    logger.info("Saved figure -> %s", fig_path)
    for pol in ("linear", "naive", "dag"):
        s = full_summary[pol]
        logger.info(
            "  [%s] Completion: %.2f%% +/- %.2f%% | Churn: %.2f%% | At-Risk: %.2f%% | Retention: %.2f%%",
            pol.upper(),
            s["completion_mean_pct"],
            s["completion_std_pct"],
            s["churn_mean_pct"],
            s["cohorts"]["at_risk"]["mean_pct"],
            s["foundational_retention_pct"],
        )


if __name__ == "__main__":
    main()

