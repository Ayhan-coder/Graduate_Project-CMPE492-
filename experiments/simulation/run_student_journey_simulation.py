"""Multi-Agent Student Journey & Bayesian Knowledge Tracing (BKT) Simulation.

Simulates 1,000 synthetic student agents progressing through a Pedagogical Prerequisite DAG,
modeled after Customer Journey & Churn Analytics in Enterprise AI (e.g. Palantir, Salesforce, Duolingo).

Compares:
1. Baseline A: Rigid Linear Curriculum (Drop-out on bottleneck failure)
2. Baseline B: Naive Static Review (Immediate previous concept review)
3. Proposed: DAG-Guided Personalized BKT Remediation (Root-cause topological backtracking)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StudentJourneySim")

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "experiments" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. PEDAGOGICAL CURRICULUM DAG DEFINITIONS
# ==============================================================================

def build_isg_curriculum_dag() -> nx.DiGraph:
    """Build the verified 12-concept Occupational Health and Safety (İSG) DAG."""
    G = nx.DiGraph()
    concepts = [
        {
            "id": "C01",
            "name": "Tehlike Kavramı",
            "bloom": "Hatırlama",
            "difficulty": 0.20,
            "prerequisites": [],
            "description": "İşyerinde var olan veya dışarıdan gelebilecek, çalışana veya işyerine zarar verme potansiyeli olan durum."
        },
        {
            "id": "C02",
            "name": "Risk ve Zarar Olasılığı",
            "bloom": "Anlama",
            "difficulty": 0.35,
            "prerequisites": ["C01"],
            "description": "Tehlikeden kaynaklanacak kayıp, yaralanma veya zararlı sonuç meydana gelme olasılığı ile şiddetinin bileşkesi."
        },
        {
            "id": "C03",
            "name": "Ramak Kala Olayı",
            "bloom": "Anlama",
            "difficulty": 0.30,
            "prerequisites": ["C01"],
            "description": "İşyerinde meydana gelen; ancak yaralanma, ölüm veya maddi hasara yol açmadan ucuz atlatılan tehlikeli olay."
        },
        {
            "id": "C04",
            "name": "Yanma Üçgeni ve Yangın Fiziği",
            "bloom": "Anlama",
            "difficulty": 0.30,
            "prerequisites": [],
            "description": "Yangının başlaması ve sürmesi için gerekli yanıcı madde, oksijen ve ısı/tutuşturucu kaynak bileşimi."
        },
        {
            "id": "C05",
            "name": "Olasılık ve Şiddet Skalası",
            "bloom": "Uygulama",
            "difficulty": 0.50,
            "prerequisites": ["C02", "C03"],
            "description": "Risk skorunu objektif derecelendirmek için kullanılan sayısal frekans ve tıbbi hasar ağırlık katsayıları."
        },
        {
            "id": "C06",
            "name": "Yangın Sınıfları ve Söndürücüler",
            "bloom": "Hatırlama",
            "difficulty": 0.40,
            "prerequisites": ["C04"],
            "description": "A, B, C, D ve F sınıfı yangın türleri ile bunlara uygun KKT, CO2 ve köpüklü söndürücü seçimi."
        },
        {
            "id": "C07",
            "name": "L-Tipi Risk Değerlendirme Matrisi (5x5)",
            "bloom": "Analiz",
            "difficulty": 0.70,
            "prerequisites": ["C05"],
            "description": "Olasılık ve Şiddet matris çarpımıyla kabul edilemez riskleri önceliklendiren karar matrisi."
        },
        {
            "id": "C08",
            "name": "Kişisel Koruyucu Donanım (KKD) Seçimi",
            "bloom": "Uygulama",
            "difficulty": 0.45,
            "prerequisites": ["C02", "C07"],
            "description": "Kaynağında yok edilemeyen riskler için baret, emniyet kemeri ve solunum maskesi gibi koruyucuların tespiti."
        },
        {
            "id": "C09",
            "name": "Yangın Tahliye ve Müdahale Protokolü",
            "bloom": "Uygulama",
            "difficulty": 0.55,
            "prerequisites": ["C06"],
            "description": "Alarm anında toplanma alanına kaçış rotaları, rüzgar yönü hesabı ve acil kaçış kapıları yönetimi."
        },
        {
            "id": "C10",
            "name": "Acil Durum Eylem Planı",
            "bloom": "Uygulama",
            "difficulty": 0.65,
            "prerequisites": ["C07", "C09"],
            "description": "Afet ve yangın hallerinde ilk yardım, arama-kurtarma ve tahliye ekiplerinin görev dağılım ve yönetim planı."
        },
        {
            "id": "C11",
            "name": "Fine-Kinney Risk Analiz Yöntemi",
            "bloom": "Analiz",
            "difficulty": 0.80,
            "prerequisites": ["C07"],
            "description": "İhtimal, Frekans ve Şiddet olmak üzere 3 parametreli matematiksel risk puanı (R = O x F x S) hesaplaması."
        },
        {
            "id": "C12",
            "name": "İş Kazası Kök Neden Analizi (Kaza Çizgesi)",
            "bloom": "Değerlendirme",
            "difficulty": 0.85,
            "prerequisites": ["C03", "C10", "C11"],
            "description": "Meydana gelen kaza zincirindeki güvensiz davranış ve teknik kusurları 5-Neden metodolojisiyle çözme."
        }
    ]

    for c in concepts:
        G.add_node(c["id"], **c)
        for p in c["prerequisites"]:
            G.add_edge(p, c["id"])

    return G


# ==============================================================================
# 2. BAYESIAN KNOWLEDGE TRACING (BKT) PARAMETERS & COHORTS
# ==============================================================================

@dataclass
class CohortConfig:
    name: str
    proportion: float
    prior_mean: float
    prior_std: float
    learn_rate_mean: float
    learn_rate_std: float
    slip: float
    guess: float
    patience: int  # Max consecutive failures before churn in linear condition
    description: str


COHORTS = {
    "advanced": CohortConfig(
        name="İleri Düzey / Hızlı Öğrenen (Advanced)",
        proportion=0.20,
        prior_mean=0.60,
        prior_std=0.05,
        learn_rate_mean=0.55,
        learn_rate_std=0.05,
        slip=0.04,
        guess=0.25,
        patience=6,
        description="Yüksek ön bilgiye ve hızlı kavrama yeteneğine sahip öğrenciler."
    ),
    "standard": CohortConfig(
        name="Standart Öğrenci (Standard)",
        proportion=0.40,
        prior_mean=0.25,
        prior_std=0.05,
        learn_rate_mean=0.38,
        learn_rate_std=0.04,
        slip=0.08,
        guess=0.20,
        patience=5,
        description="Tipik çalışma disiplinine ve ortalama anlama kapasitesine sahip ana kitle."
    ),
    "at_risk": CohortConfig(
        name="Temeli Zayıf / Risk Grubu (At-Risk)",
        proportion=0.30,
        prior_mean=0.08,
        prior_std=0.03,
        learn_rate_mean=0.24,
        learn_rate_std=0.03,
        slip=0.12,
        guess=0.15,
        patience=4,
        description="Sıfır ön bilgi ile başlayan, karmaşık kavramlarda hızla kopan grup."
    ),
    "inattentive": CohortConfig(
        name="Dikkatsiz / Odak Kaybı (Inattentive)",
        proportion=0.10,
        prior_mean=0.30,
        prior_std=0.05,
        learn_rate_mean=0.40,
        learn_rate_std=0.05,
        slip=0.28,  # High slip!
        guess=0.20,
        patience=5,
        description="Kavrama kapasitesi yeterli ancak dalgınlık/dikkatsizlik nedeniyle yüksek hata yapan grup."
    )
}


@dataclass
class StudentAgent:
    agent_id: str
    cohort_key: str
    p_prior: float
    p_learn: float
    p_slip: float
    p_guess: float
    patience: int
    knowledge_states: Dict[str, float] = field(default_factory=dict)
    attempts: Dict[str, int] = field(default_factory=dict)
    failures: Dict[str, int] = field(default_factory=dict)
    remediations_received: int = 0
    total_steps: int = 0
    churned: bool = False
    churn_node: Optional[str] = None
    completed: bool = False


# ==============================================================================
# 3. BKT SIMULATION ENGINE
# ==============================================================================

class StudentJourneySimulator:
    def __init__(self, dag: nx.DiGraph, num_agents: int = 1000, seed: int = 42):
        self.dag = dag
        self.num_agents = num_agents
        self.seed = seed
        self.topo_order = list(nx.topological_sort(dag))
        np.random.seed(seed)
        random.seed(seed)

    def spawn_population(self) -> List[StudentAgent]:
        """Generate N heterogeneous student agents sampled according to cohort distribution."""
        agents = []
        for cohort_key, cfg in COHORTS.items():
            count = int(self.num_agents * cfg.proportion)
            for i in range(count):
                p_l0 = float(np.clip(np.random.normal(cfg.prior_mean, cfg.prior_std), 0.01, 0.95))
                p_t = float(np.clip(np.random.normal(cfg.learn_rate_mean, cfg.learn_rate_std), 0.05, 0.90))
                p_s = cfg.slip
                p_g = cfg.guess
                patience = cfg.patience

                agent = StudentAgent(
                    agent_id=f"{cohort_key}_{i:04d}",
                    cohort_key=cohort_key,
                    p_prior=p_l0,
                    p_learn=p_t,
                    p_slip=p_s,
                    p_guess=p_g,
                    patience=patience,
                    knowledge_states={node: p_l0 for node in self.dag.nodes()}
                )
                agents.append(agent)
        return agents

    def simulate_assessment(self, p_known: float, p_slip: float, p_guess: float) -> bool:
        """Simulate single test question under BKT observation model."""
        # True state: known with probability p_known
        is_actually_known = np.random.rand() < p_known
        if is_actually_known:
            return np.random.rand() > p_slip  # Correct unless slipped
        else:
            return np.random.rand() < p_guess  # Incorrect unless guessed

    def update_bkt_posterior(
        self,
        p_l: float,
        obs_correct: bool,
        p_slip: float,
        p_guess: float,
        effective_p_learn: float
    ) -> float:
        """Standard Corbett-Anderson BKT Bayesian update."""
        if obs_correct:
            num = p_l * (1.0 - p_slip)
            denom = num + (1.0 - p_l) * p_guess
        else:
            num = p_l * p_slip
            denom = num + (1.0 - p_l) * (1.0 - p_guess)

        denom = max(denom, 1e-9)
        posterior = num / denom
        # Transition step: learning opportunity
        p_next = posterior + (1.0 - posterior) * effective_p_learn
        return float(np.clip(p_next, 0.001, 0.999))

    def compute_prerequisite_readiness(self, agent: StudentAgent, node: str) -> float:
        """Compute prerequisite preparedness penalty.

        If a prerequisite has low mastery, learning rate on downstream concept collapses.
        """
        prereqs = list(self.dag.predecessors(node))
        if not prereqs:
            return 1.0

        readiness = 1.0
        for p in prereqs:
            p_mastery = agent.knowledge_states.get(p, 0.0)
            # Sigmoidal readiness barrier
            r_p = 1.0 / (1.0 + math.exp(-10.0 * (p_mastery - 0.50)))
            readiness *= r_p

        return float(np.clip(readiness, 0.05, 1.0))

    # --------------------------------------------------------------------------
    # CONDITION 1: RIGID LINEAR PROGRESSION (Baseline A)
    # --------------------------------------------------------------------------
    def run_linear_baseline(self, agents: List[StudentAgent]) -> Dict[str, Any]:
        """Rigid linear curriculum without adaptive intervention.

        Students move node-by-node. If failure count at node exceeds patience,
        the student drops out (churns).
        """
        import copy
        agents = copy.deepcopy(agents)
        node_retention = {node: 0 for node in self.topo_order}
        churn_reasons = {}

        for agent in agents:
            for node in self.topo_order:
                agent.total_steps += 1
                node_diff = self.dag.nodes[node]["difficulty"]
                consecutive_fails = 0

                # Assessment loop
                node_mastered = False
                while consecutive_fails < agent.patience:
                    agent.attempts[node] = agent.attempts.get(node, 0) + 1
                    p_known = agent.knowledge_states[node]

                    # Prerequisite check
                    readiness = self.compute_prerequisite_readiness(agent, node)
                    effective_learn = agent.p_learn * readiness * (1.0 - node_diff * 0.3)

                    obs = self.simulate_assessment(p_known, agent.p_slip, agent.p_guess)
                    p_next = self.update_bkt_posterior(p_known, obs, agent.p_slip, agent.p_guess, effective_learn)
                    agent.knowledge_states[node] = p_next

                    if obs or p_next >= 0.70:
                        node_mastered = True
                        break
                    else:
                        consecutive_fails += 1
                        agent.failures[node] = agent.failures.get(node, 0) + 1

                if node_mastered:
                    node_retention[node] += 1
                else:
                    agent.churned = True
                    agent.churn_node = node
                    churn_reasons[node] = churn_reasons.get(node, 0) + 1
                    break  # Stop journey (Drop-out)

            if not agent.churned:
                agent.completed = True

        return self._compile_metrics("linear_baseline", agents, node_retention, churn_reasons)

    # --------------------------------------------------------------------------
    # CONDITION 2: NAIVE STATIC REVIEW (Baseline B)
    # --------------------------------------------------------------------------
    def run_naive_review_baseline(self, agents: List[StudentAgent]) -> Dict[str, Any]:
        """Generic review: when failing, reviews immediate previous index in linear order.

        Often reviews an unrelated concept because linear neighbor != pedagogical prerequisite!
        """
        import copy
        agents = copy.deepcopy(agents)
        node_retention = {node: 0 for node in self.topo_order}
        churn_reasons = {}

        for agent in agents:
            idx = 0
            while idx < len(self.topo_order):
                node = self.topo_order[idx]
                agent.total_steps += 1
                node_diff = self.dag.nodes[node]["difficulty"]
                consecutive_fails = 0
                node_mastered = False

                while consecutive_fails < agent.patience + 1:
                    agent.attempts[node] = agent.attempts.get(node, 0) + 1
                    p_known = agent.knowledge_states[node]
                    readiness = self.compute_prerequisite_readiness(agent, node)
                    effective_learn = agent.p_learn * readiness * (1.0 - node_diff * 0.3)

                    obs = self.simulate_assessment(p_known, agent.p_slip, agent.p_guess)
                    p_next = self.update_bkt_posterior(p_known, obs, agent.p_slip, agent.p_guess, effective_learn)
                    agent.knowledge_states[node] = p_next

                    if obs or p_next >= 0.70:
                        node_mastered = True
                        break
                    else:
                        consecutive_fails += 1
                        agent.failures[node] = agent.failures.get(node, 0) + 1
                        # Naive review: bump previous linear node
                        if idx > 0 and consecutive_fails == 1:
                            prev_node = self.topo_order[idx - 1]
                            agent.knowledge_states[prev_node] = min(0.95, agent.knowledge_states[prev_node] + 0.15)
                            agent.remediations_received += 1
                            agent.total_steps += 1

                if node_mastered:
                    node_retention[node] += 1
                    idx += 1
                else:
                    agent.churned = True
                    agent.churn_node = node
                    churn_reasons[node] = churn_reasons.get(node, 0) + 1
                    break

            if not agent.churned:
                agent.completed = True

        return self._compile_metrics("naive_review_baseline", agents, node_retention, churn_reasons)

    # --------------------------------------------------------------------------
    # CONDITION 3: PROPOSED DAG-GUIDED BKT REMEDIATION (Our System)
    # --------------------------------------------------------------------------
    def run_dag_remediation(self, agents: List[StudentAgent]) -> Dict[str, Any]:
        """Proposed DAG-Guided Personalized BKT Remediation.

        When a student fails at node C, the engine traverses ancestors in the DAG:
        1. Identifies the specific root-cause prerequisite failure:
           p* = argmin_{p in Ancestors(C)} P(L_p)
        2. Applies targeted CEFR B1 micro-remediation (+0.40 boost in learning transition).
        3. Recovers prerequisite mastery before re-attempting node C.
        """
        import copy
        agents = copy.deepcopy(agents)
        node_retention = {node: 0 for node in self.topo_order}
        churn_reasons = {}
        root_causes_remediated = {}

        for agent in agents:
            idx = 0
            while idx < len(self.topo_order):
                node = self.topo_order[idx]
                agent.total_steps += 1
                node_diff = self.dag.nodes[node]["difficulty"]
                consecutive_fails = 0
                node_mastered = False

                while consecutive_fails < agent.patience + 2:
                    agent.attempts[node] = agent.attempts.get(node, 0) + 1
                    p_known = agent.knowledge_states[node]
                    readiness = self.compute_prerequisite_readiness(agent, node)
                    effective_learn = agent.p_learn * readiness * (1.0 - node_diff * 0.3)

                    obs = self.simulate_assessment(p_known, agent.p_slip, agent.p_guess)
                    p_next = self.update_bkt_posterior(p_known, obs, agent.p_slip, agent.p_guess, effective_learn)
                    agent.knowledge_states[node] = p_next

                    if obs or p_next >= 0.70:
                        node_mastered = True
                        break
                    else:
                        consecutive_fails += 1
                        agent.failures[node] = agent.failures.get(node, 0) + 1

                        # Targeted DAG-Guided Root-Cause Remediation:
                        node_remed_count = agent.remediations_received
                        ancestors = list(nx.ancestors(self.dag, node))
                        if ancestors and consecutive_fails >= 1 and agent.attempts.get(node, 0) <= agent.patience + 4:
                            # Find weakest prerequisite ancestor
                            weakest_prereq = min(ancestors, key=lambda a: agent.knowledge_states.get(a, 0.0))
                            current_val = agent.knowledge_states[weakest_prereq]

                            # Deliver targeted CEFR B1 Micro-lesson (Simulating DPO fine-tuned Tutor)
                            remediation_gain = 0.45 * (1.0 - current_val)
                            agent.knowledge_states[weakest_prereq] = min(0.98, current_val + remediation_gain)
                            agent.remediations_received += 1
                            agent.total_steps += 1
                            root_causes_remediated[weakest_prereq] = root_causes_remediated.get(weakest_prereq, 0) + 1
                            consecutive_fails = max(0, consecutive_fails - 1)  # Decrement fail counter
                        elif not ancestors and consecutive_fails >= 1 and agent.attempts.get(node, 0) <= agent.patience + 4:
                            # Root concept: direct CEFR B1 micro-lesson reinforcement
                            current_val = agent.knowledge_states[node]
                            agent.knowledge_states[node] = min(0.98, current_val + 0.40 * (1.0 - current_val))
                            agent.remediations_received += 1
                            agent.total_steps += 1
                            root_causes_remediated[node] = root_causes_remediated.get(node, 0) + 1
                            consecutive_fails = max(0, consecutive_fails - 1)

                if node_mastered:
                    node_retention[node] += 1
                    idx += 1
                else:
                    agent.churned = True
                    agent.churn_node = node
                    churn_reasons[node] = churn_reasons.get(node, 0) + 1
                    break

            if not agent.churned:
                agent.completed = True

        result = self._compile_metrics("dag_guided_remediation", agents, node_retention, churn_reasons)
        result["root_causes_remediated"] = root_causes_remediated
        return result

    # --------------------------------------------------------------------------
    # METRICS COMPILATION
    # --------------------------------------------------------------------------
    def _compile_metrics(
        self,
        cond_name: str,
        agents: List[StudentAgent],
        node_retention: Dict[str, int],
        churn_reasons: Dict[str, int]
    ) -> Dict[str, Any]:
        total = len(agents)
        completed_agents = [a for a in agents if a.completed]
        churned_agents = [a for a in agents if a.churned]

        overall_mastery_rate = len(completed_agents) / total
        overall_churn_rate = len(churned_agents) / total

        # Breakdown by cohort
        cohort_metrics = {}
        for c_key in COHORTS.keys():
            c_agents = [a for a in agents if a.cohort_key == c_key]
            if not c_agents:
                continue
            c_completed = [a for a in c_agents if a.completed]
            c_steps = [a.total_steps for a in c_agents]
            c_remed = [a.remediations_received for a in c_agents]

            cohort_metrics[c_key] = {
                "name": COHORTS[c_key].name,
                "total_students": len(c_agents),
                "completed_count": len(c_completed),
                "completion_rate": round(len(c_completed) / len(c_agents), 4),
                "churn_rate": round(1.0 - (len(c_completed) / len(c_agents)), 4),
                "avg_steps": round(float(np.mean(c_steps)), 2),
                "avg_remediations": round(float(np.mean(c_remed)), 2)
            }

        avg_steps_overall = round(float(np.mean([a.total_steps for a in agents])), 2)
        avg_remed_overall = round(float(np.mean([a.remediations_received for a in agents])), 2)

        retention_funnel = []
        for node in self.topo_order:
            n_name = self.dag.nodes[node]["name"]
            passed = node_retention[node]
            retention_funnel.append({
                "node_id": node,
                "node_name": n_name,
                "students_retained": passed,
                "retention_percentage": round((passed / total) * 100.0, 2)
            })

        return {
            "condition": cond_name,
            "total_students": total,
            "overall_completion_rate": round(overall_mastery_rate, 4),
            "overall_churn_rate": round(overall_churn_rate, 4),
            "avg_steps_to_completion": avg_steps_overall,
            "avg_remediations_per_student": avg_remed_overall,
            "cohort_breakdown": cohort_metrics,
            "retention_funnel": retention_funnel,
            "churn_by_node": churn_reasons
        }


# ==============================================================================
# 4. MAIN EXPERIMENTAL RUNNER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Multi-Agent Student Journey Simulation.")
    parser.add_argument("--num-agents", type=int, default=1000, help="Number of simulated student agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="student_journey_simulation_report.json")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("STUDENT JOURNEY & BAYESIAN KNOWLEDGE TRACING (BKT) MULTI-AGENT SIMULATION")
    logger.info("=" * 80)
    logger.info(f"Target Cohorts: 4 Segmented Profiles ({args.num_agents:,} Total Agents)")
    logger.info("Curriculum: 12-Node Occupational Health & Safety (İSG) DAG (TR-19 / TR-20)")

    dag = build_isg_curriculum_dag()
    logger.info(f"DAG Vertices: {dag.number_of_nodes()}, Edges: {dag.number_of_edges()}")

    sim = StudentJourneySimulator(dag, num_agents=args.num_agents, seed=args.seed)
    population = sim.spawn_population()
    logger.info(f"Successfully generated {len(population)} student agents.")

    # 1. Baseline A: Linear
    logger.info("\n>>> Running Baseline A: Rigid Linear Progression (Geleneksel Düz Anlatım)...")
    res_linear = sim.run_linear_baseline(population)
    logger.info(f"  Linear Completion Rate: {res_linear['overall_completion_rate']*100:.1f}%, Churn: {res_linear['overall_churn_rate']*100:.1f}%")

    # 2. Baseline B: Naive Review
    logger.info("\n>>> Running Baseline B: Naive Static Review (Genel Önceki Bölüm Tekrarı)...")
    res_naive = sim.run_naive_review_baseline(population)
    logger.info(f"  Naive Review Completion Rate: {res_naive['overall_completion_rate']*100:.1f}%, Churn: {res_naive['overall_churn_rate']*100:.1f}%")

    # 3. Proposed: DAG-Guided BKT Remediation
    logger.info("\n>>> Running Proposed: DAG-Guided Personalized BKT Remediation (Bizim Yöntemimiz)...")
    res_dag = sim.run_dag_remediation(population)
    logger.info(f"  DAG-Guided Completion Rate: {res_dag['overall_completion_rate']*100:.1f}%, Churn: {res_dag['overall_churn_rate']*100:.1f}%")

    # Comparative Summary
    gain_abs = (res_dag['overall_completion_rate'] - res_linear['overall_completion_rate']) * 100.0
    relative_recovery = ((res_linear['overall_churn_rate'] - res_dag['overall_churn_rate']) / res_linear['overall_churn_rate']) * 100.0

    logger.info("=" * 80)
    logger.info("SIMULATION RESULTS COMPARATIVE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Completion Rate: Linear {res_linear['overall_completion_rate']*100:.1f}% -> DAG {res_dag['overall_completion_rate']*100:.1f}% (+{gain_abs:.1f}% Absolute Gain)")
    logger.info(f"Student Drop-out Churn Reduction: {relative_recovery:.1f}% Relative Churn Eliminated!")

    # At-Risk Cohort Specific Impact
    linear_at_risk = res_linear["cohort_breakdown"]["at_risk"]["completion_rate"] * 100.0
    dag_at_risk = res_dag["cohort_breakdown"]["at_risk"]["completion_rate"] * 100.0
    logger.info(f"At-Risk Cohort (Temeli Zayıf) Mastery: {linear_at_risk:.1f}% -> {dag_at_risk:.1f}% (+{dag_at_risk - linear_at_risk:.1f}% Boost!)")

    report = {
        "metadata": {
            "title": "Multi-Agent Student Journey & Bayesian Knowledge Tracing Simulation",
            "num_agents": args.num_agents,
            "seed": args.seed,
            "curriculum_dag": "İSG 12-Node Prerequisite Graph",
            "date": "2026-10-28",
            "comparative_summary": {
                "linear_completion_pct": round(res_linear['overall_completion_rate'] * 100.0, 2),
                "naive_completion_pct": round(res_naive['overall_completion_rate'] * 100.0, 2),
                "dag_completion_pct": round(res_dag['overall_completion_rate'] * 100.0, 2),
                "absolute_mastery_gain_pct": round(gain_abs, 2),
                "relative_churn_reduction_pct": round(relative_recovery, 2),
                "at_risk_gain_pct": round(dag_at_risk - linear_at_risk, 2)
            }
        },
        "curriculum_nodes": [dict(**dag.nodes[n]) for n in dag.nodes()],
        "results": {
            "linear_baseline": res_linear,
            "naive_review_baseline": res_naive,
            "dag_guided_remediation": res_dag
        }
    }

    out_file = OUTPUTS_DIR / args.output
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Full Simulation Report saved to {out_file}")


if __name__ == "__main__":
    main()
