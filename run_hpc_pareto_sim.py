#!/usr/bin/env python3
"""
HPC Pedagogical Multi-Objective Pareto Optimization Simulation
Computes the optimal operational Pareto Frontier across VRAM, Power, Latency, F1, and Morphological Robustness.
Conducts 10,000 Monte-Carlo utility sweeps for different educational deployment personas:
1. Ultra-Low-Budget Rural School (Resource-Constrained)
2. Edge University Classroom (Balanced Interactive)
3. Central Ministry / National Cloud (Quality-Maximized)
"""

import os
import sys
import json
import logging
import math
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_EMPIRICAL_PROFILES = {
    "Kumru-2B": {
        "vram_gb": 4.52,
        "power_watt": 122.7,
        "energy_joules_query": 207.42,
        "latency_sec": 1.69,
        "throughput_tok_s": 37.9,
        "f1_score": 51.3,
        "faithfulness_index": 57.0,
        "morph_robustness": 63.0,
        "json_compliance": 100.0
    },
    "Trendyol-8B": {
        "vram_gb": 15.51,
        "power_watt": 169.7,
        "energy_joules_query": 270.22,
        "latency_sec": 1.59,
        "throughput_tok_s": 20.4,
        "f1_score": 27.1,
        "faithfulness_index": 43.3,
        "morph_robustness": 20.0,
        "json_compliance": 62.5
    },
    "EduTerm-14B": {
        "vram_gb": 27.90,
        "power_watt": 216.5,
        "energy_joules_query": 908.53,
        "latency_sec": 4.19,
        "throughput_tok_s": 15.3,
        "f1_score": 40.8,
        "faithfulness_index": 30.0,
        "morph_robustness": 101.5,
        "json_compliance": 100.0
    }
}

DEPLOYMENT_PERSONAS = [
    {
        "persona": "Rural School / Low-End Edge Node",
        "description": "High weight on low power, low VRAM, low latency; moderate weight on F1",
        "weights": {"f1": 0.25, "morph": 0.10, "vram_penalty": 0.35, "energy_penalty": 0.30}
    },
    {
        "persona": "Interactive Smart Classroom",
        "description": "Balanced latency, token throughput, and extraction fidelity",
        "weights": {"f1": 0.40, "morph": 0.20, "vram_penalty": 0.20, "energy_penalty": 0.20}
    },
    {
        "persona": "National Educational Cloud / MoNE",
        "description": "Maximum linguistic precision and morphological declension handling; unlimited compute",
        "weights": {"f1": 0.50, "morph": 0.40, "vram_penalty": 0.05, "energy_penalty": 0.05}
    }
]

def compute_utility(profile, weights):
    # Normalized benefits (0 - 1)
    norm_f1 = profile["f1_score"] / 60.0
    norm_morph = profile["morph_robustness"] / 105.0
    
    # Normalized costs (0 - 1)
    norm_vram = profile["vram_gb"] / 32.0
    norm_energy = profile["energy_joules_query"] / 1000.0
    
    utility = (
        (weights["f1"] * norm_f1) +
        (weights["morph"] * norm_morph) -
        (weights["vram_penalty"] * norm_vram) -
        (weights["energy_penalty"] * norm_energy)
    )
    return round(float(utility), 3)

def run_pareto_simulation():
    results = {}
    
    for p in DEPLOYMENT_PERSONAS:
        p_name = p["persona"]
        results[p_name] = {
            "description": p["description"],
            "model_rankings": []
        }
        scored = []
        for m_name, prof in MODELS_EMPIRICAL_PROFILES.items():
            u = compute_utility(prof, p["weights"])
            scored.append({"model": m_name, "utility_score": u})
        
        scored.sort(key=lambda x: x["utility_score"], reverse=True)
        results[p_name]["model_rankings"] = scored
        results[p_name]["optimal_pareto_choice"] = scored[0]["model"]

    # Monte-Carlo weight sensitivity (10,000 draws)
    np.random.seed(42)
    win_counts = {m: 0 for m in MODELS_EMPIRICAL_PROFILES}
    
    for _ in range(10000):
        # Sample Dirichlet random weights
        w = np.random.dirichlet([2, 2, 2, 2])
        w_dict = {"f1": w[0], "morph": w[1], "vram_penalty": w[2], "energy_penalty": w[3]}
        scores = {m: compute_utility(prof, w_dict) for m, prof in MODELS_EMPIRICAL_PROFILES.items()}
        winner = max(scores, key=scores.get)
        win_counts[winner] += 1

    pareto_share_pct = {m: round((c / 10000.0) * 100, 1) for m, c in win_counts.items()}
    
    output_data = {
        "deployment_personas_analysis": results,
        "monte_carlo_pareto_optimality_share_pct": pareto_share_pct
    }
    return output_data

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/pareto_optimization_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    report = run_pareto_simulation()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logging.info(f"Pareto multi-objective simulation complete. Saved to {args.output}")

if __name__ == "__main__":
    main()

