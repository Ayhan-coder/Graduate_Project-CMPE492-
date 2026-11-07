"""Generate 4-panel publication-ready comparison chart for Student Journey & BKT Simulation.

Generates:
  notes/figures/student_journey_simulation.png
  notes/img/student_journey_simulation.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GenStudentJourneyChart")

ROOT = Path(__file__).resolve().parents[1]
REPORT_FILE = ROOT / "experiments" / "outputs" / "student_journey_simulation_report.json"
FIG_DIR = ROOT / "notes" / "figures"
IMG_DIR = ROOT / "notes" / "img"
FIG_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not REPORT_FILE.exists():
        logger.error(f"Report file {REPORT_FILE} not found!")
        return

    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    res_linear = data["results"]["linear_baseline"]
    res_naive = data["results"]["naive_review_baseline"]
    res_dag = data["results"]["dag_guided_remediation"]

    # Set up matplotlib style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --------------------------------------------------------------------------
    # PANEL A: Retention Funnel Across Curriculum
    # --------------------------------------------------------------------------
    ax_a = axes[0, 0]
    funnel_linear = [item["retention_percentage"] for item in res_linear["retention_funnel"]]
    funnel_naive = [item["retention_percentage"] for item in res_naive["retention_funnel"]]
    funnel_dag = [item["retention_percentage"] for item in res_dag["retention_funnel"]]
    node_labels = [item["node_id"] for item in res_linear["retention_funnel"]]

    x_idx = np.arange(len(node_labels))
    ax_a.plot(x_idx, funnel_linear, marker="o", color="#e74c3c", linewidth=2.2, label="Doğrusal Müfredat (Linear Baseline)")
    ax_a.plot(x_idx, funnel_naive, marker="s", color="#f39c12", linewidth=2.0, linestyle="--", label="Genel Tekrar (Naive Review)")
    ax_a.plot(x_idx, funnel_dag, marker="^", color="#27ae60", linewidth=2.5, label="DAG Kök Neden Telafisi (Önerilen)")

    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(node_labels, rotation=45, ha="right")
    ax_a.set_xlabel("Müfredat Adımları (Topolojik Sıralama)")
    ax_a.set_ylabel("Kalan Öğrenci Oranı (%)")
    ax_a.set_title("(A) Öğrenci Kalıcılık Hunisi (Student Retention Funnel)")
    ax_a.set_ylim(20, 105)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="lower left", framealpha=0.9)

    # Annotate drop-off gap
    ax_a.annotate(
        f"Kalıcılık Kazancı: +{funnel_dag[-1] - funnel_linear[-1]:.1f}%\n(Terk %{res_linear['overall_churn_rate']*100:.1f} → %{res_dag['overall_churn_rate']*100:.1f})",
        xy=(len(node_labels) - 1, funnel_dag[-1]),
        xytext=(len(node_labels) - 4, 35),
        arrowprops=dict(facecolor="#27ae60", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f8f5", edgecolor="#27ae60", alpha=0.9),
        fontsize=9, fontweight="bold"
    )

    # --------------------------------------------------------------------------
    # PANEL B: Final Mastery Rate by Cohort
    # --------------------------------------------------------------------------
    ax_b = axes[0, 1]
    cohort_keys = ["advanced", "standard", "at_risk", "inattentive"]
    cohort_names = ["İleri Düzey\n(%20)", "Standart\n(%40)", "Temeli Zayıf\n(%30)", "Dikkatsiz\n(%10)"]

    rates_linear = [res_linear["cohort_breakdown"][k]["completion_rate"] * 100 for k in cohort_keys]
    rates_naive = [res_naive["cohort_breakdown"][k]["completion_rate"] * 100 for k in cohort_keys]
    rates_dag = [res_dag["cohort_breakdown"][k]["completion_rate"] * 100 for k in cohort_keys]

    bar_width = 0.25
    x_b = np.arange(len(cohort_keys))

    bars1 = ax_b.bar(x_b - bar_width, rates_linear, width=bar_width, color="#e74c3c", label="Doğrusal", alpha=0.85)
    bars2 = ax_b.bar(x_b, rates_naive, width=bar_width, color="#f39c12", label="Genel Tekrar", alpha=0.85)
    bars3 = ax_b.bar(x_b + bar_width, rates_dag, width=bar_width, color="#27ae60", label="DAG Telafisi", alpha=0.9)

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(cohort_names)
    ax_b.set_ylabel("Müfredat Başarı Oranı (%)")
    ax_b.set_title("(B) Öğrenci Kohortlarına Göre Tamamlama Oranı")
    ax_b.set_ylim(0, 115)
    ax_b.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper left")

    # Add labels on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_b.text(
                    bar.get_x() + bar.get_width() / 2.0, height + 1.5,
                    f"%{height:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold"
                )

    # --------------------------------------------------------------------------
    # PANEL C: Bottleneck Concept Churn Distribution (Linear Baseline)
    # --------------------------------------------------------------------------
    ax_c = axes[1, 0]
    churn_dict = res_linear["churn_by_node"]
    # Sort by churn count descending
    sorted_churn = sorted(churn_dict.items(), key=lambda x: x[1], reverse=True)[:6]
    c_nodes = [item[0] for item in sorted_churn]
    c_counts = [item[1] for item in sorted_churn]
    # Map node id to friendly name
    node_name_map = {item["id"]: item["name"] for item in data["curriculum_nodes"]}
    c_labels = [f"{n}\n({node_name_map.get(n, n)[:16]})" for n in c_nodes]

    colors_c = ["#c0392b", "#d35400", "#e67e22", "#f39c12", "#7f8c8d", "#95a5a6"]
    bars_c = ax_c.bar(range(len(c_nodes)), c_counts, color=colors_c, alpha=0.88, edgecolor="black", linewidth=0.5)
    ax_c.set_xticks(range(len(c_nodes)))
    ax_c.set_xticklabels(c_labels, rotation=15, ha="right", fontsize=8.5)
    ax_c.set_ylabel("Dersi Terk Eden Öğrenci Sayısı (Churn)")
    ax_c.set_title("(C) Müfredattaki Temel Tıkanma / Düşüş Noktaları (Doğrusal)")
    ax_c.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar in bars_c:
        h = bar.get_height()
        ax_c.text(
            bar.get_x() + bar.get_width() / 2.0, h + 2.0,
            f"{int(h)} ({h/10:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )

    # --------------------------------------------------------------------------
    # PANEL D: Root-Cause Prerequisites Remediated (Our Model)
    # --------------------------------------------------------------------------
    ax_d = axes[1, 1]
    remed_dict = res_dag.get("root_causes_remediated", {})
    sorted_remed = sorted(remed_dict.items(), key=lambda x: x[1], reverse=True)[:6]
    r_nodes = [item[0] for item in sorted_remed]
    r_counts = [item[1] for item in sorted_remed]
    r_labels = [f"{n}\n({node_name_map.get(n, n)[:16]})" for n in r_nodes]

    colors_d = ["#16a085", "#1abc9c", "#2ecc71", "#27ae60", "#3498db", "#2980b9"]
    bars_d = ax_d.bar(range(len(r_nodes)), r_counts, color=colors_d, alpha=0.88, edgecolor="black", linewidth=0.5)
    ax_d.set_xticks(range(len(r_nodes)))
    ax_d.set_xticklabels(r_labels, rotation=15, ha="right", fontsize=8.5)
    ax_d.set_ylabel("Müdahale Sayısı (Targeted Interventions)")
    ax_d.set_title("(D) En Çok Telafi Edilen Kök Neden Önkoşullar (DAG Motoru)")
    ax_d.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar in bars_d:
        h = bar.get_height()
        ax_d.text(
            bar.get_x() + bar.get_width() / 2.0, h + 3.0,
            f"{int(h)} kez", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )

    # Overall figure title
    fig.suptitle(
        "Müşteri Analitiği Modeliyle Öğrenci Yolculuğu & Bayesian Knowledge Tracing (BKT) Simülasyonu\n"
        "(1.000 Sentetik Öğrenci, 4 Kohort, 12 Düğümlü Pedagojik İSG Çizgesi)",
        fontsize=13, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_fig = FIG_DIR / "student_journey_simulation.png"
    out_img = IMG_DIR / "student_journey_simulation.png"
    plt.savefig(out_fig, dpi=300)
    plt.savefig(out_img, dpi=300)
    plt.close()

    logger.info(f"Saved figure to {out_fig} and {out_img}")


if __name__ == "__main__":
    main()
