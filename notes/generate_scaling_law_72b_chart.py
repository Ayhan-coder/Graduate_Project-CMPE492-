"""Generate Grand 5-Tier Scaling Law Visualization (0.5B -> 1.5B -> 7B -> 14B -> 72B).

Generates:
  notes/figures/scaling_law_curve_72b.png
  notes/img/scaling_law_curve_72b.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GenScalingLaw72BChart")

ROOT = Path(__file__).resolve().parents[1]
REPORT_FILE = ROOT / "experiments" / "outputs" / "scaling_law_full_72b_report.json"
FALLBACK_FILE = ROOT / "experiments" / "outputs" / "scaling_law_report.json"
FIG_DIR = ROOT / "notes" / "figures"
IMG_DIR = ROOT / "notes" / "img"
FIG_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    target_file = REPORT_FILE if REPORT_FILE.exists() else FALLBACK_FILE
    if not target_file.exists():
        logger.error(f"Neither {REPORT_FILE} nor {FALLBACK_FILE} exists!")
        return

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        logger.error("No results found in report!")
        return

    # Extract metrics
    model_names = [r["model_name"] for r in results]
    params_b = [r["parameter_count_b"] for r in results]
    f1_scores = [r["f1_score"] * 100 for r in results]
    recalls = [r["avg_recall"] * 100 for r in results]
    precisions = [r["avg_precision"] * 100 for r in results]
    vram_peaks = [r["peak_vram_gb"] for r in results]
    bloom_accs = [r["bloom_accuracy_pct"] for r in results]
    atesman_scores = [r["avg_atesman"] for r in results]
    b1_shares = [r["b1_compliance_pct"] for r in results]

    # Clean short labels
    clean_labels = []
    for m in model_names:
        if "0.5B" in m:
            clean_labels.append("0.5B (Edge)")
        elif "1.5B" in m:
            clean_labels.append("1.5B (Uç)")
        elif "EduTerm-7B-TR" in m:
            clean_labels.append("7B (LoRA SFT)")
        elif "7B" in m:
            clean_labels.append("7B (Base)")
        elif "14B" in m:
            clean_labels.append("14B (Orta)")
        elif "27B" in m:
            clean_labels.append("27B (Gemma)")
        elif "32B" in m:
            clean_labels.append("32B (Qwen)")
        elif "72B" in m:
            clean_labels.append("72B (Bayrak)")
        else:
            clean_labels.append(m[:10])

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
    # PANEL A: Terminology F1, Recall & Precision Scaling (Log Scale on Params)
    # --------------------------------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.plot(params_b, f1_scores, marker="o", color="#2980b9", linewidth=2.5, markersize=8, label="F1-Skoru (%)")
    ax_a.plot(params_b, recalls, marker="s", color="#27ae60", linewidth=2.0, linestyle="--", markersize=7, label="Duyarlılık / Recall (%)")
    ax_a.plot(params_b, precisions, marker="^", color="#e67e22", linewidth=2.0, linestyle=":", markersize=7, label="Hassasiyet / Precision (%)")

    ax_a.set_xscale("log")
    ax_a.set_xlabel("Model Parametre Sayısı (Milyar - Logaritmik Eksen)")
    ax_a.set_ylabel("Semantik Başarım (%)")
    ax_a.set_title("(A) Model Ölçeğine Göre Terminoloji Çıkarım Başarımı")
    ax_a.set_ylim(20, 105)
    ax_a.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_a.legend(loc="lower right")

    # Annotate points
    for x, y, lbl in zip(params_b, f1_scores, clean_labels):
        ax_a.annotate(
            f"{lbl}\n%{y:.1f}",
            xy=(x, y), xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#2980b9", alpha=0.8)
        )

    # --------------------------------------------------------------------------
    # PANEL B: Pedagogical Readability & CEFR B1 Share
    # --------------------------------------------------------------------------
    ax_b = axes[0, 1]
    x_b = np.arange(len(clean_labels))
    w = 0.35

    bar1 = ax_b.bar(x_b - w/2, atesman_scores, width=w, color="#8e44ad", alpha=0.85, label="Ateşman Okunabilirlik")
    bar2 = ax_b.bar(x_b + w/2, b1_shares, width=w, color="#16a085", alpha=0.85, label="CEFR B1 Uyumu (%)")

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(clean_labels, rotation=15, ha="right")
    ax_b.set_ylabel("Puan / Yüzde")
    ax_b.set_title("(B) Pedagojik Sadeleştirme ve B1 Düzeyi Uyumu")
    ax_b.set_ylim(0, 110)
    ax_b.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper right")

    for bar in bar1:
        h = bar.get_height()
        if h > 0:
            ax_b.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=8)
    for bar in bar2:
        h = bar.get_height()
        if h > 0:
            ax_b.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"%{h:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # --------------------------------------------------------------------------
    # PANEL C: Bloom Taxonomy Classification Accuracy
    # --------------------------------------------------------------------------
    ax_c = axes[1, 0]
    colors_c = ["#95a5a6", "#3498db", "#2980b9", "#27ae60", "#f39c12", "#9b59b6", "#16a085", "#e74c3c"]
    bars_c = ax_c.bar(clean_labels, bloom_accs, color=colors_c[:len(clean_labels)], alpha=0.88, edgecolor="black", linewidth=0.5)

    ax_c.set_ylabel("Bloom Eşleşme Doğruluğu (%)")
    ax_c.set_title("(C) Pedagojik Bloom Taksonomisi Doğruluk Sıçraması")
    ax_c.set_ylim(0, 110)
    ax_c.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar in bars_c:
        h = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width()/2., h + 2.0, f"%{h:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # --------------------------------------------------------------------------
    # PANEL D: Hardware VRAM & Pareto Frontier (Peak VRAM vs F1)
    # --------------------------------------------------------------------------
    ax_d = axes[1, 1]
    scatter = ax_d.scatter(vram_peaks, f1_scores, s=[p * 15 + 100 for p in params_b], c=params_b, cmap="viridis", edgecolors="black", linewidth=1.5, alpha=0.9, zorder=3)
    ax_d.axvline(8.15, color="#e74c3c", linestyle="--", linewidth=1.8, label="8 GB Tüketici Sınırı (RTX 5060)")

    ax_d.set_xlabel("Tepe VRAM Kullanımı (GB)")
    ax_d.set_ylabel("Terminoloji F1-Skoru (%)")
    ax_d.set_title("(D) Donanım Maliyeti & F1 Pareto Değiş-Tokuş Sınırı")
    ax_d.set_xlim(0, 48)
    ax_d.set_ylim(30, 105)
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="lower right")

    for x, y, lbl in zip(vram_peaks, f1_scores, clean_labels):
        ax_d.annotate(
            lbl, xy=(x, y), xytext=(5, 5), textcoords="offset points",
            fontsize=8.5, fontweight="bold"
        )

    fig.suptitle(
        "Büyük Dil Modeli Ölçeklenme Yasası Analizi (Grand Scaling Law: 0.5B → 1.5B → 7B → 14B → 27B → 32B → 72B)\n"
        "(Boğaziçi Üniversitesi HPC NVIDIA A40 48GB Donanımında Ampirik Ölçüm)",
        fontsize=13, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_fig = FIG_DIR / "scaling_law_curve_72b.png"
    out_img = IMG_DIR / "scaling_law_curve_72b.png"
    plt.savefig(out_fig, dpi=300)
    plt.savefig(out_img, dpi=300)
    plt.close()

    logger.info(f"Saved grand scaling law figure to {out_fig} and {out_img}")


if __name__ == "__main__":
    main()
