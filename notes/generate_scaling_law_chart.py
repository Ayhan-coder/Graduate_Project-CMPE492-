"""Generate publication-ready Scaling Law visualization charts for EduTerm thesis.

Empirical evaluation on Boğaziçi HPC NVIDIA A40 (48GB):
- Qwen2.5-0.5B-Instruct (0.49B params) - Edge SLM
- Qwen2.5-1.5B-Instruct (1.54B params) - Uç Cihaz Hedefi
- Qwen2.5-7B-Instruct (7.62B params) - HPC Standart Temel Model
- EduTerm-7B-TR (7.62B + SFT LoRA) - Dikey Alan Uyarlanmış Model (95.7% Recall)
- Qwen2.5-14B-Instruct (14.77B params) - Büyük HPC Modeli (85.4% Recall)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.9

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "notes" / "img"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_PATH = OUTPUT_DIR / "scaling_law_curve.png"


def plot_scaling_law_charts():
    # Empirical measured metrics from Boğaziçi HPC A40 experiments
    models = ["0.5B (Edge)", "1.5B (Uç Cihaz)", "7B (Base)", "7B (EduTerm SFT)", "14B (Büyük HPC)"]
    params = [0.49, 1.54, 7.62, 7.66, 14.77]
    vram = [0.95, 2.90, 14.28, 14.39, 27.74]
    latency = [1.50, 2.25, 4.16, 1.68, 7.11]
    recall = [41.7, 68.8, 45.8, 95.7, 85.4]
    precision = [37.5, 56.2, 50.0, 95.8, 71.7]
    f1 = [39.5, 61.9, 47.8, 95.7, 77.9]
    atesman = [70.9, 45.5, 60.3, 66.4, 38.7]
    b1_comp = [12.5, 50.0, 100.0, 100.0, 25.0]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # ------------------ PANEL 1: Parameter Scaling vs. Term Recall & F1 ------------------
    ax1 = axes[0, 0]
    bar_colors = ["#94A3B8", "#38BDF8", "#0284C7", "#10B981", "#6366F1"]
    bars = ax1.bar(models, recall, color=bar_colors, width=0.55, edgecolor="#0F172A", linewidth=1.2, alpha=0.9)
    for bar, val in zip(bars, recall):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 2.0, f"%{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.plot(models, f1, color="#DC2626", marker="o", linewidth=2.5, markersize=8, label="F1-Score (%)")
    ax1.set_title("(a) Model Ölçeğine Göre Terim Doğruluğu (Recall & F1)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Yakalama Başarımı (%)", fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # ------------------ PANEL 2: Readability (Ateşman) & CEFR B1 Target ------------------
    ax2 = axes[0, 1]
    ax2.plot(models, atesman, color="#0284C7", marker="s", linewidth=2.5, markersize=8, label="Ateşman Okunabilirlik İndeksi")
    ax2.axhspan(50, 70, color="#10B981", alpha=0.18, label="Hedef CEFR B1 Bandı (50-70 Ateşman)")
    for i, val in enumerate(atesman):
        offset = 2.5 if val < 50 else -4.0
        ax2.annotate(f"{val:.1f}", (models[i], val + offset), ha="center", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Pedagojik Sadeleştirme: Ateşman Okunabilirlik Seyri", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Ateşman İndeksi (0-100)", fontsize=11)
    ax2.set_ylim(30, 85)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # ------------------ PANEL 3: VRAM Footprint & Hardware Feasibility ------------------
    ax3 = axes[1, 0]
    bars_vram = ax3.bar(models, vram, color="#F59E0B", width=0.55, edgecolor="#78350F", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars_vram, vram):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val:.2f} GB", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.axhline(8.0, color="#DC2626", linestyle="--", linewidth=2.0, label="RTX 5060 Laptop Donanım Sınırı (8 GB)")
    ax3.axhline(48.0, color="#059669", linestyle=":", linewidth=2.0, label="NVIDIA A40 HPC Kümesi Sınırı (48 GB)")
    ax3.set_title("(c) Donanım Bellek Ayak İzi: Çıkarım Zirve VRAM Tüketimi (BF16)", fontsize=12, fontweight="bold", pad=12)
    ax3.set_ylabel("Peak VRAM (GB)", fontsize=11)
    ax3.set_ylim(0, 52)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # ------------------ PANEL 4: Latency vs. Domain Accuracy (Pareto Frontier) ------------------
    ax4 = axes[1, 1]
    for i, m in enumerate(models):
        c = "#10B981" if "EduTerm" in m else ("#6366F1" if "14B" in m else ("#38BDF8" if "1.5B" in m else "#0284C7"))
        ax4.scatter(latency[i], recall[i], s=250, color=c, edgecolors="#0F172A", linewidth=1.6, zorder=5)
        offset_y = 3.5 if "EduTerm" not in m else -6.0
        ax4.annotate(m, (latency[i], recall[i] + offset_y), ha="center", fontsize=9.5, fontweight="bold")
    ax4.set_title("(d) Gecikme - Doğruluk Takası (Latency vs Recall Pareto Eğrisi)", fontsize=12, fontweight="bold", pad=12)
    ax4.set_xlabel("Sorgu Başına Ortalama Çıkarım Süresi (saniye)", fontsize=11)
    ax4.set_ylabel("Dikey Alan Terim Yakalama Recall (%)", fontsize=11)
    ax4.set_ylim(30, 108)
    ax4.set_xlim(0.8, 8.0)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=300)
    plt.close()
    print(f"Empirical Scaling Law chart generated successfully: {CHART_PATH}")


if __name__ == "__main__":
    plot_scaling_law_charts()
