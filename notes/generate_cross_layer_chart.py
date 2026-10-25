"""Generate publication-ready Cross-Layer Error Propagation visualization for EduTerm thesis.

Empirical evaluation on Boğaziçi HPC NVIDIA A40 (48GB):
- ASR Models: Whisper Tiny, Base, Small, Medium, Large-v3 Turbo, Large-v3, and Gold Reference.
- Downstream LLM: EduTerm-7B-TR (Qwen2.5-7B + SFT LoRA).
- Evaluation Metrics: ASR WER/CER vs. LLM Term Extraction Recall, Precision, F1-Score, and Hallucination Count.
- Empirical Regression: d(F1) / d(WER) sensitivity slope.
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
CHART_PATH = OUTPUT_DIR / "cross_layer_curve.png"
FIGURES_PATH = ROOT / "notes" / "figures" / "cross_layer_curve.png"
FIGURES_PATH.parent.mkdir(parents=True, exist_ok=True)

REPORT_PATH = ROOT / "experiments" / "outputs" / "cross_layer_measured_report.json"


def plot_cross_layer_charts(data=None):
    if data is None and REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not data:
        print("No report data found. Please run benchmark on HPC first.")
        return

    results = data.get("data_points", data.get("results", []))
    if not results:
        return

    # Extract TR-1 and EN-1 results
    tr_results = [r for r in results if r.get("clip_id", r.get("clip")) == "TR-1"]
    en_results = [r for r in results if r.get("clip_id", r.get("clip")) == "EN-1"]

    # Helper to get percent
    def to_pct(val):
        return round(val * 100, 2) if val <= 1.0 else round(val, 2)

    # Sort TR-1 by WER
    tr_results.sort(key=lambda x: x.get("wer_pct", to_pct(x.get("wer", 0.0))))

    tr_models = [r.get("asr_label", r.get("asr_model", "Unknown")) for r in tr_results]
    tr_wer = [r.get("wer_pct", to_pct(r.get("wer", 0.0))) for r in tr_results]
    tr_recall = [to_pct(r.get("recall", r.get("term_recall", 0.0))) for r in tr_results]
    tr_precision = [to_pct(r.get("precision", r.get("term_precision", 0.0))) for r in tr_results]
    tr_f1 = [to_pct(r.get("f1", r.get("term_f1", 0.0))) for r in tr_results]
    tr_hallucinations = [len(r.get("false_positives", [])) if "false_positives" in r else r.get("hallucinations", 0) for r in tr_results]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # ------------------ PANEL 1: Continuous Quality Curve (WER vs. F1 & Recall) ------------------
    ax1 = axes[0, 0]
    ax1.plot(tr_wer, tr_f1, color="#DC2626", marker="o", linewidth=2.8, markersize=9, label="LLM Terim Çıkarımı F1-Skoru (%)")
    ax1.plot(tr_wer, tr_recall, color="#0284C7", marker="s", linewidth=2.2, markersize=8, linestyle="--", label="Terim Yakalama (Recall %)")

    # Linear regression fit for dF1 / dWER
    if len(tr_wer) >= 3:
        w_arr = np.array(tr_wer)
        f_arr = np.array(tr_f1)
        slope, intercept = np.polyfit(w_arr, f_arr, 1)
        x_vals = np.linspace(min(tr_wer), max(tr_wer), 100)
        y_vals = slope * x_vals + intercept
        ax1.plot(x_vals, y_vals, color="#94A3B8", linestyle=":", linewidth=1.8,
                 label=f"Regresyon Eğilimi (Eğim: {slope:.2f}% F1 / 1% WER)")

    for r in tr_results:
        m_label = r.get("asr_label", r.get("asr_model", "")).replace("openai/", "").replace("Whisper ", "")
        if "Gold" in m_label:
            m_label = "Gold (0%)"
        w_val = r.get("wer_pct", to_pct(r.get("wer", 0.0)))
        f_val = to_pct(r.get("f1", r.get("term_f1", 0.0)))
        ax1.annotate(m_label, (w_val, f_val), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8.5, fontweight="bold", color="#1E293B")

    ax1.set_title("(a) Çapraz Katman Hata Yayılım Eğrisi (ASR WER % → LLM Terim F1 %)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("ASR Kelime Hata Oranı (WER %)", fontsize=11)
    ax1.set_ylabel("LLM Terim Başarımı (%)", fontsize=11)
    ax1.set_ylim(0, 110)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ------------------ PANEL 2: Hallucination Count & Spurious Terms ------------------
    ax2 = axes[0, 1]
    bar_colors = ["#10B981" if w < 10 else "#F59E0B" if w < 40 else "#EF4444" for w in tr_wer]
    bars = ax2.bar([r.get("asr_label", r.get("asr_model", "")).replace("openai/", "").replace("Whisper ", "") for r in tr_results],
                   tr_hallucinations, color=bar_colors, width=0.55, edgecolor="#0F172A", linewidth=1.2, alpha=0.9)
    for bar, val in zip(bars, tr_hallucinations):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3, str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Akustik Hata Bozulmasına Bağlı Halüsinasyon Terim Sayısı", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Üretilen Halüsinasyon / Hatalı Terim Sayısı", fontsize=11)
    ax2.set_ylim(0, max(tr_hallucinations + [5]) + 4)
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # ------------------ PANEL 3: Multilingual Error Ladder Comparison (TR-1 vs EN-1) ------------------
    ax3 = axes[1, 0]
    en_results.sort(key=lambda x: x.get("wer_pct", to_pct(x.get("wer", 0.0))))
    common_models = [r.get("asr_label", r.get("asr_model", "")).replace("openai/", "").replace("Whisper ", "") for r in tr_results if not "Gold" in r.get("asr_label", r.get("asr_model", ""))]
    tr_map = {r.get("asr_label", r.get("asr_model", "")).replace("openai/", "").replace("Whisper ", ""): r.get("wer_pct", to_pct(r.get("wer", 0.0))) for r in tr_results}
    en_map = {r.get("asr_label", r.get("asr_model", "")).replace("openai/", "").replace("Whisper ", ""): r.get("wer_pct", to_pct(r.get("wer", 0.0))) for r in en_results}

    x = np.arange(len(common_models))
    w = 0.35
    tr_bar_vals = [tr_map.get(m, 0) for m in common_models]
    en_bar_vals = [en_map.get(m, 0) for m in common_models]

    b1 = ax3.bar(x - w/2, tr_bar_vals, width=w, label="Türkçe (TR-1) WER %", color="#E11D48", edgecolor="#0F172A", alpha=0.9)
    b2 = ax3.bar(x + w/2, en_bar_vals, width=w, label="İngilizce (EN-1) WER %", color="#2563EB", edgecolor="#0F172A", alpha=0.9)

    for bar, val in zip(b1, tr_bar_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"%{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, val in zip(b2, en_bar_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"%{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax3.set_title("(c) Diller Arası Akustik Model Hata Merdiveni (TR vs. EN)", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(common_models, rotation=25, ha="right")
    ax3.set_ylabel("Kelime Hata Oranı (WER %)", fontsize=11)
    ax3.set_ylim(0, 105)
    ax3.legend(loc="upper right", frameon=True, facecolor="white")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # ------------------ PANEL 4: Resilience & Degradation Zones ------------------
    ax4 = axes[1, 1]
    scatter = ax4.scatter(tr_recall, tr_precision, s=[(w + 10)*4 for w in tr_wer],
                          c=tr_wer, cmap="viridis_r", edgecolor="#0F172A", linewidth=1.5, zorder=3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("ASR WER (%)", fontsize=10)

    for r in tr_results:
        m_label = r.get("asr_label", r.get("asr_model", "")).replace("openai/", "").replace("Whisper ", "")
        rec_val = to_pct(r.get("recall", r.get("term_recall", 0.0)))
        prec_val = to_pct(r.get("precision", r.get("term_precision", 0.0)))
        ax4.annotate(m_label, (rec_val, prec_val),
                     textcoords="offset points", xytext=(8, -4), fontsize=8.5, fontweight="bold")

    ax4.axvspan(70, 105, color="#10B981", alpha=0.15, label="Yüksek Direnç Bölgesi (WER < %15)")
    ax4.axvspan(40, 70, color="#F59E0B", alpha=0.15, label="Dönüşüm / Bozulma Bölgesi (%15-50)")
    ax4.axvspan(0, 40, color="#EF4444", alpha=0.15, label="Kritik Çöküş Bölgesi (WER > %50)")

    ax4.set_title("(d) Akustik Bozulma Karşısında LLM Hassasiyet & Direnç Bölgeleri", fontsize=12, fontweight="bold", pad=12)
    ax4.set_xlabel("Terim Yakalama Duyarlılığı (Recall %)", fontsize=11)
    ax4.set_ylabel("Terim Doğruluk Oranı (Precision %)", fontsize=11)
    ax4.set_xlim(0, 110)
    ax4.set_ylim(0, 110)
    ax4.legend(loc="lower left", frameon=True, facecolor="white", fontsize=8.5)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(CHART_PATH, bbox_inches="tight")
    plt.savefig(FIGURES_PATH, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated cross-layer error propagation curve charts at {CHART_PATH} and {FIGURES_PATH}")


if __name__ == "__main__":
    plot_cross_layer_charts()
