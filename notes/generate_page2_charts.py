"""Generate visual charts for Page 2 of the Academic Poster."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

IMG_DIR = Path(__file__).resolve().parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'


def generate_rag_chart():
    """Generate bar chart comparing Baseline Zero-Shot vs RAG-Augmented Extraction matching current artifact."""
    metrics = ['Hassasiyet (P)', 'Duyarlılık (R)', 'F1 Skoru', 'Halüsinasyon (H)']
    baseline = [0.933, 1.000, 0.966, 0.067]
    rag = [1.000, 0.929, 0.963, 0.000]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.6, 2.7), dpi=300)
    fig.patch.set_facecolor('white')

    rects1 = ax.bar(x - width/2, baseline, width, label='Temel LLM (Kısıtsız)', color='#78909C', edgecolor='white', linewidth=1.2)
    rects2 = ax.bar(x + width/2, rag, width, label='Hibrit RAG (Kanonik)', color='#00838F', edgecolor='white', linewidth=1.2)

    ax.set_ylabel('Skor (0.0 - 1.0)', fontsize=8, fontweight='bold', color='#1B365D')
    ax.set_title('RAG Ontolojik Bağlama Karşılaştırması (Sentetik Ön-Test)', fontsize=8.5, fontweight='bold', color='#1B365D', pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7.5, fontweight='bold')
    ax.legend(fontsize=7.2, loc='upper right', frameon=True, facecolor='#F4F7FB')
    ax.set_ylim(0, 1.18)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Value labels
    for r in rects1:
        h = r.get_height()
        ax.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha='center', va='bottom', fontsize=6.5, fontweight='bold', color='#00695C')

    plt.tight_layout()
    out_path = IMG_DIR / "rag_performance_chart.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_finetune_gain_chart():
    """Generate bar chart comparing Base Qwen 2.5 vs Fine-Tuned Qwen 2.5 Target Projections."""
    categories = ['Ders Terim\nRecall', 'Terim\nPrecision', 'Genel F1\nSkoru', 'Bloom\nDoğruluğu', 'B1 Uyum\n(Ateşman)']
    base_scores = [0.533, 0.615, 0.571, 0.400, 0.385]
    ft_scores = [0.933, 0.965, 0.949, 0.900, 0.865]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.6, 2.9), dpi=300)
    fig.patch.set_facecolor('white')

    rects1 = ax.bar(x - width/2, base_scores, width, label='Qwen 2.5-1.5B (Temel)', color='#78909C', edgecolor='white')
    rects2 = ax.bar(x + width/2, ft_scores, width, label='Qwen 2.5 (LoRA Projeksiyonu)', color='#2E7D32', edgecolor='white')

    ax.set_ylabel('Başarım Oranı', fontsize=8, fontweight='bold', color='#1B365D')
    ax.set_title('Ders Notları LoRA İnce Ayar Tasarım Projeksiyonu (Dry-Run)', fontsize=8.2, fontweight='bold', color='#1B365D', pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7.2, fontweight='bold')
    
    # Place legend centered at top without colliding with first bar (0.933)
    ax.legend(fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True, facecolor='#F4F7FB')
    ax.set_ylim(0, 1.30)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for r in rects2:
        h = r.get_height()
        ax.annotate(f'{h:.1%}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha='center', va='bottom', fontsize=6.5, fontweight='bold', color='#1B5E20')

    # Footnote below x-axis to avoid crossing bars
    fig.text(0.5, 0.02, "* Tasarım Projeksiyonu: RTX 5060 dry-run simülasyonu yapılmış, A100 hedef modelidir",
             ha='center', fontsize=6, fontstyle='italic', color='#555555')

    plt.subplots_adjust(bottom=0.15, top=0.90)
    out_path = IMG_DIR / "finetune_gain_chart.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    generate_rag_chart()
    generate_finetune_gain_chart()
