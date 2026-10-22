"""Generate high-resolution academic figures for the graduation project poster (No Manimator)."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

IMG_DIR = Path(__file__).resolve().parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#C4C8CF'
plt.rcParams['axes.linewidth'] = 0.8

def generate_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.axis('off')
    
    # Colors
    c_blue = "#1B365D"    # Layer 1 - Navy
    c_teal = "#1A8AA2"    # Layer 2 - Teal
    c_amber = "#B36B00"   # Layer 3 / Downstream - Amber/Ochre
    c_card_bg = "#FFFFFF"
    
    fig.patch.set_facecolor('white')
    
    # 3 Pipeline Stage Boxes
    # Stage 1: ASR
    rect1 = patches.FancyBboxPatch((0.02, 0.1), 0.28, 0.8, boxstyle="round,pad=0.02,rounding_size=0.03", 
                                  facecolor=c_card_bg, edgecolor=c_blue, linewidth=2, zorder=2)
    ax.add_patch(rect1)
    
    # Stage 2: LLM Terminology
    rect2 = patches.FancyBboxPatch((0.36, 0.1), 0.28, 0.8, boxstyle="round,pad=0.02,rounding_size=0.03", 
                                  facecolor=c_card_bg, edgecolor=c_teal, linewidth=2, zorder=2)
    ax.add_patch(rect2)
    
    # Stage 3: Downstream Educational Schema & DAG
    rect3 = patches.FancyBboxPatch((0.70, 0.1), 0.28, 0.8, boxstyle="round,pad=0.02,rounding_size=0.03", 
                                  facecolor=c_card_bg, edgecolor=c_amber, linewidth=2, zorder=2)
    ax.add_patch(rect3)
    
    # Connectors
    arr1 = patches.FancyArrowPatch((0.305, 0.5), (0.355, 0.5), arrowstyle='-|>', mutation_scale=20, 
                                  color='#5A5F66', lw=2.5, zorder=3)
    ax.add_patch(arr1)
    
    arr2 = patches.FancyArrowPatch((0.645, 0.5), (0.695, 0.5), arrowstyle='-|>', mutation_scale=20, 
                                  color='#5A5F66', lw=2.5, zorder=3)
    ax.add_patch(arr2)
    
    # Headers
    ax.text(0.16, 0.83, "KATMAN 1: KONUŞMA TANIMA (ASR)", ha='center', va='center', fontsize=9, fontweight='bold', color=c_blue)
    ax.text(0.16, 0.75, "16 kHz Mono Ses $\\to$ Transkript", ha='center', va='center', fontsize=8, color='#5A5F66')
    
    ax.text(0.50, 0.83, "KATMAN 2: SEMANTİK ÇIKARIM", ha='center', va='center', fontsize=9, fontweight='bold', color=c_teal)
    ax.text(0.50, 0.75, "LLM Tabanlı Terim Çıkarımı", ha='center', va='center', fontsize=8, color='#5A5F66')
    
    ax.text(0.84, 0.83, "AŞAĞI AKIŞ: PEDAGOJİK ŞEMA", ha='center', va='center', fontsize=9, fontweight='bold', color=c_amber)
    ax.text(0.84, 0.75, "Kavram Önkoşul DAG & Bloom", ha='center', va='center', fontsize=8, color='#5A5F66')
    
    # Content bullets
    l1_text = (
        "• Modeller: Whisper Large-v3/Turbo,\n"
        "  Qwen3-ASR (0.6B/1.7B), Nemotron\n"
        "• Sabit 30s Pencereli Kod Çözme\n"
        "• TR WER: %2.1 (Large-v3), %3.7 (Turbo)\n"
        "• EN WER: %1.0 (Large-v3), %1.1 (Turbo)\n"
        "• Hız (RTF): 0.025-0.044 (Turbo 23-40×)"
    )
    ax.text(0.04, 0.45, l1_text, ha='left', va='center', fontsize=7.5, color='#2C3E50', linespacing=1.4)
    
    l2_text = (
        "• Modeller: Qwen 3.5 (4B/9B) +\n"
        "  EVREN Millî YZ Platformu (H200)\n"
        "• İstem: Zero-Shot vs Chain-of-Thought\n"
        "• Kapalı Altın Liste: TR-1 / EN-1 (N=32)\n"
        "• Çift Etiketli IAA: $\\kappa_R=0.782, \\kappa_D=1.000$\n"
        "• Okunabilirlik: Ateşman (TR) & Flesch (EN)"
    )
    ax.text(0.38, 0.45, l2_text, ha='left', va='center', fontsize=7.5, color='#2C3E50', linespacing=1.4)
    
    l3_text = (
        "• Zaman Damgalı Terim Eşleme\n"
        "• 4 Katmanlı Açıklama (B1 Analoji,\n"
        "  Akademik Tanım, Asansör, Sınav İpucu)\n"
        "• Bloom Taksonomisi (1-6 Seviye)\n"
        "• Yönlü Çevrimsiz Önkoşul Grafiği (DAG)\n"
        "• Topolojik Sıralama ile Ders Yolu"
    )
    ax.text(0.72, 0.45, l3_text, ha='left', va='center', fontsize=7.5, color='#2C3E50', linespacing=1.4)
    
    plt.tight_layout()
    out_path = IMG_DIR / "pipeline_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")

def generate_concept_dag():
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Node definition: (id, label, x, y, level_text, bg_color, border_color)
    nodes = [
        ("N1", "Proje Yönetimi", 0.10, 0.50, "Bloom 1: Hatırlama", "#EBF3FB", "#1B365D"),
        ("N2", "Proje Yönetim Planı", 0.32, 0.75, "Bloom 2: Anlama", "#EBF3FB", "#1B365D"),
        ("N3", "İzleme ve Kontrol", 0.32, 0.25, "Bloom 2: Anlama", "#EBF3FB", "#1B365D"),
        ("N4", "Temel Çizgi (Baseline)", 0.54, 0.75, "Bloom 3: Uygulama", "#E8F8F5", "#1A8AA2"),
        ("N5", "Kazanılmış Değer (EVM)", 0.54, 0.25, "Bloom 3: Uygulama", "#E8F8F5", "#1A8AA2"),
        ("N6", "Planlanmış Değer (PV)", 0.74, 0.85, "Bloom 4: Analiz", "#FEF9E7", "#B36B00"),
        ("N7", "Gerçekleşen Maliyet (AC)", 0.74, 0.50, "Bloom 4: Analiz", "#FEF9E7", "#B36B00"),
        ("N8", "Sapma İndeksleri (CPI/SPI)", 0.74, 0.15, "Bloom 5: Değerlendirme", "#FADBD8", "#B91E1E"),
    ]
    
    node_coords = {n[0]: (n[2], n[3]) for n in nodes}
    
    # Directed Edges (prerequisite relationships)
    edges = [
        ("N1", "N2"),
        ("N1", "N3"),
        ("N2", "N4"),
        ("N3", "N5"),
        ("N4", "N5"),
        ("N4", "N6"),
        ("N5", "N7"),
        ("N5", "N8"),
        ("N6", "N8"),
        ("N7", "N8"),
    ]
    
    # Draw Edges first
    for u, v in edges:
        p1 = node_coords[u]
        p2 = node_coords[v]
        arr = patches.FancyArrowPatch(
            (p1[0] + 0.07, p1[1]), (p2[0] - 0.07, p2[1]),
            arrowstyle='-|>', mutation_scale=14, color='#7F8C8D', lw=1.5,
            connectionstyle="arc3,rad=0.08" if abs(p1[1]-p2[1]) > 0.4 else "arc3,rad=0.0",
            zorder=2
        )
        ax.add_patch(arr)
        
    # Draw Node Boxes
    for nid, label, x, y, bloom, bg, border in nodes:
        w_box, h_box = 0.16, 0.16
        box = patches.FancyBboxPatch(
            (x - w_box/2, y - h_box/2), w_box, h_box,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=bg, edgecolor=border, lw=1.6, zorder=3
        )
        ax.add_patch(box)
        ax.text(x, y + 0.02, label, ha='center', va='center', fontsize=7.2, fontweight='bold', color='#1A1A1A', zorder=4)
        ax.text(x, y - 0.035, bloom, ha='center', va='center', fontsize=5.8, fontstyle='italic', color=border, zorder=4)
        
    # Legend for Pedagogical DAG
    ax.text(0.50, 0.97, "Zenginleştirilmiş Pedagojik Şema: Kavram Önkoşul Grafiği (Prerequisite DAG)",
            ha='center', va='center', fontsize=9, fontweight='bold', color='#1B365D')
            
    plt.tight_layout()
    out_path = IMG_DIR / "concept_dag_graph.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")

def generate_asr_chart():
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
    
    models = ['Whisper\nLarge-v3', 'Whisper\nTurbo', 'Whisper\nMedium', 'Qwen3-ASR\n0.6B', 'Nemotron\n3.5']
    tr_wer = [2.1, 3.7, 4.9, 13.4, 21.4]
    tr_ci_low = [1.4, 1.1, 3.0, 8.6, 11.7]
    tr_ci_high = [3.2, 5.2, 8.1, 18.0, 28.7]
    
    en_wer = [1.0, 1.1, 1.1, 2.1, 3.5]
    en_ci_low = [0.7, 0.7, 0.8, 1.3, 2.6]
    en_ci_high = [1.2, 1.7, 1.3, 2.9, 4.9]
    
    x = np.arange(len(models))
    width = 0.36
    
    # Asymmetric error bars for 95% CI
    tr_err = [
        [tr_wer[i] - tr_ci_low[i] for i in range(len(models))],
        [tr_ci_high[i] - tr_wer[i] for i in range(len(models))]
    ]
    en_err = [
        [en_wer[i] - en_ci_low[i] for i in range(len(models))],
        [en_ci_high[i] - en_wer[i] for i in range(len(models))]
    ]
    
    rects1 = ax.bar(x - width/2, tr_wer, width, yerr=tr_err, capsize=3, label='Türkçe (TR) Normalized WER %', color='#1B365D', edgecolor='#0F223D', alpha=0.95)
    rects2 = ax.bar(x + width/2, en_wer, width, yerr=en_err, capsize=3, label='İngilizce (EN) Normalized WER %', color='#1A8AA2', edgecolor='#115E6E', alpha=0.95)
    
    ax.set_ylabel('Kelime Hata Oranı (Normalized WER %)', fontsize=8.5, fontweight='bold', color='#1A1A1A')
    ax.set_title('Çok Dilli ASR Doğruluğu ve %95 Bootstrap Güven Aralıkları', fontsize=9.5, fontweight='bold', pad=10, color='#1B365D')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.legend(frameon=True, facecolor='#F8F9FA', edgecolor='#C4C8CF', fontsize=7.5)
    ax.set_ylim(0, 32)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#C4C8CF')
    
    # Add data labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'%{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 10), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, fontweight='bold', color='#1B365D')
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'%{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, fontweight='bold', color='#1A8AA2')
                    
    plt.tight_layout()
    out_path = IMG_DIR / "asr_benchmark_chart.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_pipeline_diagram()
    generate_concept_dag()
    generate_asr_chart()
