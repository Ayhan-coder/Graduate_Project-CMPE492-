"""Generate donut charts and visual elements matching the user's template layout."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

IMG_DIR = Path(__file__).resolve().parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

def generate_donut_chart_a():
    # 5 Discipline clusters in the 40-video inventory (20 TR, 20 EN, 13.6 hours)
    labels = ['Bilgisayar & YZ', 'Mühendislik & Devre', 'Fizik & Matematik', 'Sosyal & Turizm', 'Kurumsal Uyum (KVKK/İSG)']
    sizes = [10, 8, 8, 6, 8]
    colors = ['#E65100', '#F57C00', '#FF9800', '#FFB74D', '#FFE082']
    
    fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=300)
    fig.patch.set_facecolor('white')
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.0f%%', startangle=140,
        colors=colors, pctdistance=0.75,
        textprops={'fontsize': 6.8, 'color': '#1A1A1A', 'weight': 'bold'},
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(7.0)
        at.set_weight('bold')
        at.set_color('#1A1A1A')
        
    ax.set_title("40 Videoluk Korpus Envanteri Disiplin Dağılımı", fontsize=8.2, fontweight='bold', color='#1B365D', pad=6)
    
    plt.tight_layout()
    out_path = IMG_DIR / "chart_a_donut.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")

def generate_donuts_bc():
    # Side-by-side smaller Donut Charts B & C
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.6), dpi=300)
    fig.patch.set_facecolor('white')
    
    # Chart B: Ses Durumu (Diskte Hazır WAV vs Beklemede)
    sizes_b = [7, 25]
    labels_b = ['Diskte (7)', 'Beklemede (25)']
    colors_b = ['#00838F', '#4DD0E1']
    
    wedges1, texts1, autotexts1 = ax1.pie(
        sizes_b, labels=labels_b, autopct='%1.1f%%', startangle=90,
        colors=colors_b, pctdistance=0.72,
        textprops={'fontsize': 7, 'weight': 'bold'},
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5)
    )
    ax1.set_title("Ses Durumu (Chart B)", fontsize=8, fontweight='bold', color='#1B365D')
    
    # Chart C: Transkript Durumu (Altın vs Taslak)
    sizes_c = [7, 25]
    labels_c = ['Altın (7)', 'Taslak (25)']
    colors_c = ['#C2185B', '#F48FB1']
    
    wedges2, texts2, autotexts2 = ax2.pie(
        sizes_c, labels=labels_c, autopct='%1.1f%%', startangle=90,
        colors=colors_c, pctdistance=0.72,
        textprops={'fontsize': 7, 'weight': 'bold'},
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5)
    )
    ax2.set_title("Transkript Durumu (Chart C)", fontsize=8, fontweight='bold', color='#1B365D')
    
    plt.tight_layout()
    out_path = IMG_DIR / "chart_bc_donuts.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")

def generate_logo():
    # Generate clean circular university / project logo
    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('none')
    
    # Green circle matching user's template logo
    circle = plt.Circle((0.5, 0.5), 0.46, color='#2CDA9D', ec='white', lw=3)
    ax.add_patch(circle)
    ax.text(0.5, 0.58, "BOUN", ha='center', va='center', fontsize=15, fontweight='black', color='#1B365D')
    ax.text(0.5, 0.40, "EduTerm", ha='center', va='center', fontsize=9, fontweight='bold', color='#1B365D')
    ax.text(0.5, 0.28, "2026", ha='center', va='center', fontsize=7, fontweight='bold', color='#1B365D')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    out_path = IMG_DIR / "logo_badge.png"
    plt.savefig(out_path, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_donut_chart_a()
    generate_donuts_bc()
    generate_logo()
