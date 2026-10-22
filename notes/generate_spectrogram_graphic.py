import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_spectrogram_sample():
    # Set high DPI and aesthetic styling
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.2), dpi=300, gridspec_kw={'height_ratios': [1.2, 1]})
    fig.patch.set_facecolor('#ffffff')
    
    # 1. Synthetic realistic Mel-Spectrogram & Speech signal
    t = np.linspace(0, 4.0, 1000)
    # Speech envelope with formants
    carrier = np.sin(2 * np.pi * 180 * t) * 0.4 + np.sin(2 * np.pi * 540 * t) * 0.3 + np.sin(2 * np.pi * 1800 * t) * 0.2
    env = np.exp(-((t - 1.2)/0.4)**2) + 0.8 * np.exp(-((t - 2.6)/0.5)**2) + 0.3 * np.random.normal(0, 0.05, len(t))
    signal = carrier * env
    
    # 2D Spectrogram data
    freq_bins = 64
    time_bins = 200
    np.random.seed(42)
    spec_data = np.zeros((freq_bins, time_bins))
    for i in range(time_bins):
        tt = i / time_bins * 4.0
        e = np.exp(-((tt - 1.2)/0.4)**2) + 0.8 * np.exp(-((tt - 2.6)/0.5)**2)
        # Formants around bin 12, 28, 45
        formants = np.exp(-((np.arange(freq_bins) - 12)/3.5)**2) * 1.5 + \
                   np.exp(-((np.arange(freq_bins) - 28)/4.0)**2) * 1.2 + \
                   np.exp(-((np.arange(freq_bins) - 45)/5.0)**2) * 0.8
        spec_data[:, i] = formants * e * 80 + np.random.normal(15, 4, freq_bins)
    
    spec_data = np.clip(spec_data, 0, 100)
    
    # Display Mel-Spectrogram
    im = ax1.imshow(spec_data, origin='lower', aspect='auto', extent=[0, 4.0, 0, 8000], cmap='viridis')
    ax1.set_ylabel('Frekans (Hz)', fontsize=9, fontweight='bold', color='#1A2B4C')
    ax1.set_title('16 kHz Akustik Sinyal & Mel-Spektrogram (Kirchhoff Devre Analizi Dersi)', fontsize=9.5, fontweight='bold', color='#0D233A', pad=6)
    ax1.tick_params(colors='#333333', labelsize=8)
    
    # Overlay timestamp boundaries
    ax1.axvline(0.7, color='#FF5722', linestyle='--', linewidth=1.5, alpha=0.85)
    ax1.axvline(2.1, color='#FF5722', linestyle='--', linewidth=1.5, alpha=0.85)
    ax1.axvline(2.4, color='#00A86B', linestyle='--', linewidth=1.5, alpha=0.85)
    ax1.axvline(3.7, color='#00A86B', linestyle='--', linewidth=1.5, alpha=0.85)
    
    ax1.text(1.4, 6800, 'Segment 1 (1.4s)', color='white', fontsize=8, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#FF5722', alpha=0.85, edgecolor='none'))
    ax1.text(3.05, 6800, 'Segment 2 (1.3s)', color='white', fontsize=8, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#00A86B', alpha=0.85, edgecolor='none'))
    
    # 2. Alignment & LLM Extraction Bar / Tokens
    ax2.set_facecolor('#F8FBFD')
    ax2.set_xlim(0, 4.0)
    ax2.set_ylim(0, 2.8)
    ax2.set_xlabel('Zaman (saniye)', fontsize=9, fontweight='bold', color='#1A2B4C')
    ax2.set_yticks([])
    ax2.tick_params(colors='#333333', labelsize=8)
    
    # Box 1: Whisper ASR tokens
    ax2.text(0.1, 2.1, 'ASR Çıktısı:', fontsize=8.5, fontweight='bold', color='#1A2B4C', va='center')
    ax2.text(1.4, 2.1, '"...Kirchhoff akım kanununa göre..."', fontsize=8.5, color='#0D233A', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBE6', edgecolor='#FF8A65', linewidth=1))
    ax2.text(3.05, 2.1, '"...düğüm gerilimleri toplamı..."', fontsize=8.5, color='#0D233A', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#81C784', linewidth=1))
    
    # Box 2: LLM Canonical Term & DAG Node
    ax2.text(0.1, 0.9, 'LLM Terim & DAG:', fontsize=8.5, fontweight='bold', color='#1A2B4C', va='center')
    ax2.text(1.4, 0.9, 'Terim: "Kirchhoff Yasası"\nKanonik ID: KCL-01 | Skor: 0.98', fontsize=7.5, color='#0D233A', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1F5FE', edgecolor='#4FC3F7', linewidth=1.2))
    ax2.text(3.05, 0.9, 'Terim: "Düğüm Analizi"\nKanonik ID: NDM-04 | Skor: 0.95', fontsize=7.5, color='#0D233A', ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#EDE7F6', edgecolor='#B39DDB', linewidth=1.2))
    
    plt.tight_layout()
    plt.savefig('img/acoustic_spectrogram_extraction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated img/acoustic_spectrogram_extraction.png successfully.")

if __name__ == '__main__':
    generate_spectrogram_sample()
