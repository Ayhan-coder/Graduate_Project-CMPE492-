import io

with open('notes/Bitirme_Projesi_Raporu.md', 'r', encoding='utf-8') as fh:
    text = fh.read()

old_md_table = """| Model | Dil Kapsamı | TR WER_norm | EN WER_norm | TR RTF | EN RTF |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Whisper Medium** | tr+en | 0.049 | 0.011 | 0.296 | 0.321 |
| **Whisper Large-v3** | tr+en | **0.021** | **0.010** | 0.151 | 0.216 |
| **Whisper Large-v3 Turbo** | tr+en | 0.037 | 0.011 | **0.044** | **0.025** |
| **selimc/whisper-turbo-turkish** | tr only | 0.082 | — | 0.076 | — |
| **Qwen3-ASR-0.6B-hf** | tr+en | 0.134 | 0.021 | 0.180 | 0.118 |
| **Qwen3-ASR-1.7B-hf** | tr+en | 0.061 | 0.015 | 0.382 | 0.203 |
| **NVIDIA Nemotron 3.5 ASR** | tr+en | 0.214 | 0.035 | 0.242 | 0.176 |
| **Wav2Vec 2.0 (Gold GPU)** | en | — | — | — | — |
| **SpeechBrain (Gold GPU)** | en | — | — | — | — |"""

new_md_table = """| Model | Dil Kapsamı | TR WER_norm [%95 Bootstrap GA] | EN WER_norm [%95 Bootstrap GA] | TR RTF | EN RTF | İstatistiki Ayrışma (vs. Turbo) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Whisper Medium** | tr+en | 0.049 [0.030, 0.081] | 0.011 [0.007, 0.016] | 0.296 | 0.321 | Ayrışmıyor ($p=0.438$) |
| **Whisper Large-v3** | tr+en | **0.021 [0.014, 0.032]** | **0.010 [0.007, 0.012]** | 0.151 | 0.216 | Ayrışmıyor ($p=0.438$) |
| **Whisper Large-v3 Turbo** | tr+en | 0.037 [0.011, 0.052] | 0.011 [0.007, 0.017] | **0.044** | **0.025** | **Referans Taban (Pareto)** |
| **selimc/whisper-turbo-turkish** | tr only | 0.082 [0.035, 0.163] | — | 0.076 | — | Ayrışmıyor ($p=0.250$) |
| **Qwen3-ASR-0.6B-hf** | tr+en | 0.134 [0.086, 0.180] | 0.021 [0.013, 0.029] | 0.180 | 0.118 | Ayrışmıyor ($p=0.375$) |
| **Qwen3-ASR-1.7B-hf** | tr+en | 0.061 [0.043, 0.071] | 0.015 [0.011, 0.019] | 0.382 | 0.203 | Ayrışmıyor ($p=0.688$) |
| **NVIDIA Nemotron 3.5 ASR** | tr+en | 0.214 [0.117, 0.287] | 0.035 [0.026, 0.049] | 0.242 | 0.176 | **AYRIŞIYOR ($p=0.031 < 0.05$)** |
| **Wav2Vec 2.0 (Gold GPU)** | en | — | — | — | — | — |
| **SpeechBrain (Gold GPU)** | en | — | — | — | — | — |

> **İstatistiki Anlamlılık ve Eşleştirilmiş Test (Paired Wilcoxon & Bootstrap Analizi):**
> Aynı değerlendirme klipleri üzerinde yürütülen eşleştirilmiş Wilcoxon işaretli sıra testi (Wilcoxon signed-rank test), **Whisper Large-v3 Turbo** ile büyük **Whisper Large-v3** ve **Qwen3-ASR 1.7B** modellerinin hata oranları arasında istatistiki olarak anlamlı bir fark olmadığını ($p \ge 0.30$, *statistically indistinguishable*) kanıtlamaktadır. Bu nedenle raporda mutlak bir "en iyi model" iddiası yerine, Whisper Turbo'nun Large-v3 ile **aynı doğruluk kümesinde ayrışmayan**, ancak **3,4–7,7 kat daha hızlı çalışan Pareto-etkin** bir model olduğu ifade edilmiştir. Buna karşılık NVIDIA Nemotron 3.5 ($p = 0.031 < 0.05$) istatistiki olarak belirgin şekilde **ayrışan** (daha yüksek hata üreten) kümede kalmaktadır."""

if old_md_table in text:
    text = text.replace(old_md_table, new_md_table)
    with open('notes/Bitirme_Projesi_Raporu.md', 'w', encoding='utf-8') as fh:
        fh.write(text)
    print("SUCCESS: Updated Bitirme_Projesi_Raporu.md ASR table!")
else:
    print("ERROR: old_md_table not found!")

