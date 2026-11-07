import io

with open('notes/Bitirme_Projesi_Raporu.md', 'r', encoding='utf-8') as fh:
    text = fh.read()

scope_matrix_md = """### 8.0 Araştırma İddiaları, Veri Kümeleri ve Değerlendirici Kapsam Matrisi (Scope Matrix)

Proje kapsamında savunulan her bir bilimsel hipotez ve ampirik iddianın hangi veri kümesi, örneklem boyutu ($n$), zemin gerçek yöntemi (`gold_how`) ve değerlendirici mekanizması ile test edildiği aşağıdaki **Tek Sayfalık Kapsam Matrisi**'nde açıkça haritalanmıştır. Raporun takip eden bölümleri (§8.1–§8.15) doğrudan bu matristeki protokollere dayanmaktadır:

| Temel İddia / Hipotez | Test Edilen Alt Küme | Örneklem ($n$) | Zemin Gerçek Yöntemi (`gold_how`) | Değerlendirici Mekanizması | Rapor Bölümü |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **1. Akustik Doğruluk & Hız** | Altın Pilot Klip Havuzu | $n = 7$ video (3 TR, 4 EN) | Tam Çift İnsan Transkripsiyonu (`human_gold_transcription`) | WER, CER, RTF, Bootstrap GA, Paired Wilcoxon | §6.1 |
| **2. Çapraz Katman Hata Yayılımı** | Akustik Bozulma Merdiveni | 7 ASR seviyesi $\times$ 2 dil ($n = 14$) | Altın Terim Listesi (`gold_terms.json`) | $F_1$, Kesinlik, Duyarlılık, Duyarlılık Eğimi ($\Delta F_1$) | §8.6 |
| **3. Terim Çıkarımı Doğrulaması** | İnsan Etiketli Terim Havuzu | $n = 454$ terim-açıklama | Çift-Kör İnsan Etiketleme (`double_blind_annotation`) | Cohen's Kappa ($\kappa=0.531$), PABAK ($0.669$) | §8.9.2 |
| **4. Hakem Bağımsızlığı (Cross-Family)** | Çapraz-Menşe Hakem Paneli | $n = 454$ terim çifti | İnsan Zemin Gerçeği + Negatif Denetimi | Qwen2.5-7B vs. Mistral-7B-Instruct (A40 GPU) | §8.9.3 |
| **5. Model Parametre Ölçeklenmesi** | Grand Scaling Benchmark | 7 Model Ailesi ($0.5\text{B} \to 72\text{B}$) | Standart Altın Şablon (`gold_schema`) | JSON Şema Uyumu, Ateşman B1 Oranı, VRAM Tavanı | §8.12 |
| **6. Dikey Adaptasyon (SFT/QLoRA)** | Çok Alanlı Eğitim Seti | $n = 2.296$ (7B) / $4.666$ (14B) | LeakageGuard Korumalı Sentetik ChatML | Doğrulama Kaybı, Bloom Bilişsel Seviye Artışı | §8.3, §8.13 |
| **7. Çok Düzeyli CEFR Kontrolü** | Multi-Tier Kontrollü Set | $n = 4.666$ eğit., $586$ val., $24$ test | Seviye Koşullu B1/B2/C1 Şablonu | Otomatik Ateşman Metriği (%91.7 Seviye Uyumu) | §8.14 |
| **8. Çok Turlu Sokratik Eğitmenlik**| Sentetik Öğrenci Personaları | 50 Persona $\times$ 5 Tur ($n = 250$ tur) | 5 Alan Kavram Yanılgısı Senaryoları | Sokratik Soru ($SQ$), İskeleleme ($HQ$), Sıfır Spoon-Feed ($SF$) | §8.15 |
| **9. Müfredat DAG & Önkoşul Analizi**| Genişletilmiş Korpus | $n = 33 \to 133$ tam transkript | Makine Transkripti (`gold_how=machine_only`)| NetworkX Yönlü Asiklik Çizge ve Eksik Önkoşul Motoru | §8.1, §8.10 |

---

"""

target = "### 8.1 Veri Korpusunun 7'den 40 Videoya Genişletilmesi (Kurumsal Uyum, KVKK ve İSG)"
if target in text:
    text = text.replace(target, scope_matrix_md + target)
    with open('notes/Bitirme_Projesi_Raporu.md', 'w', encoding='utf-8') as fh:
        fh.write(text)
    print("SUCCESS: Added Scope Matrix to Bitirme_Projesi_Raporu.md!")
else:
    print("ERROR: Target not found in Bitirme_Projesi_Raporu.md")

