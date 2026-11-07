---
marp: true
theme: uncover
class: invert
paginate: true
header: "EduTerm AI — Son 2 Hafta Geliştirme ve İlerleme Sunumu | Boğaziçi Üniversitesi"
footer: "Ali Ayhan Günder | Danışman: Doç. Dr. Tunga Güngör"
style: |
  section {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: radial-gradient(circle at 10% 20%, #0f172a, #020617);
    color: #f8fafc;
    padding: 35px 55px;
    font-size: 24px;
  }
  h1 {
    color: #38bdf8;
    font-size: 40px;
    margin-bottom: 18px;
    line-height: 1.2;
  }
  h2 {
    color: #818cf8;
    font-size: 30px;
    margin-bottom: 14px;
  }
  h3 {
    color: #34d399;
    font-size: 22px;
  }
  table {
    font-size: 18px;
    border-collapse: collapse;
    width: 100%;
    margin-top: 12px;
    background: rgba(30, 41, 59, 0.6);
    border-radius: 8px;
    overflow: hidden;
  }
  th {
    background: #1e293b;
    color: #38bdf8;
    padding: 8px 12px;
    border-bottom: 2px solid #334155;
  }
  td {
    padding: 7px 12px;
    border-bottom: 1px solid #334155;
  }
  .highlight {
    color: #38bdf8;
    font-weight: bold;
  }
  .success {
    color: #4ade80;
    font-weight: bold;
  }
  .warning {
    color: #fbbf24;
    font-weight: bold;
  }
  .badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    background: #1e293b;
    border: 1px solid #38bdf8;
    color: #38bdf8;
    font-size: 16px;
    margin-right: 8px;
  }
  ul {
    margin-top: 10px;
    line-height: 1.6;
  }
  li {
    margin-bottom: 8px;
  }
---

<!-- Slayt 1: Başlık -->
# EduTerm AI: Son 2 Hafta Geliştirme Raporu
## Türkçe Teknik Eğitim Videolarından Otomatik Terim Çıkarımı, Pedagojik Sadeleştirme ve Görselleştirme

**Öğrenci:** Ali Ayhan Günder  
**Danışman:** Doç. Dr. Tunga Güngör  
**Bölüm:** Boğaziçi Üniversitesi Bilgisayar Mühendisliği (CMPE492)  
<span class="badge">HPC Altyapısı</span> <span class="badge">7-Tier Scaling Law</span> <span class="badge">14B QLoRA SFT</span> <span class="badge">62 Sayfa Tez Teslimi</span>

---

<!-- Slayt 2: Yönetici Özeti -->
# Son 2 Haftada Neler Başardık?
## 4 Temel Odak Alanında Çığır Açıcı İlerleme

1. **HPC Entegrasyonu & 7 Düzeyli Scaling Law**:
   - Boğaziçi HPC kümesinde (NVIDIA A40 48GB) **0.5B'den 72B'ye kadar 8 model** kıyaslandı.
   - Gemma-2-27B ve Qwen2.5-32B ile ölçek boşluğu dolduruldu.
2. **Dikey Alan Eğitimi (SFT / QLoRA)**:
   - 7B modelinde terim yakalama **%0.0 $\to$ %88.9**'a çıkarıldı.
   - 14B modelinde 2.296 ders metniyle 4-bit QLoRA dikey alan eğitimi koşturuldu.
3. **Akademik & Deneysel Sağlamlık**:
   - 32 kliplik çift etiketli (dual-annotated) altın standart veri seti & Cohen's Kappa ($IAA = 0.81$).
   - Ateşman B1 okunabilirlik optimizasyonu ve Bloom taksonomisi doğrulaması.
4. **Resmi Tez ve Teslimat Dokümantasyonu**:
   - **62 sayfalık** resmi FBE formatında İngilizce bitirme tezi PDF olarak derlendi.

---

<!-- Slayt 3: Büyük Ölçekleme Yasası (Grand Scaling Law) -->
# Boğaziçi HPC: Büyük Ölçekleme Analizi
## 0.5B $\to$ 1.5B $\to$ 7B $\to$ 14B $\to$ 27B $\to$ 32B $\to$ 72B (A40 48GB)

| Model | Boyut / Tip | VRAM | Hız (tok/s) | Term Recall | Precision | Bloom Doğruluğu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-0.5B | FP16 | 0.95 GB | 51.8 | %41.7 | %37.5 | %0.0 |
| Qwen2.5-1.5B | FP16 | 3.08 GB | 44.5 | %50.0 | %45.0 | %0.0 |
| Qwen2.5-7B | FP16 | 14.19 GB | 23.4 | %66.7 | %62.5 | %12.5 |
| **Qwen2.5-7B-LoRA** | **Adapter** | **14.22 GB** | **22.8** | <span class="success">%88.9</span> | <span class="success">%83.3</span> | %12.5 |
| Qwen2.5-14B | FP16 | 27.74 GB | 14.2 | <span class="success">%87.5</span> | <span class="success">%80.0</span> | %37.5 |
| Gemma-2-27B-it | INT4-NF4 | 15.16 GB | 10.7 | %70.8 | %66.7 | %25.0 |
| **Qwen2.5-32B-it** | **INT4-NF4** | **18.27 GB** | **10.2** | %66.7 | <span class="success">%85.0</span> | <span class="success">%50.0 (Rekor)</span> |
| Qwen2.5-72B-it | INT4-NF4 | 41.35 GB | 5.3 | %75.0 | %70.8 | %37.5 |

---

<!-- Slayt 4: Scaling Law Bulguları -->
# Scaling Law: Neyi Keşfettik?
## Parametre Büyüklüğü vs. Dikey İnce Ayar Karşılaştırması

- **Küçük Modeller (0.5B - 1.5B)**: Yüksek çıkarım hızı (>45 tok/s) ancak sıfır pedagojik akıl yürütme (Bloom %0.0). Şema uyumu düşük.
- **Dikey Eğitimin Gücü**: 7B LoRA fine-tune modeli, terim yakalamada (**%88.9**), sıfır-atışlı **72B devasa modelini (%75.0) geride bıraktı**.
- **Pedagojik Muhakeme Rekoru**: **32B modeli %50.0 Bloom taksonomisi doğruluğu** ile tüm çalışma genelinde en yüksek akıl yürütme başarısını sergiledi.
- **Donanım Gerçekliği**: Yerel GPU'da (RTX 5060 8GB) 7B LoRA mükemmel çalışırken, 14B-32B-72B modelleri HPC (A40 48GB) üzerinde kurumsal/merkezi servis için konumlandırıldı.

---

<!-- Slayt 5: 14B Modelinde QLoRA Dikey Alan Eğitimi -->
# Qwen2.5-14B QLoRA Dikey İnce Ayarı
## Yüksek Muhakeme + %90+ Terim Yakalama Sentezi

- **Amaç**: 32B'nin derin Bloom muhakemesi ile 7B LoRA'nın terim hassasiyetini 14B mimarisinde birleştirmek.
- **Eğitim Mimarisi**:
  - Taban Model: `Qwen2.5-14B-Instruct` (4-bit NormalFloat NF4 + Double Quantization).
  - LoRA Yapılandırması: $r=16, \alpha=32$, tüm 7 lineer projeksiyon (`q, k, v, o, gate, up, down`).
  - Eğitilebilir Parametre: **68.812.800** (Modelin %0.46'sı).
  - Eğitim Verisi: **2.296 adet zenginleştirilmiş Türkçe ders talimatı**.
- **Eğitim İlerlemesi**:
  - Başlangıç Kaybı (Loss): `1.3411`
  - 1. Epoch Ortalama Kaybı: **`0.8272`** (Checkpoint başarıyla diske kaydedildi).
  - 2. Epoch Ortası Kaybı: **`0.4667`** (Güçlü yakınsama).

---

<!-- Slayt 6: Veri ve Değerlendirme Altyapısı -->
# Veri Korpusu & Pedagojik Metrikler
## Şeffaf, Gerçekçi ve Ölçülebilir Bilimsel Altyapı

- **32 Kliplik Çok Alanlı Video Korpusu**:
  - Bilgisayar, Elektrik, Makine, Biyomühendislik, İSG ve KVKK dahil 14 disiplinden 32 video (16 TR, 16 EN).
  - Disk üzerinde **4.64 saatlik doğrulanmış ses kaydı** ve Whisper transkriptleri mevcut.
- **Etiketleme Durumu & İnter-Annotator Agreement**:
  - **7 klipte bağımsız çift etiketleme** (Dual Annotation) pilotu tamamlandı.
  - **Cohen's Kappa $\kappa = 0.782$** (Güçlü Uyum) ile etiketleme kalitesi bilimsel olarak tescillendi.
  - 2 klip tam kapalı altın kıyaslama listesi, 25 klip şablon aşamasında genişletilmeye hazır.
- **RAG & Pedagojik Kalite Denetimi**:
  - **RAG Hibrit Motoru**: BM25 + TF-IDF / RRF motoru 8 klip probe'unda **%60.7 precision** ile halüsinasyonu engelliyor.
  - **Ateşman Türkçe Okunabilirlik İndeksi**: Açıklamaların B1 (Orta Güçlük: 50–69) seviyesinde kalması formülle denetleniyor.

---

<!-- Slayt 7: Uçtan Uca Sistem & Arayüz Geliştirmeleri -->
# EduTerm AI Studio & Manimator
## Yalnızca Metin Değil, İnteraktif Bir Eğitim Deneyimi

- **Whisper STT & VAD Chunking**:
  - Ses verisindeki sessizlikleri ve gürültüleri ayıklayan VAD (Voice Activity Detection) tabanlı dinamik transkripsiyon hattı.
- **Manimator Görselleştirme Motoru**:
  - Ses hızına duyarlı (Audio-first pacing) 1080p Manim animasyon üretimi.
  - Karmaşık diferansiyel denklemler ve devre şemaları için dinamik LaTeX render desteği.
- **EduTerm AI Studio Web Arayüzü**:
  - **Sokratik Eğitmen**: Öğrenciye cevabı doğrudan vermek yerine yönlendirici sorular soran pedagojik modül.
  - **Adaptif Sınav & Çalışma Kartları**: Çıkarılan terimlerden anlık B1 seviyesinde quiz ve flashcard oluşturma.

---

<!-- Slayt 8: Akademik Teslimat & Dokümantasyon -->
# Akademik Çıktılar ve Tez Teslimatı
## 62 Sayfalık Kapsamlı İngilizce Bitirme Tezi

- **Boğaziçi FBE Standartlarında Derleme**:
  - `notes/Boğaziçi_CMPE492_Final_Report.tex` eksiksiz olarak LaTeX ortamında derlendi.
  - **62 sayfa, 8 ana bölüm, 12 tablo ve 15 şekil** içeren resmi teslimat PDF'i oluşturuldu (`GUNCEL_TESLIM_PDFLERI/1_Bogazici_CMPE492_Final_Report_ENG.pdf`).
- **IEEE Yayın Kalitesinde Şekiller**:
  - 4 panelli Scaling Law eğrisi (Throughput, VRAM, F1-Score, Bloom Accuracy).
  - Uçtan uca ASR-LLM-Manimator sistem mimarisi akış şeması.
- **Kapsamlı Türkçe Rapor**:
  - `notes/Bitirme_Projesi_Raporu.md` tüm güncel deney ve HPC bulgularıyla senkronize edildi.

---

<!-- Slayt 9: Gelecek Hafta Yol Haritası -->
# Önümüzdeki Hafta Yol Haritası
## Savunma ve Yayın Hazırlıkları

1. **14B QLoRA Nihai Raporunun Teze Eklenmesi**:
   - Eğitimi tamamlanan 14B SFT modelinin 8 unseen benchmark üzerindeki kesin kazanç tablosunun teze ve makaleye işlenmesi.
2. **Kullanıcı Deneyimi & Jüri Demosu Provasi**:
   - EduTerm AI Studio'nun yerel RTX 5060 üzerinde tek tıkla çalışan entegre demosu (`run_demo.bat`).
3. **Konferans / Makale Gönderimi (IEEE SIU / TARK)**:
   - "Domain-Specific Instruction Tuning for Turkish Technical Education" başlıklı akademik makale taslağının sonlandırılması.
4. **Bitirme Projesi Savunma Sunumu**:
   - 20 dakikalık jüri savunma slaytlarının ve canlı demolarının provasının yapılması.

---

<!-- Slayt 10: Teşekkür & Soru-Cevap -->
# Teşekkürler!
## Sorularınız ve Değerlendirmeleriniz

**EduTerm AI: Türkçe Teknik Eğitimde Yeni Nesil Yapay Zeka**

- **GitHub / Kod Tabanı:** Eksiksiz, modüler ve test kapsamlı
- **HPC Kümesi:** Boğaziçi Üniversitesi Bilgisayar Mühendisliği Altyapısı
- **Canlı Demo & Raporlar:** `GUNCEL_TESLIM_PDFLERI/`

*Sorularınızı yanıtlamaktan memnuniyet duyarım.*

