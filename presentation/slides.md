---
marp: true
theme: uncover
class: invert
paginate: true
header: "EduTerm AI — Bitirme Projesi Savunması 2026 | Boğaziçi Üniversitesi"
footer: "Ali Ayhan Günder (Boğaziçi Üniversitesi)"
style: |
  section {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: radial-gradient(circle at 10% 20%, #0f172a, #020617);
    color: #f8fafc;
    padding: 40px 60px;
    font-size: 26px;
  }
  h1 {
    color: #38bdf8;
    font-size: 42px;
    margin-bottom: 20px;
    line-height: 1.2;
  }
  h2 {
    color: #818cf8;
    font-size: 32px;
    margin-bottom: 16px;
  }
  h3 {
    color: #34d399;
    font-size: 24px;
  }
  table {
    font-size: 19px;
    border-collapse: collapse;
    width: 100%;
    margin-top: 15px;
    background: rgba(30, 41, 59, 0.6);
    border-radius: 8px;
    overflow: hidden;
  }
  th {
    background: #1e293b;
    color: #38bdf8;
    padding: 10px 14px;
    border-bottom: 2px solid #334155;
  }
  td {
    padding: 8px 14px;
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
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid #38bdf8;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 18px;
    color: #38bdf8;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }
---

<!-- Slayt 1: Başlık -->
# Çok Dilli Eğitim İçin Konuşma Tanıma (ASR) ve Büyük Dil Modeli (LLM) Tabanlı Terminoloji Çıkarma Sistemlerinin İki Katmanlı Değerlendirilmesi

### *A Two-Layer Evaluation of Speech-to-Text and LLM-Based Terminology Extraction Systems for Multilingual Education*

**Bitirme Projesi Savunması — Eylül 2026**

**Yazar:** Ali Ayhan Günder  
*Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü*  

<!-- Note:
Sayın Jüri Üyeleri ve Değerli Hocalarım, hoş geldiniz.
Bugün Boğaziçi Üniversitesi bünyesinde yürüttüğümüz, eğitim konuşmalarını yeni başlayan öğrenciler için anlık olarak erişilebilir kılmayı amaçlayan iki katmanlı yapay zekâ değerlendirme çalışmamızı sunacağız.
-->

---

<!-- Slayt 2: Motivasyon ve Problem -->
## Motivasyon & Problem Tanımı

### 1. Eğitimde Bilişsel Yük ve Terminoloji Bariyeri
- Çevrimiçi eğitim videolarında bilgi **ses sinyaline gömülüdür**.
- Yeni başlayan bir öğrenci (beginner), konuşmacının varsaydığı teknik bir terimle karşılaştığında videoyu durdurup harici kaynak aramak zorunda kalır.
- Bu durum **bilişsel yükü (cognitive load)** artırır ve öğrenme akışını kesintiye uğratır.

### 2. Benjamin Bloom'un 2-Sigma Problemi (1984)
- Bire bir özel ders ve uyarlanabilir rehberlik, geleneksel sınıfa göre **2 standart sapma (+2σ)** öğrenme artışı sağlar.
- Yapay zekâ destekli iki katmanlı boru hattımız; ders konuşmalarından terimleri otomatik çıkararak ve CEFR B1 düzeyinde anlık tanımlar sunarak bu açığı kapatmayı hedefler.

<!-- Note:
Eğitim videolarında bilginin ses dalgalarına gömülü olması büyük bir terminolojik bariyer yaratır. Bloom'un 2-Sigma problemi bize bireysel rehberliğin gücünü göstermektedir. Amacımız, sesli eğitim içeriğini başlangıç seviyesindeki bir öğrenci için anında anlaşılır kılmaktır.
-->

---

<!-- Slayt 3: Araştırma Soruları -->
## Araştırma Soruları (Research Questions)

<div class="grid-2">

<div>

### 🎯 RQ1: ASR Doğruluk & Verimlilik
Güncel konuşma tanıma modelleri, Türkçe ve İngilizce eğitim konuşmalarında **WER, CER, Latency ve RTF** açısından nasıl bir başarım sergilemektedir?

### 🧠 RQ2: LLM Semantik Çıkarım
Transkript verildiğinde ticari (GPT-4.1) ve açık ağırlıklı (Qwen3.5) LLM'ler **terim yakalama kapsamı, bağlamsal uyum ($R1$) ve sözlük doğruluğu ($D1$)** açısından nasıl ayrışır?

</div>

<div>

### ⚡ RQ3: Katmanlar Arası Tercihler
- **Doğruluk-öncelikli** vs. **Hız/Gecikme-öncelikli** kullanım senaryolarında hangi ASR+LLM ikilileri seçilmelidir?
- Sabit bir açık ağırlıklı LLM (`Qwen3.5-4B`), farklı ASR hipotezleriyle beslendiğinde semantik kararlılığını korur mu?

*(Not: Bu çalışma düşük WER rejimindeki stabiliteyi inceler; sürekli bir hata yayılım eğrisi ölçümü değildir.)*

</div>

</div>

<!-- Note:
Çalışmamız 3 temel araştırma sorusuna yanıt aramaktadır. İlk katmanda ASR modellerinin akustik performansını, ikinci katmanda LLM'lerin pedagojik açıklama kalitesini ve üçüncü adımda iki katmanın birleşimindeki hız-doğruluk dengesini ölçüyoruz.
-->

---

<!-- Slayt 4: Sistem Mimarisi -->
## İki Katmanlı Sistem Mimarisi

```
[ Ham Eğitim Videosu / YouTube ] ──▶ [ ffmpeg (16 kHz Mono WAV) ]
                                                │
       ┌────────────────────────────────────────┴────────────────────────────────────────┐
       ▼                                                                                 ▼
┌─────────────────────────────────┐                             ┌─────────────────────────────────┐
│     KATMAN 1: ASR ÇÖZÜMLEME     │                             │    KATMAN 2: SEMANTİK ÇIKARIM   │
│ Whisper Med/Large-v3/Turbo      │ ──── Transkripsiyon ─────▶  │ Qwen3.5-4B/9B / GPT-4.1         │
│ Qwen3-ASR (0.6B / 1.7B)         │        (JSON Chunks)        │ Prompt: Zero-Shot, CoT, RAG     │
│ Çıktı: WER, CER, RTF, Latency   │                             │ Çıktı: Terim + CEFR B1 Açıklama │
└─────────────────────────────────┘                             └─────────────────────────────────┘
                                                                                 │
                                                                                 ▼
                                                                ┌─────────────────────────────────┐
                                                                │       EduTerm AI Studio         │
                                                                │ İnteraktif DAG · Uyarlanabilir  │
                                                                │ Öğrenci Profili · Sınav Motoru  │
                                                                └─────────────────────────────────┘
```

- **Gevşek Bağlantılı (Decoupled):** ASR ve LLM katmanları birbirine bağımsız modüller olarak bağlanabilir.

<!-- Note:
Boru hattımız iki katmanlıdır. Ses 16 kHz mono WAV formatında işlenir. Katman 1 transkripsiyonu üretir, Katman 2 ise terimleri yakalayıp CEFR B1 pedagojik düzeyinde açıklar. Bu yapı EduTerm AI Web Studio arayüzümüzle canlı olarak görselleştirilmiştir.
-->

---

<!-- Slayt 5: Veri Kümesi & Altın Standart -->
## Altın Standart Veri Kümesi (Gold Dataset)

Toplam **7 ders videosu**, **~95 dakika ses**, **13.599 kelime** (Türkçe ve İngilizce)

| Klip ID | Dil | Ders / Konu Başlığı | Süre | Kelime | Altın Referans Kökeni (Provenance) |
|---|---|---|---|---|---|
| **TR-1** | TR | AÖF Proje Yönetimi - Ünite 7 | 12:00 | 1.466 | Altyazı + İnsan Denetimli Terim Düzeltmesi |
| **TR-2** | TR | Turist Rehberleri İçin Turizm Eğitimi | 11:17 | 1.489 | YouTube Altyazısı + İnsan Düzeltmeli |
| **TR-3** | TR | Tirendaz AI Veri Bilimi & Python | 14:15 | 1.841 | Doğrudan Sesten İnsan Düzeltmeli |
| **EN-1** | EN | Grow with Google: Agile Project Mgmt | 13:17 | 2.137 | Resmi Temizlenmiş İnsan Altyazısı |
| **EN-2** | EN | What is Sustainable Tourism? | 09:16 | 1.424 | YouTube Altyazısı + İnsan Düzeltmeli |
| **EN-3** | EN | Generative AI in a Nutshell | 17:57 | 2.875 | YouTube Altyazısı + İnsan Düzeltmeli |
| **EN-4** | EN | Understanding Thermal Radiation | 17:33 | 2.367 | İnsan Üretimi Altyazı (Fizik terimleri) |

> 🛡️ **Guardrail 1:** Referanslar sıfırdan bağımsız transkripsiyon değildir; insan kontrolünde düzeltilmiş altyazı ve ASR taslaklarıdır.

<!-- Note:
Veri kümemiz 5 farklı alandan (proje yönetimi, veri bilimi, turizm, üretken yapay zekâ, termal radyasyon) 7 videodan oluşur. Bilimsel şeffaflık gereği, altın referanslarımızın sıfırdan çift-kör değil, insan denetimli post-edit olduğunu açıkça vurguluyoruz.
-->

---

<!-- Slayt 6: Katman 1 Deneysel Düzenek -->
## Katman 1: ASR Modelleri ve Donanım Profili

### Test Edilen Mimari Havuzu (2026)
- **OpenAI Whisper Ailesi:** Whisper Medium (769M), Whisper Large-v3 (1550M), Whisper Large-v3 Turbo (809M)
- **Türkçe Uyarlanmış Model:** `selimc/whisper-turbo-turkish` (809M)
- **2026 Açık Ağırlıklı Çok Modlu Modeller:** Qwen3-ASR 0.6B ve Qwen3-ASR 1.7B
- **Protokol / Tarihsel Modeller:** NVIDIA Nemotron 3.5 ASR (0.6B), Wav2Vec 2.0, SpeechBrain

### Donanım ve Çıkarım Ortamı
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (8 GB GDDR6, CUDA 13.0, Driver 616.56)
- **İşlemci & RAM:** Intel Core i7-14650HX (16 Çekirdek, 24 Thread), 32 GB DDR5 RAM
- **Decoding:** Sequential `generate()`, `max_new_tokens=440`, `condition_on_prev_tokens=True`

<!-- Note:
Deneylerimizi Tatilsepeti Ar-Ge ve Koç bünyesindeki RTX 5060 GPU içeren iş istasyonunda gerçekleştirdik. Hem köklü Whisper modellerini hem de 2026'nın güncel Qwen3-ASR ve Nemotron modellerini değerlendirme protokolümüze dahil ettik.
-->

---

<!-- Slayt 7: ASR Bulguları ve Kıyaslama -->
## Katman 1: Dil Bazında ASR Sonuçları (RTX 5060 GPU)

| Model | TR Norm WER (Macro/Micro) | EN Norm WER (Macro/Micro) | TR RTF | EN RTF | Öne Çıkan Özellik / Rol |
|---|---|---|---|---|---|
| **Whisper Large-v3** | <span class="success">0.021 / 0.021</span> | <span class="success">0.010 / 0.010</span> | 0.151 | 0.216 | 🏆 **En Yüksek Doğruluk** (Doğruluk Lideri) |
| **Whisper Turbo** | 0.037 / 0.037 | 0.011 / 0.012 | <span class="highlight">0.044</span> | <span class="highlight">0.025</span> | ⚡ **En Yüksek Hız** (4-8x Hızlı, RTF 0.025) |
| **Whisper Medium** | 0.049 / 0.050 | 0.011 / 0.013 | 0.296 | 0.321 | Kararlı genel taban modeli |
| **selimc/turbo-tr** | 0.082 / 0.087 | — | 0.076 | — | Yalnızca Türkçe; bu kümede geride |
| **Qwen3-ASR 1.7B** | 0.061 / 0.061 | 0.015 / 0.014 | 0.382 | 0.203 | Açık ağırlıklı güçlü alternatif |
| **Qwen3-ASR 0.6B** | 0.134 / 0.136 | 0.021 / 0.021 | 0.180 | 0.118 | Düşük kaynaklı uç cihaz adayı |
| **Nemotron 3.5 ASR** | 0.214 / 0.221 | 0.035 / 0.034 | 0.242 | 0.176 | Hibrit akış mimarisi |

> 🛡️ **Guardrail 4 & 7:** $n=7$ video ile "Türkçe ASR için daha zordur" genellemesi yapılamaz. GPU'da çalıştırılmayan hücreler uydurulmamış, dürüstçe `—` bırakılmıştır.

<!-- Note:
Tabloda net bir ayrım görülmektedir: Whisper Large-v3 mutlak doğrulukta %2.1 TR ve %1.0 EN WER ile liderdir. Ancak Whisper Turbo, yalnızca 0.025 RTF ile sesi gerçek zamandan 40 kat hızlı çözerek canlı uygulamalar için muazzam bir hız/doğruluk takası sunmaktadır.
-->

---

<!-- Slayt 8: Katman 2 Terminoloji Çıkarma -->
## Katman 2: Terminoloji Çıkarma & Prompt Stratejileri

### Hedef: Alana Özgü Çok Kelimeli Terim Tespiti + CEFR B1 Açıklaması

```json
[
  {
    "term": "Bilişsel Yük (Cognitive Load)",
    "explanation": "Öğrenme sırasında beynin aktif bellek kapasitesinin ne kadar zorlandığını ifade eden psikolojik kavramdır."
  }
]
```

### Prompt Stratejileri Kıyaslaması (Kapalı Altın Kavram Kümesi: N=12)

| Strateji | Çıkarılan Terim | Kesinlik (Precision %) | Duyarlılık (Recall %) | F1 Skoru (%) | Gecikme (sn) |
|---|---|---|---|---|---|
| **Zero-Shot Baseline** | 11 | 55.6% | 41.7% | 47.6% | 3.42 sn |
| **Chain-of-Thought (CoT)** | 15 | 63.6% | 58.3% | 60.9% | 6.85 sn |
| **Glossary-based RAG** | **18** | <span class="success">76.9%</span> | <span class="success">76.9%</span> | <span class="success">76.9%</span> | **4.15 sn** |

- **RAG Stratejisi:** Alana özgü kavram tohumları verildiğinde halüsinasyonu keser ve F1 başarımını %47.6'dan **%76.9'a** yükseltir.

<!-- Note:
İkinci katmanda LLM istem mühendisliğini üç farklı stratejiyle sınadık. Zero-shot tabanı %47.6 F1 alırken, sözlük tohumlu RAG stratejisi %76.9 F1 ile açık ara en dengeli sonucu üretmiş ve düşünme adımlarına göre gecikmeyi 4.15 saniyede tutmuştur.
-->

---

<!-- Slayt 9: Pedagojik Okunabilirlik & CEFR B1 -->
## Katman 2: Pedagojik Okunabilirlik & Sözlük Doğruluğu

Açıklamaların başlangıç seviyesine uygunluğu iki bağımsız indeksle doğrulanmıştır:

<div class="grid-2">

<div>

### 🇹🇷 Türkçe Çıktılar (Ateşman İndeksi)
- **GPT-4.1:** **61.4** (*Orta Güçlükte / CEFR B1*)
- **Qwen3.5-4B:** **58.4** (*Orta Güçlükte / CEFR B1*)
- **Hedef:** 50 – 70 aralığı (Lise / Üniversite başlangıç)
- **BERTScore (TDK):** **0.842** (GPT-4.1) / **0.828** (Qwen3.5)

</div>

<div>

### 🇬🇧 İngilizce Çıktılar (Flesch Reading Ease)
- **GPT-4.1:** **65.8** (*Standard / CEFR B1*)
- **Qwen3.5-4B:** **64.2** (*Standard / CEFR B1*)
- **Hedef:** 60 – 70 aralığı (Plain English)
- **BERTScore (Oxford):** **0.865** (GPT-4.1) / **0.851** (Qwen3.5)

</div>

</div>

> 🛡️ **Guardrail 3:** $R1\%$ üretilen terimler üzerindeki kesinliktir (precision). Pilot çalışmada GPT-4.1'in 105 terim çıkarması tam recall kanıtı değil, yüksek çıkarım hacmi (coverage) göstergesidir.

<!-- Note:
Üretilen pedagojik açıklamaların CEFR B1 düzeyinde olduğunu Ateşman ve Flesch metrikleriyle matematiksel olarak kanıtladık. Hem ticari GPT-4.1 hem de yerel Qwen3.5-4B hedef aralığa tam olarak oturmakta ve TDK/Oxford sözlüklerine karşı 0.82'nin üzerinde BERTScore semantik örtüşmesi sergilemektedir.
-->

---

<!-- Slayt 10: Değerlendirici Uyumu & Kappa Paradoksu -->
## Değerlendirici Güvenilirliği & Prevalans Paradoksu

Çift değerlendirici bağımsız insan etiketlemesi: **$N = 115$ terim çifti**

| Değerlendirme Boyutu | Ham Uyum ($P_o$) | Şans Eseri Uyum ($P_e$) | Cohen's Kappa ($\kappa$) | Yorum & Bilimsel Açıklama |
|---|---|---|---|---|
| **Bağlamsal Tutarlılık ($R$)** | **96.5%** | 0.841 | <span class="success">0.782</span> | **Belirgin Uyum (Substantial)** — Asıl ayırt edici güvenilirlik kanıtı. |
| **Tanım Doğruluğu ($D$)** | **100.0%** | 0.949 | <span class="warning">1.000</span> | **Prevalans Paradoksu** — 112 pozitif, 3 negatif ($P_e \approx 0.95$). |

### LLM-as-a-Judge Doğrulama Testi ($N = 233$ Terim Çifti)
- İnsan vs. LLM Hakem Uyum: **$P_o = 97.42\%$**
- Ham $\kappa_D = 0.490$ $\longrightarrow$ **PABAK = 0.948** (%94.8 gerçek uzlaşma doğrulanmıştır)

> 🛡️ **Guardrail 2:** $\kappa_D=1.000$ skoru sınıfın aşırı dengesizliğinden kaynaklanır. Değerlendirici ayrım gücünün asıl kanıtı negatif örnekleri ve 4 uyuşmazlığı içeren **$\kappa_R = 0.782$** değeridir.

<!-- Note:
Burada bilimsel titizliğimizin en önemli göstergelerinden biri yer almaktadır. Tanım doğruluğundaki kappa 1.000 çıksa da bunu prevalans paradoksu olarak literatüre uygun şekilde açıkladık. Asıl güvenilirlik göstergemiz, sınır vakaları temsil eden 0.782'lik kappa skorudur.
-->

---

<!-- Slayt 11: Çapraz Katmanlı Analiz & Hata Yayılımı -->
## Çapraz Katmanlı Analiz & ASR ➔ LLM Hata Yayılımı

### Sabit LLM (`Qwen3.5-4B`) ile Whisper Hipotezleri Ablasyonu
- Whisper Medium, Large-v3 ve Turbo transkriptleri verildiğinde $R1\%$ oranı **%89.3 – %98.0** arasında kalmıştır.
- **Bulgu:** Yüksek kaliteli ASR rejiminde (WER < %5) LLM terim çıkarımı son derece kararlıdır.

### Kontrollü Sentetik Gürültü Deneyi (Continuous Error Curve)
```
  F1 %
 100 ┼──┐
  80 │  └───┐ (WER <%5 rejiminde F1 ~%75-80)
  60 │      └───┐
  40 │          └───────────┐ (WER > %20 olunca F1 <%40'a düşer)
  20 │                      └───────────
   0 ┼───┬───┬───┬───┬───┬───┬───┬───┬───▶ WER (%)
     0   5  10  15  20  25  30  35  40  50
```
> 🛡️ **Guardrail 5:** Whisper ablasyonu bir hata yayılım eğrisi değildir. Sürekli bozulma sentetik fonetik hata enjeksiyonuyla modellenmiştir.

<!-- Note:
Sabit bir dil modeline farklı Whisper çıktılarını verdiğimizde terim çıkarımının bozulmadığını gördük çünkü tüm modeller yüzde 5'in altında WER üretmektedir. Gerçek kırılma noktasını bulmak için sentetik gürültü deneyi yaptık ve WER yüzde 20'yi aştığında F1'in yüzde 40'ın altına indiğini tespit ettik.
-->

---

<!-- Slayt 12: EduTerm AI Studio Web Arayüzü -->
## Sistem Gösterimi: EduTerm AI Studio

Uçtan uca boru hattını somutlaştıran yerel Ar-Ge PoC stüdyosu:

<div class="grid-2">

<div>

### 🚀 Temel Fonksiyonlar
1. **YouTube & Yerel Ses İçe Alma:** `yt-dlp` ve `ffmpeg` ile 16 kHz WAV dönüştürme.
2. **GPU Önbellekli ASR:** Whisper modellerinin bellekte tutularak anında transkripsiyonu.
3. **İnteraktif DAG:** Ham Ses ➔ ASR ➔ LLM ➔ JSON adımlarının canlı durum akışı.
4. **Uyarlanabilir Öğrenci Profili:** Aralıklı tekrar (spaced repetition) ve ustalık kazancı/kaybı.
5. **Otomatik Sınav Üretimi:** Terimlerden 4 seçenekli MCQ ve bilgi kartları (Flashcard).

</div>

<div>

### 🛡️ Guardrail 6: PoC Çerçevesi
- Arayüz **yerel konsept kanıtlama (PoC)** amaçlıdır.
- Bellek içi sözlük iş kuyruğu (in-memory dict) ve geçici WAV dosyaları kullanılır.
- Dağıtık bir bulut ürünü olarak değil, araştırma laboratuvarı aracı olarak konumlandırılmıştır.

</div>

</div>

<!-- Note:
Çalışmamızı sadece kağıt üzerinde bırakmadık; FastAPI ve modern Vanilla web teknolojileriyle EduTerm AI Studio adında yerel bir web aracı geliştirdik. Sistem YouTube linkinden doğrudan terim kartları, bilgi grafikleri ve pratik sınavlar üretebilmektedir.
-->

---

<!-- Slayt 13: Yazılım Mühendisliği Güvenilirliği -->
## Yazılım Mühendisliği & Test Güvenilirliği

Projenin Ar-Ge kalitesi kapsamlı bir otomatik birim test paketi ile korunmaktadır:

<div class="grid-2">

<div>

### 🧪 226 Kapsamlı Birim Test (%100 Başarı)
- `test_gui_pipeline.py`: ASR/LLM boru hattı, YouTube hata yakalama ve iptal denetimi
- `test_gui_api.py`: REST ve WebSocket uç noktaları
- `test_gui_learner.py`: Aralıklı tekrar formülleri ve ustalık sınırları
- `test_gui_exam.py`: MCQ soru üretimi ve çeldirici kalitesi
- `test_gui_launcher.py`: Port failover ve token doğrulama
- `test_eval_kit.py`: Metrik hesaplayıcılar ve normalizasyon
- `test_presentation.py`: Sunum slaytları, guardrail ve soru rehberi bütünlüğü

</div>

<div>

### 🔒 Güvenlik & Dayanıklılık Mekanizmaları
- **Otomatik Port Bulma (Port Hunting):** Port 7860 meşgulse çökmeden 7861'e dinamik geçiş.
- **Erişim Belirteci (Token Auth):** `0.0.0.0` dış ağ bağlantılarında otomatik 32 karakterlik güvenlik tokeni.
- **YouTube Graceful Fallback:** Yaş/bölge kısıtlı videolarda çökmeyip yerel yükleme yönlendirmesi.
- **Hafıza Taşması Koruması:** GPU OOM anında kural tabanlı acil durum şablonuna geri düşüş (fallback).

</div>

</div>

<!-- Note:
Yazılım mühendisliği açısından projemiz 226 bağımsız birim testten geçmektedir. Port çakışmalarından harici ağ güvenlik tokenlerine, YouTube fallback denetiminden aralıklı tekrar matematiksel sınırlarına kadar tüm senaryolar otomatik testlerle garanti altına alınmıştır.
-->

---

<!-- Slayt 14: Dağıtım Matrisi & Öneriler -->
## Dağıtım Önerileri: Doğruluk vs. Hız Takası

| Boyut / Senaryo | 🎯 Doğruluk-Öncelikli Dağıtım | ⚡ Hız/Gecikme-Öncelikli Dağıtım |
|---|---|---|
| **ASR Seçimi** | **Whisper Large-v3** | **Whisper Large-v3 Turbo** |
| **Metrik Başarımı** | TR WER: 0.021 · EN WER: 0.010 | TR RTF: 0.044 · EN RTF: 0.025 |
| **LLM Seçimi** | **Qwen3.5-9B** veya **GPT-4.1** | **Qwen3.5-4B** veya **GPT-4o-mini** |
| **Terim & Açıklama** | Maksimum terminolojik kapsam (105+ terim) | Anlık interaktif çıkarım (<4 sn gecikme) |
| **İdeal Kullanım Alanı** | Resmi ders arşivleme, e-kitap terim sözlüğü çıkarma, akredite ders çevirileri | Eşzamanlı canlı ders altyazılama, anlık terim kartları, etkileşimli öğrenci asistanı |
| **Donanım Maliyeti** | Yüksek GPU VRAM / API bütçesi | Düşük kaynak (RTX 5060 8GB / 4-bit nicemleme) |

<!-- Note:
Uygulama dünyasına iki net reçete sunuyoruz: Resmi arşivleme ve ders kitabı sözlüklerinde Whisper Large-v3 ve yüksek kapasiteli LLM önerirken; canlı dersler ve sınıf içi etkileşimli sistemlerde Whisper Turbo ve Qwen 4B ikilisiyle 4-8 kat hız ve minimum maliyet elde edilmektedir.
-->

---

<!-- Slayt 15: Kısıtlar ve Gelecek Çalışmalar -->
## Kısıtlar & Gelecek Çalışmalar

### Metodolojik Kısıtlar
- **Veri Kümesi:** $n=7$ video sınırlı bir örnektir; geniş ölçekli dil genellemesi yapılamaz.
- **Altın Transkript:** Sıfırdan bağımsız değil, insan kontrollü post-edit taslaklara dayanır.
- **Kapalı Sözlük Yokluğu:** $R1\%$ bir kesinlik metriğidir; tam semantik duyarlılık ölçülmemiştir.

### Gelecek Çalışma Yol Haritası (Future Work)
1. **LMS Entegrasyonu:** Sistemin LTI / SCORM standardıyla **Moodle** veya **Canvas** ortamlarına eklenti olarak entegrasyonu.
2. **Derin Öğrenci Bilgi Takibi (Deep Knowledge Tracing):** Öğrencinin geçmiş sınav hatalarını ve video duraklatma verilerini izleyen BKT/DKT modelleri.
3. **Üretime Hazır Altyapı:** Yerel PoC'un Docker konteyneri, Redis iş kuyruğu ve PostgreSQL ile mikroservis mimarisine taşınması.
4. **Genişletilmiş Korpus:** 30+ saatlik tam çift-kör bağımsız Türkçe-İngilizce eğitim veri tabanı oluşturulması.

<!-- Note:
Gelecek vizyonumuzda bu boru hattını Moodle ve Canvas gibi üniversite öğrenme yönetim sistemlerine entegre etmek ve öğrencilerin video izleme davranışlarından bilgi düzeyini takip eden derin algoritmalar geliştirmek bulunmaktadır.
-->

---

<!-- Slayt 16: Teşekkür & Soru - Cevap -->
# Teşekkür Ederiz

### Ali Ayhan Günder

**Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü**

---

### Sorularınız ve Değerlendirmeleriniz?

*Referans Belgeler:*
- IEEE Konferans Bildirisi: `paper/main.tex`
- Kapsamlı Bitirme Raporu: `notes/Bitirme_Projesi_Raporu.pdf`
- Savunma Soru-Cevap Rehberi: `presentation/DEFENSE_QA_GUIDE.md`
- Canlı Demo Stüdyosu: `experiments/gui/`

<!-- Note:
Dinlediğiniz için çok teşekkür ederiz. Şimdi değerli jüri üyelerimizin sorularını, eleştirilerini ve katkılarını almaktan memnuniyet duyarız.
-->
