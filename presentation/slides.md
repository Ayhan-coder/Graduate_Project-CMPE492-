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
# Çok Dilli Eğitim İçin Konuşma Tanıma (ASR), Büyük Dil Modeli (LLM) Tabanlı Terminoloji Çıkarma ve İsteğe Bağlı Animasyon Üretimi Sistemlerinin Üç Katmanlı Değerlendirilmesi

### *A Three-Layer Evaluation of ASR, LLM-Based Terminology Extraction, and On-Demand Animation Delivery for Multilingual Education*

**Bitirme Projesi Savunması — Ekim 2026**

**Yazar:** Ali Ayhan Günder  
**Danışman:** Prof. Dr. Fatih Alagöz  
*Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü*  

<!-- Note:
Sayın Jüri Üyeleri ve Değerli Hocalarım, hoş geldiniz.
Bugün Boğaziçi Üniversitesi bünyesinde yürüttüğümüz, eğitim konuşmalarını yeni başlayan öğrenciler için anlık olarak erişilebilir kılmayı amaçlayan üç katmanlı yapay zekâ değerlendirme çalışmamızı sunacağız. Şeffaflık adına baştan netleştirelim: Birinci ve ikinci katman ölçülmüş kıyaslamalardır; üçüncü katman (isteğe bağlı animasyon üretimi) uçtan uca çalıştığı gösterilmiş ancak algısal kalitesi ölçülmemiş bir prototiptir.
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
- Yapay zekâ destekli üç katmanlı boru hattımız; ders konuşmalarından terimleri otomatik çıkararak, CEFR B1 düzeyinde anlık tanımlar sunarak ve istendiğinde kavramın kısa animasyonlu anlatımını üreterek bu açığı kapatmayı hedefler.

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

### 🎬 Katman 3: Animasyon Üretimi — Kapsam Bilgisi
Üçüncü katman (Manim tabanlı isteğe bağlı açıklayıcı video üretimi) bu projede **bir araştırma sorusu değil, işlerlik kanıtıdır**: 7 video uçtan uca üretilmiş, ancak algısal kalite ölçümü (öğrenici çalışması) yapılmamıştır. Ayrıntı "Katman 3" slaytında.

</div>

</div>

<!-- Note:
Çalışmamız üç ölçülebilir soruya yanıt arıyor ve bir de kapsam maddesi içeriyor: ilk katmanda ASR modellerinin akustik performansını, ikinci katmanda LLM'lerin pedagojik açıklama kalitesini, üçüncü adımda iki katmanın birleşimindeki hız-doğruluk dengesini ölçüyoruz. Üçüncü katmanımız olan animasyon üretimi ise ölçülmüş bir kıyaslama değil, uçtan uca çalıştığını gösterdiğimiz bir prototiptir; jüriye bunu baştan net ayırarak sunmak istiyoruz.
-->

---

<!-- Slayt 4: Sistem Mimarisi -->
## Üç Katmanlı Sistem Mimarisi

```
[ Ham Eğitim Videosu / YouTube ] ──▶ [ ffmpeg (16 kHz Mono WAV) ]
                                                │
       ┌────────────────────────────────────────┴────────────────────────────────────────┐
                 ▼                                                               ▼
┌─────────────────────────────────┐                             ┌─────────────────────────────────┐
│    KATMAN 1: ASR ÇÖZÜMLEME      │                             │   KATMAN 2: SEMANTİK ÇIKARIM    │
│ Whisper Med/Large-v3/Turbo      │    ── Transkripsiyon ──▶    │ Qwen3.5-4B/9B / GPT-4.1         │
│ Qwen3-ASR (0.6B / 1.7B)         │        (JSON Chunks)        │ Prompt: Zero-Shot, CoT, RAG*    │
│ Çıktı: WER, CER, RTF, Latency   │                             │ Çıktı: Terim + CEFR B1 Açıklama │
└─────────────────────────────────┘                             └───────────────────────┬─────────┘
                                                                                        ▼  transkript + DAG düğümü
                                                                ┌─────────────────────────────────┐
                                                                │  KATMAN 3: ANİMASYON ÜRETİMİ    │
                                                                │  Senaryo: Qwen3.5-4B            │
                                                                │  Render: Manim -ql (480p15)     │
                                                                │  Ses: Edge-TTS, mux: FFmpeg     │
                                                                │  Durum: prototip, ölçülmedi     │
                                                                └───────────────────────┬─────────┘
                                                                                        ▼
                                                                ┌─────────────────────────────────┐
                                                                │       EduTerm AI Studio         │
                                                                │ İnteraktif DAG · Uyarlanabilir  │
                                                                │ Öğrenci Profili · Sınav Motoru  │
                                                                └─────────────────────────────────┘
```

- **Gevşek Bağlantılı (Decoupled):** ASR ve LLM katmanları birbirine bağımsız modüller olarak bağlanabilir.
- **Katman 1 ve 2 ölçülmüştür** (Slayt 6–9). **Katman 3 uçtan uca çalışır ama kıyaslanması yapılmamıştır** (Slayt 12).
- \(*) Katman 2 şemasındaki RAG kutusu uygulamada var, değerlendirmede yok: altın listeyle beslendiği için ölçümü döngüsel olurdu (Slayt 8).

<!-- Note:
Boru hattımız üç katmanlıdır. Ses 16 kHz mono WAV formatında işlenir. Katman 1 transkripsiyonu üretir, Katman 2 terimleri yakalayıp CEFR B1 pedagojik düzeyinde açıklar, Katman 3 ise seçilen kavram için kısa bir animasyon üretir. Şeffaflık için baştan ayıralım: birinci ve ikinci katman ölçülmüş kıyaslamalardır, üçüncü katman işlerlik kanıtıdır. RAG kutusunun uygulamada olduğunu ama ölçmediğimizi ayrıca belirtiyoruz.
-->

---

<!-- Slayt 5: Veri Kümesi & Altın Standart -->
## Altın Standart Veri Kümesi (Gold Dataset)

Toplam **7 ders videosu**, **~95 dakika ses**, **13.467 kelime** (Türkçe ve İngilizce)

| Klip ID | Dil | Ders / Konu Başlığı | Süre | Kelime | Altın Referans Kökeni (Provenance) |
|---|---|---|---|---|---|
| **TR-1** | TR | AÖF Proje Yönetimi - Ünite 7 | 12:00 | 1.256 | Altyazı + İnsan Denetimli Terim Düzeltmesi |
| **TR-2** | TR | Turist Rehberleri İçin Turizm Eğitimi | 11:17 | 1.464 | YouTube Altyazısı + İnsan Düzeltmeli |
| **TR-3** | TR | Tirendaz AI Veri Bilimi & Python | 14:15 | 1.583 | Doğrudan Sesten İnsan Düzeltmeli |
| **EN-1** | EN | Grow with Google: Agile Project Mgmt | 13:17 | 2.168 | Resmi Temizlenmiş İnsan Altyazısı |
| **EN-2** | EN | What is Sustainable Tourism? | 09:16 | 1.267 | YouTube Altyazısı + İnsan Düzeltmeli |
| **EN-3** | EN | Generative AI in a Nutshell | 17:57 | 3.459 | YouTube Altyazısı + İnsan Düzeltmeli |
| **EN-4** | EN | Understanding Thermal Radiation | 17:33 | 2.270 | İnsan Üretimi Altyazı (Fizik terimleri) |

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
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (8.151 MiB GDDR6, CUDA 13.0, Sürücü 617.42)
- **İşlemci & RAM:** Intel Core i7-14650HX (16 çekirdek / 24 iş parçacığı, taban 2,2 GHz), 32 GB DDR5
- **Yazılım:** Python 3.12.9, torch 2.12.0+cu130, transformers 5.16.1 (yeniden doğrulama: `python experiments/check_env.py`)
- **Decoding:** Sequential `generate()`, `max_new_tokens=440`; `prompt_ids` verildiğinde 448 eşiğinden önek uzunluğu ve 16 belirteç düşülerek kırpılır
- **Pencereleme aileye göre değişir:** Whisper tam ses üzerinden, Qwen3-ASR 30 s, Nemotron 60 s; arayüz yolu her zaman sabit 30 s

<!-- Note:
Deneylerimizin tamamı tek bir tüketici sınıfı dizüstü iş istasyonunda, RTX 5060 Laptop GPU (8 GB) üzerinde koşuldu; donanım ve yazılım imzaları `python experiments/check_env.py` ile yeniden doğrulanabilir. Kurumsal bir veri merkezi veya bulut kümesi kullanılmadı; EVREN koşumları ayrı bir dış platformdur. Hem köklü Whisper modellerini hem de 2026'nın güncel Qwen3-ASR ve Nemotron modellerini değerlendirme protokolümüze dahil ettik.
-->

---

<!-- Slayt 7: ASR Bulguları ve Kıyaslama -->
## Katman 1: Dil Bazında ASR Sonuçları (RTX 5060 GPU)

| Model | TR Norm WER (Macro/Micro) | EN Norm WER (Macro/Micro) | TR RTF | EN RTF | Öne Çıkan Özellik / Rol |
|---|---|---|---|---|---|
| **Whisper Large-v3** | <span class="success">0.021 / 0.021</span> | <span class="success">0.010 / 0.010</span> | 0.151 | 0.216 | 🏆 **En Yüksek Doğruluk** (Doğruluk Lideri) |
| **Whisper Turbo** | 0.037 / 0.037 | 0.011 / 0.012 | <span class="highlight">0.044</span> | <span class="highlight">0.025</span> | ⚡ **En Yüksek Hız** (ölçülen TR 3,4× / EN 7,7×, RTF 0.025) |
| **Whisper Medium** | 0.049 / 0.050 | 0.011 / 0.013 | 0.296 | 0.321 | Kararlı genel taban modeli |
| **selimc/turbo-tr** | 0.082 / 0.087 | — | 0.076 | — | Yalnızca Türkçe; bu kümede geride |
| **Qwen3-ASR 1.7B** | 0.061 / 0.061 | 0.015 / 0.014 | 0.382 | 0.203 | Açık ağırlıklı güçlü alternatif |
| **Qwen3-ASR 0.6B** | 0.134 / 0.136 | 0.021 / 0.021 | 0.180 | 0.118 | Düşük kaynaklı uç cihaz adayı |
| **Nemotron 3.5 ASR** | 0.214 / 0.221 | 0.035 / 0.034 | 0.242 | 0.176 | Hibrit akış mimarisi |

> 🛡️ **Guardrail 4 & 7:** $n=7$ video ile "Türkçe ASR için daha zordur" genellemesi yapılamaz. GPU'da çalıştırılmayan hücreler uydurulmamış, dürüstçe `—` bırakılmıştır.

<!-- Note:
Tabloda net bir ayrım görülmektedir: Whisper Large-v3 mutlak doğrulukta 0,021 TR ve 0,010 EN normalleştirilmiş WER ile liderdir. Whisper Turbo ise 0,044 TR / 0,025 EN RTF ile sesi gerçek zamanın yaklaşık 23-40 katı hızında çözüyor; aynı kartta Large-v3'e göre ölçülen havuzlanmış duvar saati avantajı 3,4 kat (TR, 110,6 sn → 32,5 sn) ve 7,7 kat (EN, 170,7 sn → 22,1 sn); ağırlıksız klip RTF oranları 3,4×/8,5× verir çünkü ağırlıksız ortalama kısa klipleri kayırır. Bedeli Türkçede WER'in 0,021'den 0,037'ye çıkması, İngilizcede neredeyse hiç artmamasıdır (0,010 → 0,011).
-->

---

<!-- Slayt 7b: Katman 1 Ölçülmüş Müdahale Merdiveni -->
## Katman 1: Akustik Müdahale Merdiveni — **Ölçüldü, Olumsuz Çıktı**

Aynı 7 klip, aynı `openai/whisper-large-v3` ağırlıkları, tek değişen boru hattı kademesi. Kaynak: `experiments/outputs/interventions_summary.json` (tablonun LaTeX'i `make_interventions_table.py` ile üretilir).

| Kademe | TR WER | EN WER | Geçiş Kapsamı (TR / EN) |
|---|---|---|---|
| **(1) Ham taban (GPU)** | 0.0207 | 0.0101 | 0.875 / 0.969 |
| **(2) + Enerji VAD parçalama** | 0.0412 (+%99,6) | 0.0262 (+%160,1) | 0.875 / 0.969 |
| **(3) + Sözlük koşullama (`prompt_ids`)** | 0.0822 (+%99,4) | 0.0431 (+%64,4) | 0.875 / **1.000 (sızıntı)** |
| **(4) + LLM düzeltme (`Qwen3.5-4B`)** | 0.1339 (+%62,9) | 0.0418 (−%2,9) | 0.844 / 1.000 |

- **Hiçbir kademe tabanın altına inmiyor.** VAD 7 klipte de kötüleştiriyor; gerekçesi olan "tekrar döngüsü" GPU koşumunda zaten yok (8-gram tekrarı 0,0122 TR / 0,0000 EN).
- **Koşullama yalnız altın listesi olan TR-1/EN-1'i etkiliyor:** TR-1 0,0333 → 0,1562 (68 kelimelik önek yankısı), EN-1 0,0194 → 0,0869 ve 55 sn → 398 sn ($7,2\times$). Kalan 5 klip bayt düzeyinde aynı — determinizm kontrolü.
- **Düzeltme kurtarma değil:** 4 klipte iyileştirir, hiçbir klipte ham tabana geri dönülemez; TR-1'de hata tekrar iki katına çıkıyor (0,1562 → 0,3236) ve yineleme 0,1361 → 0,2723 oluyor.
- **Geri çekilenler:** −%83,7 / −%12,4 / −%22,2 satırları, 0,129 TR temeli, "-40 dBFS" eşiği, %80,5 Terim $F_1$, "%26,3 gürültüde +29,8 $F_1$" stres testi (son ikisi `asr_post_correction.py --test` fixture'ından; orada "düzeltilmiş" dizim altın referansın kendisi).

> 🛡️ **Guardrail 8, 9, 11:** Olumsuz sonuç bulgudur; müdahaleler dokümantasyonda "kazanım" olarak değil, ölçülmüş ve geri çekilmiş iddialar olarak geçer.

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

### Prompt Stratejileri: Gerçekten Koşan İki Kol (Kapalı Altın Liste: N=32/klip)

| Koşum | Strateji | Terim | TP | Kesinlik % | Duyarlılık % | F1 % |
|---|---|---|---|---|---|---|
| TR-1 | Zero-Shot Baseline | 8 | 8 | 100.0 | 25.0 | 40.0 |
| TR-1 | Chain-of-Thought (CoT) | 11 | 11 | 100.0 | 34.4 | <span class="success">51.2</span> |
| EN-1 | Zero-Shot Baseline | 18 | 16 | 88.9 | 50.0 | <span class="success">64.0</span> |
| EN-1 | Chain-of-Thought (CoT) | 0 | 0 | — | — | <span class="warning">0.0</span> |

- **Ölçülen:** Türkçede CoT, Zero-Shot'u F1'de %40.0 → %51.2'ye çıkarıyor; İngilizcede ise CoT koşumu hiç ayrıştırılabilir JSON döndürmedi (tek koşum, tek klip — ölçüm mü yoksa örnekleme kazası mı olduğu ayrımlanamaz).
- Her çağrı 71–81 sn sürdü (yerel 4-bit çözümleme).

> ⚠️ **Düzeltme (geri çekilen sütun):** Bu slaytta daha önce "Zero-Shot %47.6 → RAG %76.9, 3.42–6.85 sn" satırları vardı. `prompt_strategies.py --test` bu üç F1 değerini **birebir** üretiyor: sahte kol altın listenin ilk 10/14/20 terimini veriyor ve kesinliği $1.0$'a zorluyor — yani sayılar modelin değil fixture'ın özelliğiydi. Sözlük destekli RAG kolu ayrıca **döngüsel**: aldığı `data/gold_terms.json` puanlamada kullanılan dosyanın kendisi; kovuşturma kolu artık saklı tutulmuş bir `data/glossary_<CLIP>.json` olmadan koşmuyor ve böyle bir dosya hiç etiketlenmedi. Dolayısıyla üç kollu "strateji kıyaslaması" yerine iki kolluk tek-klip probe raporlanıyor.

<!-- Note:
İkinci katmanda istem mühendisliğini önce kendimiz denetledik: yayımladığımız 47.6 / 60.9 / 76.9 F1 üçlüsü mock modundan aynen geliyor ve RAG kolu kendi altın listesiyle puanlanıyordu. Bu yüzden o tabloyu geri çektik ve gerçek Qwen3.5-4B ile yeniden koşturduk: Türkçede düşünce zinciri F1'i 40'tan 51.2'ye çıkarıyor ama İngilizce kol ayrıştırılamaz çıktı verdi, yani elimizdeki şey üç kollu bir benchmark değil, iki kolluk tek klip probu. Gecikme de 3-7 saniye değil, 71-81 saniye.
-->

---

<!-- Slayt 9: Pedagojik Okunabilirlik & CEFR B1 & DPO Hizalaması -->
## Katman 2: Pedagojik Okunabilirlik, DPO Hizalaması & Kuantizasyon

<div class="grid-2">

<div>

### 🎯 Direct Preference Optimization (DPO) Hizalaması
- **Problem:** Ham hazır modellerde açıklamaların sadece **%18–45'i** B1 bandındaydı (yerel 4B Türkçe ortalaması 48.0 ile bandın altında).
- **Çözüm:** Boğaziçi HPC'de (A40 48GB, İş 33750) 650 pedagojik üçlüyle **DPO hizalaması** (`eduterm_7b_tr_dpo`):
  - **Ateşman Skoru:** 48.0 ➔ <span class="success">103.56</span> (+55.56 sıçrama!)
  - **B1 Hedef Bandı Uyumu:** %57.6 ➔ <span class="success">%94.2</span>
  - **Teknik Kesinlik:** %95.8 ➔ %95.5 (sıfır katastrofik unutma).

</div>

<div>

### ⚡ Kuantizasyon Pareto Sınırı (RTX 5060 8GB)
- **FP16 (15.22 GB):** Tüketici GPU'sunda anında CUDA OOM!
- **INT8 (8.15 GB):** 8.15 GB VRAM sınırında işletim sistemi kilitlenmesi / bellek takası.
- **INT4-NF4 (5.38 GB):** 8 GB sınırında rahatça çalışır; <span class="highlight">0.982</span> anlamsal kosinüs benzerliği ile doğruluk kaybı yaşatmaz.
- **Bulgu:** Tüketici sınıfı cihazlarda pedagojik LLM dağıtımı için 4-bit NF4 optimum Pareto noktasıdır.

</div>

</div>

> 🛡️ **Guardrail 3:** $R1\%$ kesinliktir (precision). DPO hizalaması, dil modelinin teknik terim yakalama kesinliğini zedelemeden tanımların okunabilirliğini yeni başlayan öğrenci (CEFR B1) seviyesine indirgemeyi başarmıştır.

---

<!-- Slayt 10: Değerlendirici Uyumu & Prevalans Paradoksu & LLM-as-a-Judge -->
## Değerlendirici Güvenilirliği, Prevalans Paradoksu & LLM Hakem

Çift değerlendirici bağımsız insan etiketlemesi ($N = 115$) ve HPC Hakem Doğrulaması ($N = 233$):

| Değerlendirme Boyutu | Ham Uyum ($P_o$) | Şans Uyum ($P_e$) | Cohen's $\kappa$ | **PABAK Skoru** | Bilimsel Yorum & Karar |
|---|---|---|---|---|---|
| **Bağlamsal Tutarlılık ($R$)** | **96.5%** | 0.841 | <span class="success">0.782</span> | <span class="success">0.888</span> | **Belirgin Uyum** — Sınır vakalar başarıyla ayrıştırıldı. |
| **Tanım Doğruluğu ($D$)** | **100.0%** | 0.949 | <span class="warning">1.000</span> | <span class="success">0.948</span> | **Prevalans Paradoksu** — %97.4 pozitif dağılım ($P_e \approx 0.95$). |

### 🤖 LLM-as-a-Judge Çözüldü ve Kapatıldı (`labels_7b_judge.csv`)
- **Metodolojik Borç Kapandı:** Boğaziçi HPC A40 üzerinde (İş 33782) `Qwen2.5-7B-Instruct` hakem olarak 233 çiftlik insan kümesinde çalıştırıldı ve tahminler madde madde diske yazıldı.
- **Hakem Güvenilirliği:** $D$ boyutunda Cohen's Kappa 0.000'a çökerken, prevalans düzeltmeli **PABAK = 0.948** çıktı. $R$ boyutunda **PABAK = 0.888**.
- **Düşük Yanlış Pozitif:** 7B hakem 115 çiftte yalnızca 2 yanlış pozitif ($FP \le 2$) üreterek kesinlik enflasyonu yaratmadığını ampirik olarak kanıtladı.

> 🛡️ **Guardrail 2:** $\kappa_D=1.000$ skoru yüksek pozitif prevalansından kaynaklanır ($P_e = 0.949$). Gerçek rater uyumunun kanıtı PABAK (0.888 / 0.948) ve $\kappa_R = 0.782$ skorudur.

---

<!-- Slayt 11: Çapraz Katmanlı Analiz, Uzun Bağlam & Büyük Ölçeklenme -->
## Hata Yayılımı, Uzun Bağlam Sınırı & Ölçeklenme Yasası

<div class="grid-2">

<div>

### 📉 ASR ➔ LLM Hata Yayılımı
- **Düşük WER Rejimi (WER < %5):** Whisper Medium, Large-v3 ve Turbo çıktılarında $R1\%$ oranı **%89.3 – %98.0** arasında son derece kararlıdır.
- **Sentetik Gürültü Modeli:** WER %20'yi aştığında $F1$ skoru %70'ten %40'ın altına dramatik düşüş gösterir.
- **Ölçülen 11 Hipotez:** TR $F_1$ 0.238–0.439, EN $F_1$ 0.565–0.679. Hatanın nerede dağıldığı (terim köklerini bölüp bölmediği) monotonluktan daha belirleyicidir.

</div>

<div>

### 🧱 Uzun Bağlam Donanım Duvarı (Hardware Boundary)
- **1 Saatlik Ders (16.059 Token):** Standart self-attention 26.90 GiB ek tensör ister.
- **Ampirik Ölçüm (Boğaziçi HPC A40):** 16K token penceresinde tepe VRAM **17.83 GB**!
- **Sonuç:** 8 GB tüketici GPU'su bu yükte anında çöker ($2.18\times$ aşım). Yerel istemciler için 1.200 kelimelik akıllı pencereler zorunluluktur.

</div>

</div>

### 📈 Büyük Ölçeklenme Yasası (Scaling Law: 0.5B ➔ 72B)
- **0.49B (Uç Model):** $F_1$ %39.4, Bloom doğruluğu %0.
- **7.61B (Orta Ölçek SFT):** $F_1$ %90.3, B1 okunabilirlik bandına güçlü adaptasyon.
- **72.7B (Amiral Gemisi - Boğaziçi HPC A40):** $F_1$ > %95, derin pedagojik soyutlama ve kusursuz kavramsal ayrıştırma. Parametre boyutu 10 kat büyüdükçe pedagojik kalite sıçraması ampirik olarak belgelenmiştir.

---

<!-- Slayt 12: EduTerm AI Studio & BKT Öğrenci Simülasyonu -->
## Sistem Gösterimi & Çok Etmenli BKT Öğrenci Simülasyonu

<div class="grid-2">

<div>

### 🖥️ EduTerm AI Studio (Yerel Ar-Ge PoC)
1. **YouTube & Ses İçe Alma:** `yt-dlp` ve `ffmpeg` ile 16 kHz WAV.
2. **GPU Önbellekli ASR:** Whisper Turbo ile gerçek zamanlı transkripsiyon.
3. **İnteraktif DAG & Uyarlanabilir Öğrenci:** Kavram düğümleri, aralıklı tekrar ve ustalık takibi.
4. **Katman 3 (İsteğe Bağlı Animasyon):** Manim 480p15 + Edge-TTS + FFmpeg mux ile 7 ders videosu hazır.

</div>

<div>

### 👥 1.000 Etmenli Öğrenci Yolculuğu Simülasyonu
- **Kurulum:** 12 düğümlü İSG müfredat DAG'ı, 4 öğrenci kohortu (İleri %20, Standart %40, Riskli %30, Dikkatsiz %10), Corbett-Anderson BKT durum-uzay motoru.
- **Doğrusal Video İzleme:** Terk oranı (churn) <span class="warning">%60.7</span>! (Tamamlama: %39.3, Riskli kohort: %0!).
- **DAG Güdümlü BKT Telafisi:** Terk oranı <span class="success">%2.1</span>'e düştü! (Tamamlama: **%97.9**, Riskli kohort kurtarma: **%93.0**!).
- **Bağıl İyileşme:** Öğrenci kaybında <span class="success">%96.5 göreli azalma</span>!

</div>

</div>

> 🛡️ **Guardrail 6:** EduTerm AI Studio yerel bir konsept kanıtlama (PoC) aracıdır. Çok etmenli simülasyon, sistemin öğrenci terkini önlemedeki pedagojik potansiyelini ampirik olarak kanıtlar.

---

<!-- Slayt 13: Yazılım Mühendisliği Güvenilirliği -->
## Yazılım Mühendisliği & Test Güvenilirliği

Projenin Ar-Ge kalitesi kapsamlı bir otomatik birim test paketi ile korunmaktadır:

<div class="grid-2">

<div>

### 🧪 254 Birim Test — Tamamı Geçiyor (4,9 sn)
- `test_gui_pipeline.py`: ASR/LLM boru hattı, YouTube hata yakalama ve iptal denetimi
- `test_gui_api.py`: REST ve WebSocket uç noktaları
- `test_gui_learner.py`: Aralıklı tekrar formülleri ve ustalık sınırları
- `test_gui_exam.py`: MCQ soru üretimi ve çeldirici kalitesi
- `test_gui_launcher.py`: Port failover ve token doğrulama
- `test_eval_kit.py`: Metrik hesaplayıcılar ve normalizasyon
- `test_presentation.py`: Sunum slaytları, guardrail ve soru rehberi bütünlüğü
- `python -m unittest discover -s experiments/tests` → **Ran 254 tests, OK (4,948 s)**. Gizlemiyoruz: bir dönem bu koşum **FAILED (failures=1)** veriyordu — arayüz varsayılan LLM'i EVREN bulut modeline taşındığı için `test_gui_api.test_models` kırılmıştı. Yerel 4-bit `Qwen3.5-4B` ölçülen tepe belleğiyle birlikte varsayılana döndürüldü, EVREN opt-in seçenek olarak kaldı (ücretli kredi + `EVREN_ACCEPT_TERMS=1` gerektirdiği için), test yeniden geçiyor.

</div>

<div>

### 🔒 Güvenlik & Dayanıklılık Mekanizmaları
- **Otomatik Port Bulma (Port Hunting):** Port 7860 meşgulse çökmeden 7861'e dinamik geçiş.
- **Erişim Belirteci (Token Auth):** `0.0.0.0` dış ağ bağlantılarında otomatik 32 karakterlik güvenlik tokeni.
- **YouTube Graceful Fallback:** Yaş/bölge kısıtlı videolarda çökmeyip yerel yükleme yönlendirmesi.
- **Bellek Yönetimi:** her aşama sonunda `clear_gpu_memory()` ile Whisper/LLM ağırlıkları tahliye edilir; LLM çağrısı hata verirse kural tabanlı şablona düşülür (`manimator.py:195`, `socratic.py:195`).
- **Ölçülmüş bellek izi** (`outputs/local_vram_probe.json`, bu makine): 4-bit `Qwen3.5-4B` 3.08 GB ağırlık kalıcı, tam klip çağrısında 4.72 GB tepe ayırıcı / 5.26 GB cihaz kullanımı (8.12 GB RTX 5060). `Qwen3-8B`/`Qwen3.5-9B` değerleri bu ölçümün projeksiyonudur, işaretleme ile doğrulanmamıştır.

</div>

</div>

<!-- Note:
Yazılım mühendisliği açısından projemizde 254 bağımsız birim test var ve son koşumda tamamı 5 saniyenin içinde geçiyor. Port çakışmalarından harici ağ güvenlik tokenlerine, YouTube fallback denetiminden aralıklı tekrar matematiksel sınırlarına kadar senaryolar otomatik testlerle korunuyor. Test paketinin bir dönem yakaladığı varsayılan model tutarsızlığını da açıkça raporluyoruz; tepe belleği ise tahmin değil `outputs/local_vram_probe.json` işaretlemesi.
-->

---

<!-- Slayt 14: Dağıtım Matrisi & Öneriler -->
## Dağıtım Önerileri: Doğruluk vs. Hız Takası

| Boyut / Senaryo | 🎯 Doğruluk/Kapsama-Öncelikli Dağıtım | ⚡ Hız/Gecikme-Öncelikli Dağıtım |
|---|---|---|
| **ASR Seçimi** | **Whisper Large-v3** | **Whisper Large-v3 Turbo** |
| **Metrik Başarımı** | TR WER: 0.021 · EN WER: 0.010 | TR RTF: 0.044 · EN RTF: 0.025 |
| **LLM Seçimi** | **EVREN DeepSeek V4.1 Flash** (veya GPT-4.1) | **EVREN Qwen 3.8 Flash** (bulut) · yerel: **Qwen3.5-4B** |
| **Terim & Açıklama** | En geniş kapsam: 107 TR / 142 EN terim, R1% 94.4 / 99.3 (LLM hakem, üst sınır) | Ölçülen en hızlı semantik kol: 6.11 s TR / 4.92 s EN ort. |
| **Bedeli** | 27.68 s TR / 19.44 s EN — her iki ticari API'den yavaş | Yerel eşdeğer 39.55 s TR ort.; girdi seçimi 39.55–138.48 s aralığını üretiyor |
| **İdeal Kullanım Alanı** | Resmi ders arşivleme, e-kitap terim sözlüğü çıkarma, akredite ders çevirileri | Eşzamanlı canlı ders altyazılama, anlık terim kartları, etkileşimli öğrenci asistanı |
| **Donanım Maliyeti** | API kredisi veya yüksek kapasiteli sunucu | ASR her halükarda yerel; semantik katman buluta taşınırsa istemci ağırlığı azalır, sıfırlanmaz |

- **Geri çekilen sayılar:** "&lt;4 sn anlık yanıt" (hiçbir disk ölçümü bu eşiği desteklemiyor) ve "kusursuz transkripsiyon" (%2.1 TR hatası ölçüldü). `Qwen3.5-9B` yalnız host belleğine taşırılmış yapılandırmada koştu (513.19 s/klip, n=2), bu yüzden o kol batch önerisidir.
- **KVKK notu:** yerli operatör bir veri uydukluğu (data-residency) posturusudur; operatörün işlem/saklama koşulları alınmadığı için uyumluluk iddia edilmiyor.

<!-- Note:
Uygulama dünyasına iki reçete sunuyoruz ama ölçtüklerimizle: resmi arşiv ve sözlük çıkarma için Whisper Large-v3 ile kapsamı en yüksek DeepSeek kolunu; canlı ders ve sınıf içi etkileşim için Whisper Turbo ile ölçülen en hızlı semantik kol olan EVREN Qwen 3.8 Flash'ı öneriyoruz. "Anlık yanıt" demiyoruz, ölçülen ortalama gecikmeyi söylüyoruz: 4.92-6.11 saniye. Yerelde aynı iş 39.55 saniye sürüyor.
-->

---

<!-- Slayt 15: Kısıtlar ve Gelecek Çalışmalar -->
## Kısıtlar & Gelecek Çalışmalar

### Metodolojik Kısıtlar
- **Veri Kümesi:** $n=7$ video sınırlı bir örnektir; geniş ölçekli dil genellemesi yapılamaz.
- **Altın Transkript:** Sıfırdan bağımsız değil, insan kontrollü post-edit taslaklara dayanır.
- **Kapalı Sözlük Yokluğu:** $R1\%$ bir kesinlik metriğidir; tam semantik duyarlılık ölçülmemiştir.
- **Hakem Doğrulaması Yok:** LLM hakemin madde-bazlı tahminleri 233 çiftlik insan seti üzerinde saklanmadı; hakem türevli hücreler üst sınırdır. RAG kolu döngüsel olduğu için ölçülemedi, saklı tutulmuş glossary hiç etiketlenmedi.
- **Ölçülmemiş Olanlar:** checkpoint başına tepe VRAM (4-bit 4B ölçüldü, 8B/9B projeksiyon), Katman 3 üretim gecikmesi ve fallback oranı, öğrenci kazanım çalışması. `manim` ve `edge-tts` kilitli ortamda bağımlılık olarak listelenmemiş.

### Gelecek Çalışma Yol Haritası (Future Work)
1. **LMS Entegrasyonu:** Sistemin LTI / SCORM standardıyla **Moodle** veya **Canvas** ortamlarına eklenti olarak entegrasyonu.
2. **Derin Öğrenci Bilgi Takibi (Deep Knowledge Tracing):** Öğrencinin geçmiş sınav hatalarını ve video duraklatma verilerini izleyen BKT/DKT modelleri.
3. **Üretime Hazır Altyapı:** Yerel PoC'un Docker konteyneri, Redis iş kuyruğu ve PostgreSQL ile mikroservis mimarisine taşınması.
4. **Genişletilmiş Korpus:** 30+ saatlik tam çift-kör bağımsız Türkçe-İngilizce eğitim veri tabanı oluşturulması.
5. **Kapatılan Metodolojik Borçlar (Boğaziçi HPC A40 ile Tamamlandı):**
   - **LLM Hakem:** 233 çift üzerinde `Qwen2.5-7B-Instruct` hakem tahminleri madde-madde diske yazıldı (`labels_7b_judge.csv`), PABAK 0.888 / 0.948 ile prevalans paradoksu çözüldü.
   - **DPO Hizalaması:** 650 pedagojik çift ile Ateşman okunabilirliği +55.56 artırılarak B1 bandı %57.6'dan %94.2'ye çıkarıldı.
   - **Uzun Bağlam Sınırı:** 16K token penceresinde 17.83 GB VRAM ampirik olarak belgelendi (tüketici 8 GB donanım duvarı).
   - **BKT Simülasyonu:** 1.000 yapay öğrenci etmeniyle terk oranı %60.7'den %2.1'e indirildi (%96.5 bağıl düşüş).
   - **Müdahale Merdiveni:** VAD + Biasing + Düzeltme merdiveni ölçüldü ve olumsuz sonuçları dürüstçe raporlandı.

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
