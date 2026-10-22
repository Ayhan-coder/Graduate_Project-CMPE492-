# Proje Geliştirme Adımı 1: Kapsam, Veri Envanteri ve Deney Protokolü

**Belge Sürümü:** 1.0  
**Tarih:** 2026-10-15  
**Yazar:** Ali Ayhan Günder  
**Bağlam:** Bitirme Projesi 2026 — Çok Dilli Eğitim İçin Konuşma Tanıma (ASR), LLM Tabanlı Terminoloji Çıkarma ve Üretken Video Katmanlarının Değerlendirilmesi

---

## 1. Giriş ve Amaç

Bu doküman, **Proje Geliştirme Planı (Seçenek 1 - Adım 1: "Kapsamı ve mevcut durumu sabitleyin | 1 hafta")** gereksinimlerini eksiksiz karşılamak üzere hazırlanmıştır. 

Projenin temel felsefesi: **"Hiçbir varsayım kanıtlanmış sonuç gibi sunulamaz; veri envanteri gerçek dosya durumunu yansıtmalı ve her veri öğesi izlenebilir olmalıdır."**

Bu kapsamda:
1. 32 derslik manifest (`manifest_30plus.jsonl`), YouTube kaynakları (`sources_30plus.json`), yerel sesler, transkriptler ve terim etiketleri **tek birleştirilmiş envanterde** (`experiments/data/data_inventory.json` ve `data_inventory.csv`) eşleştirilmiştir.
2. Her bir kayıt için **kaynak kaydı**, **indirilebilir ses**, **referans transkript** ve **anotasyon durumları** birbirinden bağımsız 4 durum bayrağı ile işaretlenmiştir.
3. Terim eşleşmesi, zaman aralığı, Cohen's kappa ($\kappa$), F1, WER/CER ve bootstrap güven aralığı ($\text{CI}_{95\%}$) matematiksel ve operasyonel olarak önceden dondurulmuştur.
4. Sistemin tüm bilimsel ve deneysel sınırlılıkları ve geçiş ölçütü kriterleri açıkça listelenmiştir.

---

## 2. 40 Kayıtlık Birleşik Veri Envanteri (Data Inventory)

> **[GÜNCELLEME VE ÜSTÜNE YAZILMA NOTU - 22 Ekim 2026]:** Bu belgedeki ilk 32 kayıtlık envanter tasarımı, kurumsal zorunlu eğitimlerin (KVKK: TR-17..18, EN-17..18; İSG: TR-19..20, EN-19..20) eklenmesiyle **40 videoluk envantere (20 TR + 20 EN, kaynak üstverisinde 10,1 saat; diskte süre doğrulaması yapılmış 16/40 videoda 4,6 saat)** genişletilmiştir. Disk üzerinde **16 doğrulanmış yerel WAV** (7 gold + 8 kurumsal + EN-11) + **18 süre uyumsuz ham kayıt** (pending slice trim) + **6 henüz indirilmemiş kayıt** (downloadable pending fetch) yer almaktadır; 15 transkript kalite eşiğini geçmiştir (7 gold + 8 tam Whisper Turbo kurumsal transkripti). Güncel tek doğruluk kaynağı `experiments/data/data_inventory.json` ve `data_inventory.csv` dosyalarıdır.

Envanter, 14 farklı disiplin ve kurumsal alandan seçilmiş 20 Türkçe (`TR-01` .. `TR-20`) ve 20 İngilizce (`EN-01` .. `EN-20`) olmak üzere toplam **40 video kaydını** içermektedir.

### 2.1 Dört Bağımsız Durum Boyutu

Her veri öğesi için aşağıdaki 4 durum ekseni ayrı ayrı izlenir:

1. **Kaynak Kaydı Durumu (`source_status`):**
   - `CONFIRMED_PUBLIC_URL`: YouTube üzerinde doğrulanmış açık erişimli ders videosu (URL, kanal, konuşmacı profili ve süre kayıtlıdır).
2. **İndirilebilir Ses Durumu (`audio_status`):**
   - `LOCAL_PCM_WAV_EXISTS`: Yerel diskte 16 kHz, Mono, 16-bit PCM WAV olarak hazır ve doğrulanmış ses dosyası. (Mevcut durumda 7 pilot klip: TR-01..03, EN-01..04).
   - `DOWNLOADABLE_PENDING_FETCH`: YouTube URL'si erişilebilir, ancak ses dosyası henüz yerel ortama indirilmemiştir (25 aday klip).
3. **Referans Transkript Durumu (`transcript_status`):**
   - `HUMAN_VERIFIED_GOLD`: İnsan denetiminden geçmiş ve doğrulanmış altın referans transkript (`experiments/data/gold/{ID}.txt`).
   - `CURATED_DRAFT_CAPTION`: Altyazı veya sentetik ders metni taslağı; henüz tam insan düzeltmesi yapılmamış.
4. **Anotasyon / Terim Etiketi Durumu (`annotation_status`):**
   - `CLOSED_GOLD_BENCHMARK`: $N=32$ terimlik kapalı altın terminoloji listesi hazırlanmış (TR-01 ve EN-01).
   - `DUAL_ANNOTATED_SAMPLE`: İki bağımsız değerlendirici tarafından $R$ ve $D$ metrikleriyle etiketlenmiş ve $\kappa$ hesaplanmış örneklem (115 terim çifti, TR-01..03 ve EN-01..04).
   - `TEMPLATE_STARTER_ONLY`: Şablon dosyada (`annotation_template_30plus.jsonl`) 4 aday terim ile açılmış, hakem anotasyonu bekleyen kayıtlar.

---

### 2.2 Veri Envanteri Özet Tablosu *(Tarihsel 32 Kayıtlık Çekirdek Tasarım)*

> **[TARİHSEL TABLO NOTU]:** Aşağıdaki tablo, projenin ilk aşamasındaki 32 videoluk çekirdek tasarım durumunu yansıtmaktadır. Genişletilmiş güncel envanterde 40 kayıt (20 TR + 20 EN, kaynak üstverisinde 10,1 saat; diskte süre doğrulanmış 16/40 klipte 4,6 saat; 7 altın + 8 kurumsal QA onaylı transkript) yer almakta olup, tek doğruluk kaynağı `experiments/data/data_inventory.json` ve `data_inventory.csv` dosyalarıdır.

| Metrik / Durum | Türkçe (TR) | İngilizce (EN) | Başlangıç (32 Kayıt) | Güncel Genişletme (40 Kayıt) | Açıklama |
|---|:---:|:---:|:---:|:---:|---|
| **Toplam Kayıt Sayısı** | 16 $\to$ 20 | 16 $\to$ 20 | **32** | **40** | 12 disiplin + KVKK ve İSG kurumsal eğitimleri |
| **Doğrulanmış Kaynak URL'si** | 16 $\to$ 20 | 16 $\to$ 20 | **32 (%100)** | **40 (%100)** | Doğrulanmış YouTube açık eğitim ve kurumsal videoları |
| **Doğrulanmış Yerel WAV (Süre Uyumlu)** | 3 (`TR-01..03`) + 4 (`TR-17..20`) | 4 (`EN-01..04`) + 4 (`EN-17..20`) + 1 (`EN-11`) | **7 (%21.9)** | **16 (%40.0)** | 7 gold + 8 kurumsal + EN-11 doğrulanmış yerel PCM WAV |
| **İndirilebilir / İnceleme Bekleyen Ses** | 13 | 11 | **25 (%78.1)** | **24 (%60.0)** | Ham indirme dilimleme/doğrulama veya fetch bekleyen |
| **İnsan Onaylı Altın Transkript** | 3 (`TR-01..03`) | 4 (`EN-01..04`) | **7 (%21.9)** | **7 (%17.5)** | 13.467 kelime insan denetimli altın referans |
| **Tam ASR Transkripti (QA Onaylı)** | 4 (`TR-17..20`) | 4 (`EN-17..20`) | **0** | **8 (%20.0)** | Whisper Turbo ile uzun form ASR transkripti |
| **Kürate Edilmiş Taslak Transkript** | 13 | 12 | **25 (%78.1)** | **25 (%62.5)** | Konu özetleri ve taslak altyazılar (düşük WPM) |
| **Kapalı Altın Sözlük ($N=32$)** | 1 (`TR-01`) | 1 (`EN-01`) | **2 (%6.3)** | **2 (%5.0)** | 32 kanonik terim + alternatif eşanlamlılar |
| **Çift Etiketli IAA Örneklemi** | 3 | 4 | **7 (%21.9)** | **7 (%17.5)** | 115 madde-bazlı $R/D$ çift etiket |
| **Aday Şablon Terimleri (N=4)** | 16 $\to$ 20 | 16 $\to$ 20 | **32 (%100)** | **40 (%100)** | 160 aday terim şablonu hazırlandı |

---

### 2.3 40 Kaydın Detaylı Matrisi (Güncel Disk ve Transkript Durumu)

> **[GÜNCEL MATRİS NOTU]:** Aşağıdaki matris, 40 kaydın disk üzerindeki gerçek durumunu (16 doğrulanmış yerel WAV, 18 süre uyumsuz ham kayıt, 6 fetch bekleyen) ve transkript durumunu yansıtmaktadır. Tek doğruluk kaynağı experiments/data/data_inventory.json ve data_inventory.csv dosyalarıdır.

| ID | Eski ID | Dil | Alan (Domain) | Süre | Ses Durumu | Transkript Durumu | Anotasyon Durumu | Kaynak Kanalı |
|---|---|:---:|---|:---:|:---:|:---:|:---:|---|
| **TR-01** | TR-1 | TR | Software Engineering & Project Management | 12:01 | LOCAL WAV (22.0 MB, Doğrulanmış) | HUMAN GOLD (1.256 kelime) | CLOSED GOLD (32 terim) + IAA | Açıköğretim Fakültesi - Anadolu Üniversitesi |
| **TR-02** | TR-2 | TR | Tourism & Sustainable Hospitality | 11:17 | LOCAL WAV (20.7 MB, Doğrulanmış) | HUMAN GOLD (1.464 kelime) | DUAL ANNOTATED (IAA) | Forteliber |
| **TR-03** | TR-3 | TR | Computer Science & Data Science | 14:15 | LOCAL WAV (26.1 MB, Doğrulanmış) | HUMAN GOLD (1.583 kelime) | DUAL ANNOTATED (IAA) | Tirendaz AI / Tirendaz Academy |
| **TR-04** | TR-4 | TR | Artificial Intelligence & Machine Learning | 13:40 | HAM SES (5153.2 sn, Dilimleme Bekliyor) | CURATED DRAFT (66 kelime) | TEMPLATE ONLY (4 aday terim) | Bilişim Vadisi Akademi |
| **TR-05** | TR-5 | TR | Computer Science & Data Science | 15:20 | DOWNLOADABLE PENDING | CURATED DRAFT (66 kelime) | TEMPLATE ONLY (4 aday terim) | ODTÜ Bilgisayar Mühendisliği Açık Ders |
| **TR-06** | TR-6 | TR | Computer Science & Data Science | 12:50 | DOWNLOADABLE PENDING | CURATED DRAFT (70 kelime) | TEMPLATE ONLY (4 aday terim) | Boğaziçi Mezunlar Akademisi |
| **TR-07** | TR-7 | TR | Artificial Intelligence & Machine Learning | 14:05 | DOWNLOADABLE PENDING | CURATED DRAFT (63 kelime) | TEMPLATE ONLY (4 aday terim) | Yapay Zeka Araştırmacıları Derneği |
| **TR-08** | TR-8 | TR | Mechanical & Thermal Engineering | 16:10 | HAM SES (2974.5 sn, Dilimleme Bekliyor) | ASR REJECTED (36 kelime) | TEMPLATE ONLY (4 aday terim) | İTÜ Açık Ders Malzemeleri |
| **TR-09** | TR-9 | TR | Electrical Engineering & Circuit Theory | 11:45 | HAM SES (764.5 sn, Dilimleme Bekliyor) | ASR REJECTED (3 kelime) | TEMPLATE ONLY (4 aday terim) | Mühendislik Dersleri |
| **TR-10** | TR-10 | TR | Mathematics (Linear Algebra, Calculus) | 13:30 | HAM SES (1754.8 sn, Dilimleme Bekliyor) | ASR REJECTED (2 kelime) | TEMPLATE ONLY (4 aday terim) | Matematik Köyü Seminerleri |
| **TR-11** | TR-11 | TR | Physics (Mechanics, Electromagnetism) | 15:00 | HAM SES (963.5 sn, Dilimleme Bekliyor) | ASR REJECTED (40 kelime) | TEMPLATE ONLY (4 aday terim) | TÜBİTAK Temel Bilimler |
| **TR-12** | TR-12 | TR | Economics & Finance | 10:55 | HAM SES (782.2 sn, Dilimleme Bekliyor) | ASR REJECTED (56 kelime) | TEMPLATE ONLY (4 aday terim) | Açıköğretim Fakültesi - Anadolu Üniversitesi |
| **TR-13** | TR-13 | TR | Cognitive Psychology & Neuroscience | 12:15 | HAM SES (607.5 sn, Dilimleme Bekliyor) | ASR REJECTED (45 kelime) | TEMPLATE ONLY (4 aday terim) | Psikoloji Platformu TR |
| **TR-14** | TR-14 | TR | History & Cultural Heritage | 13:50 | DOWNLOADABLE PENDING | CURATED DRAFT (41 kelime) | TEMPLATE ONLY (4 aday terim) | Tarih Vakfı Konferansları |
| **TR-15** | TR-15 | TR | History & Cultural Heritage | 11:30 | DOWNLOADABLE PENDING | CURATED DRAFT (44 kelime) | TEMPLATE ONLY (4 aday terim) | Felsefe Açık Ders |
| **TR-16** | TR-16 | TR | Biology & Genetics | 14:40 | HAM SES (1400.0 sn, Dilimleme Bekliyor) | ASR REJECTED (60 kelime) | TEMPLATE ONLY (4 aday terim) | Biyoloji Günlüğü |
| **TR-17** | TR-17 | TR | Corporate Compliance & Personal Data Protection (KVKK/GDPR) | 21:13 | LOCAL WAV (38.8 MB, Doğrulanmış) | ASR QA PASSED (2.375 kelime) | TEMPLATE ONLY (4 aday terim) | Arda Çetin |
| **TR-18** | TR-18 | TR | Corporate Compliance & Personal Data Protection (KVKK/GDPR) | 10:17 | LOCAL WAV (18.8 MB, Doğrulanmış) | ASR QA PASSED (812 kelime) | TEMPLATE ONLY (4 aday terim) | İK Kütüphanem Filiz Erboğa |
| **TR-19** | TR-19 | TR | Occupational Health and Safety (İSG / OHS) | 62:10 | LOCAL WAV (113.8 MB, Doğrulanmış) | ASR QA PASSED (6.572 kelime) | TEMPLATE ONLY (4 aday terim) | İzka Osgb TV |
| **TR-20** | TR-20 | TR | Occupational Health and Safety (İSG / OHS) | 45:50 | LOCAL WAV (83.9 MB, Doğrulanmış) | ASR QA PASSED (5.508 kelime) | TEMPLATE ONLY (4 aday terim) | Fatma AYAZ DEĞERLİ |
| **EN-01** | EN-1 | EN | Software Engineering & Project Management | 13:17 | LOCAL WAV (24.3 MB, Doğrulanmış) | HUMAN GOLD (2.168 kelime) | CLOSED GOLD (32 terim) + IAA | Grow with Google |
| **EN-02** | EN-2 | EN | Tourism & Sustainable Hospitality | 09:16 | LOCAL WAV (17.0 MB, Doğrulanmış) | HUMAN GOLD (1.267 kelime) | DUAL ANNOTATED (IAA) | Dr Hayley Stainton | Tourism Teacher |
| **EN-03** | EN-3 | EN | Artificial Intelligence & Machine Learning | 17:57 | LOCAL WAV (32.9 MB, Doğrulanmış) | HUMAN GOLD (3.459 kelime) | DUAL ANNOTATED (IAA) | Henrik Kniberg |
| **EN-04** | EN-4 | EN | Mechanical & Thermal Engineering | 17:33 | LOCAL WAV (32.1 MB, Doğrulanmış) | HUMAN GOLD (2.270 kelime) | DUAL ANNOTATED (IAA) | The Efficient Engineer |
| **EN-05** | EN-5 | EN | Artificial Intelligence & Machine Learning | 14:20 | HAM SES (766.7 sn, Dilimleme Bekliyor) | ASR REJECTED (93 kelime) | TEMPLATE ONLY (4 aday terim) | 3Blue1Brown |
| **EN-06** | EN-6 | EN | Computer Science & Data Science | 12:40 | HAM SES (21283.4 sn, Dilimleme Bekliyor) | ASR REJECTED (66 kelime) | TEMPLATE ONLY (4 aday terim) | freeCodeCamp |
| **EN-07** | EN-7 | EN | Computer Science & Data Science | 13:05 | HAM SES (3030.1 sn, Dilimleme Bekliyor) | ASR REJECTED (61 kelime) | TEMPLATE ONLY (4 aday terim) | MIT OpenCourseWare (6.006) |
| **EN-08** | EN-8 | EN | Mathematics (Linear Algebra, Calculus) | 15:10 | HAM SES (482.5 sn, Dilimleme Bekliyor) | ASR REJECTED (84 kelime) | TEMPLATE ONLY (4 aday terim) | Khan Academy |
| **EN-09** | EN-9 | EN | Physics (Mechanics, Electromagnetism) | 16:45 | HAM SES (258.8 sn, Dilimleme Bekliyor) | ASR REJECTED (44 kelime) | TEMPLATE ONLY (4 aday terim) | Veritasium |
| **EN-10** | EN-10 | EN | Computer Science & Data Science | 14:15 | DOWNLOADABLE PENDING | CURATED DRAFT (52 kelime) | TEMPLATE ONLY (4 aday terim) | Stanford Online (CS110) |
| **EN-11** | EN-11 | EN | Economics & Finance | 11:50 | LOCAL WAV (22.3 MB, Doğrulanmış) | ASR REJECTED (73 kelime) | TEMPLATE ONLY (4 aday terim) | CrashCourse Economics |
| **EN-12** | EN-12 | EN | Cognitive Psychology & Neuroscience | 12:35 | HAM SES (117.0 sn, Dilimleme Bekliyor) | ASR REJECTED (57 kelime) | TEMPLATE ONLY (4 aday terim) | Harvard University Center for Brain Science |
| **EN-13** | EN-13 | EN | Mechanical & Thermal Engineering | 13:20 | HAM SES (982.9 sn, Dilimleme Bekliyor) | ASR REJECTED (81 kelime) | TEMPLATE ONLY (4 aday terim) | StructureFree |
| **EN-14** | EN-14 | EN | Biology & Genetics | 13:55 | HAM SES (271.6 sn, Dilimleme Bekliyor) | ASR REJECTED (43 kelime) | TEMPLATE ONLY (4 aday terim) | Nature Video |
| **EN-15** | EN-15 | EN | History & Cultural Heritage | 12:05 | HAM SES (343.5 sn, Dilimleme Bekliyor) | ASR REJECTED (50 kelime) | TEMPLATE ONLY (4 aday terim) | Wireless Philosophy |
| **EN-16** | EN-16 | EN | Electrical Engineering & Circuit Theory | 15:30 | HAM SES (1702.5 sn, Dilimleme Bekliyor) | ASR REJECTED (96 kelime) | TEMPLATE ONLY (4 aday terim) | Reducible |
| **EN-17** | EN-17 | EN | Corporate Compliance & Personal Data Protection (KVKK/GDPR) | 08:00 | LOCAL WAV (14.6 MB, Doğrulanmış) | ASR QA PASSED (1.500 kelime) | TEMPLATE ONLY (4 aday terim) | Data Protection World Forum |
| **EN-18** | EN-18 | EN | Corporate Compliance & Personal Data Protection (KVKK/GDPR) | 01:33 | LOCAL WAV (2.8 MB, Doğrulanmış) | ASR QA PASSED (201 kelime) | TEMPLATE ONLY (4 aday terim) | IT Governance Ltd |
| **EN-19** | EN-19 | EN | Occupational Health and Safety (İSG / OHS) | 05:22 | LOCAL WAV (9.8 MB, Doğrulanmış) | ASR QA PASSED (645 kelime) | TEMPLATE ONLY (4 aday terim) | Health and Safety Executive (HSE UK) |
| **EN-20** | EN-20 | EN | Occupational Health and Safety (İSG / OHS) | 16:22 | LOCAL WAV (30.0 MB, Doğrulanmış) | ASR QA PASSED (1.843 kelime) | TEMPLATE ONLY (4 aday terim) | Safety Training Videos |

---

## 3. Önceden Sabitlenen Deney Protokolü ve Formel Tanımlar

Adım 1'in temel gereksinimi olarak; sonraki adımlarda (çapraz katman, müdahale karşılaştırmaları, model kıyaslamaları) kullanılacak tüm metrikler ve protokol kuralları önceden sabitlenmiştir.

### 3.1 Terim Eşleşmesi (Terminology Matching)

Modelin ürettiği aday terim $t_{hyp}$ ile referans altın sözlükteki $t_{ref}$ terimi arasındaki eşleşme algoritması `experiments/eval_gold_terms.py` dosyasında kilitlenmiştir:

1. **Metin Normalizasyonu:**
   - Unicode `NFC` normalizasyonu uygulanır.
   - Türkçe noktalı/noktasız harf ayrımı korunarak küçük harfe çevrilir: `İ -> i`, `I -> ı`. (İngilizce doğrudan `lower()` yapılır).
   - Başta yer alan tanımlıklar temizlenir: `"the "`, `"a "`, `"an "`, `"bir "`.
   - Tüm noktalama işaretleri boşlukla değiştirilir ve aradaki fazla boşluklar tek boşluğa indirgenir.
2. **Eşleşme Kademeleri:**
   - **Kademe 1 (Kanonik veya Eşanlamlı Birebir Eşleşme):** Normalleştirilmiş aday metin, kanonik terim veya tanımlı alternatif eşanlamlı (alias) kümelerinden birine tam olarak eşitse eşleşme kabul edilir ($t_{norm} \in \text{Aliases}$).
   - **Kademe 2 (Belirteç Sınırı / Token-Boundary Match):** Eğer tam eşitlik yoksa, sözlükteki kanonik terim aday metin içinde tam kelime sınırlarıyla (`\b... \b`) alt dize olarak geçiyorsa eşleşme kabul edilir.
   - **Uzunluk Önceliği:** Yanlış eşleşmeleri engellemek için alias listesi karakter uzunluğuna göre azalan sırada sıralanır; önce daha spesifik çok kelimeli terimler (ör. *"Kazanılmış Değer Analizi"*), sonra tek kelimelik terimler (ör. *"Analiz"*) kontrol edilir.

---

### 3.2 Zaman Aralığı ve Parçalama (Time Alignment & Windowing)

1. **Segment Eşleme Toleransı ($\Delta t$):**
   - Çıkarılan bir terimin konuşma içindeki konumu ile altyazı/ses zaman damgası arasındaki tolerans $\Delta t = \pm 2.0$ saniye olarak dondurulmuştur. Zaman örtüşmesi için Intersection-over-Union ($\text{IoU}$) eşiği $\ge 0.5$ şartı aranır.
2. **Çıkarım ve Arayüz Pencereleme:**
   - Web arayüzü (`gui/pipeline.py`) sabit, örtüşmesiz $30.0$ saniyelik pencereler (`CHUNK_S = 30.0`) kullanır.
   - Whisper modelleri tam ses üzerinde `sequential_generate` ile çalıştırılır; `max_new_tokens=440` olarak ayarlanır. Prompt sağlandığında `prompt_ids` bütçesi $448 - \text{prompt\_len} - 16$ değerine kırpılır.
   - Qwen3-ASR sabit 30 saniyelik, Nemotron 3.5 ise akışkan 60 saniyelik pencerelerle çalışır.
3. **Akustik VAD Müdahalesi Sınırlılığı:**
   - Enerji tabanlı VAD (`vad_chunker.py`) tüm kliplerde WER'i yükselttiği için (`TR` makro $0.0207 \rightarrow 0.0412$, `EN` $0.0101 \rightarrow 0.0262$) bu yöntem **iyileştirme değil, ölçülmüş olumsuz müdahale** olarak işaretlenmiştir.

---

### 3.3 Değerlendiriciler Arası Uyum: Cohen's Kappa ($\kappa$)

İki bağımsız etiketleyicinin ($A_1$ ve $A_2$) terimler üzerindeki ikili kararları `experiments/iaa.py` protokolüne göre değerlendirilir:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

Burada:
- $P_o$: Gözlenen göreli uyum oranı ($P_o = \frac{\sum \text{Uyum}}{N}$).
- $P_e$: Şans eseri beklenen rastgele uyum oranı ($P_e = \sum_{k \in \{0, 1\}} P(A_1 = k) \cdot P(A_2 = k)$).

**Etiketleme Boyutları:**
- **$R$ (Relevance - Bağlamsal İlgililik):** Çıkarılan terim, transkript bağlamında gerçekten alana özgü teknik bir kavram mıdır? ($R1=1$, $R0=0$).
- **$D$ (Definition - Tanım Doğruluğu):** Üretilen açıklama TDK (Türkçe) veya Oxford English Dictionary (İngilizce) sözlüklerine göre doğru mudur? ($D1=1$, $D0=0$).

**Sabitlenmiş Sayısal Doğrulama (115 Çift Etiketli Veri):**
- $R$ metriği: $P_o = 0.965$, $\kappa_R = \mathbf{0.782}$ (Yüksek ayrıştırıcı güvenilirlik, Landis & Koch "Önemli Düzeyde Uyum").
- $D$ metriği: $P_o = 1.000$, $\kappa_D = \mathbf{1.000}$ ($112$ pozitif, $3$ negatif etiket; yüksek sınıf dengesizliği nedeniyle $P_e \approx 0.95$ düzeyindedir. Bu nedenle $\kappa_D$ tek başına model kalitesinin kanıtı sayılmaz; daima $P_o$ ve $\kappa_R$ ile birlikte sunulmalıdır).

---

### 3.4 F1 Skoru (Hassasiyet, Duyarlılık ve F1 Tanımları)

1. **Kapalı Altın Sözlük F1 (TR-01 ve EN-01, $N=32$):**
   Modelin transkriptten çıkardığı terim kümesi $E$, altın terim kümesi $G$ olmak üzere:
   $$\text{Precision} (P) = \frac{|E \cap G|}{|E|}, \quad \text{Recall} (R) = \frac{|E \cap G|}{|G|}$$
   $$F_1 = \frac{2 \cdot P \cdot R}{P + R} \quad (P+R > 0 \text{ ise; aksi halde } 0.0)$$

2. **Açık Uçlu Çıkarım ve $R1\%$ Ayrımı:**
   Eğitim konuşmalarında terimsellik bir sürekliliktir. Kapalı altın listesi bulunmayan kliplerde (TR-02..16, EN-02..16) recall hesaplanamaz. Bu kliplerde $R1\%$, LLM'nin ürettiği terimlerin transkript bağlamında teknik bir kavram olup olmadığını ölçen bir **Kesinlik (Precision)** metriğidir. Modelin terim üretme hacmi ise üretilen toplam terim adediyle izlenir.

3. **Makro ve Mikro Toplama:**
   - **Makro $F_1$:** Kliplerin $F_1$ puanlarının ağırlıksız aritmetik ortalamasıdır.
   - **Mikro $F_1$:** Tüm kliplerdeki toplam $TP, FP, FN$ değerlerinin havuzlanarak tek bir oranda hesaplanmasıdır.

---

### 3.5 WER ve CER (Kelime ve Karakter Hata Oranları)

Levenshtein dinamik programlama hizalaması üzerinden hesaplanır (`experiments/metrics.py`):

$$\text{WER} = \frac{S + D + I}{N_{ref}}, \quad \text{CER} = \frac{S_c + D_c + I_c}{N_{ref\_char}}$$

Burada:
- $S$: Yerine koyma (Substitutions), $D$: Silme (Deletions), $I$: Ekleme (Insertions), $N$: Referans kelime/karakter sayısı.

**İki Ayrık Varyant:**
- **Ham (Raw) WER/CER:** Büyük/küçük harf, noktalama ve boşluklar korunarak birebir Levenshtein mesafesi.
- **Normalize (Normalized) WER/CER:** Küçük harfe dönüştürülmüş (Türkçe `I/ı` ve `İ/i` kuralları korunarak), tüm noktalama işaretleri temizlenmiş ve ardışık boşlukları teke indirilmiş metinler üzerinden hesaplanır. Raporlanan tüm akademik kıyaslamalar normalize WER kullanır.

---

### 3.6 Bootstrap Güven Aralıkları ($\text{CI}_{95\%}$)

Küçük örneklem boyutuna bağlı belirsizliği şeffafça belgelemek amacıyla parametrik olmayan Bootstrap yöntemi (`experiments/report.py` içindeki `bootstrap_ci`) uygulanır:

1. **Yeniden Örnekleme:** $N_{\text{boot}} = 10{,}000$ iterasyon boyunca, klip hata oranları kümesinden iadeli (with replacement) rastgele seçim yapılır.
2. **Ortalama Havuzu:** Her iterasyonda örneklem ortalaması $\bar{x}_b$ hesaplanır.
3. **Persentil Sınırları:** 
   $$\text{Alt Sınır} = \text{Percentile}(\{\bar{x}_b\}, 2.5\%), \quad \text{Üst Sınır} = \text{Percentile}(\{\bar{x}_b\}, 97.5\%)$$
4. **Kelime Ağırlıklı Mikro WER:** Kliplerin uzunluk farklarını (ör. TR-3'ün 1.583 kelime olması) yansıtmak için kelime ağırlıklı mikro WER bootstrap aralığıyla birlikte raporlanır.

---

## 4. Sınırlılık Listesi (Limitations & Guardrails)

Gelecekteki araştırma adımlarında hatalı veya abartılı iddiaların kurulmasını engellemek üzere aşağıdaki sınırlılıklar kesin olarak belirlenmiştir:

1. **Örneklem Kapsamı Sınırlılığı:**
   32 videoluk genişletilmiş liste şu aşamada bir **aday korpus (candidate corpus)** aşamasındadır. Diskte yalnızca 7 videonun (3 TR, 4 EN) yerel ses dosyası ve doğrulanmış altın referansı bulunmaktadır. Kalan 25 videonun sesleri yerel diskte yoktur ve transkriptleri taslak aşamasındadır.
2. **Dil Karşılaştırması Kısıtı ("Türkçe Daha Zordur" İddiasından Kaçınma):**
   $n=3$ Türkçe ve $n=4$ İngilizce klip üzerinden alınan sonuçlarda güven aralıkları geniştir ve formel bir hipotez/anlamlılık testi (significance test) uygulanmamıştır. Türkçe modellerdeki yüksek hata payı, incelenen spesifik videolardaki konuşma temposu, stüdyo/sınıf akustiği ve jargon yoğunluğundan kaynaklanıyor olabilir; bu durum "Türkçe ASR mimarileri için genel olarak İngilizceden daha zordur" şeklinde evrensel bir iddiaya dönüştürülemez.
3. **Anotasyon ve Hakem LLM Sınırlılığı (Leniency Riski):**
   Otomatik LLM hakemi (`experiments/llm_judge.py`) tarafından verilen $R1\%$ ve $D1\%$ puanları, hakemin insana göre esnek davranma (false-positive leniency) eğilimi nedeniyle **üst sınır (upper bound)** olarak okunmalıdır. Hakem tahminleri insan etiketleriyle doğrulanmadan mutlak doğruluk kabul edilemez.
4. **RAG Döngüselliği ve Veri Sızıntısı:**
   Terim çıkarma istemine altın test listesinden (`data/gold_terms.json`) besleme yapılması veri sızıntısına (leakage) yol açar. RAG kolu yalnızca test setinden tamamen bağımsız harici ders sözlükleri (`data/glossary_<CLIP>.json`) ile ölçülmelidir.
5. **Akustik Müdahalelerin Tabanı Geçememesi:**
   Enerji tabanlı VAD, altın liste biasing'i ve LLM post-correction kademeleri ham Whisper tabanının altına inememiştir. Düzeltme katmanı bir "kurtarma" mekanizması olarak sunulamaz.

---

## 5. Geçiş Ölçütü ve İzlenebilirlik Matrisi

> **Geçiş Ölçütü:** Her veri öğesinin kaynağı ve mevcut durumu izlenebilir olmalıdır.

| Veri Öğesi | Dosya / Veri Kaynağı | Fiziksel Durum / İzlenebilirlik | Doğrulama Yolu |
|---|---|---|---|
| **Birleşik Veri Envanteri** | `experiments/data/data_inventory.json` | 40 kayıt (20 TR + 20 EN), 4 durum boyutu, makine okunabilir JSON | `python -m json.tool experiments/data/data_inventory.json` |
| **Envanter CSV Tablosu** | `experiments/data/data_inventory.csv` | 40 satır x 19 sütun eksiksiz CSV | Excel / pandas okunabilir |
| **Aday Kaynak Manifesti** | `experiments/data/manifest_30plus.jsonl` | 32 akademik + 8 kurumsal manifest kaydı, YouTube ID ve konuşmacı profili | `python experiments/data/build_manifest_30plus.py` |
| **Orijinal Video Kaynakları** | `experiments/data/sources_30plus.json` | 40 video URL, kanal, süre ve alan eşlemesi | Web üzerinden erişilebilir doğrulanmış URL'ler |
| **Doğrulanmış 16 Yerel Ses** | `experiments/data/{TR-1..3, EN-1..4}.wav` + `experiments/data/audio/{TR-17..20, EN-11, EN-17..20}.wav` | 16 kHz Mono 16-bit PCM WAV (Toplam 4.64 saat süre doğrulamalı ses) | `ls -la experiments/data/*.wav` |
| **Doğrulanmış 7 Altın + 8 Kurumsal Transkript**| `experiments/data/gold/*.txt` + `outputs/transcripts/.../*.json` | 7 insan onaylı altın (13.467 kelime) + 8 tam ASR transkripti (QA onaylı) | `experiments/data/DATASET_CARD.md` |
| **Kapalı Altın Sözlük** | `experiments/data/gold_terms.json` | TR-1 ve EN-1 için 32'şer doğrulanmış kanonik terim | `experiments/eval_gold_terms.py` |
| **Çift Etiketli Anotasyonlar**| `experiments/annotation/labels_a1.csv`, `labels_a2.csv` | 115 ortak madde, $R$ ve $D$ etiketleri | `python experiments/iaa.py --a1 ... --a2 ...` |
| **40 Video Aday Şablonu** | `annotation/annotation_template_30plus.jsonl` | 160 aday terim kaydı (video başına 4 terim) | Anotasyon pilotu için hazır şablon |

---
**Sonuç:** Adım 1 başarıyla tamamlanmış; veri envanteri, deney protokolü tanımları ve sınırlılık listesi tam izlenebilirlikle sabitlenmiştir. Adım 2'ye ("Altın veri kümesini tamamlayın") geçiş için zemin hazırdır.
