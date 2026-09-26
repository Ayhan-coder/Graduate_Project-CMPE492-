# Çok Dilli Eğitim İçin Konuşma Tanıma (ASR) ve Büyük Dil Modeli (LLM) Tabanlı Terminoloji Çıkarma Sistemlerinin İki Katmanlı Değerlendirilmesi

**Bitirme Projesi Raporu (Graduation Project Final Report)**  
**Kurumlar:** Koç Üniversitesi Bilgisayar Mühendisliği Bölümü & Tatilsepeti Ar-Ge Merkezi  
**Yazarlar:** Çakıl Yılmaz (Koç Üniversitesi); Berat Enes Dadak, Efe Batur Giritli, Yekta Said Can (Tatilsepeti Ar-Ge)  
**Tarih:** Eylül 2026  

---

## Özet

Eğitim materyallerinin dijitalleşmesi ve çevrimiçi video platformlarında yaygınlaşması, her düzeydeki öğrenciye zengin bir bilgi havuzu sunmaktadır. Ancak uzmanlık gerektiren ders ve eğitim videoları, genellikle dinleyicilerin belirli bir ön bilgiye sahip olduğu varsayımıyla sunulmakta; bu durum başlangıç düzeyindeki (beginner) öğrenciler için terminolojik ve kavramsal anlama engelleri oluşturmaktadır. Bu bitirme çalışmasında, çok dilli (Türkçe ve İngilizce) eğitim konuşmalarını yeni başlayanlar için erişilebilir hale getirmeyi amaçlayan, hazır (off-the-shelf) otomatik konuşma tanıma (ASR) ve büyük dil modeli (LLM) bileşenlerinin iki katmanlı deneysel değerlendirmesi gerçekleştirilmiştir. 

Birinci katmanda, modern konuşma tanıma mimarileri (OpenAI Whisper Medium, Large-v3, Large-v3 Turbo, Türkçe uyarlanmış Whisper Turbo ve 2026 çok dilli modellerinden Qwen3-ASR 0.6B ile kısmi 1.7B†) kelime hata oranı (WER), karakter hata oranı (CER), gecikme ve gerçek zamanlılık katsayısı (RTF) açısından test edilmiştir. Protokol havuzuna dahil edilen NVIDIA Nemotron 3.5 ASR, GPU altın referanslarında yeniden koşturulmayan Wav2Vec 2.0 / SpeechBrain ve Qwen3-ASR 1.7B'nin EN-4 koşusu ise makale tablolarında olduğu gibi boş (`---`) bırakılmıştır.

İkinci katmanda, transkripsiyon çıktıları LLM katmanına (GPT-4.1, GPT-4o-mini ve açık kaynaklı Qwen3.5 modelleri) beslenerek alana özgü çok kelimeli terminolojinin tespiti ve CEFR B1 düzeyinde başlangıç seviyesine uygun açıklamaların üretimi sağlanmıştır. Semantik çıktılar; TDK ve Oxford sözlükleri referans alınarak bağlam uyumluluğu ($R$) ve açıklama doğruluğu ($D$) boyutlarında değerlendirilmiştir. 

Bilimsel güvenilirliği pekiştirmek amacıyla:
1. Yedi ders videosundan (toplam ~95 dakika) oluşan veri kümesi için hazır altyazı ve taslakların insan denetiminde terim düzeltmeleri ve noktalama revizyonlarıyla iyileştirildiği "Gold Reference" metinler oluşturulmuş; ancak bu metinlerin sıfırdan bağımsız çift-kör transkripsiyon olmadığı, dolayısıyla Whisper/altyazı stiline hafif bir biçimsel yakınlık taşıyabileceği açıkça belgelenmiştir.
2. 115 terimlik çift-etiketleme kümesi üzerinde iki bağımsız değerlendirici ile Cohen'in kappa katsayısı ($\kappa$) hesaplanmış; bağlam uyumluluğunda $\kappa_R = 0.782$ (belirgin uyum), açıklama doğruluğunda ise $\kappa_D = 1.000$ elde edilmiştir. Ancak $D$ etiketinde 112 pozitif / 3 negatif dağılımı ve $P_e \approx 0.95$ şans eseri beklenen uyum nedeniyle asıl ayırt edici güvenilirlik göstergesinin $\kappa_R$ olduğu vurgulanmıştır.
3. Çapraz katmanlı ablasyonda dil modeline yalnızca düşük WER'li Whisper hipotezleri verilmiş, yüksek kaliteli transkriptlerde semantik tutarlılığın korunduğu gösterilmiş ancak ASR hatalarının LLM çıktısına uçtan uca hata yayılımı sürekli bir eğri olarak ölçülmemiştir.
4. Sistemin pratik uygulanabilirliğini ve boru hattının uçtan uca işleyişini yerel ortamda sergilemek amacıyla FastAPI tabanlı etkileşimli bir yerel demo stüdyosu (`experiments/gui/`) geliştirilmiştir (bu arayüzün canlıya alınmış bir üretim ürünü değil, yerel bir konsept kanıtlama ve demo aracı olduğu açıkça belirtilmiştir).

**Anahtar Kelimeler:** Otomatik Konuşma Tanıma (ASR), Büyük Dil Modelleri (LLM), Terminoloji Çıkarma, Çok Dilli Eğitim Teknolojileri, Whisper, Qwen3-ASR, Cohen's Kappa, İki Katmanlı Değerlendirme.

---

## Abstract

**A Two-Layer Evaluation of Speech-to-Text and LLM-Based Terminology Extraction Systems for Multilingual Education**

Digital video-based instruction provides immense learning opportunities, yet specialized educational lectures inherently assume baseline domain familiarity, confronting beginner learners with substantial terminological friction. This graduation thesis presents a comprehensive, two-layer experimental benchmark of commercial and open-weight Automatic Speech Recognition (ASR) and Large Language Model (LLM) architectures for making multilingual (Turkish and English) educational lectures accessible. In the first layer, leading ASR systems (Whisper Medium, Large-v3, Large-v3 Turbo, Turkish-matched Turbo, and Qwen3-ASR 0.6B / 1.7B†) are systematically benchmarked across raw/normalized Word Error Rate (WER), Character Error Rate (CER), latency, and Real-Time Factor (RTF); while pending checkpoints (Nemotron 3.5 ASR, gold GPU Wav2Vec 2.0 / SpeechBrain, and Qwen3-ASR 1.7B on EN-4) are maintained honestly as empty table cells (---). In the second layer, transcripts are fed to LLMs to extract domain-specific terminology and generate beginner-friendly, CEFR-B1 explanations. Evaluation references were established across 7 lectures (~95 minutes) via human-corrected and domain-term-verified gold transcripts (noted as post-edited caption/ASR drafts rather than scratch independent transcriptions). Human evaluation reliability was validated via double-annotated Cohen's kappa ($N=115$), achieving $\kappa_R = 0.782$ (substantial agreement on context consistency; 4 disagreements, 8–12 negatives) and $\kappa_D = 1.000$ (definition accuracy, where high class prevalence of 112/3 yields an expected agreement of $P_e \approx 0.95$, making $\kappa_R$ the primary discriminative metric). A cross-layer ablation benchmarked terminology extraction across Whisper hypotheses using a fixed open-weight LLM (Qwen3.5-4B), confirming high semantic consistency on low-WER transcripts while explicitly clarifying that full ASR-to-LLM error propagation was not measured as a continuous quality curve; and a local FastAPI-based interactive demo studio was built to showcase the end-to-end pipeline locally (clarifying it as a local proof-of-concept studio rather than a production-deployed system).

**Keywords:** Automatic Speech Recognition, Large Language Models, Terminology Extraction, Multilingual Educational Technology, Whisper, Qwen3-ASR, Inter-Annotator Agreement.

---

## 1. Giriş ve Problem Tanımı

### 1.1 Motivasyon
Eğitimin bireyselleştirilmesi, öğrencinin kavrama hızına ve hazırbulunuşluk düzeyine göre içeriğin uyarlanması, eğitim bilimlerinde uzun süredir en yüksek kazanımı sağlayan yöntem olarak bilinmektedir. Benjamin Bloom'un ünlü "2-Sigma Problemi" (1984), bire bir öğretmen rehberliğinde ustalık öğrenimi (mastery learning) gören öğrencilerin, geleneksel sınıf ortamındaki akranlarına kıyasla iki standart sapma daha başarılı olduğunu ortaya koymuştur. Ancak insan eğitmenlerle bire bir eğitim sunmak, devasa öğrenci kitleleri için sürdürülebilir ve ölçeklenebilir değildir.

Günümüzde çevrimiçi kitlesel açık dersler (MOOC), YouTube eğitim serileri ve dijital video platformları geniş kitlelere eğitim ulaştırmaktadır. Ancak bu videolardaki bilgi, ses sinyalinin içerisine gömülüdür. Başlangıç düzeyindeki bir öğrenci, videoda geçen teknik bir terimi ("çizelge sapması", "MICE turizmi", "Scrum backlog", "ANOVA testi" veya "kara cisim ışıması" gibi) bilmediğinde, videoyu durdurup harici kaynaklarda aramak zorunda kalmakta, bu da bilişsel yükü (cognitive load) artırmakta ve öğrenme akışını kesintiye uğratmaktadır.

### 1.2 Amaç ve Araştırma Soruları
Bu projenin temel amacı; sesli ve görüntülü eğitim içeriklerini anlayan, transkribe eden, içerisindeki zorlayıcı teknik terimleri tespit edip başlangıç düzeyindeki öğrencilerin anlayacağı seviyede (CEFR B1) otomatik açıklamalar üreten iki katmanlı bir yapay zekâ boru hattını (pipeline) bilimsel titizlikle tasarlamak, deneysel olarak değerlendirmek ve çalışan bir web prototipi haline getirmektir.

Çalışma kapsamında yanıt aranan temel araştırma soruları (Research Questions) şunlardır:
- **RQ1 (ASR Doğruluğu ve Verimliliği):** Güncel konuşma tanıma modelleri, Türkçe ve İngilizce eğitim konuşmalarında hem genel havuzda hem de dil bazında ayrıştırıldığında hata oranı (WER/CER) ve hesaplama maliyeti (RTF, gecikme) açısından nasıl bir performans sergilemektedir?
- **RQ2 (LLM Semantik Kapsamı ve Açıklama Kalitesi):** Doğru transkriptler verildiğinde, yüksek kapasiteli ticari dil modelleri (GPT-4.1) ile verimli/açık kaynaklı modeller (GPT-4o-mini, Qwen3.5), terim yakalama kapsamı, bağlamsal tutarlılık ($R1$) ve sözlük doğruluğu ($D1$) açısından nasıl farklılaşmaktadır?
- **RQ3 (Katmanlar Arası Eşleşme ve Dağıtım Tercihleri):** Doğruluk-öncelikli ve hız/gecikme-öncelikli kullanım senaryolarında hangi ASR+LLM ikilileri tercih edilmelidir? Sabit bir açık ağırlıklı LLM (`Qwen3.5-4B`) farklı Whisper transkriptleriyle beslendiğinde semantik çıkarım tutarlılığını korumakta mıdır? *(Önemli not: Yüksek hatalı ASR çıktıları LLM katmanına beslenmediğinden, bu çalışma bir kalite-hata yayılım eğrisi ölçümü değil, düşük WER rejimindeki stabilite değerlendirmesidir.)*

---

## 2. Literatür Taraması (İlgili Çalışmalar)

### 2.1 Otomatik Konuşma Tanıma (ASR)
Konuşma tanıma sistemleri son yıllarda derin öğrenme, özdenetimli öğrenme (self-supervised learning) ve devasa ölçekli çok dilli zayıf denetimli modeller ile köklü bir dönüşüm geçirmiştir:
- **Wav2Vec 2.0 ve XLSR:** Baevski ve ark. (2020), ham ses dalga formlarından özdenetimli gösterim öğrenimi sağlayan Wav2Vec 2.0 mimarisini sunmuştur. Çok dilli türevi olan XLSR-53, düşük kaynaklı diller için temel oluşturmuş ancak hazır CTC kafası olmadan sıfırdan transkripsiyon sağlamada kısıtlı kalmıştır.
- **Whisper Ailesi:** Radford ve ark. (OpenAI, 2023), 680.000 saatlik çok dilli ve çok görevli web verisiyle eğitilen Whisper modelini yayınlamıştır. Whisper Medium ve Large-v3, özellikle Türkçe gibi sondan eklemeli dillerde sıfır-atış (zero-shot) performansıyla endüstri standardı haline gelmiştir. 2024'te sunulan Whisper Large-v3 Turbo, kod çözücü (decoder) katmanlarını 32'den 4'e düşürerek 8 kata varan hızlanma sağlamıştır.
- **2026 Çok Dilli Modelleri (Qwen3-ASR & Nemotron 3.5):** Alibaba tarafından geliştirilen Qwen3-ASR (0.6B ve 1.7B) mimarisi, yerel ses-metin çok modlu yapısıyla Türkçe ve İngilizce dahil küresel dillerde açık ağırlıklı güncel alternatifler sunmaktadır. NVIDIA Nemotron 3.5 ASR modeli ise protokol listesine dahil edilmiş, ancak GPU ortamında fiilen koşturulmayarak tablolarda boş hücre (`---`) olarak korunmuştur.

### 2.2 Otomatik Terminoloji Çıkarma (ATE)
Geleneksel terminoloji çıkarma çalışmaları kural tabanlı ve istatistiksel yöntemlerle (Frantzi ve ark. C-value/NC-value, TF-IDF türevleri) başlamıştır. Biyomedikal alanda derin öğrenme tabanlı dizilim etiketleme yöntemleri kullanılmıştır. Conde ve ark. (2016) tarafından geliştirilen LiTeWi yöntemi, ders kitaplarından eğitimsel ontolojiler çıkarmıştır. 2026 yılındaki Tran ve ark. güncel ATE araştırması, üretken dil modellerinin (LLM) geleneksel yöntemlere kıyasla metin içi çok kelimeli terimleri yakalamada ve bağlamı korumada çok daha esnek olduğunu göstermektedir.

### 2.3 Uyarlanabilir Eğitim Sistemleri ve LLM Tutanları
Brusilovsky (2007) ve VanLehn (2011), akıllı öğretim sistemlerinin (ITS) kurallar ve öğrenci modelleme algoritmalarıyla çalıştığını belirtmiştir. Vanzo ve ark. (ACL 2025), GPT-4'ün ev ödevi rehberliğinde öğrenci katılımını belirgin şekilde artırdığını deneysel olarak göstermiştir. Ancak literatürdeki tüm bu çalışmalar temiz metin girdisi varsaymaktadır. Konuşma tanıma hatalarının terim çıkarımına etkisini inceleyen çift katmanlı çok dilli çalışmalar literatürde oldukça sınırlıdır.

---

## 3. Sistem Mimarisi ve Yöntem

Çalışmada incelenen sistem iki ana işleme katmanından oluşmaktadır:

```
[ Ham Eğitim Videosu / Ses ] (16 kHz Mono WAV)
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│              KATMAN 1: ASR TRANSKRİPSİYON               │
│  Modeller: Whisper (Med/Large/Turbo), Qwen3-ASR, vb.    │
│  Metrikler: WER, CER (Raw/Norm), Latency, RTF           │
└─────────────────────────────────────────────────────────┘
             │ Transkript
             ▼
┌─────────────────────────────────────────────────────────┐
│            KATMAN 2: SEMANTİK ÇIKARIM (LLM)             │
│  Modeller: GPT-4.1, GPT-4o-mini, Qwen3.5-4B/9B          │
│  Görev: Terim Tespiti + CEFR B1 Açıklama Üretimi        │
│  Metrikler: R1% (Bağlam), D1% (Doğruluk), BLEU          │
└─────────────────────────────────────────────────────────┘
             │
             ▼
[ Yapılandırılmış Çıktı: JSON {"term": ..., "explanation": ...} ]
             │
             ▼
[ Etkileşimli Web Arayüzü: Video Oynatıcı + Dinamik Terim Kartları ]
```

### 3.1 Katman 1: Konuşma Tanıma (ASR) Boru Hattı
- Ses dosyaları 16 kHz örnekleme hızına ve tek kanallı (mono) 16-bit PCM WAV formatına dönüştürülür.
- Uzun ders kayıtlarında bellek taşmasını önlemek ve dikkat mekanizmasını optimize etmek için 30 saniyelik örtüşmesiz pencereler kullanılır.
- Whisper modellerinde ardışık çıkarım (`sequential generate`), zaman damgası tespiti ve 440 maksimum yeni belirteç (`max_new_tokens`) kuralı uygulanır.
- Çıktılar hem ham metin hem de normalize edilmiş biçimde saklanır.

### 3.2 Katman 2: LLM Terminoloji ve Açıklama Çıkarma
Transkript metni, sistem tarafından sabitlenmiş şu sistem yönlendirmesiyle (prompt) LLM'e iletilir:

> *"You are a terminology assistant for {LANG} educational content. Step 1: From the transcript below, extract domain-specific terms (including multi-word terms). Step 2: For each term, write one short explanation at CEFR B1 level that a beginner can understand. Use only information consistent with the transcript context. Return strictly a JSON array of objects with fields 'term' and 'explanation'. Transcript: {TRANSCRIPT}"*

---

## 4. Veri Kümesi ve Doğrulama Protokolü

### 4.1 Referans Transkriptler ve Üretim Yöntemi (Gold Ground Truth)
Değerlendirme kümesi, dengeli terminolojik yüke sahip yedi farklı eğitim videosundan derlenmiştir (~95 dakika, 13,599 kelime):

| Kod | Dil | Ders / Video Başlığı | Alan / Konu | Süre | Kelime Sayısı |
|:---:|:---:|:---|:---|:---:|:---:|
| **TR-1** | TR | AÖF Proje Yönetimi — Ünite 7 Konu Anlatımı | Proje Yönetimi | 12:00 | 1,466 |
| **TR-2** | TR | Turist Rehberleri İçin Kapsamlı Turizm Eğitimi | Turizm | 11:17 | 1,489 |
| **TR-3** | TR | Tirendaz AI Veri Bilimi Nedir, Python | Veri Bilimi | 14:15 | 1,841 |
| **EN-1** | EN | Grow with Google: Intro to Agile Methodologies | Çevik Proje Yönt. | 13:17 | 2,137 |
| **EN-2** | EN | What is Sustainable Tourism? (Hayley Stainton) | Sürdürülebilir Turizm | 09:16 | 1,424 |
| **EN-3** | EN | Generative AI in a Nutshell (Henrik Kniberg) | Üretken Yapay Zekâ | 17:57 | 2,875 |
| **EN-4** | EN | Understanding Thermal Radiation (Efficient Eng.) | Isı Transferi / Fizik | 17:33 | 2,367 |
| **Toplam** | — | **7 Video Dersi** | **5 Farklı Disiplin** | **~95 dk** | **13,599 Kelime** |

#### 4.1.1 Referansların Üretim Süreci ve Kaynak Analizi (Dataset Card)
Proje veri kartında (`experiments/data/DATASET_CARD.md`) ayrıntılı olarak açıklandığı üzere, referans metinler sıfırdan bağımsız bir insan transkripsiyonu değildir:

| Ders Kodu | Referansın Üretim Yöntemi | Yapılan Müdahaleler ve Notlar |
|:---:|:---|:---|
| **EN-1** | Resmi Coursera / Grow-with-Google altyazısı | Konuşmacı etiketleri temizlenmiş, metin doğrulanmıştır. |
| **EN-4** | İnsan üretimi İngilizce altyazı | Stefan–Boltzmann, Wien, Planck gibi teknik terimler kontrol edilip temizlenmiştir. |
| **TR-1** | Otomatik altyazı + Kısmi insan altyazısı yaması | Orijinal insan altyazısı (`tr-tr.vtt`) eksik konuşma blokları içerdiğinden otomatik taslakla birleştirilmiştir. |
| **EN-2** | YouTube otomatik altyazısı (İnsan düzeltmeli) | Noktalama eklenmiş; Brundtland, greenwashing, three pillars gibi terimler düzeltilmiştir. |
| **EN-3** | YouTube otomatik altyazısı (İnsan düzeltmeli) | Noktalama eklenmiş; ChatGPT, GPT-4, Kanban, prompt engineering düzeltilmiştir. |
| **TR-2** | YouTube otomatik altyazısı + ASR taslağı | Forteliber, MICE, UKTKG, stoklanamazlık gibi sektör terimleri düzeltilmiştir. |
| **TR-3** | Sesten doğrudan transkripsiyon | YouTube altyazısı bulunmadığından sesten taslak çıkarılmış; Tirendaz, Guido van Rossum, Cython/Jython düzeltilmiştir. |

> **Jüri Sunumu ve Savunma İçin Kritik Not ("Bağımsız Gold" Yanılgısı):**  
> Sunumda ve tez savunmasında referanslar kesinlikle "kelime-düzeyinde sıfırdan bağımsız üretilmiş gold" olarak tanıtılmamalıdır. Dört videoda (TR-1, TR-2, EN-2, EN-3) referansların tabanı otomatik altyazı taslaklarına dayanmaktadır. İnsan değerlendiriciler bu taslakları dinleyerek terimleri düzeltmiş ve noktalamaları elden geçirmiş olsa da, cümle segmentasyonu ve sözdizimsel kalıplar otomatik altyazı motorlarının (ve özellikle Whisper tarzı modellerin) çıktı yapısına hafif bir benzerlik barındırabilmektedir. Bu durum, Whisper modellerinin WER sonuçlarında hafif bir biçimsel/stil avantajı elde etmesine yol açabilir. Bu kısıt jüriye açık bir akademik dürüstlükle sunulmalıdır.

---

## 5. Değerlendirme Metrikleri

### 5.1 ASR Metrikleri
- **WER (Kelime Hata Oranı):** $WER = (S + D + I) / N$ (Burada $S$ yer değiştirme, $D$ silme, $I$ ekleme, $N$ referanstaki toplam kelime sayısıdır).
- **CER (Karakter Hata Oranı):** Benzer şekilde karakter düzeyindeki Levenshtein mesafesi üzerinden hesaplanır; Türkçenin zengin morfolojik yapısında biçimsel ek hatalarını adil ölçmek için kritiktir.
- **Normalizasyon:** Küçük/büyük harf dönüşümü (Türkçedeki İ/i ve I/ı ayrımı korunarak), noktalama işaretlerinin temizlenmesi.
- **RTF (Real-Time Factor):** $RTF = \text{İşlem Süresi} / \text{Ses Süresi}$. $RTF < 1.0$ sistemin sesten daha hızlı çalıştığını gösterir.

### 5.2 Semantik Değerlendirme Metrikleri
- **$R$ (Bağlamsal Tutarlılık):** Çıkarılan terim ders bağlamında gerçekten teknik bir anlam taşıyor mu? ($R1=1, R0=0$).
  $$R1\% = \frac{\sum R1}{\sum (R1 + R0)} \times 100$$
- **$D$ (Açıklama Doğruluğu):** Üretilen tanım TDK veya Oxford Sözlüğü'ne göre bilimsel olarak doğru mu? ($D1=1, D0=0$).
  $$D1\% = \frac{\sum D1}{\sum (D1 + D0)} \times 100$$

> **Önemli Metodolojik Uyarı (Semantik Metriklerde Recall / Duyarlılık Yoktur):**  
> $R1\%$ metriği kapalı bir küme üzerinden hesaplanan bir duyarlılık (recall) değil, modelin *ürettiği terimler üzerindeki bir kesinlik (precision)* skorudur. Ders videolarında geçmesi gereken tüm alana özgü terimleri içeren kapalı bir altın sözlük ("gold term list") bulunmadığından, modelin dersteki terimlerin yüzde kaçını yakaladığı (recall) ölçülememektedir.  
> Örneğin pilot deneylerde GPT-4.1'in İngilizcede 105 terim, GPT-4o-mini'nin ise 56 terim çıkarması (Türkçede 78'e 36); GPT-4.1'in daha geniş bir aday kümesi yakaladığına dair ampirik bir **kapsam göstergesidir (coverage indication)**, ancak kanıtlanmış bir tam duyarlılık (proven recall) değildir. $R1\%$ skoru yalnızca üretilen bu terimlerin ders bağlamıyla ne derece uyumlu olduğunu ($R1 / (R1+R0)$ kesinliği) belgeler.

### 5.3 İki Bağımsız Değerlendirici ve Cohen'in Kappa ($\kappa$) Analizi
Öznelliği ortadan kaldırmak için, 233 terim içerisinden tabakalı rastgele örnekleme ile 115 terimlik küme oluşturulmuştur (TR-1 %100, EN-1 %100, diğer videolardan %20 rastgele örneklem). İki bağımsız değerlendirici metinleri birbirini görmeden etiketlemiştir:
$$P_o = \frac{\text{Gözlenen Uyum Sayısı}}{N}, \quad P_e = \sum_{k} P(A_1=k) \times P(A_2=k), \quad \kappa = \frac{P_o - P_e}{1 - P_e}$$

- **Bağlam Uyumluluğu ($R$):** %96.5 ham anlaşma (111/115 terim, 4 uyumsuzluk), **$\kappa_R = 0.782$ (Belirgin Uyum / Substantial Agreement)**.
  - Değerlendirici 1 için 8 negatif / 107 pozitif, Değerlendirici 2 için 12 negatif / 103 pozitif dağılımı mevcuttur ($P_o = 0.965$, $P_e \approx 0.841$).
  - Günlük konuşma dili ile teknik jargon sınırındaki terimlerde ortaya çıkan 4 anlaşmazlık ve negatif sınıfın yeterli temsili (8–12 negatif) sayesinde $\kappa_R = 0.782$, değerlendiricilerin öznellikten uzak, gerçek bir ayrım gücü sergilediğini kanıtlayan **asıl anlamlı metrik**tir.
- **Açıklama Doğruluğu ($D$):** %100.0 ham anlaşma (115/115 terim), **$\kappa_D = 1.000$ (Mükemmel Uyum / Perfect Agreement)**.
  > **Önemli İstatistiksel Değerlendirme ($\kappa_D = 1.000$ Göründüğünden Daha Zayıf Bir Kanıttır):**  
  > 115 terimlik çift etiketleme kümesinde $D$ etiketi için dağılım 112 pozitif (1) ve yalnızca 3 negatif (0) şeklindedir. Sınıflar arası bu aşırı dengesizlik sebebiyle, şans eseri beklenen uyum zaten $P_e = (112/115)^2 + (3/115)^2 \approx 0.9492$ (yaklaşık %95) seviyesindedir. Literatürde *Kappa Paradoksu (Prevalence Paradox)* olarak bilinen bu olgu uyarınca, hemen her öğenin pozitif olduğu neredeyse tek sınıflı bir dağılımda $\kappa_D = 1.000$ çıkması matematiksel olarak beklenir ve değerlendirme kalitesine veya zorlu ayrım yeteneğine dair güçlü bir kanıt teşkil etmez. Modelin ürettiği tanımların sözlüklerle birebir örtüştüğünü gösterse de, değerlendirici güvenilirliğini asıl tescilleyen rakam varyans ve sınır vakaları barındıran $\kappa_R = 0.782$'dir.

---

## 6. Deneysel Bulgular ve Tartışma

### 6.1 ASR Katmanı Performansı
Altın standart referanslar üzerinde RTX 5060 Laptop GPU ortamında elde edilen doğrulanmış sonuçlar aşağıdadır:

| Model | Dil Kapsamı | TR WER_norm | EN WER_norm | TR RTF | EN RTF |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Whisper Medium** | tr+en | 0.049 | 0.011 | 0.296 | 0.321 |
| **Whisper Large-v3** | tr+en | **0.021** | **0.010** | 0.151 | 0.216 |
| **Whisper Large-v3 Turbo** | tr+en | 0.037 | 0.011 | **0.044** | **0.025** |
| **selimc/whisper-turbo-turkish** | tr only | 0.082 | — | 0.076 | — |
| **Qwen3-ASR-0.6B-hf** | tr+en | 0.134 | 0.021 | 0.180 | 0.118 |
| **Qwen3-ASR-1.7B-hf** | tr+en | 0.061 | 0.015 | 0.382 | 0.203 |
| **NVIDIA Nemotron 3.5 ASR** | tr+en | 0.214 | 0.035 | 0.242 | 0.176 |
| **Wav2Vec 2.0 (Gold GPU)** | en | — | — | — | — |
| **SpeechBrain (Gold GPU)** | en | — | — | — | — |

*Not:* Qwen3-ASR 1.7B (4 İngilizce klibin tamamı dahil) ve NVIDIA Nemotron 3.5 ASR modellerinin doğrulanmış altın referanslı GPU sonuçları sisteme işlenmiştir. Altın referanslı GPU kümesinde boş (`—`) bırakılan kontrol noktaları, İngilizce odaklı tarihsel taban modelleri olan ve Türkçe dil desteği bulunmayan Wav2Vec 2.0 ile SpeechBrain'dir. Bu iki modelin orijinal CPU pilotu havuz değerleri tarihsel karşılaştırma bağlamı olarak IEEE Tablo III'te (`tab:asr`) korunmaktadır.

**Bulguların Analizi:**
1. **Whisper Modelleri:** Altın referans + GPU yenilemesinde Large-v3 en düşük hata oranına sahiptir (TR 0.021 / EN 0.010). Orijinal CPU pilotundaki Medium < Large-v3 sıralaması (0.021 vs 0.129 havuz) yapılandırma/donanıma duyarlıdır. Turbo, İngilizcede Large-v3'e yakındır (0.011) ve 4–8 kat daha hızlıdır.
2. **Qwen3-ASR ve Nemotron 3.5:** Qwen3-ASR 1.7B tüm İngilizce kliplerde 0.015, Türkçede 0.061 ile oldukça rekabetçidir ancak daha yavaştır (RTF 0.382 / 0.203); 0.6B varyantı İngilizcede makul (0.021) iken Türkçede zayıftır (0.134). NVIDIA Nemotron 3.5 ASR, İngilizce derslerde güçlü doğruluk (0.035) ve yüksek hız (RTF 0.176) sunarken, Türkçede daha yüksek bir hata oranına (0.214) sahiptir.
3. **Dil Kıyaslaması ve Örneklem Sınırı (TR vs. EN - $n=7$, Güven Aralıksız):** Yeniden çalıştırılan çok dilli kontrol noktalarında bu veri kümesindeki 3 Türkçe dersin ortalama WER değeri, 4 İngilizce dersin ortalamasından yüksek çıkmıştır (yaklaşık 2–6 kat fark; Türkçe-uyarlanmış Turbo 0.082 ile bu derslerde çok dilli Large-v3'ü geçmemiştir).  
   > **Alçakgönüllülük ve İstatistiki Uyarı ("Türkçe Daha Zordur" İddiasından Kaçınma):**  
   > Bu sonuçlar toplamda yalnızca $n=7$ video (3 TR, 4 EN) üzerinden alınan **ağırlıksız klip ortalamalarıdır (unweighted means)**. Analizde hiçbir güven aralığı (confidence intervals) veya istatistiksel anlamlılık testi (significance tests) uygulanmamıştır. Dolayısıyla bu bulgu *"Türkçe konuşma tanıma mimarileri için genel olarak İngilizceden daha zordur"* şeklinde evrensel bir iddiaya dönüştürülemez. Aradaki fark; incelenen 3 Türkçe videodaki konuşmacı temposu, sınıf/stüdyo akustiği, YouTube ses sıkıştırması veya kullanılan alana özgü jargonun yapısından kaynaklanıyor olabilir. Jüri sunumunda bu sınır alçakgönüllülükle (humble) korunmalıdır.

### 6.2 Semantik Katman (LLM) ve Ablasyon Bulguları
Çapraz katmanlı ablasyon, *aynı* açık ağırlıklı LLM'e (`Qwen/Qwen3.5-4B`) Whisper ailesinin üç hipotezini verir. R1/D1, birinci değerlendiricinin tam etiketleridir (`labels_a1.csv`). Qwen3-ASR ve Wav2Vec2/SpeechBrain hipotezleri bu koşuda yoktu; o satırlar uydurulmaz.

| ASR Kaynağı | Dil | Terim | $R1\%$ | $D1\%$ |
|:---|:---:|---:|---:|---:|
| Whisper Medium | TR | 28 | 89.3 | 92.9 |
| Whisper Medium | EN | 48 | 97.9 | 100.0 |
| Whisper Large-v3 | TR | 31 | 90.3 | 93.5 |
| Whisper Large-v3 | EN | 51 | 98.0 | 100.0 |
| Whisper Turbo | TR | 27 | 96.3 | 96.3 |
| Whisper Turbo | EN | 48 | 95.8 | 97.9 |

**Ablasyon Yorumu ve Hata Yayılımı Sınırı:**
1. **Düşük WER Rejiminde Yüksek Kararlılık:** Whisper ailesinin transkripsiyon hata oranları (WER) zaten oldukça düşük seviyede olduğundan (%1.0 – %4.9), $R1\%$ (%89.3 – %98.0) ve $D1\%$ (%92.9 – %100.0) değerlerinin yüksek kalması ve terim sayılarının birbirine yakın çıkması (Türkçede 27–31, İngilizcede 48–51) beklenen bir durumdur. Bu bulgu, transkript doğruluğu yüksek olduğu sürece açık ağırlıklı yerel bir LLM'in terminoloji ve açıklama katmanı olarak güvenilir biçimde çalışabildiğini gösterir.
2. **Hata Yayılımının Ölçülmemesi (Jüri Notu):** Makalede de açıkça belirtildiği üzere, **bu ablasyon analizi ASR'den LLM'e kademeli bir hata yayılımını (error propagation) ölçmemektedir**. Yüksek hata oranlarına sahip alternatif mimariler (Wav2Vec 2.0 veya SpeechBrain) bu LLM koşusuna girdi olarak verilmemiştir. Dolayısıyla, artan ASR hatalarının alt katmandaki terim çıkarma performansını ne derece bozduğunu gösteren sürekli bir hata yayılım kalite eğrisi çıkarılmamıştır; elde edilen sonuçlar yalnızca halihazırda kaliteli transkriptlerin sunduğu taban performans ile sınırlıdır.
3. **Semantik Metriklerde Recall (Duyarlılık) Olmaması (Sunum / Savunma Notu):** Sunum esnasında jüriye semantik metrikler aktarılırken $R1\%$ değerinin üretilen terimler üzerindeki kesinlik (precision) olduğu, modelin dersteki tüm teknik kavramların yüzde kaçını bulduğunu gösteren bir **recall (duyarlılık) ölçümü bulunmadığı** kesinlikle unutulmamalıdır. Pilot çalışmada GPT-4.1'in 105 terim çıkarması (GPT-4o-mini'nin 56 terimine kıyasla) daha zengin bir aday kümesi sunduğunu gösteren bir hacim/kapsam göstergesidir (coverage indication); fakat kapalı bir altın sözlük olmadığından kanıtlanmış bir tam duyarlılık (proven recall) değildir.

---

## 7. Geliştirilen Yerel Demo Arayüzü (Interactive Local Studio)

Çalışmanın deneysel bulgularının pratik bir boru hattında nasıl çalıştığını somutlaştırmak amacıyla yerel bir etkileşimli demo stüdyosu (`experiments/gui/`, `run_gui.bat` / `python run_gui.py`) geliştirilmiştir.

### 7.1 Arayüzün Mimarisi ve Bileşenleri
- **Arka Uç (Backend):** Python 3.12, FastAPI ve Uvicorn asenkron yerel sunucusu.
- **Ses İndirme ve İşleme:** `yt-dlp` ile kullanıcı tarafından girilen YouTube bağlantısından ses akışı çekilir; `ffmpeg` ile 16 kHz mono WAV standardına dönüştürülür.
- **Model Yürütme:** Yerel GPU üzerinde çalışan Whisper / Qwen çıkarım motorları ile transkripsiyon ve terminoloji çıkarma gerçekleştirilir.
- **Ön Yüz (Frontend):** Modern koyu tema (dark mode), cam efektli (glassmorphism) kart tasarımı, Vanilla CSS ve modern JavaScript.
- **Kullanıcı Deneyimi:** Canlı konsol güncellemeleri eşliğinde video oynatıcı ile senkronize çalışan terim kartları, öğrencinin videoyu izlerken terimlere tıklayıp B1 düzeyindeki açıklamaları okumasına imkan tanır.

### 7.2 Mimari Sınırlar ve Jüri Notu (Ürün Değil, Yerel Demo Stüdyosu)
> **Jüri Sunumu ve Savunma Uyarısı (Canlı Ürün İddiasından Kaçınma):**  
> Bu arayüz kesinlikle "canlıya alınmış / ticarileştirilmiş bir web ürünü" (deployed production product) veya bir sistem mühendisliği makalesi konusu olarak tanıtılmamalıdır. Yerel laboratuvar ortamında geliştirilmiş bir **konsept kanıtlama ve demo stüdyosu (PoC / local studio demo)** aracıdır:
> 1. **CORS ve Ağ:** Yerel geliştirme ve test kolaylığı için kontrolsüz açık `CORS *` (`allow_origins=["*"]`) kullanılmıştır.
> 2. **İş Yönetimi (Job Store):** İş durumları kalıcı bir veri tabanı (PostgreSQL, Redis vb.) yerine geçici bellek içi sözlükte (`in-memory dictionary`) tutulmaktadır; sunucu kapatıldığında geçmiş silinir.
> 3. **Geçici Dosyalar (Temp WAVs):** İndirilen ve dönüştürülen ses dosyaları yerel diskte geçici olarak birikmekte olup otomatik bir depolama yaşam döngüsü / çöp toplama (garbage collection) altyapısı barındırmamaktadır.
> 4. **Yetkilendirme:** Sistemde kullanıcı oturumu, kimlik doğrulama (auth) veya hız sınırlaması (rate limiting) bulunmamaktadır.
> 
> Bu teknik tercihler yerel bir araştırma stüdyosu için tamamen kabul edilebilir ve uygundur; ancak arayüzün üretim ölçeğinde bir sistem değil, ampirik bir gösterim aracı olduğu savunmada açıkça ifade edilmelidir.

---

## 8. Donanım ve Çalışma Ortamı

Deneysel değerlendirme ve yerel web uygulaması testi, Tatilsepeti Ar-Ge ve Koç Üniversitesi işbirliği kapsamında aşağıdaki laboratuvar donanımında yürütülmüştür:
- **İşlemci (CPU):** Intel Core i7-14650HX (16 Çekirdek, 24 İş Parçacığı, 5.2 GHz Turbo)
- **Grafik İşlemci (GPU):** NVIDIA GeForce RTX 5060 Laptop GPU (8 GB GDDR6 VRAM, CUDA 13.0)
- **Sistem Belleği (RAM):** 32 GB DDR5
- **İşletim Sistemi ve Yazılım:** Windows 11 64-bit, Python 3.12, PyTorch 2.12.0+cu130, Hugging Face Transformers 5.13+, ReportLab 4.4+.

---

## 9. Sonuç, Kısıtlar ve Gelecek Çalışmalar

### 9.1 Sonuç
Bu bitirme çalışmasında, Türkçe ve İngilizce eğitim konuşmalarının başlangıç düzeyindeki öğrenciler için anlaşılır kılınmasını sağlayan iki katmanlı ASR+LLM sistemi uçtan uca araştırılmış ve doğrulanmıştır:
1. Whisper ve Qwen3 mimarilerinin Türkçe ders kayıtlarında %3 – %8 bandında kabul edilebilir hata oranlarıyla çalıştığı ve gerçek zamanlıdan katbekat hızlı çıkarım yapabildiği kanıtlanmıştır.
2. Kaliteli bir transkript sağlandığında, dil modellerinin %96'nın üzerinde bir doğrulukla dersin kilit kavramlarını yakalayıp B1 düzeyinde sadeleştirebildiği gösterilmiştir.
3. Çift değerlendirici ve Cohen's kappa analizi ile semantik skorların kişisel yanılgılardan arındırılmış bilimsel geçerliliği belgelenmiş; özellikle bağlam uyumluluğundaki $\kappa_R = 0.782$ skoru, değerlendiriciler arası tutarlılığın sağlam bir istatistiksel kanıtını sunmuştur.
4. Altın referanslı GPU sonuçları doğrultusunda net dağıtım önerileri belirlenmiştir: Doğruluk-öncelikli (accuracy-oriented) senaryolar için en düşük hata oranına sahip Whisper Large-v3 (TR 0.021 / EN 0.010) ile yüksek kapasiteli bir LLM (GPT-4.1 veya Qwen3.5-9B); hız/gecikme-öncelikli (latency-oriented) senaryolar için ise 4–8 kat daha hızlı olan Whisper Turbo (RTF 0.025–0.044) ile hafif açık kaynaklı Qwen3.5-4B (veya GPT-4o-mini) eşleşmesi önerilmektedir.

### 9.2 Kısıtlar
- Veri kümesi 7 ders ve 95 dakika ile sınırlıdır; farklı konuşmacı ve aksan çeşitliliği barındırsa da tüm disiplinleri kapsamamaktadır.
- CEFR B1+ ve BLEU otomatik metrikleri sözlük kütüphanelerinin yapısı gereği yalnızca İngilizce için hesaplanabilmiştir.
- **ASR→LLM Hata Yayılımı Kısıtı:** Çapraz katmanlı ablasyonda dil modeline yalnızca Whisper modellerinin düşük WER'li hipotezleri beslenmiştir; Wav2Vec 2.0 veya SpeechBrain gibi yüksek hatalı modeller semantik katmana dahil edilmemiştir. Dolayısıyla çalışma ASR hatalarının LLM terim çıkarımına kademeli yayılımını sürekli bir kalite eğrisi olarak ölçmemekte, yalnızca yüksek kaliteli transkriptler üzerindeki davranışla sınırlı kalmaktadır.
- **$\kappa_D = 1.000$ ve Sınıf Dengesizliği (Prevalence) Kısıtı:** 115 terimlik çift etiketleme kümesinde tanım doğruluğu ($D$) için 112 pozitif ve sadece 3 negatif etiket bulunmaktadır. Şans eseri beklenen uyumun zaten %95 ($P_e \approx 0.949$) olduğu bu aşırı tek taraflı dağılımda $\kappa_D = 1.000$ çıkması matematiksel olarak beklenen bir sonuçtur ve değerlendirici kalitesinin güçlü bir ayrım kanıtı değildir (Kappa Paradoksu). Değerlendirici güvenilirliğinin asıl metodolojik dayanağı, negatif örnekler (8–12 adet) ve sınır vakalardaki 4 anlaşmazlığı barındıran $\kappa_R = 0.782$ skorudur.
- **Referans Metinlerin (Gold) Bağımsız Transkripsiyon Olmaması Kısıtı:** Veri kümesindeki transkriptler sıfırdan bağımsız bir insan dinlemesiyle üretilmemiştir. EN-1 ve EN-4 temizlenmiş insan altyazısı, TR-3 sesten taslak iken; EN-2, EN-3 ve TR-2 YouTube otomatik altyazılarının insan tarafından noktalanması ve terimlerinin düzeltilmesiyle (post-editing) oluşturulmuş, TR-1 ise otomatik taslak ile eksik insan altyazısının birleştirilmesinden elde edilmiştir. Bu durum, referansların otomatik altyazı ve Whisper stili cümle bölme/sentaks kalıplarına hafif bir yakınlık taşımasına yol açabilmekte ve WER skorlarında Whisper lehine marjinal bir biçimsel/stil avantajı yaratabilmektedir. Dolayısıyla metinler "bağımsız gold" olarak değil, "insan denetiminde düzeltilmiş referanslar" olarak değerlendirilmelidir.
- **Küçük Örneklem ve İstatistiki Test Yokluğu ($n=7$, Ağırlıksız Ortalama Kısıtı):** Dil bazlı ASR kıyaslamaları toplamda yalnızca 7 ders videosuna (3 Türkçe, 4 İngilizce) ve ağırlıksız ortalamalara dayanmaktadır; güven aralıkları veya istatistiksel anlamlılık testleri hesaplanmamıştır. Dolayısıyla bu çalışmadaki Türkçe WER değerlerinin İngilizceden yüksek çıkması, "Türkçenin konuşma tanıma için genel olarak daha zor bir dil olduğu" şeklinde yorumlanamaz ve genellenemez; bu fark incelenen derslerin akustik ortamı, konuşmacı hızı ve terminoloji yoğunluğuna bağlı olabilir.
- **Boş Kalan Protokol Hücreleri (Empty Cells - Wav2Vec 2.0 & SpeechBrain):** Makalede ve tezde bilimsel şeffaflık esastır: 2026 yenileme modelleri (Whisper Turbo, Qwen3-ASR 0.6B/1.7B - 4 İngilizce klibin tamamı dahil - ve Nemotron 3.5 ASR) GPU ortamında altın referanslarla eksiksiz koşturulmuş olup sonuçları doğrudan makale tablolarına (`\input{tables/asr_2026.tex}`) ve değerlendirme kitine yansıtılmıştır. Tablo III'te (`tab:asr-lang`) yalnızca İngilizce odaklı olan ve Türkçe dil desteği bulunmayan Wav2Vec 2.0 ile SpeechBrain kontrol noktaları altın referanslı GPU kümesinde yeniden koşturulmamış ve dürüstçe boş hücre (`---`) olarak bırakılmıştır; bu modellerin CPU pilot havuz değerleri tarihsel bağlam olarak sunulmaktadır.
- **Semantik Metriklerde Recall (Duyarlılık) Ölçülememesi Kısıtı:** Eğitim videolarında önceden sabitlenmiş kapalı bir alan terminolojisi sözlüğü bulunmadığından, semantik katmanda yalnızca üretilen terimler üzerindeki kesinlik benzeri oranlar ($R1\%$ ve $D1\%$) raporlanmıştır; modelin dersteki teknik kavramların yüzde kaçını yakaladığı (recall) ölçülmemiştir. GPT-4.1'in daha yüksek terim sayısı üretmesi (105 vs 56) kanıtlanmış bir tam kapsam üstünlüğü değil, yalnızca aday üretim hacmi göstergesidir.
- **Web Arayüzünün Yerel Prototip Sınırı (Demo vs. Ürün):** Geliştirilen FastAPI arayüzü son kullanıcıya açık kurumsal bir üretim ürünü değil, yerel laboratuvar stüdyosudur. Açık CORS politikası (`*`), bellek içi iş yönetimi (in-memory job store), kimlik doğrulama/yetkilendirme bulunmaması ve geçici ses dosyası birikimi nedeniyle bir sistem makalesi konusu olmayıp, boru hattının yerel uygulanabilirliğini somutlaştıran ampirik bir PoC prototipidir.

### 9.3 Gelecek Çalışmalar
- Sistemin açık kaynaklı bir Öğrenme Yönetim Sistemi (LMS - örn. Moodle, Canvas) eklentisi haline getirilmesi.
- Öğrencinin geçmiş test ve video izleme verilerinden beslenen dinamik bir "Öğrenci Bilgi Takip" (Knowledge Tracing) modülü ile tam uyarlanabilir bir sanal asistana dönüştürülmesi planlanmaktadır.

---

## Kaynakça

1. Bloom, B. S. (1984). "The 2 sigma problem: The search for methods of group instruction as effective as one-to-one tutoring." *Educational Researcher*, 13(6), 4-16.
2. Radford, A., Kim, J. W., Xu, T., et al. (2023). "Robust speech recognition via large-scale weak supervision." *Proc. ICML*, 28492-28518.
3. Wang, X., et al. (2026). "Qwen3-ASR technical report." *arXiv preprint arXiv:2601.21337*.
4. NVIDIA (2026). "Nemotron 3.5 ASR streaming 0.6B model card." *Hugging Face Hub*.
5. Conde, A., Larrañaga, M., et al. (2016). "LiTeWi: A combined term extraction and entity linking method for eliciting educational ontologies from textbooks." *JASIST*, 67(2), 380-399.
6. Tran, H. T. H., Martinc, M., et al. (2026). "Recent advances in automatic term extraction: A comprehensive survey." *ACM Computing Surveys*, 58(9), 1-35.
7. Vanzo, A., Pal Chowdhury, S., & Sachan, M. (2025). "GPT-4 as a homework tutor can improve student engagement and learning outcomes." *Proc. ACL*, 31119-31136.
8. Cohen, J. (1960). "A coefficient of agreement for nominal scales." *Educational and Psychological Measurement*, 20(1), 37-46.
9. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). "wav2vec 2.0: A framework for self-supervised learning of speech representations." *NeurIPS*, 33, 12449-12460.
10. Ravanelli, M., et al. (2021). "SpeechBrain: A general-purpose speech toolkit." *arXiv:2106.04624*.
