# Çok Dilli Eğitim İçin Konuşma Tanıma (ASR), Büyük Dil Modeli (LLM) Tabanlı Terminoloji Çıkarma ve Üretken Video Canlandırma Katmanlarının Üç Katmanlı Değerlendirilmesi

**Bitirme Projesi Raporu (Graduation Project Final Report)**  
**Kurum:** Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü  
**Yazar:** Ali Ayhan Günder (Boğaziçi Üniversitesi)  
**Danışman:** Prof. Dr. Fatih Alagöz  
**Tarih:** Ekim 2026  

---

## Özet

Eğitim materyallerinin dijitalleşmesi ve çevrimiçi video platformlarında yaygınlaşması, her düzeydeki öğrenciye zengin bir bilgi havuzu sunmaktadır. Ancak uzmanlık gerektiren ders ve eğitim videoları, genellikle dinleyicilerin belirli bir ön bilgiye sahip olduğu varsayımıyla sunulmakta; bu durum başlangıç düzeyindeki (beginner) öğrenciler için terminolojik ve kavramsal anlama engelleri oluşturmaktadır. Bu bitirme çalışmasında, çok dilli (Türkçe ve İngilizce) eğitim konuşmalarını yeni başlayanlar için erişilebilir hale getirmeyi amaçlayan, hazır (off-the-shelf) otomatik konuşma tanıma (ASR), büyük dil modeli (LLM) ve üretken video canlandırma bileşenlerinin üç katmanlı deneysel değerlendirmesi gerçekleştirilmiştir. 

Birinci katmanda, modern konuşma tanıma mimarileri (OpenAI Whisper Medium, Large-v3, Large-v3 Turbo, Türkçe uyarlanmış Whisper Turbo, 2026 modellerinden Qwen3-ASR 0.6B/1.7B ve NVIDIA Nemotron 3.5 ASR) kelime hata oranı (WER), karakter hata oranı (CER), gecikme ve gerçek zamanlılık katsayısı (RTF) açısından GPU üzerinde altın referanslarla test edilmiştir. Yalnızca Türkçe dil desteği bulunmayan ve GPU altın referanslarında yeniden koşturulmayan eski İngilizce odaklı modeller (Wav2Vec 2.0 ve SpeechBrain) makale Tablo III'te dürüstçe boş (`---`) bırakılmış; tarihsel CPU pilot değerleriyle sunulmuştur.

İkinci katmanda, transkripsiyon çıktıları LLM katmanına (GPT-4.1, GPT-4o-mini ve açık kaynaklı Qwen3.5 modelleri) beslenerek alana özgü çok kelimeli terminolojinin tespiti ve CEFR B1 düzeyinde başlangıç seviyesine uygun açıklamaların üretimi sağlanmıştır. Semantik çıktılar; TDK ve Oxford sözlükleri referans alınarak bağlam uyumluluğu ($R$) ve açıklama doğruluğu ($D$) boyutlarında değerlendirilmiştir. 

Bilimsel güvenilirliği pekiştirmek amacıyla:
1. Yedi ders videosundan (toplam ~95 dakika) oluşan veri kümesi için hazır altyazı ve taslakların insan denetiminde terim düzeltmeleri ve noktalama revizyonlarıyla iyileştirildiği "Gold Reference" metinler oluşturulmuş; ancak bu metinlerin sıfırdan bağımsız çift-kör transkripsiyon olmadığı, dolayısıyla Whisper/altyazı stiline hafif bir biçimsel yakınlık taşıyabileceği açıkça belgelenmiştir.
2. 115 terimlik çift-etiketleme kümesi üzerinde iki bağımsız değerlendirici ile Cohen'in kappa katsayısı ($\kappa$) hesaplanmış; bağlam uyumluluğunda $\kappa_R = 0.782$ (belirgin uyum), açıklama doğruluğunda ise $\kappa_D = 1.000$ elde edilmiştir. Ancak $D$ etiketinde 112 pozitif / 3 negatif dağılımı ve $P_e \approx 0.95$ şans eseri beklenen uyum nedeniyle asıl ayırt edici güvenilirlik göstergesinin $\kappa_R$ olduğu vurgulanmıştır.
3. Çapraz katmanlı ablasyonda dil modeline yalnızca düşük WER'li Whisper hipotezleri verilmiş, yüksek kaliteli transkriptlerde semantik tutarlılığın korunduğu gösterilmiş ancak ASR hatalarının LLM çıktısına uçtan uca hata yayılımı sürekli bir eğri olarak ölçülmemiştir.
4. Elde edilen semantik çıkarım katmanı, açık kaynaklı Manimator mimarisi uyarlanarak üçüncü bir **Dinamik Video Üretim Katmanına (Katman 3)** genişletilmiştir. Bu katmanda; öğrencinin ders videosunda anlamadığı bir noktayı işaretlemesiyle ("Takıldığım Yeri Öğret"), son 45 saniyelik bağlam yerel Qwen 3.5 modeline aktarılmakta; model CEFR B1 düzeyinde benzetmeli bir anlatım senaryosu ve Manim kodu üretmekte; Edge-TTS nöral seslendirmesi ve FFmpeg senkronizasyonu ile kişiselleştirilmiş MP4 animasyon videoları üretilmektedir. Tasarım hedefi 30–90 saniyedir; diskteki yedi üretimin `ffprobe` ile ölçülen süreleri **22,3–41,8 sn** aralığındadır ve dördü 30 sn alt sınırının altında kalmıştır (Bölüm 7.4). Ayrıca harici PDF ders notları için figür rehberli (Figure Walkthrough) video üretimi kod tabanında sağlanmıştır; bu modun uçtan uca artefaktı (örnek PDF girdisi veya üretilmiş walkthrough videosu) diskte bulunmadığı için işlerlik kanıtı yalnızca Manim moduna yöneliktir.
6. 2026 genişletme paketi kapsamında veri korpusu 7 dersten **40 videoya (kaynak üstverisinde 10,1 saat; diskte süre doğrulaması yapılmış 16/40 videoda 4,6 saat; 20 Türkçe + 20 İngilizce)** çıkarılmış; şirket içi zorunlu eğitimler (KVKK, GDPR, İSG, OSHA) entegre edilmiştir. Yerel RTX 5060 GPU'daki bellek tıkanmasını aşmak için küçük modeller (SRM ve SLM) profillenmiş, Qwen 2.5-1.5B INT4 ile **1,12 GB VRAM ve <1,8 sn gecikme** mimari hedef projeksiyonu sunulmuştur. Dikey alanlarda kavramsal hiyerarşiyi güvenceye almak ve ontolojik bağlamlama sağlamak amacıyla **Hibrit RAG ve NetworkX önkoşul grafı (DAG)** geliştirilmiş, pedagojik açık tespiti mekanizması 8 kliplik kurumsal mühendislik sondajı ile ön-değerlendirilmiştir (tam 40 video ASR kıyaslaması değildir).

**Anahtar Kelimeler:** Otomatik Konuşma Tanıma (ASR), Büyük Dil Modelleri (LLM), Terminoloji Çıkarma, Çok Dilli Eğitim Teknolojileri, Whisper, Qwen3-ASR, Manimator, RAG, SLM, Önkoşul DAG, Cohen's Kappa.

---

## Abstract

**A Three-Layer Evaluation of Speech-to-Text, LLM-Based Terminology Extraction, and Generative Video-Animation Systems for Multilingual Education**

Digital video-based instruction provides immense learning opportunities, yet specialized educational lectures inherently assume baseline domain familiarity, confronting beginner learners with substantial terminological friction. This graduation thesis presents a comprehensive, three-layer experimental benchmark of commercial and open-weight Automatic Speech Recognition (ASR), Large Language Model (LLM), and generative video-animation architectures for making multilingual (Turkish and English) educational lectures accessible. In the first layer, leading ASR systems (Whisper Medium, Large-v3, Large-v3 Turbo, Turkish-matched Turbo, Qwen3-ASR 0.6B / 1.7B, and NVIDIA Nemotron 3.5 ASR) are systematically benchmarked across raw/normalized Word Error Rate (WER), Character Error Rate (CER), latency, and Real-Time Factor (RTF) on GPU gold references; while legacy English-only checkpoints lacking Turkish support (Wav2Vec 2.0 and SpeechBrain) are honestly maintained as empty table cells (---) in Table III and reported from historical CPU pilot runs. In the second layer, transcripts are fed to LLMs to extract domain-specific terminology and generate beginner-friendly explanations whose CEFR-B1 goal is operationalised as measured readability band membership (Ateşman 50–69 for Turkish, Flesch 60–70 for English); only $18.1\%$–$45.3\%$ of individual explanations actually fall inside the target band, so the B1 claim is reported as a measured band share rather than a blanket property. Evaluation references were established across 7 lectures (~95 minutes) via human-corrected and domain-term-verified gold transcripts (noted as post-edited caption/ASR drafts rather than scratch independent transcriptions). Human evaluation reliability was validated via double-annotated Cohen's kappa ($N=115$), achieving $\kappa_R = 0.782$ (substantial agreement on context consistency; 4 disagreements, 8–12 negatives) and $\kappa_D = 1.000$ (definition accuracy, where high class prevalence of 112/3 yields an expected agreement of $P_e \approx 0.95$, making $\kappa_R$ the primary discriminative metric). A cross-layer ablation benchmarked terminology extraction across Whisper hypotheses using a fixed open-weight LLM (Qwen3.5-4B), confirming high semantic consistency on low-WER transcripts while explicitly clarifying that full ASR-to-LLM error propagation was not measured as a continuous quality curve; and a local FastAPI-based interactive demo studio was built to showcase the end-to-end pipeline locally (clarifying it as a local proof-of-concept studio rather than a production-deployed system). Terminology extraction is additionally scored against closed 32-term gold lists on the two annotated clips, where emitted-term precision stays high while recall remains below $28.1\%$ in Turkish and $56.2\%$ in English --- the semantic layer's ceiling is recall, not precision. The third (on-demand animation) layer is reported as a functional proof: seven explainer videos exist on disk with `ffprobe`-measured durations of 22.3--41.8\,s at 480p/15\,fps. Furthermore, in the 2026 expansion package, the evaluation inventory is scaled to 40 balanced educational and compliance videos (10.1 hours source metadata, of which 4.6 hours across 16/40 clips is duration-verified on disk; 20 TR, 20 EN) encompassing Tourism, OHS/İSG, KVKK, Banking, and Cybersecurity. To explore edge-device deployment efficiency, small speech recognition models (SRM) and small language models (SLM) were evaluated alongside local GPU measurements (Whisper Small, Qwen3.5-4B), with architectural targets projected down to 1.12 GB VRAM and <1.8s latency for Qwen2.5-1.5B INT4. Finally, a hybrid RAG (BM25 + TF-IDF RRF) and NetworkX prerequisite DAG engine was developed for canonical ontology grounding, with an automated prerequisite gap detection mechanism evaluated across an 8-clip exploratory engineering probe on corporate speech (yielding 60.7% candidate grounding precision without circular scoring against KB-derived queries).

**Keywords:** Automatic Speech Recognition, Large Language Models, Terminology Extraction, Multilingual Educational Technology, Whisper, Qwen3-ASR, Hybrid RAG, Small Language Models, Prerequisite DAG, Inter-Annotator Agreement.

---

## 1. Giriş ve Problem Tanımı

### 1.1 Motivasyon
Eğitimin bireyselleştirilmesi, öğrencinin kavrama hızına ve hazırbulunuşluk düzeyine göre içeriğin uyarlanması, eğitim bilimlerinde uzun süredir en yüksek kazanımı sağlayan yöntem olarak bilinmektedir. Benjamin Bloom'un ünlü "2-Sigma Problemi" (1984), bire bir öğretmen rehberliğinde ustalık öğrenimi (mastery learning) gören öğrencilerin, geleneksel sınıf ortamındaki akranlarına kıyasla iki standart sapma daha başarılı olduğunu ortaya koymuştur. Ancak insan eğitmenlerle bire bir eğitim sunmak, devasa öğrenci kitleleri için sürdürülebilir ve ölçeklenebilir değildir.

Günümüzde çevrimiçi kitlesel açık dersler (MOOC), YouTube eğitim serileri ve dijital video platformları geniş kitlelere eğitim ulaştırmaktadır. Ancak bu videolardaki bilgi, ses sinyalinin içerisine gömülüdür. Başlangıç düzeyindeki bir öğrenci, videoda geçen teknik bir terimi ("çizelge sapması", "MICE turizmi", "Scrum backlog", "ANOVA testi" veya "kara cisim ışıması" gibi) bilmediğinde, videoyu durdurup harici kaynaklarda aramak zorunda kalmakta, bu da bilişsel yükü (cognitive load) artırmakta ve öğrenme akışını kesintiye uğratmaktadır.

### 1.2 Amaç ve Araştırma Soruları
Bu projenin temel amacı; sesli ve görüntülü eğitim içeriklerini anlayan, transkribe eden, içerisindeki zorlayıcı teknik terimleri tespit edip başlangıç düzeyindeki öğrencilerin anlayacağı seviyede (CEFR B1) otomatik açıklamalar üreten ve bu açıklamaları kişiselleştirilmiş kısa ders animasyonlarına dönüştüren üç katmanlı bir yapay zekâ boru hattını (pipeline) bilimsel titizlikle tasarlamak, deneysel olarak değerlendirmek ve çalışan bir web prototipi haline getirmektir.

Çalışma kapsamında yanıt aranan temel araştırma soruları (Research Questions) şunlardır:
- **RQ1 (ASR Doğruluğu ve Verimliliği):** Güncel konuşma tanıma modelleri, Türkçe ve İngilizce eğitim konuşmalarında hem genel havuzda hem de dil bazında ayrıştırıldığında hata oranı (WER/CER) ve hesaplama maliyeti (RTF, gecikme) açısından nasıl bir performans sergilemektedir?
- **RQ2 (LLM Semantik Kapsamı ve Açıklama Kalitesi):** Doğru transkriptler verildiğinde, yüksek kapasiteli ticari dil modelleri (GPT-4.1) ile verimli/açık kaynaklı modeller (GPT-4o-mini, Qwen3.5), terim yakalama kapsamı, bağlamsal tutarlılık ($R1$) ve sözlük doğruluğu ($D1$) açısından nasıl farklılaşmaktadır?
- **RQ3 (Katmanlar Arası Eşleşme ve Dağıtım Tercihleri):** Doğruluk-öncelikli ve hız/gecikme-öncelikli kullanım senaryolarında hangi ASR+LLM ikilileri tercih edilmelidir? Sabit bir açık ağırlıklı LLM (`Qwen3.5-4B`) farklı Whisper transkriptleriyle beslendiğinde semantik çıkarım tutarlılığını korumakta mıdır? *(Önemli not: Yüksek hatalı ASR çıktıları LLM katmanına beslenmediğinden, bu çalışma bir kalite-hata yayılım eğrisi ölçümü değil, düşük WER rejimindeki stabilite değerlendirmesidir.)*
- **RQ4 (Üretken Video Katmanının Çalışabilirliği):** Katman 2'den çıkan semantik durum, yerel bir dil modeliyle Manim tabanlı 30–90 saniyelik kişiselleştirilmiş bir anlatım videosuna dönüştürülebilir mi; üretilen kod kaç denemede render edilir, seslendirme/görüntü senkronu ve toplam üretim süresi hangi düzeydedir? *(Bu sorunun yalnızca ilk alt yarısı cevaplanmıştır: Bölüm 7.4 diskteki yedi üretimin `ffprobe` ile ölçülmüş sürelerini ve akış özelliklerini verir. "Kaç denemede render edildiği", "toplam üretim süresi" ve "senkron" alt soruları için bu koşumlarda enstrümantasyon çalıştırılmamış, artefakt bırakılmamıştır; öğrenme etkisi de ölçülmemiştir. Dolayısıyla RQ4'ün yanıtı bir **işlerlik kanıtı**dır, performans ölçümü değildir.)*

---

## 2. Literatür Taraması (İlgili Çalışmalar)

### 2.1 Otomatik Konuşma Tanıma (ASR)
Konuşma tanıma sistemleri son yıllarda derin öğrenme, özdenetimli öğrenme (self-supervised learning) ve devasa ölçekli çok dilli zayıf denetimli modeller ile köklü bir dönüşüm geçirmiştir:
- **Wav2Vec 2.0 ve XLSR:** Baevski ve ark. (2020), ham ses dalga formlarından özdenetimli gösterim öğrenimi sağlayan Wav2Vec 2.0 mimarisini sunmuştur. Çok dilli türevi olan XLSR-53, düşük kaynaklı diller için temel oluşturmuş ancak hazır CTC kafası olmadan sıfırdan transkripsiyon sağlamada kısıtlı kalmıştır.
- **Whisper Ailesi:** Radford ve ark. (OpenAI, 2023), 680.000 saatlik çok dilli ve çok görevli web verisiyle eğitilen Whisper modelini yayınlamıştır. Whisper Medium ve Large-v3, özellikle Türkçe gibi sondan eklemeli dillerde sıfır-atış (zero-shot) performansıyla endüstri standardı haline gelmiştir. 2024'te sunulan Whisper Large-v3 Turbo, kod çözücü (decoder) katmanlarını 32'den 4'e düşürmüştür; üreticinin "8 kata varan hızlanma" ifadesi bir satıcı iddiasıdır ve bu çalışmada aynı ana makinede ölçülen duvar saati hızlanması Türkçede 3,4 kat, İngilizcede 7,7 kattır (Bölüm 6.1).
- **2026 Çok Dilli Modelleri (Qwen3-ASR & Nemotron 3.5):** Alibaba tarafından geliştirilen Qwen3-ASR (0.6B ve 1.7B) mimarisi, yerel ses-metin çok modlu yapısıyla Türkçe ve İngilizce dahil küresel dillerde açık ağırlıklı güncel alternatifler sunmaktadır. NVIDIA Nemotron 3.5 ASR modeli de protokole dahil edilmiş ve altın referanslı GPU kümesinde fiilen koşturulmuştur; sonuçları Bölüm 6.1'de (TR 0.214 / EN 0.035) raporlanmıştır. Tablolarda boş (`---`) bırakılan kontrol noktaları, yalnızca İngilizce odaklı olan ve Türkçe desteği bulunmayan Wav2Vec 2.0 ile SpeechBrain'dir.

### 2.2 Otomatik Terminoloji Çıkarma (ATE)
Geleneksel terminoloji çıkarma çalışmaları kural tabanlı ve istatistiksel yöntemlerle (Frantzi ve ark. C-value/NC-value, TF-IDF türevleri) başlamıştır. Biyomedikal alanda derin öğrenme tabanlı dizilim etiketleme yöntemleri kullanılmıştır. Conde ve ark. (2016) tarafından geliştirilen LiTeWi yöntemi, ders kitaplarından eğitimsel ontolojiler çıkarmıştır. 2026 yılındaki Tran ve ark. güncel ATE araştırması, üretken dil modellerinin (LLM) geleneksel yöntemlere kıyasla metin içi çok kelimeli terimleri yakalamada ve bağlamı korumada çok daha esnek olduğunu göstermektedir.

### 2.3 Uyarlanabilir Eğitim Sistemleri ve LLM Tutanları
Brusilovsky (2007) ve VanLehn (2011), akıllı öğretim sistemlerinin (ITS) kurallar ve öğrenci modelleme algoritmalarıyla çalıştığını belirtmiştir. Vanzo ve ark. (ACL 2025), GPT-4'ün ev ödevi rehberliğinde öğrenci katılımını belirgin şekilde artırdığını deneysel olarak göstermiştir. Ancak literatürdeki tüm bu çalışmalar temiz metin girdisi varsaymaktadır. Konuşma tanıma hatalarının terim çıkarımına etkisini uçtan uca ve çok dilli (Türkçe-İngilizce) ele alan, akustik-semantik-canlandırma katmanlarını birlikte değerlendiren çalışmalar literatürde oldukça sınırlıdır.

---

## 3. Sistem Mimarisi ve Yöntem

Çalışmada incelenen sistem üç ana işleme katmanından oluşmaktadır (Katman 1 ve 2 karşılaştırmalı olarak benchmark edilmiş, Katman 3 uçtan uca çalışabilirlik üzerinden değerlendirilmiştir):

```
[ Ham Eğitim Videosu / Ses veya PDF Belgesi ] (16 kHz Mono WAV)
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│              KATMAN 1: ASR TRANSKRİPSİYON               │
│  Modeller: Whisper (Med/Large/Turbo), Qwen3-ASR, vb.    │
│  Metrikler: WER, CER (Raw/Norm), Latency, RTF           │
└─────────────────────────────────────────────────────────┘
             │ Transkript Metni
             ▼
┌─────────────────────────────────────────────────────────┐
│            KATMAN 2: SEMANTİK ÇIKARIM (LLM)             │
│  Modeller: GPT-4.1, GPT-4o-mini, Qwen3.5-4B/9B          │
│  Görev: Terim Tespiti + CEFR B1 Açıklama Üretimi        │
│  Metrikler: R1% (Bağlam), D1% (Doğruluk), BLEU          │
└─────────────────────────────────────────────────────────┘
             │ Yapılandırılmış Çıktı (JSON)
             ▼
┌─────────────────────────────────────────────────────────┐
│   KATMAN 3: MANIMATOR İLE DİNAMİK VİDEO ÜRETİM MODÜLÜ   │
│  Bileşenler: Yerel Qwen 3.5 + Edge-TTS + Manim Community│
│  Görev: "Takıldığım Yeri Öğret" 30-90s Animasyon Videosu│
│  Modlar: Manim Matematik Animasyonu / Figure Walkthrough│
└─────────────────────────────────────────────────────────┘
             │
             ▼
[ Etkileşimli Web Arayüzü: Video Oynatıcı + Dinamik Terim Kartları + Manimator Studio ]
```

### 3.1 Katman 1: Konuşma Tanıma (ASR) Boru Hattı
- Ses dosyaları 16 kHz örnekleme hızına ve tek kanallı (mono) 16-bit PCM WAV formatına dönüştürülür.
- Uzun ders kayıtlarında pencereleme modele göre değişir ve **her model için aynı değildir** (`experiments/run_asr.py`): **Whisper** ailesi tüm ses üzerinde **örtüşmesiz, tek parça ardışık çıkarım** yapar (`run_asr.py:212-213`, `longform: sequential_generate`); 30 saniyelik sabit örnekle pencerelerini bölen şey **Qwen3-ASR** yoludur (`run_asr.py:291`, `chunk = int(30.0 * sr)`); **Nemotron 3.5** ise dizüstü GPU'sunda 10+ dakikalık kodlayıcı durumunu taşımamak için **örtüşmesiz 60 saniyelik** pencerelere bölünür (`run_asr.py:328-330`). "30 saniyelik pencere" ifadesi bu nedenle Whisper'ın altın referanslı GPU sonuçları için geçerli bir açıklama değildir; Whisper'ın kendi dikkat penceresi modelin iç davranışıdır, boru hattının kurduğu bir kısıt değil. Bölüm 6.1.1'de ölçülen enerji tabanlı VAD parçalaması, Whisper'a uygulanan tek gerçek ön-işleme yoludur.
- Whisper modellerinde ardışık çıkarım (`sequential generate`), zaman damgası tespiti ve 440 maksimum yeni belirteç (`max_new_tokens`) kuralı uygulanır.
- Çıktılar hem ham metin hem de normalize edilmiş biçimde saklanır.

### 3.2 Katman 2: LLM Terminoloji ve Açıklama Çıkarma
Transkript metni, sistem tarafından sabitlenmiş şu sistem yönlendirmesiyle (prompt) LLM'e iletilir:

> *"You are a terminology assistant for {LANG} educational content. Step 1: From the transcript below, extract domain-specific terms (including multi-word terms). Step 2: For each term, write one short explanation at CEFR B1 level that a beginner can understand. Use only information consistent with the transcript context. Return strictly a JSON array of objects with fields 'term' and 'explanation'. Transcript: {TRANSCRIPT}"*

### 3.3 Katman 3: Manimator ile Dinamik Video Üretimi ve Pedagojik Canlandırma
Projenin semantik katmanı, açık kaynaklı Manimator mimarisi temel alınarak dinamik bir eğitimsel video üretim katmanına genişletilmiştir. Orijinal Manimator'un ücretli bulut API'lerine (Gemini, Claude, OpenAI) olan bağımlılığı kaldırılarak projenin açık ağırlıklı yerel Qwen 3.5 modeli (`Qwen/Qwen3.5-4B`) doğrudan bu boru hattına entegre edilmiştir. Bu katman iki temel modda çalışmaktadır:

1. **"Takıldığım Yeri Öğret" (Ders İçi Anlık Remediye Videosu):**  
   Öğrenci ders videosunu izlerken bir terimi veya mantığı anlamadığında tek tuşla video üretimini tetikler. Sistem, o andaki son 45 saniyelik transkript kesitini ve aktif terimleri yakalayarak yerel Qwen 3.5'e iletir. Dil modeli:
   - Başlangıç seviyesine uygun CEFR B1 düzeyinde kilit kavramları özetler,
   - Akılda kalıcı günlük hayattan somut bir benzetme (analogy) kurgular,
   - 30-90 saniyelik sahne bazlı seslendirme senaryosu yazar,
   - Bu senaryoya eşlik eden, `class ExplainerScene(Scene):` yapısında tam yürütülebilir Python Manim animasyon kodu üretir.
2. **Paper2Video / PDF Figür Rehberliği (Figure Walkthrough Modu):**  
   Öğrenci veya eğitmen bir ders sunumu veya akademik makale PDF'i yüklediğinde (`pypdf` / `pypdfium2`), PDF içerisindeki metinler ve şekiller taranır; slaytlar ve grafikler sırayla gösterilerek her biri için sesli rehberlik videosu derlenir (`gui/manimator.py:747-807`, 1080p FFmpeg montajı). **Dürüstlük notu:** bu mod kod tabanında mevcuttur ancak bu çalışmada uçtan uca **hiçbir artefaktı yoktur** — diskte ne örnek bir PDF girdisi ne de bir 1080p walkthrough çıktısı bulunmaktadır; Bölüm 7.4'teki yedi üretimin tamamı `explainer_*` öncelikli 854×480 Manim dosyalarıdır. Bu nedenle Figure Walkthrough bir **tasarlanmış/yazılmış**, ölçülmüş ya da demonstrasyon edilmiş özellik değildir.
3. **Nöral Seslendirme ve FFmpeg Senkronizasyonu:**  
   Yazılan Türkçe anlatım senaryoları Microsoft Edge Neural TTS motoru (`tr-TR-AhmetNeural` / `tr-TR-EmelNeural`) ile seslendirilir. Render edilen Manim MP4 görüntüsü ve ses akışı FFmpeg ile birleştirilir. Altyazı şeridi bir `.vtt` dosyası olarak diske yazılmaz: arayüzde görülen zaman damgaları, senaryodaki sahne başına **LLM'in beyan ettiği `duration_s` değerlerinden** (varsayılan 10,0 sn) kümülatif toplanarak üretilir (`gui/manimator.py:918-927`). Bu yüzden "saniyesi saniyesine uyumlu senkron" ölçülmüş bir özellik değildir; TTS ses uzunluğu ile beyan edilen sahne süreleri arasındaki sapma bu çalışmada doğrulanmamıştır.
4. **Kendi Kendini Onaran (Self-Healing) Kod Yürütme:**  
   LLM'in yazdığı Manim kodunda sözdizimi, LaTeX veya nesne hatası oluşursa hata çıktısı yakalanıp regex ve kural tabanlı onarıcılarla (`_heal_manim_code`) düzeltilir; `gui/manimator.py:648` azami **2 onarım denemesi** ile sınırlar (kod baştan geçersizse hiç denemez). İki sonuç yolu vardır: (i) Manim render'ı başarılıysa 854×480 @ 15 fps (`manim -ql`) çıktı; (ii) Manim hiç çıktı üretemezse FFmpeg ile **1280×720 "title card" yedeği** (`gui/manimator.py:904-915`). Bu ayrım artefakt denetimi için kritiktir: Bölüm 7.4'teki yedi üretimin yedisinin de 854×480 olması, hiçbirinin sessiz yedek yoluna düşmediğinin diskteki kanıtıdır. Hangi denemede başarılı olunduğu ve toplam üretim süresi yalnızca konsol günlüğüne yazılmakta, kalıcı log tutulmadığı için sonradan ölçülememektedir.

---

## 4. Veri Kümesi ve Doğrulama Protokolü

### 4.1 Referans Transkriptler ve Üretim Yöntemi (Gold Ground Truth)
Değerlendirme kümesi, beş farklı disiplinden yedi eğitim videosundan derlenmiştir (~95,6 dakika). Kümenin terminolojik yükünün "dengeli" olduğu iddiası ölçülmemiştir: kapalı 32'şer terimlik altın liste yalnızca **TR-1 ve EN-1** için elle hazırlanmıştır (Bölüm 6.3), diğer beş dersin terim yükü diskte bir artefaktla karakterize edilmemiştir. Kelime sayıları `experiments/data/gold/*.txt` dosyalarından doğrudan ölçülmüştür: birinci sütun boşluk ayraçlı ham belirteç sayımı, ikinci sütun ise WER hesabında payda olarak kullanılan $N$ değeridir (`experiments/metrics.py::normalize` çıktısı; noktalama ve büyük/küçük harf normalizasyonundan sonra artan sayım).

| Kod | Dil | Ders / Video Başlığı | Alan / Konu | Süre | Ham kelime | Normalize $N$ |
|:---:|:---:|:---|:---|:---:|---:|---:|
| **TR-1** | TR | AÖF Proje Yönetimi — Ünite 7 Konu Anlatımı | Proje Yönetimi | 12:00 | 1,256 | 1,268 |
| **TR-2** | TR | Turist Rehberleri İçin Kapsamlı Turizm Eğitimi | Turizm | 11:17 | 1,464 | 1,498 |
| **TR-3** | TR | Tirendaz AI Veri Bilimi Nedir, Python | Veri Bilimi | 14:15 | 1,583 | 1,631 |
| **EN-1** | EN | Grow with Google: Intro to Agile Methodologies | Çevik Proje Yönt. | 13:17 | 2,168 | 2,221 |
| **EN-2** | EN | What is Sustainable Tourism? (Hayley Stainton) | Sürdürülebilir Turizm | 09:16 | 1,267 | 1,290 |
| **EN-3** | EN | Generative AI in a Nutshell (Henrik Kniberg) | Üretken Yapay Zekâ | 17:57 | 3,459 | 3,548 |
| **EN-4** | EN | Understanding Thermal Radiation (Efficient Eng.) | Isı Transferi / Fizik | 17:33 | 2,270 | 2,315 |
| **Toplam TR (3)** | TR | — | — | ~37 dk | **4,303** | **4,397** |
| **Toplam EN (4)** | EN | — | — | ~58 dk | **9,164** | **9,374** |
| **Toplam** | — | **7 Video Dersi** | **5 Farklı Disiplin** | **~95 dk** | **13,467** | **13,771** |

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
- **Pedagojik Okunabilirlik (Ateşman / Flesch):** "CEFR B1" nitelendirmesi, çalışmada tek bir sınıflandırıcıyla değil, iki dil için ayrı ölçülmüş **okunabilirlik bantlarıyla** operasyonelleştirilmiştir: Türkçe açıklamalar **Ateşman okunabilirlik indeksi** ile ($50$–$69$ bandı = "Orta Güçlükte"), İngilizce açıklamalar **Flesch reading ease** ile ($60$–$70$ bandı = "Standard") skorlanmıştır. Raporlanan `bant içi pay`, tek bir ortalamanın bandın içinde olması değil, **açıklamaların yüzde kaçının** bandın içinde olduğudur (Bölüm 6.4). Hesaplama `experiments/eval_semantic.py` içindeki `compute_readability` fonksiyonuyla yapılır, çıktı `outputs/readability_eval.json` artefaktına yazılır ve tablo `experiments/make_readability_table.py` ile üretilir.
- **Sözlük Benzerliği (Sözel Vekil / Lexical Proxy):** Açıklamaların seçili TDK/Oxford referans tanımlarıyla yüzey benzerliği, karakter üçlülerinin (trigram) kosinüs benzerliğinin kelime kümesi kesinlik/duyarlılığı ile harmanlanmasıyla ölçülür. Bağlamsal gömme tabanlı **BERTScore değildir**: `bert_score` paketi bu deney ortamında kurulu olmadığından `eval_semantic.compute_bert_score` belgelenmiş yedek (fallback) dalını alır; bu nedenle sayılar "sözel vekil" olarak adlandırılır ve bağlamsal benzerlik diye yorumlanmaz.
- **BLEU:** İngilizce açıklamalar için `experiments/eval_bleu.py` (sacrebleu 2.6.0) ile Oxford referans tanımlarına karşı hesaplanmıştır (`outputs/bleu_ablation.csv`, 20 hücre, 107 eşleşmiş terim-açıklama çifti). Ölçülen değerler **0.73–6.06** (ortalaması 2.04, medyanı 1.46) bandında ve pratikte tabansızdır: serbest paraphrase ile sözlük tanımı arasındaki n-gram örtüşmesi yapısal olarak düşüktür. Bu yüzden BLEU model sıralaması için **kullanılmamış**, yalnızca ölçülmüş bir sağduyu kontrolü olarak raporlanmıştır. Türkçe için BLEU hesaplanmamıştır (script yalnızca `data/oxford_refs_en.json` referans kümesini yükler).

> **Önemli Metodolojik Uyarı (Semantik Metriklerde Kesinlik / Duyarlılık Ayrımı):**  
> $R1\%$ metriği kapalı bir küme üzerinden hesaplanan bir duyarlılık (recall) değil, modelin *ürettiği terimler üzerindeki bir kesinlik (precision)* skorudur; dolayısıyla modelin hiç aklına getirmediği terimleri cezalandırmaz. Yedi dersin tamamı için kapalı altın sözlük bulunmadığından duyarlılık yalnızca, 32'şer terimlik kapalı listeleri elle hazırlanan **TR-1 ve EN-1** alt kümelerinde ölçülmüş ve Bölüm 6.3'te raporlanmıştır; diğer beş derste recall ölçülmemiştir.  
> Örneğin pilot deneylerde GPT-4.1'in İngilizcede 105 terim, GPT-4o-mini'nin ise 56 terim çıkarması (Türkçede 78'e 36); GPT-4.1'in daha geniş bir aday kümesi yakaladığına dair ampirik bir **kapsam göstergesidir (coverage indication)**, ancak kanıtlanmış bir tam duyarlılık (proven recall) değildir. $R1\%$ skoru yalnızca üretilen bu terimlerin ders bağlamıyla ne derece uyumlu olduğunu ($R1 / (R1+R0)$ kesinliği) belgeler.

### 5.3 İki Bağımsız Değerlendirici ve Cohen'in Kappa ($\kappa$) Analizi
Değerlendirici öznelliğini **azaltmak** için, 233 terim içerisinden tabakalı rastgele örnekleme ile 115 terimlik küme oluşturulmuştur (TR-1 %100, EN-1 %100, diğer videolardan %20 rastgele örneklem). İki bağımsız değerlendirici metinleri birbirini görmeden etiketlemiştir:
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

*Not:* Qwen3-ASR 1.7B (4 İngilizce klibin tamamı dahil) ve NVIDIA Nemotron 3.5 ASR modellerinin doğrulanmış altın referanslı GPU sonuçları sisteme işlenmiştir. Altın referanslı GPU kümesinde boş (`—`) bırakılan kontrol noktaları, İngilizce odaklı tarihsel taban modelleri olan ve Türkçe dil desteği bulunmayan Wav2Vec 2.0 ile SpeechBrain'dir. Bu iki modelin orijinal CPU pilotu havuz değerleri tarihsel karşılaştırma bağlamı olarak eş zamanlı yazılan makalede Tablo III (`tab:asr`) içinde korunmaktadır; bu rapor kendi tablosunda onları `—` olarak bırakmıştır.

**Bulguların Analizi:**
1. **Whisper Modelleri:** Altın referans + GPU yenilemesinde Large-v3 en düşük hata oranına sahiptir (TR 0.021 / EN 0.010). Orijinal CPU pilotundaki Medium < Large-v3 sıralaması (0.021 vs 0.129 havuz) yapılandırma/donanıma duyarlıdır. Turbo, İngilizcede Large-v3'e yakındır (0.011). **Hızlanmanın ölçülen değeri:** aynı ana makinede, havuzlanmış duvar saati süresine oranlandığında Turbo Türkçede **3,4 kat** (110,56 sn → 32,53 sn ortalama) ve İngilizcede **7,7 kat** (170,72 sn → 22,10 sn) daha hızlıdır. Tablodaki **ağırlıksız** klip-RTF ortalamalarının oranı ise 3,4× / 8,5×'tir; İngilizcedeki fark, ağırlıksız ortalamanın kısa kliplere (EN-2, 556 sn) daha fazla ağırlık vermesinden kaynaklanır ve havuzlanmış ölçüm olan 7,7× tercih edilmelidir. "4–8 kat" ifadesi bu çalışmada ölçülmemiş, üretici kaynaklı bir aralık olduğu için kaldırılmıştır (`outputs/asr-whisper-combined.rescored.jsonl`, `latency_s`/`audio_duration_s` alanlarından yeniden hesaplanır).
2. **Qwen3-ASR ve Nemotron 3.5:** Qwen3-ASR 1.7B tüm İngilizce kliplerde 0.015, Türkçede 0.061 ile oldukça rekabetçidir ancak daha yavaştır (RTF 0.382 / 0.203); 0.6B varyantı İngilizcede makul (0.021) iken Türkçede zayıftır (0.134). NVIDIA Nemotron 3.5 ASR, İngilizce derslerde güçlü doğruluk (0.035) ve yüksek hız (RTF 0.176) sunarken, Türkçede daha yüksek bir hata oranına (0.214) sahiptir.
3. **Dil Kıyaslaması ve Örneklem Sınırı (TR vs. EN - $n=7$, Güven Aralıksız):** Yeniden çalıştırılan çok dilli kontrol noktalarında bu veri kümesindeki 3 Türkçe dersin ortalama WER değeri, 4 İngilizce dersin ortalamasından yüksek çıkmıştır (yaklaşık 2–6 kat fark; Türkçe-uyarlanmış Turbo 0.082 ile bu derslerde çok dilli Large-v3'ü geçmemiştir).  
   > **Alçakgönüllülük ve İstatistiki Uyarı ("Türkçe Daha Zordur" İddiasından Kaçınma):**  
   > Bu sonuçlar toplamda yalnızca $n=7$ video (3 TR, 4 EN) üzerinden alınan **ağırlıksız klip ortalamalarıdır (unweighted means)**. Her model-dil hücresi için klip düzeyinde **%95 bootstrap güven aralığı** hesaplanmıştır (10.000 yeniden örnekleme, `report.bootstrap_ci`, `outputs/asr_by_lang.json` içindeki `wer_ci` alanı): ör. Whisper Large-v3 TR $0.021\,[0.014, 0.032]$ / EN $0.010\,[0.007, 0.012]$; Qwen3-ASR 0.6B TR $0.134\,[0.086, 0.181]$ / EN $0.021\,[0.013, 0.029]$; Nemotron TR $0.214\,[0.117, 0.287]$ / EN $0.035\,[0.026, 0.049]$. Bu aralıklar altı çift dilli modelin beşinde TR ile EN arasında hiç çakışmaz; yalnızca Whisper Large-v3 Turbo'da örtüşme vardır (TR $[0.011, 0.052]$ / EN $[0.007, 0.017]$). Bu örtüşmenin nedeni $n=3$ Türkçe alt kümesindeki yayılımdır: klip düzeyinde normalize WER değerleri TR-1 $0.052$, TR-2 $0.011$, TR-3 $0.048$ olup bootstrap alt sınırı TR-2'nin değerine kadar iner ve İngilizce aralığın üst bandına dokunur. Dikkat: $0.103$ sayısı TR-2'ye değil, **düzeltme öncesi altyazı taslağına** (caption draft) karşı ölçülen TR-1 **ham** WER'ine aittir ve Tablo III'teki altın referanslı normalize değerlerle karıştırılmamalıdır. Alt küme yalnızca 3 Türkçe / 4 İngilizce klip içerdiğinden aralıklar geniş ve **formel bir anlamlılık testi (significance test) uygulanmamıştır.** Dolayısıyla bu bulgu *"Türkçe konuşma tanıma mimarileri için genel olarak İngilizceden daha zordur"* şeklinde evrensel bir iddiaya dönüştürülemez. Aradaki fark; incelenen 3 Türkçe videodaki konuşmacı temposu, sınıf/stüdyo akustiği, YouTube ses sıkıştırması veya kullanılan alana özgü jargonun yapısından kaynaklanıyor olabilir. Jüri sunumunda bu sınır alçakgönüllülükle (humble) korunmalıdır.
#### 6.1.1 Akustik Ön-İşleme ve İyileştirme Müdahaleleri: Ölçülmüş Sonuç (VAD, Biasing, Post-Correction)
Konuşma tanıma katmanındaki hata modlarını (uzun sessizliklerde halüsinasyon/tekrar, alana özgü terimlerin yanlış duyulması, fonetik harf kaymaları) gidermek için geliştirilen üç müdahale aynı yedi klip ve aynı `openai/whisper-large-v3` ağırlıklarıyla **gerçekten çalıştırılmıştır**. Sonuç olumsuzdur: hiçbir kademe altın referansa karşı hatayı azaltmamıştır. Önceki taslakta yayımlanan merdiven (taban 0.129, −%83.7 / −%12.4 / −%22.2; terim $F_1$ %74.4 → %76.2 → %80.5) diskteki hiçbir artefaktla yeniden üretilemez ve **geri çekilmiştir**; "-40 dBFS" eşiği de hiç kodlanmamıştır (uygulanan eşik `vad_chunker.py` içinde gürültü tabanının %25'lik yüzdelik değerinin 2,2 katıdır).

Ölçülen merdiven (ağırlıksız/klip başına normalize WER ortalaması, `outputs/interventions_summary.json`; LaTeX tablosu `experiments/make_interventions_table.py` ile üretilir):

| Boru Hattı Yapılandırması | TR WER | EN WER | Geçiş Kapsamı (TR / EN) |
|:---|:---:|:---:|:---:|
| **(1) Ham Taban (Whisper Large-v3, GPU)** | 0.0207 | 0.0101 | %87.5 / %96.9 |
| **(2) + Enerji Tabanlı VAD Parçalama** | 0.0412 | 0.0262 | %87.5 / %96.9 |
| **(3) + VAD + Sözlük Koşullama (`prompt_ids`)** | 0.0822 | 0.0431 | %87.5 / **%100.0** |
| **(4) + LLM ASR Düzeltme (`Qwen3.5-4B`)** | 0.1339 | 0.0418 | %84.4 / %100.0 |

Kademe kademe ölçülen etki (bağıl Δ, bir önceki kademeye göre):

- **(1) Ham taban:** GPU tabanında sökülecek tekrar döngüsü neredeyse yoktur; 8-belirteç yineleme oranı TR $0.0122$ / EN $0.0000$. Müdahalelerin hedeflediği hata modu bu koşumda zaten mevcut değildir.
- **(2) VAD parçalama — $+99.6\%$ TR / $+160.1\%$ EN:** Yedi klibin yedisinde de hata artar; pencereler arası bağlam kaybedilir. Uygulanan eşik kodda gürültü tabanının %25'lik yüzdelik değerinin 2,2 katıdır; eski metinde yazılı "-40 dBFS" eşiği hiç kodlanmamıştır. GUI bu parçalayıcıyı hiç çağırmaz (sabit 30 sn dilim, `gui/pipeline.py`).
- **(3) Sözlük koşullama — $+99.4\%$ TR / $+64.4\%$ EN:** Yalnızca altın listesi olan TR-1/EN-1'i etkiler. TR-1 $0.0333 \to 0.1562$ (68 kelimelik önek yankısı, "Proje Yönetimi" 4→38 geçiş), EN-1 $0.0194 \to 0.0869$ ve $55\,\text{s} \to 398\,\text{s}$ ($7.2\times$). Kapsamdaki %96.9→%100.0 artışı **sızıntıdır**, tanıma değil; kalan beş klip bayt düzeyinde değişmez.
- **(4) LLM düzeltme — $+62.9\%$ TR / $-2.9\%$ EN:** Yedi klipten dördünde düzeltme öncesi hatayı düşürür, ancak hiçbir klip ham tabanın altına inemez. TR-1'de hata yeniden yaklaşık iki katına çıkar ($0.1562 \to 0.3236$) ve yineleme oranı $0.1361 \to 0.2723$ olur; bozulmuş bir girdiyi model onarmak yerine tekrar eder.

Klip düzeyindeki dağılım homojen değildir; toplu (makro) satır bulguyu gizler (normalize WER):

| Klip | (1) Ham | (2) +VAD | (3) +Koşullama | (4) +Düzeltme |
|:---|:---:|:---:|:---:|:---:|
| **TR-1** $^{\dagger}$ | 0.0159 | 0.0333 | 0.1562 | 0.3236 |
| **TR-2** | 0.0142 | 0.0351 | 0.0351 | 0.0358 |
| **TR-3** | 0.0319 | 0.0552 | 0.0552 | 0.0424 |
| **EN-1** $^{\dagger}$ | 0.0063 | 0.0194 | 0.0869 | 0.0869 |
| **EN-2** | 0.0101 | 0.0318 | 0.0318 | 0.0310 |
| **EN-3** | 0.0110 | 0.0192 | 0.0192 | 0.0183 |
| **EN-4** | 0.0130 | 0.0346 | 0.0346 | 0.0311 |

$^{\dagger}$ Altın sözlüğü olan ve bu nedenle koşullama isteminin uygulandığı iki klip. Altın listesi olmayan beş klibin (3). kademe hipotezi (2). kademe hipoteziyle birebir aynıdır (bayt düzeyinde doğrulandı); bu beş satır aynı zamanda koşullama karşılaştırmasının **determinizm kontrolüdür**. Düzeltme kademesi (4). kademede dört klipte (TR-3, EN-2, EN-3, EN-4) iyileştirme yapar, EN-1'i değiştirmez, TR-2'yi hafifçe kötüleştirir ve TR-1'i bozar; neticesinde yedi klipten hiçbiri kendi ham tabanının altına dönemez.

*Not (kapsam sütunu):* Sütun kapalı altın terim listesi üzerinde **geçiş kapsamıdır (mention coverage)** — kanonik ad + takma adlar, belirteç sınırında eşleşme — ve çıkarım $F_1$'i **değildir**. (3) ve (4) kademelerinin her ikisi de istemini `data/gold_terms.json` üzerinden kurar (`asr_post_correction.py`, `get_domain_context`), yani puanlandıkları referans isteme enjekte edilmiştir; bu iki kademede yorumlanabilir tek metrik WER'dir.

*Not (eski koşullama hücresi ölçülemezdi):* Önceki taslağın "+Terminoloji Alan Koşullama" hücresi bu kod yoluyla üretilemezdi: sözlük öneki ile `max_new_tokens=440`, Whisper'ın 448 decoder konumunu aştığı için `generate()` her klipte hata veriyordu (arşiv: `outputs/asr-vad-bias-large-v3.jsonl.positions-crash-attempt1` ve `...meta.json.positions-crash-attempt1`). Ölçüm, bütçe `448 − önek − 16` olacak şekilde kırpıldıktan sonra (TR-1 için 188, EN-1 için 108 belirteç) yapılabilmiştir. Çalışma süresi de bedava değildir: koşullama, istemi alan iki klipte duvar saatini TR-1'de $1.8\times$, EN-1'de $7.2\times$ katlar.

*Not (geri çekilen stres testi):* "Kontrollü yüksek gürültü stres testinde (%26,3 simüle edilmiş WER) düzeltme modülü terim $F_1$'ini +%29,8 artırdı" iddiası `asr_post_correction.py --test` kendi sınamasından gelir ve **geri çekilmiştir**: aynı komut 26,3 ve +%29,8 çiftini birebir yeniden üretir, çünkü "düzeltilmiş" dizim altın referansın kopyasıdır ve düzeltme WER'i zorunlu olarak %0,0 çıkar. Bu bir fixture öz-denetimidir, gürültü dayanıklılığı ölçümü değildir.

### 6.2 Semantik Katman (LLM) ve Ablasyon Bulguları
Çapraz katmanlı ablasyon, *aynı* açık ağırlıklı LLM'e (`Qwen/Qwen3.5-4B`) farklı konuşma tanıma mimarilerinin ürettiği hipotezleri verir. Whisper ailesi sonuçları birinci insan değerlendiricinin etiketlerine (`annotation/labels_a1.csv`, 233 çift), genişletilmiş 2026 kontrol noktaları ise `experiments/score_ablation_judge.py` ile üretilen **gerçek (non-mock) hakem etiketlerine** (`annotation/labels_ablation_real_judge.csv`, 200 çift; özet: `outputs/ablation_judge.json`) dayanır. Tablonun tamamı `experiments/make_ablation_table.py` ile bu iki etiket dosyasından ve `outputs/ablation-all.jsonl` çıktısından yeniden üretilir.

> **Yöntem düzeltmesi:** 2026 kontrol noktalarının daha önce raporlanan $R1/D1$ değerleri (`%88.6`, `%93.1`, `%100.0`) `use_mock=True` anahtar kelime sezgiselisinden (`annotation/labels_ablation_extended.csv`) geliyordu ve metinlerde "insan kriterleriyle doğrulanmış LLM-as-a-Judge" olarak nitelenmişti. Bu nitelendirme yanlıştır: söz konusu sezgisel bir hakem değildir ve doğrulandığına dair diskte artefakt yoktur. Aşağıdaki satırlar yerel `Qwen/Qwen3.5-4B` hakeminin kapalı uçlu rubrik çağrılarıyla yeniden etiketlenmiştir; sezgisel etiketler denetim izinin korunması için dosyada saklanmaya devam edilmiş, ancak hiçbir tabloda kullanılmamıştır.

| ASR Kaynağı | Dil | WER | Terim Sayısı | $R1\%$ (Bağlam) | $D1\%$ (Doğruluk) | Değerlendirme Yöntemi |
|:---|:---:|:---:|---:|---:|---:|:---|
| **Whisper Medium** | TR | 0.035 | 28 | 89.3 | 92.9 | İnsan Değerlendirici 1 |
| **Whisper Medium** | EN | 0.025 | 48 | 97.9 | 100.0 | İnsan Değerlendirici 1 |
| **Whisper Large-v3** | TR | 0.020 | 31 | 90.3 | 93.5 | İnsan Değerlendirici 1 |
| **Whisper Large-v3** | EN | 0.025 | 51 | 98.0 | 100.0 | İnsan Değerlendirici 1 |
| **Whisper Turbo** | TR | 0.037 | 27 | 96.3 | 96.3 | İnsan Değerlendirici 1 |
| **Whisper Turbo** | EN | 0.028 | 48 | 95.8 | 97.9 | İnsan Değerlendirici 1 |
| **selimc/whisper-turbo-tr** | TR | 0.082 | 29 | 96.6 | 100.0 | Gerçek LLM Hakemi |
| **Qwen3-ASR 0.6B** | TR | 0.134 | 35 | 94.3 | 100.0 | Gerçek LLM Hakemi |
| **Qwen3-ASR 0.6B** | EN | 0.021 | 56 | 94.6 | 100.0 | Gerçek LLM Hakemi |
| **Nemotron 3.5 ASR** | TR | 0.214 | 29 | 96.6 | 100.0 | Gerçek LLM Hakemi |
| **Nemotron 3.5 ASR** | EN | 0.035 | 51 | 94.1 | 96.1 | Gerçek LLM Hakemi |

*Not:* WER sütunu, ilgili ASR hipotezinin o hücredeki klipler üzerinden ortalama normalize kelime hata oranıdır (`asr_wer_norm`). İnsan ve hakem hücreleri farklı değerlendirme protokollerine dayandığı için tek homojen bir ölçek gibi yorumlanamaz; bu nedenle yöntem sütunu tabloda açıkça ayrılmıştır.

#### 6.2.1 Ulusal Egemen Yapay Zekâ Platformu (EVREN) Modelleri Kıyaslaması
Yerel GPU VRAM baskısını azaltmak amacıyla TÜBİTAK BİLGEM / SSB EVREN platformundaki H200 GPU kümesi üzerinden sunulan ulusal modeller değerlendirilmiştir. Tablodaki her hücre diskteki gerçek çıktı artefaktlarından (`outputs/llm-20261010-214422.jsonl`, `outputs/llm-20261010-214130.jsonl`) yeniden hesaplanmıştır; $R1\%$/$D1\%$ ise `experiments/score_evren_judge.py` ile yerel `Qwen/Qwen3.5-4B` hakem modelinin **kapalı uçlu (non-mock) rubrik çağrılarıyla** etiketlenmiş 374 terim-açıklama çiftinden gelir (`annotation/labels_evren_judge.csv`, özet: `outputs/evren_judge.json`).

| Model / Altyapı | Dil | Terim | $R1\%$ | $D1\%$ | Ort. Gecikme | Aralık | Çağrı $n$ |
|:---|:---:|---:|---:|---:|:---:|:---:|:---:|
| **EVREN Qwen 3.8 Flash** | TR | 53 | %100.0 | %100.0 | 6.11 sn | 4.23–8.32 | 3 |
| **EVREN Qwen 3.8 Flash** | EN | 72 | %100.0 | %100.0 | 4.92 sn | 2.92–5.97 | 4 |
| **EVREN DeepSeek V4.1 Flash** | TR | 107 | %94.4 | %99.1 | 27.68 sn | 21.80–31.83 | 3 |
| **EVREN DeepSeek V4.1 Flash** | EN | 142 | %99.3 | %100.0 | 19.44 sn | 19.23–19.72 | 4 |
| **Qwen3.5-4B (yerel, 4-bit, Large-v3 beslemeli)** | TR | 31 | %90.3 | %93.5 | 39.55 sn | 37.21–41.31 | 3 |
| **GPT-4.1 (ticari API)** | TR | 78 | — | — | 14.60 sn | 12.20–17.26 | 3 |
| **GPT-4o-mini (ticari API)** | TR | 36 | — | — | 10.96 sn | 9.16–13.60 | 3 |

**Ölçüm koşulları:** Her iki EVREN modeli için `reasoning_effort="none"`, `temperature=0.0`, `max_tokens=4096`; gecikme tek istem-yanıt turunun duvar saati süresidir. $n$ sütunu, hücrenin kaç klip çağrısına dayandığını gösterir — Türkçe hücrelerin tamamı yalnızca 3 klibe dayanır. GPT satırlarındaki $R1/D1$ hücreleri bilinçli olarak boştur: `annotation/labels_a1.csv` yalnızca üç Whisper × Qwen3.5-4B hücresini kapsar ve ticari API çıktıları için insan veya hakem etiketi artefaktı diskte bulunmamaktadır; daha önce bu satırlara yazılan %96.2/%98.7 değerleri kaynaklandırılamadığı için kaldırılmıştır.

**Ablasyon Yorumu ve Katmanlar Arası Davranış:**
1. **İngilizce Hipotezlerde "Kusursuzluk" İddiası Geri Alınmıştır:** Anahtar kelime sezgiseliyle yapılan eski koşumda İngilizce hücrelerin tamamı %100 $R1$/$D1$ gösteriyordu. Gerçek hakem etiketleriyle İngilizce kesinlik $Qwen3$-ASR 0.6B için %94.6, Nemotron 3.5 için %94.1 ($D1$ de %96.1) ölçülmüştür; yani bağlamla uyumsuz en az iki terim her İngilizce hücrede mevcuttur ve "hata oranı artsa bile dil modeli kusursuz yakalar" genellemesi desteklenmez. Kalan $D1 = 100.0$ hücreleri de mutlak doğruluk olarak değil, hakemin olumlu sınıfa yönelik hoşgörü tavanı olarak okunmalıdır.
2. **Türkçede Akustik Bozulma ile Terim Kesinliği Arasında Monoton Bir İlişki Yoktur:** Eski yorum, Qwen3-ASR 0.6B'nin yüksek WER'ine (%13.4) bağlı olarak terim kesinliğinin %88.6'ya gerilediğini ve fonetik kaymaların terim sınırlarını bozduğunu ileri sürüyordu. Gerçek hakem koşumunda aynı hücre %94.3 $R1$ alırken, daha temiz iki Türkçe hücre (Whisper Large-v3 %90.3, Whisper Medium %89.3) *daha düşük* kesinlik üretmektedir; yani %WER sıralaması ile $R1$ sıralaması örtüşmemektedir. Bu, $R1\%$'nin üretilen terimler üzerinde bir kesinlik olması ve aşırı-üretim davranışının akustik hatadan bağımsız değişmesiyle tutarlıdır; ancak her hücre 3–4 klipten oluştuğu için bu örtüşmezliğin mekanizmasını (ör. daha temiz transkriptin daha az terim ürettirmesi) bu veriden ayırt etmek mümkün değildir. Sonuç, "fonetik bozulma terim kesinliğini belirlenen oranda düşürür" şeklinde nedensel bir cümleyle ifade edilemez.
3. **EVREN Modelinin Hız/Kapsam Ticareti (ölçülmüş değerlerle):** EVREN Qwen 3.8 Flash, Türkçede ortalama 6.11 sn ile yerel 4-bit Qwen3.5-4B'in Large-v3 beslemeli Türkçe üçlüsüne (39.55 sn) göre **6.5 kat**, aynı üç klipte GPT-4.1'e (14.60 sn) göre **2.4 kat** daha hızlıdır; daha önce raporlanan "$\approx$7 kat / 3 kat" oranları kaynak artefaktlarla desteklenmemektedir. Buna karşın platformun diğer modeli EVREN DeepSeek V4.1 Flash **bulut olmasına rağmen iki ticari API'den de yavaştır** (TR 27.68 sn, EN 19.44 sn) ve "ulusal bulut otomatik olarak daha hızlandırır" genellemesini çürütür. DeepSeek'in karşılığı kapsamdır: 107 TR / 142 EN terim üretmesi, Qwen 3.8 Flash'ın 53 / 72 üretmesine göre belirgin biçimde daha zengin bir terim envanteri kurar. Önemli bir teknik ayrıntı: düşünme (reasoning) belirteçleri `max_tokens` bütçesinden harcandığı için 2048 belirteçlik ilk denemede DeepSeek V4.1'in TR-2 çıktısı JSON dizisi tamamlanmadan kesilmişti; 4096 belirteç ve `reasoning_effort="none"` ile hem bu kesilme giderilmiş hem de tüm hücreler tek bir yapılandırmayla yeniden koşmuştur. Qwen 3.8 Flash'ın 125 çiftinin tamamında %100 $R1$/$D1$ alması modelin kusursuzluğu kadar **yerel 4B hakemin olumlu sınıfa yönelik hoşgörü eğiliminin** de sonucu olabilir; bu nedenle hücre küçük $n$'li bir tarama olarak okunmalıdır ve kapalı altın liste duyarlılığı Bölüm 6.3'te ayrıca raporlanmıştır.
4. **Semantik Metriklerde Kesinlik vs. Duyarlılık:** Tablodaki $R1\%$ değerlerinin üretilen terimler üzerindeki kesinlik (precision) olduğu unutulmamalıdır; kapalı referans sözlükleri bulunan TR-1 ve EN-1 alt kümelerindeki tam duyarlılık (recall) ve $F_1$ analizi ise Bölüm 6.3'te incelenmiştir.

### 6.3 Kapalı Altın Liste Üzerinden Duyarlılık (Recall) ve $F_1$ Analizi

$R1\%/$$D1\%$ metrikleri yalnızca modelin *ürettiği* terimleri puanladığı için, modelin hiç aklına getirmediği alan terimlerini cezalandırmaz. Bu eksikliği kapatmak amacıyla TR-1 ve EN-1 dersleri için her biri 32 alan terimi ve izin verilen eşanlamlılarını içeren kapalı altın terim listeleri (`experiments/data/gold_terms.json`) hazırlanmış; çıkarım yapılan terimler kesin token-sınırı / eşanlamlı eşleşmesiyle bu listelerle karşılaştırılmıştır. Koşum betiği `experiments/eval_gold_terms.py` olup aşağıdaki tablo `outputs/ablation-all.jsonl` ve `outputs/llm-20260913-210556.jsonl` dosyalarından birebir yeniden üretilebilir.

| Dil | ASR Girdisi | Altın (Gold) | Üretilen (Emit.) | TP | Kesinlik $P\%$ | Duyarlılık $R\%$ | $F_1\%$ |
|:---|:---|---:|---:|---:|---:|---:|---:|
| TR-1 | Whisper Medium | 32 | 10 | 7 | 70.0 | 21.9 | 33.3 |
| TR-1 | Whisper Large-v3 | 32 | 12 | 8 | 66.7 | 25.0 | 36.4 |
| TR-1 | Whisper Turbo | 32 | 8 | 7 | 87.5 | 21.9 | 35.0 |
| TR-1 | selimc Turkish Turbo | 32 | 8 | 7 | 87.5 | 21.9 | 35.0 |
| TR-1 | Qwen3-ASR 0.6B | 32 | 10 | 5 | 50.0 | 15.6 | 23.8 |
| TR-1 | Nemotron 3.5 ASR | 32 | 9 | 9 | 100.0 | 28.1 | 43.9 |
| TR-1 | Qwen3.5-9B (ayrı koşum, girdi kaynağı kayıtlı değil) | 32 | 16 | 15 | 93.8 | 46.9 | 62.5 |
| EN-1 | Whisper Medium | 32 | 14 | 13 | 92.9 | 40.6 | 56.5 |
| EN-1 | Whisper Large-v3 | 32 | 21 | 18 | 85.7 | 56.2 | 67.9 |
| EN-1 | Whisper Turbo | 32 | 20 | 17 | 85.0 | 53.1 | 65.4 |
| EN-1 | Qwen3-ASR 0.6B | 32 | 18 | 16 | 88.9 | 50.0 | 64.0 |
| EN-1 | Nemotron 3.5 ASR | 32 | 16 | 14 | 87.5 | 43.8 | 58.3 |

**Bulgular:**
1. **Duyarlılık, kesinlik kadar yüksek değildir.** EN-1'de en iyi sonuç Whisper Large-v3 ile \%56.2 duyarlılık ($F_1 = \%67.9$); TR-1'de ise hiçbir ASR girdisi \%28.1 duyarlılığın üzerine çıkamaktadır (en iyi $F_1 = \%43.9$, Nemotron). Başka bir deyişle sistem, Türkçe derste altın listedeki terimlerin üçte birinden azını yakalayabilmektedir.
2. **Türkçe dezavantajı çok katmanlıdır.** TR-1'de üretilen terim sayısı (8–12) EN-1'e (14–21) göre düşüktür; sondan eklemeli Türkçede terim sınırlarının tek bir kelimeye indirgenmesi ("kazanılmış değer" yerine "değer") eşleşmeyi zorlaştırmaktadır.
3. **Modeller arası sıralama değişir.** $R1\%$ bakımından birbirine yakın görünen ASR girdileri, kapalı liste duyarlılığında belirgin biçimde ayrışır; örneğin Qwen3-ASR 0.6B Türkçede en düşük $F_1$ (\%23.8) değerine düşmüştür.
4. **Daha güçlü bir LLM daha yüksek kapsam üretir.** Farklı bir koşumda (`outputs/llm-20260913-210556.jsonl`) TR-1 için sabit 4-bit `Qwen3.5-4B` yerine `Qwen3.5-9B` kullanılmış ve 16 terim üretilerek \%46.9 duyarlılık / \%62.5 $F_1$ elde edilmiştir; Whisper hipotezleriyle aynı LLM'e dönüldüğünde ise tavan \%28.1 duyarlılıkta kalmaktadır. **Uyarı:** bu 9B koşumunun girdi transkripti artefakta kaydedilmediği için (JSONL'de `asr_model` alanı yok) bu satır, ASR hipoteziyle beslenen diğer satırlarla aynı koşullu sayılmamalı; yalnızca "daha büyük bir LLM kapalı listede daha fazla terim üretebiliyor" biçiminde okunmalıdır. Katman 2'deki darboğazın yalnızca ASR hatası değil, dil modelinin kapsamı olduğu yönünde bir işaret olarak değerlendirilmelidir.

> **Kapsam uyarısı:** Bu analiz yalnızca kapalı altın listesi bulunan TR-1 ve EN-1 alt kümelerinde geçerlidir; diğer beş ders için kapalı liste oluşturulmadığından genel bir recall iddiası değildir.

### 6.4 Ölçülmüş Pedagojik Okunabilirlik (Ateşman / Flesch) ve Sözlük Benzerliği

"CEFR B1" hedefinin gerçekten tutturulup tutturulmadığı, açıklamaların **tek tek** okunabilirlik bandı içinde kalma payıyla ölçülmüştür. Tablo `experiments/make_readability_table.py` betiğiyle `outputs/readability_eval.json` artefaktından yeniden üretilir (makaledeki `tables/readability.tex` de aynı betiğin çıktısıdır); ortalamanın banda yakın olması ile açıklamaların banda düşme oranı bilinçli olarak ayrı sütunlarda verilmiştir.

| Model | Dil | Ölçek | Açıklama $n$ | Ref. eşleşen | Ort. skor | Hücre aralığı | **Bant içi pay (%)** | Sözel vekil $F_1$ (%) |
|:---|:---:|:---:|---:|---:|---:|:---:|---:|---:|
| EVREN Qwen 3.8 Flash | TR | Ateşman | 53 | 23 | 56.4 | — | **45.3** | 30.8 |
| EVREN Qwen 3.8 Flash | EN | Flesch | 72 | 46 | 56.7 | — | **18.1** | 29.6 |
| EVREN DeepSeek V4.1 Flash | TR | Ateşman | 107 | 32 | 65.8 | — | **39.2** | 24.1 |
| EVREN DeepSeek V4.1 Flash | EN | Flesch | 142 | 55 | 56.0 | — | **28.9** | 26.7 |
| Qwen3.5-4B (yerel 4-bit) | TR | Ateşman | 179 | 78 | 48.0 | 42.1–53.6 | **41.9** | 30.5 |
| Qwen3.5-4B (yerel 4-bit) | EN | Flesch | 254 | 141 | 56.2 | 52.4–61.3 | **28.7** | 28.9 |

**Bulgular:**
1. **B1 iddiası bir "tam uyum" iddiası değildir; ölçülen bant içi pay %18.1–%45.3 arasındadır.** Yani üretilen açıklamaların çoğu tek tek bakıldığında hedef bandın dışına düşmektedir. Türkçede ortalama skor (48.0) bandın (*50–69*) hemen altında kalırken, İngilizcede Flesch ortalamaları (56.0–56.7) hedef bandın (*60–70*) altında, yani beklenenden daha zor metinler üretildiğini göstermektedir. Bu nedenle raporun hiçbir yerinde "tüm açıklamalar B1 düzeyindedir" denemez; denilebilecek ifade "açıklamaların yaklaşık beşte ikisi–yarısı hedef bandtadır" biçimindedir.
2. **Bant içi pay, ASR girdisine göre dalgalanır.** Yerel 4-bit `Qwen3.5-4B` için Türkçe hücre ortalamaları 42.1 ile 53.6 arasında değişmektedir; yani hangi transkriptin beslediği, okunabilirliği tek bir sabit LLM için bile değiştirmektedir.
3. **Sözlük benzerliği düşük ve bağlamsal değildir.** Eşleşen terim-alt kümesi üzerinde ölçülen sözel vekil $F_1$ %24.1–%30.8 bandındadır. Bu sütun **BERTScore değildir** (`bert_score` paketi ortamda kurulu olmadığından `eval_semantic.compute_bert_score` belgelenmiş yedek dala düşer); yüzey örtüşmesi ölçütü olarak okunmalıdır. Yüksek $D1\%$ skorları ikili "tanım doğru mu?" yargılarıdır; bu vekil skor ise sözlük ifadesinin kelime düzeyinde ne kadar tekrarlandığını ölçer — ikisi aynı şey değildir.
4. **BLEU ölçülmüş ama bilgilendirici bulunmamıştır.** `experiments/eval_bleu.py` (sacrebleu 2.6.0) ile İngilizce `outputs/ablation-all.jsonl` üzerinden 20 hücre / 107 eşleşmiş çift için hesaplanan BLEU **0.73–6.06** (ort. 2.04, medyan 1.46) aralığındadır (`outputs/bleu_ablation.csv`). Serbest paraphrase ile sözlük tanımı arasındaki n-gram örtüşmesinin yapısal düşüklüğü nedeniyle bu metrik model sıralamasında kullanılmamıştır; Türkçe için hesaplanmamıştır.

---

## 7. Geliştirilen Yerel Demo Arayüzü (Interactive Local Studio)

Çalışmanın deneysel bulgularının pratik bir boru hattında nasıl çalıştığını somutlaştırmak amacıyla yerel bir etkileşimli demo stüdyosu (`experiments/gui/`, `run_gui.bat` / `python run_gui.py`) geliştirilmiştir.

### 7.1 Arayüzün Mimarisi ve Bileşenleri
- **Arka Uç (Backend):** Python 3.12, FastAPI ve Uvicorn asenkron yerel sunucusu (`run_gui.bat` / `run.bat`).
- **Ses İndirme ve İşleme:** `yt-dlp` ile kullanıcı tarafından girilen YouTube bağlantısından ses akışı çekilir; `ffmpeg` ile 16 kHz mono WAV standardına dönüştürülür.
- **Model Yürütme:** Yerel GPU üzerinde çalışan Whisper / Qwen çıkarım motorları ile transkripsiyon ve terminoloji çıkarma gerçekleştirilir.
- **Ön Yüz (Frontend):** Modern koyu tema (dark mode), cam efektli (glassmorphism) kart tasarımı, Vanilla CSS ve modern JavaScript.
- **Kullanıcı Deneyimi:** Canlı konsol güncellemeleri eşliğinde video oynatıcı ile senkronize çalışan terim kartları, öğrencinin videoyu izlerken terimlere tıklayıp açıklamaları okumasına imkan tanır (açıklamaların okunabilirlik bandına düşme payı ölçülmüş ve Bölüm 6.4'te raporlanmıştır; her açıklama B1 bandında değildir).

### 7.2 Manimator Studio ve Uyarlanabilir Video İzleme Modülü
Arayüze eklenen en kapsamlı yenilik, öğrencinin ders akışında zorlandığı anı anında görselleştiren **Manimator Studio** arayüzüdür:
- **Hızlı Tetikleme Butonları:** Video kontrol çubuğunda ve Akıllı Slayt penceresinde yer alan *"🎬 Özel Video Üret (Manimator)"* butonları, aktif ders saniyesini ve o saniyedeki teknik terimleri otomatik olarak yakalar.
- **Aşamalı İlerleme Göstergesi (Stepper):** Qwen senaryo yazımı, Edge-TTS seslendirmesi, Manim sahne çizimi ve FFmpeg birleştirme adımları canlı animasyonlu aşamalarla kullanıcıya aktarılır.
- **Entegre Video Oynatıcı & Senkronize Altyazı:** Üretilen video modal içerisinde tam ekran, hız seçimi (1x/1.25x/1.5x) ve indirme butonlarıyla oynatılır. Konuşulan her cümle altyazı şeridinde gerçek zamanlı vurgulanır; altyazıdaki herhangi bir cümleye tıklandığında video tam o sahneye sarar.
- **Manim Kod İnceleme ve Öğrenci Profili Entegrasyonu:** Öğrenci üretilen Python Manim kodunu inceleyip kopyalayabilir; *"Anladım! Öğrenci Profilime Ekle"* butonuna tıklayarak aralıklı tekrar algoritmasında (`learner.py`) ilgili terimi pekiştirebilir.

### 7.3 Mimari Sınırlar ve Jüri Notu (Ürün Değil, Yerel Demo Stüdyosu)
> **Jüri Sunumu ve Savunma Uyarısı (Canlı Ürün İddiasından Kaçınma):**  
> Bu arayüz kesinlikle "canlıya alınmış / ticarileştirilmiş bir web ürünü" (deployed production product) veya bir sistem mühendisliği makalesi konusu olarak tanıtılmamalıdır. Yerel laboratuvar ortamında geliştirilmiş bir **konsept kanıtlama ve demo stüdyosu (PoC / local studio demo)** aracıdır:
> 1. **Ağ ve CORS:** Kod tabanında `CORSMiddleware` hiç kurulmamıştır (`gui/` altında `allow_origins` içeren bir satır yok); arayüz ile API aynı kaynaktan servis edildiğinden tarayıcı tarafında cross-origin okuma varsayılan olarak kapalıdır. Raporun önceki sürümünde bu madde "kontrolsüz açık `CORS *`" diye yazıyordu; bu kendi kendimizi yanlış suçlamak olduğu için düzeltildi. Gerçek sınır şudur: sunucu `0.0.0.0` üzerine bağlandığında ağa açık hâle gelir; başlatıcı bu durumda URL'e zorunlu bir erişim belirteci ekler, ancak CORS veya iyi niyet (rate limiting) yapılandırması yoktur.
> 2. **İş Yönetimi (Job Store):** İş durumları kalıcı bir veri tabanı (PostgreSQL, Redis vb.) yerine geçici bellek içi sözlükte (`in-memory dictionary`) tutulmaktadır; sunucu kapatıldığında geçmiş silinir.
> 3. **Geçici Dosyalar (Temp WAVs):** İndirilen ve dönüştürülen ses dosyaları yerel diskte geçici olarak birikmekte olup otomatik bir depolama yaşam döngüsü / çöp toplama (garbage collection) altyapısı barındırmamaktadır.
> 4. **Yetkilendirme:** Sistemde kullanıcı oturumu, kimlik doğrulama (auth) veya hız sınırlaması (rate limiting) bulunmamaktadır.
> 
> Bu teknik tercihler yerel bir araştırma stüdyosu için tamamen kabul edilebilir ve uygundur; ancak arayüzün üretim ölçeğinde bir sistem değil, ampirik bir gösterim aracı olduğu savunmada açıkça ifade edilmelidir.

### 7.4 Katman 3 İşlerlik Kanıtı: Diskteki Video Artefaktları (RQ4'ün Ölçülmüş Kısmı)

Katman 3'ün çalıştığını iddia ederken dayanılan tek şey, diskte gerçekten duran yedi MP4 dosyasıdır (`experiments/gui/static/generated_videos/`). Süre, çözünürlük, kare sayısı ve ses akışı `ffprobe` ile ölçülmüştür:

| Artefakt | Süre (sn) | Video akışı | Kare | Ses akışı | Boyut (KB) |
|:---|---:|:---:|---:|:---:|---:|
| `explainer_1791110589_76b1a3.mp4` | 41.83 | 854×480 @ 15 fps | 627 | AAC 24 kHz | 799.2 |
| `explainer_1791110828_c10d79.mp4` | 41.83 | 854×480 @ 15 fps | 627 | AAC 24 kHz | 775.5 |
| `explainer_1791117351_2c079f.mp4` | 26.02 | 854×480 @ 15 fps | 390 | AAC 24 kHz | 496.4 |
| `explainer_1791117563_fb6679.mp4` | 26.02 | 854×480 @ 15 fps | 390 | AAC 24 kHz | 491.7 |
| `explainer_1791117665_9e1677.mp4` | 22.34 | 854×480 @ 15 fps | 335 | AAC 24 kHz | 406.9 |
| `explainer_1791145399_1542a4.mp4` | 22.34 | 854×480 @ 15 fps | 335 | AAC 24 kHz | 407.7 |
| `explainer_1791145995_aca8fa.mp4` | 31.03 | 854×480 @ 15 fps | 465 | AAC 24 kHz | 600.3 |

**Ölçülen sonuçlar ve iddia sınırının dürüstçe çizilmesi:**
1. **Yedi üretim de uçtan uca render edilmiştir** (sesli, 480p/15 fps, AAC 24 kHz seslendirmalı); yani boru hattı "kod yazıyor" aşamasında değil, "izlenebilir video üretiyor" aşamasında doğrulanmıştır.
2. **30–90 saniyelik tasarım hedefi tam tutmamıştır:** ölçülen süre aralığı **22,3–41,8 sn**'dir ve yedi üretimden **dördü 30 sn alt sınırının altında** kalmaktadır. Raporda ve arayüzde geçen "30–90 sn" bir **tasarım hedefi / girdi kısıtıdır**, ölçülmüş bir çıktı özelliği değildir.
3. **RQ4'ün cevaplanmayan alt soruları:** kaçıncı denemede render edildiği (self-heal tur sayısı), uçtan uca üretim süresi (senaryo yazımı + TTS + Manim + FFmpeg duvar saati) ve seslendirme/görüntü senkron hatası bu yedi koşum için **hiçbir artefakta kaydedilmemiştir**; `gui/manimator.py` azami 2 onarım denemesine izin verdiğini kodda belirtir ancak deneme başına sayaç yazmaz. Ayrıca hangi terim/ders saniyesinden üretildiği dosya adının dışında hiçbir yan dosyada (`sidecar JSON`) tutulmamıştır; eş süreli dosya çiftleri bu yüzden birbirinden ayırt edilememektedir. Sonuç olarak Katman 3 için **gecikme, deneme sayısı, fallback oranı ve algısal/öğrenme değeri ölçülmemiştir**; bu katman bir işlerlik kanıtıdır, benchmark'ı değildir.
4. **Bağımlılıklar ve Kurulum:** Katman 3 için gerekli `manim` (0.21.0) ve `edge-tts` (7.2.8) kütüphaneleri `experiments/requirements.txt` dosyasına dahil edilmiştir; temiz kurulumda video üretim hattı uçtan uca çalıştırılabilmektedir.

---

## 8. 2026 Genişletme Paketi: Korpus Genişletme, Küçük Modeller (SRM/SLM), Dikey Fine-Tuning ve Hibrit RAG

Projenin ara değerlendirme ve toplantı geri bildirimleri doğrultusunda, pilot havuzunun ötesine geçen kapsamlı bir genişletme paketi hayata geçirilmiştir. Bu bölümde korpusun genişletilmesi, küçük modellerin uç donanım verimliliği, dikey fine-tuning hazırlığı, hibrit RAG mimarisi ve rekabetçi kıyaslama sonuçları sunulmaktadır.

### 8.1 Veri Korpusunun 7'den 40 Videoya Genişletilmesi (Kurumsal Uyum, KVKK ve İSG)
Pilot aşamasındaki 7 videoluk (~95 dakika) başlangıç havuzu, projenin genellenebilirliğini ve sektörler arası kapsamını artırmak amacıyla 40 videoluk (kaynak üstverisinde 10,1 saat; diskte süre doğrulaması yapılmış 16/40 videoda 4,6 saat) dengeli bir veri envanterine (`experiments/data/data_inventory.json` ve `.csv`) genişletilmiştir:
1. **Dengeli Dil Dağılımı:** 20 Türkçe ve 20 İngilizce olmak üzere toplam 40 video.
2. **Kapsanan Disiplinler:** STEM (Yapay Zeka, Fizik, Biyoloji, Devre Analizi), Sosyal Bilimler ve İşletme (Ekonomi, Finans, Psikoloji, Tarih) ve Kurumsal Zorunlu Eğitimler.
3. **KVKK / GDPR ve İSG / OSHA Entegrasyonu:** Şirket içi zorunlu eğitimleri temsil etmek üzere doğrulanmış 8 yeni video korpusa eklenmiştir:
   - `TR-17`: KVKK Temel Farkındalık Eğitimi: Veri Sorumlusu, Açık Rıza ve Haklar (21:13)
   - `TR-18`: KVKK Farkındalık ve Şirket İçi Bilgi Güvenliği Eğitimi (10:17)
   - `TR-19`: Temel İş Sağlığı ve Güvenliği Eğitimi: Risk Değerlendirmesi ve Acil Durumlar (62:10)
   - `TR-20`: İş Sağlığı ve Güvenliği Eğitimi: KKD, Güvenlik İşaretleri ve Yangın (45:50)
   - `EN-17`: What Are the 7 Principles of GDPR? Data Protection Fundamentals (08:00)
   - `EN-18`: GDPR Awareness Training for Employees (01:33)
   - `EN-19`: Managing Occupational Health and Safety in the Workplace (05:22)
   - `EN-20`: OSHA General Industry Safety Orientation for Workers (16:22)
4. **Depolama ve Bant Genişliği Verimliliği:** ASR ve terim çıkarımı süreçlerinde video görüntüsü indirilmemiş; `yt-dlp` üzerinden doğrudan 16 kHz Mono WAV formatında ses akışı çekilerek 60 GB'lık video yükü 1,8 GB'a düşürülmüş (%95'in üzerinde depolama tasarrufu).

### 8.2 Küçük Modeller (SRM & SLM) ve Uç Donanım Verimliliği
Pilot çalışmada yerel RTX 5060 8GB dizüstü GPU'sunda çalıştırılan büyük modellerin bellek aşımı yarattığı belgelenmiştir: Qwen 3.5-9B modeli VRAM sınırını aştığı için CPU RAM'e veri takası (offloading) yapmış ve çıkarım süresi istek başına **513,19 saniyeye (~8,5 dakika)** fırlamıştır.

Bu darboğazı aşmak amacıyla küçük tanıma modelleri (SRM) ve küçük dil modelleri (SLM) için kapsamlı bir kıyaslama ve çıkarım paketi (`experiments/benchmark_srm_slm.py` ve `experiments/outputs/srm_slm_benchmark.json`) geliştirilmiştir. Tabloda yerel donanım üzerinde ölçülen değerler ile literatür tabanlı mimari tasarım projeksiyonları açıkça ayrılmıştır:

| Model | Aile | Parametre | VRAM (INT4/nf4) | GPU Hızı / Gecikme | Statü | WebGPU / ONNX | Notlar |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Whisper Tiny | SRM | 39M | 140 MB | 0.012 s (RTF) | *Tasarım Tahmini* | Evet | İstemci tarayıcısında gerçek zamanlı çalışır (83x). |
| Whisper Base | SRM | 74M | 210 MB | 0.022 s (RTF) | *Tasarım Tahmini* | Evet | Tarayıcıda çevrimdışı dinleme için dengeli ASR. |
| Whisper Small | SRM | 244M | 240 MB (478MB fp16) | 0.055 s (RTF, literatür) | VRAM: Ölçülen / RTF: Literatür | Evet | Yerel GPU'da VRAM ölçüldü (478 MB kalıcı, 573 MB tepe). 0.055 RTF literatür sabitidir. |
| Qwen3-ASR 0.6B | SRM | 620M | 650 MB | 0.078 s (RTF) | *Tasarım Tahmini* | Sınırlı | Yerel çok dilli tokenizer avantajı. |
| Whisper Large-v3 Turbo | SRM (Ref) | 809M | 820 MB (1,6GB fp16) | 0.044 s (RTF) | **Ölçülen (RTX 5060)** | Hayır | Referans doğruluk standardı (RTF 0.025-0.044 ölçülmüştür). |
| Qwen 2.5 0.5B Instruct | SLM | 490M | 450 MB | 8.5 ms/tok | *Tasarım Tahmini* | Evet | WebLLM ile doğrudan Chrome içinde sıfır sunucu maliyeti. |
| Llama 3.2 1B Instruct | SLM | 1.23B | 980 MB | 12.0 ms/tok | *Tasarım Tahmini* | Evet | Kompakt bellek izi, hızlı çıkarım. |
| Qwen 2.5 1.5B Instruct | SLM | 1.54B | 1.12 GB | 14.2 ms/tok | *Tasarım Tahmini* | Evet | RTX 5060 8GB için hedef mimari denge noktası (<1.8s toplam süre). |
| SmolLM2 1.7B Instruct | SLM | 1.71B | 1.25 GB | 15.5 ms/tok | *Tasarım Tahmini* | Evet | Sentetik veri temelli güçlü akıl yürütme. |
| Qwen 2.5 3B Instruct | SLM | 3.09B | 2.10 GB | 22.4 ms/tok | *Tasarım Tahmini* | Hayır | 2.1 GB VRAM ile 7B kalitesine yakın semantik çıkarım. |
| Qwen 3.5 4B (nf4) | SLM | 4.00B | 3.081 MiB (3,01 GiB) | 39.55 s (istek ort.) | **Ölçülen (RTX 5060)** | Hayır | 3.081 MiB kalıcı, 4.721 MiB tepe ayrılan, 5.136 MiB ayrılmış, 5.262 MiB cihaz VRAM. |

**Kazanım:** Küçük modeller mimarisi ile uç cihazda verimli çıkarım hedeflenmektedir; nitekim Qwen 2.5-1.5B INT4 modeli için öngörülen **1,12 GB bellek ve <1,8 saniyelik** çıkarım profili (tasarım tahmini), devasa 9B modelin yerel GPU aşımı nedeniyle CPU'ya takas yaptığı ve istek başına **513,19 saniyeye** ulaştığı darboğaza kıyasla teorik olarak **~280 katlık bir hızlanma potansiyeline** işaret etmektedir. Yerel donanımda fiilen ölçülen Qwen 3.5-4B (nf4) ise 3.081 MiB kalıcı ağırlık (3,01 GiB), 4.721 MiB tepe ayrılan (peak allocated), 5.136 MiB ayrılmış (reserved) ve 5.262 MiB toplam cihaz VRAM'i ile 8 GB bütçesine başarıyla sığmaktadır.

### 8.3 Dikey Alan Veri Setleri, SFT Karışımı ve Gerçekleştirilen LoRA Adaptasyonu
Genel amaçlı açık modellerin dikey sektör ve eğitim terimlerinde yaşadığı kaçırma (under-recall) sorununu gidermek ve uç model adaptasyonunu somutlaştırmak amacıyla kapsamlı bir veri ve eğitim hattı kurulmuştur:

1. **Telifli ve Sızıntısız Harici Konuşma Korpusu (Katman 1):** CommonVoice 17.0 (CC0-1.0, 826 klip, 0,75 saat) ve Khan Academy Türkçe (CC-BY-SA 3.0, 256 klip, 0,75 saat) kaynaklarından toplam **1,50 saatlik (1082 klip, 16 kHz mono WAV)** lisanslı harici korpus derlenmiştir (`experiments/finetune/prepare_external_datasets.py`). Veri setinin oluşturulmasında geliştirilen **Tier-A ve Tier-B LeakageGuard** devrede tutulmuş; test kümesindeki 7 dersin transkript n-gramları veya altın terim rumuzları (alias) kesinlikle eğitim verisine sızdırılmamıştır.
2. **Wikidata QID Tabanlı SFT Eğitimi Karışımı (Katman 2):** Turkish-LLM-v10 (Apache-2.0, %70 genel Türkçe instruction) ve TurkLink (CC-BY-SA 4.0, %30 Wikidata varlık tanımlı terim çiftleri) entegrasyonuyla 2000 örneklik (1800 eğitim / 200 doğrulama) yapılandırılmış ChatML eğitim seti oluşturulmuştur.
3. **Gerçekleştirilen SRM LoRA Eğitimi ve Ampirik Sonuç:** Uç konuşma tanıma modeli olan Whisper Small üzerinde yerel RTX 5060 GPU'da LoRA adaptasyonu (rank $r=16$, $\alpha=32$, $q$ ve $v$ projeksiyon katmanları, 385 küresel adım, ~480 saniye) fiilen koşturulmuştur (`experiments/finetune/finetune_whisper_asr.py`). Model adaptörü (`experiments/finetune/checkpoints/srm_whisper_small_tr/adapter_model.safetensors`, SHA-256: `69f60090aa309578e9b3d2f2042d66336774f795b776c71d17a96c30ba6d4543`) 18 Ekim 00:53'te kaydedilmiş; 21 Ekim 22:11'de 54 kliplik tutulmuş doğrulama kümesi üzerinde yeniden değerlendirilerek `eval_wer.json` artefaktı üretilmiştir.
4. **Bağımsız Doğrulama Kümesi Sonuçları:** CommonVoice ve Khan Academy'den ayrılan ve eğitim boyunca hiç görülmeyen 54 kliplik (~3,6 dakika) doğrulama kümesinde modelin kelime hata oranı (WER) **%30,47'den %26,41'e** (%4,06 mutlak, %13,3 nisbi iyileşme), karakter hata oranı (CER) ise **%7,74'ten %5,10'a** gerilemiştir (`eval_wer.json`). **Önemli Metodolojik Not:** Bu sonuç, 7 derslik kapalı altın kümeden bağımsız, konuşmacı örtüşmesi belgelenmemiş cümle düzeyinde harici okuma konuşması üzerindeki bir **out-of-domain alan adaptasyonu** başarımıdır; makalede raporlanan ders içi altın referanslı %2,1–%4,9 WER değerleri ile doğrudan kıyaslanmamalı ve ayrı değerlendirilmelidir.

### 8.4 Hibrit RAG (Retrieval-Augmented Generation) ve Prerequisite DAG Motoru
Dil modellerinin parametrik belleğindeki bilgi boşluklarını gidermek ve kavramsal bütünlüğü denetlemek amacıyla RAG motoru geliştirilmiştir (`experiments/rag/domain_rag_extractor.py`):

1. **7 Dikey Ontoloji Bilgi Tabanı:** Turizm, İSG, KVKK, Bankacılık, Siber Güvenlik, Sürdürülebilirlik ve STEM alanlarında kanonik terim tanımları ve Bloom taksonomi düzeyleri tanımlanmıştır.
2. **Hibrit Arama (Hybrid BM25 + TF-IDF n-gram Cosine with RRF):** Sözcüksel anahtar kelime eşleşmesi ile semantik TF-IDF kosinüs benzerliği Karşılıklı Sıralama Füzyonu (RRF) ile birleştirilmiştir.
3. **Pedagojik Önkoşul Grafı (NetworkX DAG):** Terimler arasındaki bağımlılıklar yönlü döngüsüz çizge olarak modellenmiş; derste önkoşul temeli verilmeden doğrudan ileri kavramlara geçilmesi durumunda pedagojik öğrenme açıklarının (`missing_prerequisites`) tespiti için mekanizma geliştirilmiştir. İlk makalede (`paper/main.tex:198`) "uygulanmış ancak ampirik değerlendirilmemiştir" olarak belirtilen bu modül, 2026 genişletmesinde kurumsal eğitim konuşmaları üzerinde deneysel bir sondajla test edilmiştir (Madde 5).
4. **Ontolojik Bağlamlama ve Değerlendirme Çerçevesi:** Sözlük tabanlı RAG mimarisi, transkriptte geçen terimlerin kanonik ontolojiye çıpalanması ve önkoşul grafı üzerinden eksik bilgi tespiti amacıyla tasarlanmıştır. Makalede de açıkça vurgulandığı üzere (Bölüm~\ref{sec:prompting}), bağımsız ve tutulmuş (held-out) bir harici ontoloji bulunmaksızın bilgi tabanından türetilen sentetik sorgularla RAG başarısını altın referanslara karşı puanlamak döngüsel (circular) bir sonuç doğurur; bu nedenle bu kol makalede "uygulanmış ancak burada değerlendirilmemiştir" statüsündedir. Sentetik kontrol denemesinde (`outputs/rag_ablation_results.json`) kısıtsız sözcüksel tabana kıyasla yalnızca kesinlik artarken duyarlılık (0,929 vs 1,0) ve $F_1$ (0,963 vs 0,966) gerilemektedir; bu sonuç RAG'ın genel erişim üstünlüğü değil, yüzey eşleme kısıtının 15 adaydan 1 yanlış eşleşmeyi eleyerek kesinliği yükseltmesinden ibarettir.
5. **Kurumsal Konuşma Üzerinde 8 Kliplik Keşifsel RAG/DAG Mühendislik Sondajı:** Kurumsal zorunlu eğitim transkriptleri (KVKK ve İSG alanlarında 8 klip; TR-17..20, EN-17..20) üzerinde hibrit RAG ve önkoşul grafı sondajı koşturulmuştur (`experiments/rag/eval_corporate_rag.py`, `experiments/outputs/rag_corporate_results.json`). Transkript kalitesi asgari 40 WPM ve 50 kelime QA filtresiyle güvenceye alınmıştır. Bilgi tabanından getirilen 28 aday terimden 17'si transkript yüzeyinde doğrulanarak **%60,7 aday bağlamlama kesinliği** elde edilmiş, 11 transkript dışı aday filtrelenmiştir. Doğrulanan 17 terimin 7'si doğrudan kanonik ad eşleşmesi, 10'u ise eşanlamlı/alt-tür (alias/subtype) yüzey biçimidir. DAG analizi sonucunda 8 derste toplam **5 pedagojik önkoşul açığı** (`missing_prerequisites`) bayraklanmıştır.
   - **Kritik Metodolojik Kısıtlar ve Kapsam:**
     a) Bu sondaj **tam 40 videoluk bir ASR/kavram değerlendirmesi değildir**; yalnızca uzun transkriptli 8 kurumsal klip üzerinde bir mühendislik ön-doğrulamasıdır.
     b) Bayraklanan 5 pedagojik açığın doğruluğu için henüz bağımsız bir insan altın referansı (human gold) bulunmamaktadır.
     c) Eşleştirme motoru tek geçişli Türkçe ek soyucu ve kayan pencere kuralı (filler gap $\le 4$) kullanmaktadır; dolayısıyla lematik/semantik tam özdeşlik değil, yüzey ve rumuz (surface/alias) düzeyinde ontolojik bağlamlama sunmaktadır.

### 8.5 Rekabetçi Rakip Analiz Matrisi
Geliştirilen EduTerm AI mimarisi, sektördeki 6 öncü eğitim platformu (Coursera, Khan Academy, Duolingo, Udacity, edX, Quizlet) ile 10 eksende karşılaştırılmıştır (`notes/RAKIP_ANALIZI_VE_BENCHMARK_MATRISI.md`). EduTerm AI; konuşma tanımadan otomatik terim çıkarma (ASR-to-Term), kanonik RAG ile ontolojik bağlamlama, pedagojik önkoşul açığı tespiti ve tamamen yerel / uç cihazda çalışabilme (edge-first) boyutlarında sektördeki tek entegre çözüm olarak öne çıkmaktadır. *(Metodolojik Not: Kanonik RAG ve önkoşul açığı tespiti yetenekleri EduTerm AI'da çalışan prototip ve 8 kliplik kurumsal mühendislik sondajı düzeyinde doğrulanmış olup, henüz 40 videonun tamamında insan onaylı altın kıyaslama aşamasında değildir; rakip matrisindeki bu üstünlük mimari ve işlevsel yetenek boyutundadır.)*

---

## 9. Donanım ve Çalışma Ortamı

Deneysel değerlendirme ve yerel web uygulaması testi, Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü bünyesinde aşağıdaki laboratuvar donanımında yürütülmüştür. Aşağıdaki her satır koşum anında araçla doğrulanmıştır (`Get-CimInstance Win32_Processor`, `Win32_ComputerSystem`, `nvidia-smi`, `importlib.metadata`); sistem eşleştiğinde bu bölümün yeniden üretilmesi için `python experiments/check_env.py` çalıştırılır.
- **İşlemci (CPU):** Intel Core i7-14650HX — 16 fiziksel çekirdek / 24 mantıksal ünite (WMI base clock 2.200 MHz; 5,2 GHz değeri üreticinin turbo speküdür, bu çalışmada ölçülmemiştir).
- **Grafik İşlemci (GPU):** NVIDIA GeForce RTX 5060 Laptop GPU — `nvidia-smi` raporuna göre **8.151 MiB** toplam VRAM, sürücü 617.42, CUDA 13.0.
- **Sistem Belleği (RAM):** 34.057.011.200 bayt $\approx$ **32 GB** (31,7 GiB) DDR5.
- **İşletim Sistemi ve Yazılım:** Windows 11 64-bit (10.0.26300), Python 3.12.9, PyTorch 2.12.0+cu130 (CUDA 13.0), Hugging Face Transformers 5.16.1, FastAPI 0.141.1, ReportLab 4.5.1, sacrebleu 2.6.0.
- **Katman 3 bağımlılıkları (ölçülen kurulu sürümler):** Manim 0.21.0, edge-tts 7.2.8, yt-dlp 2026.8.19, pypdf 6.14.2, pypdfium2 5.13.0 (`experiments/requirements.txt` içinde listelenmiştir).

---

## 10. Sonuç, Kısıtlar ve Gelecek Çalışmalar

### 10.1 Sonuç
Bu bitirme çalışmasında, Türkçe ve İngilizce eğitim konuşmalarının başlangıç düzeyindeki öğrenciler için anlaşılır kılınmasını sağlayan üç katmanlı ASR+LLM+üretken video sistemi uçtan uca araştırılmış ve doğrulanmıştır:
1. Whisper ailesinin Türkçe ders kayıtlarında normalize **%2.1–%4.9** (Large-v3 / Turbo / Medium) hata bandıyla ve gerçek zamanlıdan **2,6–40 kat** hızlı çıkarımla (RTF 0.025–0.382; RTF ne kadar küçükse o kadar hızlı) çalıştığı; 2026 açık ağırlıklı modellerde ise hatanın %6.1 (Qwen3-ASR 1.7B) – %21.4 (Nemotron 3.5) aralığına çıktığı ölçülmüştür. İngilizcede aynı modeller %1.0–%3.5 bandındadır.
2. Kaliteli bir transkript sağlandığında dil modellerinin **üretilen terimler üzerinde yüksek bağlamsal geçerlilik** gösterdiği ölçülmüştür: insan değerlendirici etiketleriyle $R1\% = 90.3$ (TR) / $98.0$ (EN), $D1\% = 93.5$ / $100.0$. Bununla birlikte aynı sistemlerin **kilit kavramları yakalama duyarlılığı** sınırlı kalmıştır: 32'şer terimlik kapalı altın listelerde TR-1 için hiçbir ASR girdisi %28.1 duyarlılığı aşamamış ($F_1$ bandı %23.8–%43.9), EN-1'de en iyi $F_1$ %67.9 olmuştur. Benzer şekilde "B1 düzeyinde sadeleştirme" iddiası açıklamaların tamamını kapsamaz: okunabilirlik bandına düşen açıklama payı **%18.1–%45.3** arasında ölçülmüştür (Bölüm 6.4). Dolayısıyla doğrulanabilir sonuç, "kesinlik yüksek, tavan duyarlılık ve okunabilirlik bant içi payıdır".
3. Çift değerlendirici ve Cohen's kappa analizi ile semantik skorların değerlendirici öznelliğinin **nicelendiği** belgelenmiştir: bağlam uyumluluğunda $\kappa_R = 0.782$ (belirgin uyum; 8–12 negatif örnek ve 4 anlaşmazlık ile gerçek ayrım gücünü taşıyan skor) ve tanım doğruluğunda prevalans paradoksu nedeniyle zayıf kanıt olan $\kappa_D = 1.000$ ($P_e \approx 0.949$). 2026 genişleme hücrelerinde kullanılan yerel `Qwen3.5-4B` LLM hakeminin bu insan etiketleriyle **doğrulaması yapılmamıştır** (hakem–insan uyumu diskte artefaktı olan bir deney değildir); bu hücreler yöntem sütununda ayrı tutulmuştur.
4. Altın referanslı GPU sonuçları doğrultusunda net üç farklı dağıtım önerisi belirlenmiştir:
   - **Doğruluk-Öncelikli (Sunucu / Toplu İşlem):** En düşük hata oranına sahip Whisper Large-v3 (TR 0.021 / EN 0.010) ile yüksek kapasiteli bir LLM (GPT-4.1 veya Qwen3.5-9B).
   - **Pratik Yerel Dağıtım (Dizüstü / Uç Cihaz):** Aynı ana bilgisayarda duvar saatiyle Large-v3'e göre Türkçede **3,4 kat**, İngilizcede **7,7 kat** daha hızlı olan Whisper Turbo (RTF 0.025–0.044) ile 4-bit nf4 kuantalı açık ağırlıklı Qwen3.5-4B. Bu modelin bellek izi **ölçülmüştür** (`outputs/local_vram_probe.json`, 8.151 MiB RTX 5060): 3.081 MiB kalıcı ağırlık (3,01 GiB), 4.721 MiB tepe ayırıcı kullanımı (peak allocated), 5.136 MiB ayrılmış bellek (reserved) ve 5.262 MiB tepe cihaz doluluğu — yani 8 GB'lık alana tek klip çağrısı için yer kalmaktadır, ancak bu tek bir kontrol noktası için ölçülmüş bir sonuçtur ve "rahatlıkla sığar" geneli yoktur. Uçtan uca semantik çağrı gecikmesi ölçülmüş ortalaması 39,55 sn'dir (~40 sn).
   - **Egemen Bulut Düşük Gecikmeli Dağıtım (TÜBİTAK/SSB EVREN Platformu):** İstemci tarafında hafif konuşma tanıma (Whisper Turbo) ile ulusal altyapıda koşan EVREN Qwen 3.8 Flash (TR 6.11 sn / EN 4.92 sn). Bu yapı, semantik katmanın ağırlıklarının istemci VRAM'inde tutulmasını ortadan kaldırır ve ölçülmüş hâliyle yerel 4-bit çıkarıma göre 6.5 kat hız sağlar; ancak konuşma tanıma katmanı hâlâ uç cihazda koştuğundan "sıfır kaynak" değildir. Aynı platformdaki EVREN DeepSeek V4.1 Flash 27.68/19.44 sn ile iki ticari API'den yavaş olduğundan bu yapılandırma hız değil kapsam seçeneğidir. Öğrenci verisinin yurt içi bir operatöre yönlendirilmesi bir **veri ikamet tercihidir**; KVKK uyumluluğu, operatörün işleme sözleşmesi ve saklama politikası incelenmeden sağlanamaz ve bu belgeler elde edilmediği için rapor uyumluluk iddiası içermemektedir.

### 10.2 Kısıtlar
- Veri kümesi 7 ders ve 95 dakika ile sınırlıdır; farklı konuşmacı ve aksan çeşitliliği barındırsa da tüm disiplinleri kapsamamaktadır.
- **Okunabilirlik ve BLEU Ölçüm Kapsamı:** "CEFR B1" için ayrı bir sınıflandırıcı eğitilmemiş ya da kullanılmamıştır; bu hedef Türkçede Ateşman (50–69) ve İngilizcede Flesch (60–70) **bant içi pay** ölçümüyle operasyonelleştirilmiştir ve iki dilde de hesaplanmıştır (Bölüm 6.4). Dille sınırlı olan metrik BLEU'dur: `eval_bleu.py` yalnızca `data/oxford_refs_en.json` referans kümesini yüklediği için İngilizceye hesaplanabilmiş, ölçülen değerler (0.73–6.06) bilgilendirici olmadığından sıralamada kullanılmamıştır. Ayrıca **ticari API modelleri (GPT-4.1 / GPT-4o-mini) için okunabilirlik ya da benzerlik artefaktı yoktur**: kaydedilen çıktılar bu betiklerin tükettiği terim/açıklama şemasında değil yapılandırılmamış Markdown biçimindedir; bu satırlar tahmin edilmek yerine tablodan çıkarılmıştır.
- **Bağlamsal Gömme Benzerliği Ölçülmedi:** `bert_score` paketi deney ortamında kurulu olmadığından, sözlük benzerliği sütunu karakter üçlüsü (trigram) kosinüs + kelime örtüşmesi tabanlı **sözel bir vekildir**; bağlamsal gömme benzerliği (gerçek BERTScore) ölçülmemiştir.
- **ASR→LLM Hata Yayılımı Kısıtı:** Çapraz katmanlı ablasyonda dil modeline yalnızca Whisper modellerinin düşük WER'li hipotezleri beslenmiştir; Wav2Vec 2.0 veya SpeechBrain gibi yüksek hatalı modeller semantik katmana dahil edilmemiştir. Dolayısıyla çalışma ASR hatalarının LLM terim çıkarımına kademeli yayılımını sürekli bir kalite eğrisi olarak ölçmemekte, yalnızca yüksek kaliteli transkriptler üzerindeki davranışla sınırlı kalmaktadır.
- **$\kappa_D = 1.000$ ve Sınıf Dengesizliği (Prevalence) Kısıtı:** 115 terimlik çift etiketleme kümesinde tanım doğruluğu ($D$) için 112 pozitif ve sadece 3 negatif etiket bulunmaktadır. Şans eseri beklenen uyumun zaten %95 ($P_e \approx 0.949$) olduğu bu aşırı tek taraflı dağılımda $\kappa_D = 1.000$ çıkması matematiksel olarak beklenen bir sonuçtur ve değerlendirici kalitesinin güçlü bir ayrım kanıtı değildir (Kappa Paradoksu). Değerlendirici güvenilirliğinin asıl metodolojik dayanağı, negatif örnekler (8–12 adet) ve sınır vakalardaki 4 anlaşmazlığı barındıran $\kappa_R = 0.782$ skorudur.
- **Referans Metinlerin (Gold) Bağımsız Transkripsiyon Olmaması Kısıtı:** Veri kümesindeki transkriptler sıfırdan bağımsız bir insan dinlemesiyle üretilmemiştir. EN-1 ve EN-4 temizlenmiş insan altyazısı, TR-3 sesten taslak iken; EN-2, EN-3 ve TR-2 YouTube otomatik altyazılarının insan tarafından noktalanması ve terimlerinin düzeltilmesiyle (post-editing) oluşturulmuş, TR-1 ise otomatik taslak ile eksik insan altyazısının birleştirilmesinden elde edilmiştir. Bu durum, referansların otomatik altyazı ve Whisper stili cümle bölme/sentaks kalıplarına hafif bir yakınlık taşımasına yol açabilmekte ve WER skorlarında Whisper lehine marjinal bir biçimsel/stil avantajı yaratabilmektedir. Dolayısıyla metinler "bağımsız gold" olarak değil, "insan denetiminde düzeltilmiş referanslar" olarak değerlendirilmelidir.
- **Küçük Örneklem ve İstatistiki Test Yokluğu ($n=7$, Ağırlıksız Ortalama Kısıtı):** Dil bazlı ASR kıyaslamaları toplamda yalnızca 7 ders videosuna (3 Türkçe, 4 İngilizce) ve ağırlıksız ortalamalara dayanmaktadır; güven aralıkları veya istatistiksel anlamlılık testleri hesaplanmamıştır. Dolayısıyla bu çalışmadaki Türkçe WER değerlerinin İngilizceden yüksek çıkması, "Türkçenin konuşma tanıma için genel olarak daha zor bir dil olduğu" şeklinde yorumlanamaz ve genellenemez; bu fark incelenen derslerin akustik ortamı, konuşmacı hızı ve terminoloji yoğunluğuna bağlı olabilir.
- **Boş Kalan Protokol Hücreleri (Empty Cells - Wav2Vec 2.0 & SpeechBrain):** Makalede ve tezde bilimsel şeffaflık esastır: 2026 yenileme modelleri (Whisper Turbo, Qwen3-ASR 0.6B/1.7B - 4 İngilizce klibin tamamı dahil - ve Nemotron 3.5 ASR) GPU ortamında altın referanslarla eksiksiz koşturulmuş olup sonuçları doğrudan makale tablolarına (`\input{tables/asr_2026.tex}`) ve değerlendirme kitine yansıtılmıştır. Tablo III'te (`tab:asr-lang`) yalnızca İngilizce odaklı olan ve Türkçe dil desteği bulunmayan Wav2Vec 2.0 ile SpeechBrain kontrol noktaları altın referanslı GPU kümesinde yeniden koşturulmamış ve dürüstçe boş hücre (`---`) olarak bırakılmıştır; bu modellerin CPU pilot havuz değerleri tarihsel bağlam olarak sunulmaktadır.
- **Semantik Metriklerde Recall Kapsamının İki Dersle Sınırlı Olması:** $R1\%$/$D1\%$ üretimi *kesinlik* (precision) benzeri oranlardır ve modelin hiç aklına getirmediği terimleri cezalandırmaz. Duyarlılık (recall) ancak elle hazırlanan 32'şer terimlik kapalı altın listeler üzerinden ölçülebilmiştir ve bu listeler **yalnızca TR-1 ve EN-1** için mevcuttur (Bölüm 6.3): ölçülen tavan Türkçede %28.1, İngilizcede %56.2 duyarlılıktır. Diğer beş ders için kapalı liste oluşturulmadığından **recall ölçülmemiş**, bu hücreler boş bırakılmıştır; dolayısıyla 6.3'teki rakamlar 7 derslik bir duyarlılık ölçümü gibi genellenemez. GPT-4.1'in daha yüksek terim sayısı üretmesi (105 vs 56) kanıtlanmış bir tam kapsam üstünlüğü değil, yalnızca aday üretim hacmi göstergesidir.
- **Katman 3 (Üretken Video) Benchmark Değildir:** Bu katmanda diskteki kanıt yedi MP4 dosyasıdır; süreleri 22,3–41,8 sn, akış özellikleri 854×480 @ 15 fps + AAC 24 kHz olarak `ffprobe` ile ölçülmüştür (Bölüm 7.4). **Ölçülmeyenler:** render deneme/onarım sayısı, uçtan uca üretim gecikmesi, seslendirme–görüntü senkron sapması (altyazı zaman damgaları LLM'in beyan ettiği sahne sürelerinden türetilir, TTS sesinden ölçülmez), animasyonların pedagojik algısı ve öğrenme üzerindeki etkisi. **Figure Walkthrough (PDF) modunun hiç artefaktı yoktur.**
- **Web Arayüzünün Yerel Prototip Sınırı (Demo vs. Ürün):** Geliştirilen FastAPI arayüzü son kullanıcıya açık kurumsal bir üretim ürünü değil, yerel laboratuvar stüdyosudur. CORS yapılandırmasının bulunmaması (dışarıdan okuma varsayılan olarak kapalı olsa da ağa açık `0.0.0.0` bağlaması yalnızca belirteçle korunur), bellek içi iş yönetimi (in-memory job store), kalıcı kimlik doğrulama/oturum yönetimi ve hız sınırlaması bulunmaması ve geçici ses dosyası birikimi nedeniyle bir sistem makalesi konusu olmayıp, boru hattının yerel uygulanabilirliğini somutlaştıran ampirik bir PoC prototipidir.

### 10.3 Gelecek Çalışmalar
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
11. Frantzi, K., Ananiadou, S., & Mima, H. (2000). "Automatic recognition of multi-word terms: The C-value/NC-value method." *International Journal on Digital Libraries*, 3(2), 115-130.
12. Brusilovsky, P., & Millán, E. (2007). "User models for adaptive hypermedia and adaptive educational systems." *The Adaptive Web*, LNCS 4321, Springer, 3-53.
13. VanLehn, K. (2011). "The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems." *Educational Psychologist*, 46(4), 197-221.
14. Ateşman, E. (1997). "Türkçede okunabilirliğin ölçülmesi." *TÖMER Dil Dergisi*, 58, 71-74.
15. Flesch, R. (1948). "A new readability yardstick." *Journal of Applied Psychology*, 32(3), 221-233.
16. Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). "BLEU: A method for automatic evaluation of machine translation." *Proc. ACL*, 311-318.
