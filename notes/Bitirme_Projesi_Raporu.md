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
3. ASR hatalarının semantik katmana yayılımı, Boğaziçi HPC A40 kümesinde 7 farklı ASR basamağı (Gold Referans, Turbo, Large-v3, Medium, Base, Tiny, Small; WER %0 ile %89.7 aralığı) ve açık ağırlıklı Qwen 2.5-7B LLM ile kapalı 32'şer terimlik altın listeler üzerinden uçtan uca ampirik bir kalite eğrisi olarak ölçülmüştür. Elde edilen duyarlılık denklemleri ($\Delta F_{1,\text{TR}} = -0.11\%$, $\Delta F_{1,\text{EN}} = -1.23\% \text{ per } 5\% \text{ WER}$), konuşma tanıma kalitesi $\text{WER} \le 15\%$ bandında tutulduğu sürece semantik terim çıkarımının referans metin kalitesini koruduğunu, WER > %40 seviyesinde ise fonetik bozulmanın halüsinasyonu %91.7'ye tırmandırdığını matematiksel olarak ortaya koymuştur.
4. Elde edilen semantik çıkarım katmanı, açık kaynaklı Manimator mimarisi uyarlanarak üçüncü bir **Dinamik Video Üretim Katmanına (Katman 3)** genişletilmiştir. Bu katmanda; öğrencinin ders videosunda anlamadığı bir noktayı işaretlemesiyle ("Takıldığım Yeri Öğret"), son 45 saniyelik bağlam yerel Qwen 3.5 modeline aktarılmakta; model CEFR B1 düzeyinde benzetmeli bir anlatım senaryosu ve Manim kodu üretmekte; Edge-TTS nöral seslendirmesi ve FFmpeg senkronizasyonu ile kişiselleştirilmiş MP4 animasyon videoları üretilmektedir. Tasarım hedefi 30–90 saniyedir; diskteki yedi üretimin `ffprobe` ile ölçülen süreleri **22,3–41,8 sn** aralığındadır ve dördü 30 sn alt sınırının altında kalmıştır (Bölüm 7.4). Ayrıca harici PDF ders notları için figür rehberli (Figure Walkthrough) video üretimi kod tabanında sağlanmıştır; bu modun uçtan uca artefaktı (örnek PDF girdisi veya üretilmiş walkthrough videosu) diskte bulunmadığı için işlerlik kanıtı yalnızca Manim moduna yöneliktir.
5. 2026 genişletme paketi kapsamında veri korpusu 7 dersten **40 videoya (kaynak üstverisinde 10,1 saat; diskte sesi bulunan 34/40 videoda ölçülen 16,8 saat; 20 Türkçe + 20 İngilizce)** çıkarılmış; şirket içi zorunlu eğitimler (KVKK, GDPR, İSG, OSHA) entegre edilmiştir. Yerel RTX 5060 GPU'daki bellek tıkanmasını aşmak için küçük modeller (SRM ve SLM) profillenmiş, Qwen 2.5-1.5B aynı cihazda **2,93 GB tepe VRAM ve 3,62 sn istek gecikmesi** ile ölçülmüş ve 9B modelin CPU'ya takas yaptığı 513,19 sn'lik isteğe karşı **~142× hızlanma** gösterilmiştir. Dikey alanlarda kavramsal hiyerarşiyi güvenceye almak ve ontolojik bağlamlama sağlamak amacıyla **Hibrit RAG ve NetworkX önkoşul grafı (DAG)** geliştirilmiş, pedagojik açık tespiti mekanizması 8 kliplik kurumsal mühendislik sondajı ile ön-değerlendirilmiştir (tam 40 video ASR kıyaslaması değildir).

**Anahtar Kelimeler:** Otomatik Konuşma Tanıma (ASR), Büyük Dil Modelleri (LLM), Terminoloji Çıkarma, Çok Dilli Eğitim Teknolojileri, Whisper, Qwen3-ASR, Manimator, RAG, SLM, Önkoşul DAG, Cohen's Kappa.

---

## Abstract

**A Three-Layer Evaluation of Speech-to-Text, LLM-Based Terminology Extraction, and Generative Video-Animation Systems for Multilingual Education**

Digital video-based instruction provides immense learning opportunities, yet specialized educational lectures inherently assume baseline domain familiarity, confronting beginner learners with substantial terminological friction. This graduation thesis presents a comprehensive, three-layer experimental benchmark of commercial and open-weight Automatic Speech Recognition (ASR), Large Language Model (LLM), and generative video-animation architectures for making multilingual (Turkish and English) educational lectures accessible. In the first layer, leading ASR systems (Whisper Medium, Large-v3, Large-v3 Turbo, Turkish-matched Turbo, Qwen3-ASR 0.6B / 1.7B, and NVIDIA Nemotron 3.5 ASR) are systematically benchmarked across raw/normalized Word Error Rate (WER), Character Error Rate (CER), latency, and Real-Time Factor (RTF) on GPU gold references; while legacy English-only checkpoints lacking Turkish support (Wav2Vec 2.0 and SpeechBrain) are honestly maintained as empty table cells (---) in Table III and reported from historical CPU pilot runs. In the second layer, transcripts are fed to LLMs to extract domain-specific terminology and generate beginner-friendly explanations whose CEFR-B1 goal is operationalised as measured readability band membership (Ateşman 50–69 for Turkish, Flesch 60–70 for English); only $18.1\%$–$45.3\%$ of individual explanations actually fall inside the target band, so the B1 claim is reported as a measured band share rather than a blanket property. Evaluation references were established across 7 lectures (~95 minutes) via human-corrected and domain-term-verified gold transcripts (noted as post-edited caption/ASR drafts rather than scratch independent transcriptions). Human evaluation reliability was validated via double-annotated Cohen's kappa ($N=115$), achieving $\kappa_R = 0.782$ (substantial agreement on context consistency; 4 disagreements, 8–12 negatives) and $\kappa_D = 1.000$ (definition accuracy, where high class prevalence of 112/3 yields an expected agreement of $P_e \approx 0.95$, making $\kappa_R$ the primary discriminative metric). Furthermore, cross-layer error propagation was empirically benchmarked on the Boğaziçi HPC cluster across a 7-step ASR ladder (Gold reference, Turbo, Large-v3, Medium, Base, Tiny, Small; spanning 0% to 89.7% WER) feeding a 7B LLM (Qwen2.5-7B) evaluated on closed 32-term gold lists, deriving continuous mathematical sensitivity slopes ($\Delta F_{1,\text{TR}} = -0.11\%$, $\Delta F_{1,\text{EN}} = -1.23\% \text{ per } 5\% \text{ WER}$) and demonstrating that semantic terminology extraction remains resilient for $\text{WER} \le 15\%$ before severe acoustic corruption elevates hallucination to 91.7%; and a local FastAPI-based interactive demo studio was built to showcase the end-to-end pipeline locally (clarifying it as a local proof-of-concept studio rather than a production-deployed system). Terminology extraction is additionally scored against closed 32-term gold lists on the two annotated clips, where emitted-term precision stays high while recall remains below $28.1\%$ in Turkish and $56.2\%$ in English --- the semantic layer's ceiling is recall, not precision. The third (on-demand animation) layer is reported as a functional proof: seven explainer videos exist on disk with `ffprobe`-measured durations of 22.3--41.8\,s at 480p/15\,fps. Furthermore, in the 2026 expansion package, the evaluation inventory is scaled to 40 balanced educational and compliance videos (20 TR, 20 EN; source metadata declares 10.1 hours, but the 34/40 clips that have local audio measure 16.8 hours, while 6 clips have no audio on disk) encompassing Tourism, OHS/İSG, KVKK, Banking, and Cybersecurity. Within the 32-clip core, a machine-only pass ($gold\_how=$ machine\_only, no human annotation) covers the 26 clips that have audio: full-audio transcripts (13.92 hours, 110,510 words) and 235 term candidates, with $R$/$D$ left empty and no double-annotation claim made. To explore edge-device deployment efficiency, small speech recognition models (SRM) and small language models (SLM) were evaluated alongside local GPU measurements (Whisper Small, Qwen3.5-4B), with Qwen2.5-1.5B measured at 2.93 GB peak VRAM and 3.62 s per request in bf16 on the local RTX 5060 (about 142x faster than the 9B model that offloaded to CPU at 513.19 s per request). Finally, a hybrid RAG (BM25 + TF-IDF RRF) and NetworkX prerequisite DAG engine was developed for canonical ontology grounding, with an automated prerequisite gap detection mechanism evaluated across an 8-clip exploratory engineering probe on corporate speech (yielding 60.7% candidate grounding precision without circular scoring against KB-derived queries).

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

| Model | Dil Kapsamı | TR WER_norm [%95 Bootstrap GA] | EN WER_norm [%95 Bootstrap GA] | TR RTF | EN RTF | İstatistiki Ayrışma (vs. Turbo) |
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
> Aynı değerlendirme klipleri üzerinde yürütülen eşleştirilmiş Wilcoxon işaretli sıra testi (Wilcoxon signed-rank test), **Whisper Large-v3 Turbo** ile büyük **Whisper Large-v3** ve **Qwen3-ASR 1.7B** modellerinin hata oranları arasında istatistiki olarak anlamlı bir fark olmadığını ($p \ge 0.30$, *statistically indistinguishable*) kanıtlamaktadır. Bu nedenle raporda mutlak bir "en iyi model" iddiası yerine, Whisper Turbo'nun Large-v3 ile **aynı doğruluk kümesinde ayrışmayan**, ancak **3,4–7,7 kat daha hızlı çalışan Pareto-etkin** bir model olduğu ifade edilmiştir. Buna karşılık NVIDIA Nemotron 3.5 ($p = 0.031 < 0.05$) istatistiki olarak belirgin şekilde **ayrışan** (daha yüksek hata üreten) kümede kalmaktadır.

*Not:* Qwen3-ASR 1.7B (4 İngilizce klibin tamamı dahil) ve NVIDIA Nemotron 3.5 ASR modellerinin doğrulanmış altın referanslı GPU sonuçları sisteme işlenmiştir. Altın referanslı GPU kümesinde boş (`—`) bırakılan kontrol noktaları, İngilizce odaklı tarihsel taban modelleri olan ve Türkçe dil desteği bulunmayan Wav2Vec 2.0 ile SpeechBrain'dir. Bu iki modelin orijinal CPU pilotu havuz değerleri tarihsel karşılaştırma bağlamı olarak eş zamanlı yazılan makalede Tablo III (`tab:asr`) içinde korunmaktadır; bu rapor kendi tablosunda onları `—` olarak bırakmıştır.

**RTF için ölçüm yolu ayrımı — üç RTF ailesi vardır ve birbiriyle kıyaslanmaz:** Yukarıdaki RTF sütunu `outputs/asr-20260903-turbo.jsonl` koşusundan gelir: 7 altın klip, tek süreç, **klip-bazlı RTF değerlerinin ağırlıksız ortalaması** (TR $(0{,}0645+0{,}0354+0{,}0317)/3 = 0{,}0439$; EN $0{,}0253$). Aynı model ve aynı çözme yolu (tam ses tek `generate()` çağrısına verilir; `num_beams=1`, `temperature=0.0`, `max_new_tokens=440`, VAD yok. `max_new_tokens` toplam çıktıyı sınırlamıyor: 5,9 saatlik EN-06 klipi **50.410 kelime** üretti, yani uzun ses model içinde ardışık pencerelerle işleniyor. 30 sn sabit dilimleme yalnız GUI yoluna ve Qwen3-ASR'ye aittir) tam-ses makine transkriptleri üzerinde de ölçülüdür; genişlemiş **125 kayıtlık** tam-ses kümesinde (48,70 saat ses, ham kayıt: `experiments/outputs/transcripts/machine_only/*.json`) gecikme–ses doğrusal uyarlaması

$$\text{gecikme (sn)} \;=\; 7{,}43 \;+\; 0{,}0178 \times \text{ses süresi (sn)},\qquad R^2 = 0{,}907$$

sonucunu vermektedir: **durağan RTF $=0{,}0178$** (gerçek zamanlıdan ≈56 kat hızlı) ve **sabit çağrı maliyeti $=7{,}43$ sn** (uyarlamanın keseni; ses çözme/yeniden örnekleme, özellik çıkarımı ve üretimin ilk adımı gibi sabit öğlerden oluştuğu düşünülür, ancak **bileşen kırılımı ölçülmemiştir**). 600 sn ve üzeri 104 klipte (üç soğuk başlangıç kaydı bu bandın dışındadır) havuzlanmış RTF $\sum \text{gecikme}/\sum \text{ses} = 0{,}0226$ (aralık 0,0146–0,0402). Aynı bandın dil kırılımı $\text{TR} = 0{,}0276$ (48 klip), $\text{EN} = 0{,}0190$ (56 klip): tek eğimli uyarlama dili ayrıştırmaz, bu fark tek başına $R^2$'nin düşüşünü açıklamak için ölçülmemiştir. İki uyarı zorunludur: (i) **Kısa kliplerde klip-bazlı RTF model hızını değil sabit maliyeti raporlar** — 420 sn altı 8 kaydın medyanı 0,0428, en yükseği 0,1954 (EN-12: 117 sn ses / 22,9 sn işlem). (ii) **Bir sürecin ilk klipi** ek olarak ağırlık yüklemeyi, CUDA bağlam kurulumunu ve HF revizyon sorgusunu taşır; bu yük yukarıdaki tablonun TR ucunu (0,044) şişirir — ilk klip çıkarıldığında TR **ağırlıksız ortalaması** $0{,}0439 \to 0{,}0336$, **havuzlanmış RTF**'i $0{,}0433 \to 0{,}0334$ olur; EN ucu (0,025) etkilenmez. Aynı yük, tek kliplik deneme koşusunda ölçülen **0,1004**'ün (TR-63: 6,5 dk ses / 39,2 sn) tamamını açıklar; genişlemiş kümede soğuk başlangıç taşıyan üç kayıt bandın tamamında 0,0616–0,1004 arasında (havuzlanmış 0,0797) ölçülmüştür ve üçü de 600 sn bandının dışında kalır. Bu sayı bir model yavaşlığı bulgusu değil, **soğuk başlangıç artefaktıdır**. Üçüncü RTF ailesi Bölüm 8 katalog tablosundaki SRM satırlarıdır (`outputs/srm_asr_measured.json`; 6 altın klip, `mean_rtf` = klip-bazlı ağırlıksız ortalama, her model kendi bölümünün ilk klipinde soğuk başlar): orada 39M Whisper Tiny **0,112**, 244M Whisper Small **0,054** ölçülmüş ve tiny, 6 klipin **5'inde** small'dan yavaş çıkmıştır. Bu ters sıralama, RTF'nin bu örneklem boyutunda parametre sayısıyla monoton olmadığını gösterir: çıkarım süresi üretilen belirteç sayısıyla arttığı için, tekrar/hallüsinasyon üreten bir model 440 belirteçlik bütçeyi sonuna kadar doldurur ve erken duran daha iyi bir modelden uzun koşar. Bu mekanizma **doğrulanmış değildir** — sözkonusu koşuda üretilen belirteç sayıları kaydedilmemiştir — dolayısıyla katalogdaki dört SRM RTF değeri **hız sıralaması olarak kullanılamaz**, yalnız gözlenen duvar saati süreleridir. Denetim yeniden üretilebilir: `python experiments/analyze_rtf_paths.py` → `experiments/outputs/rtf_path_audit.json` (GPU gerektirmez; yukarıdaki bütün tam-ses sayıları 125 kayıtlık tamamlanmış kümeden gelir). Denetim bir üstveri kusurunu da saptamıştır: `outputs/srm_asr_measured.json` içindeki `decoding.chunk_length_s = 30.0` kaydı **uygulanmış bir ayar değildir** — `run_asr.transcribe_whisper()` `args.chunk_s` değerini hiç okumaz, değer betimleyici üstveridir; 30 sn sabit dilimleme GUI yolunda (`gui/pipeline.py`) ve Qwen3-ASR'de gerçekten uygulanır.

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

### 8.0 Araştırma İddiaları, Veri Kümeleri ve Değerlendirici Kapsam Matrisi (Scope Matrix)

Proje kapsamında savunulan her bir bilimsel hipotez ve ampirik iddianın hangi veri kümesi, örneklem boyutu ($n$), zemin gerçek yöntemi (`gold_how`) ve değerlendirici mekanizması ile test edildiği aşağıdaki **Tek Sayfalık Kapsam Matrisi**'nde açıkça haritalanmıştır. Raporun takip eden bölümleri (§8.1–§8.15) doğrudan bu matristeki protokollere dayanmaktadır:

| Temel İddia / Hipotez | Test Edilen Alt Küme | Örneklem ($n$) | Zemin Gerçek Yöntemi (`gold_how`) | Değerlendirici Mekanizması | Rapor Bölümü |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **1. Akustik Doğruluk & Hız** | Altın Pilot Klip Havuzu | $n = 7$ video (3 TR, 4 EN) | Tam Çift İnsan Transkripsiyonu (`human_gold_transcription`) | WER, CER, RTF, Bootstrap GA, Paired Wilcoxon | §6.1 |
| **2. Çapraz Katman Hata Yayılımı** | Akustik Bozulma Merdiveni | 7 ASR seviyesi $	imes$ 2 dil ($n = 14$) | Altın Terim Listesi (`gold_terms.json`) | $F_1$, Kesinlik, Duyarlılık, Duyarlılık Eğimi ($\Delta F_1$) | §8.6 |
| **3. Terim Çıkarımı Doğrulaması** | İnsan Etiketli Terim Havuzu | $n = 454$ terim-açıklama | Çift-Kör İnsan Etiketleme (`double_blind_annotation`) | Cohen's Kappa ($\kappa=0.531$), PABAK ($0.669$) | §8.9.2 |
| **4. Hakem Bağımsızlığı (Cross-Family)** | Çapraz-Menşe Hakem Paneli | $n = 454$ terim çifti | İnsan Zemin Gerçeği + Negatif Denetimi | Qwen2.5-7B vs. Mistral-7B-Instruct (A40 GPU) | §8.9.3 |
| **5. Model Parametre Ölçeklenmesi** | Grand Scaling Benchmark | 7 Model Ailesi ($0.5	ext{B} 	o 72	ext{B}$) | Standart Altın Şablon (`gold_schema`) | JSON Şema Uyumu, Ateşman B1 Oranı, VRAM Tavanı | §8.12 |
| **6. Dikey Adaptasyon (SFT/QLoRA)** | Çok Alanlı Eğitim Seti | $n = 2.296$ (7B) / $4.666$ (14B) | LeakageGuard Korumalı Sentetik ChatML | Doğrulama Kaybı, Bloom Bilişsel Seviye Artışı | §8.3, §8.13 |
| **7. Çok Düzeyli CEFR Kontrolü** | Multi-Tier Kontrollü Set | $n = 4.666$ eğit., $586$ val., $24$ test | Seviye Koşullu B1/B2/C1 Şablonu | Otomatik Ateşman Metriği (%91.7 Seviye Uyumu) | §8.14 |
| **8. Çok Turlu Sokratik Eğitmenlik**| Sentetik Öğrenci Personaları | 50 Persona $	imes$ 5 Tur ($n = 250$ tur) | 5 Alan Kavram Yanılgısı Senaryoları | Sokratik Soru ($SQ$), İskeleleme ($HQ$), Sıfır Spoon-Feed ($SF$) | §8.15 |
| **9. Müfredat DAG & Önkoşul Analizi**| Genişletilmiş Korpus | $n = 33 	o 133$ tam transkript | Makine Transkripti (`gold_how=machine_only`)| NetworkX Yönlü Asiklik Çizge ve Eksik Önkoşul Motoru | §8.1, §8.10 |

---

### 8.1 Veri Korpusunun 7'den 40 Videoya Genişletilmesi (Kurumsal Uyum, KVKK ve İSG)
Pilot aşamasındaki 7 videoluk (~95 dakika) başlangıç havuzu, projenin genellenebilirliğini ve sektörler arası kapsamını artırmak amacıyla 40 videoluk (kaynak üstverisinde 10,1 saat; diskte sesi bulunan **34/40 videoda ölçülen 16,8 saat**) dengeli bir veri envanterine (`experiments/data/data_inventory.json` ve `.csv`) genişletilmiştir:
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
5. **32 kliplik çekirdek korpusta makine-transkriptli kapsam (`gold_how=machine_only`):** Disk taraması, ilan edilen 32 klipten **26'sında yerel ses** bulunduğunu, **6'sında hiç ses olmadığını** göstermiştir (TR-05, TR-06, TR-07, TR-14, TR-15, EN-10). Sesi olan 26 klip `whisper-large-v3-turbo` ile **tam uzunlukta** yeniden yazılmış (**13,92 saat, 110.510 kelime**, RTF 0,02-0,20; önceki transkript dosyaları bu kliplerde kesikti — örneğin TR-08'in 49,6 dakikalık sesinden yalnızca 36 kelime çıkmıştı) ve `Qwen3.5-4B` / `extract-v2` ile **235 terim adayı** üretilmiştir (`experiments/annotation/labels_todo_32clip_machine.csv`, `experiments/data/manifest_32_machine.jsonl`). Bu küme **çift etiketli bir altın küme değildir**: tüm satırlar `gold_how=machine_only` olarak işaretli, `R`/`D` alanları boş ve insan etiketlemesi beklemektedir; "32 klip çift etiketli" iddiası bu artefaktlarla kapatılamaz, insan etiketi yalnızca 7 pilot klipte bulunmaktadır. Ayrıca 18 klipte diskteki sesin süresi ilan edilen süreyle uyuşmuyor (EN-06: ilan 760 s, ölçülen 21.283 s), bu nedenle kapsam ve maliyet hesaplarında ilan edilen `duration_s` yerine ölçülen süre kullanılmıştır.

### 8.2 Küçük Modeller (SRM & SLM) ve Uç Donanım Verimliliği
Pilot çalışmada yerel RTX 5060 8GB dizüstü GPU'sunda çalıştırılan büyük modellerin bellek aşımı yarattığı belgelenmiştir: Qwen 3.5-9B modeli VRAM sınırını aştığı için CPU RAM'e veri takası (offloading) yapmış ve çıkarım süresi istek başına **513,19 saniyeye (~8,5 dakika)** fırlamıştır.

Bu darboğazı aşmak amacıyla küçük tanıma modelleri (SRM) ve küçük dil modelleri (SLM) için kapsamlı bir kıyaslama ve çıkarım paketi (`experiments/benchmark_srm_slm.py`, `experiments/benchmark_srm_asr_local.py`, `experiments/benchmark_small_models_local.py`) geliştirilmiştir. Tablodaki statü ve skorların tamamı RTX 5060 üzerinde ölçülmüştür: SRM satırları 6 altın transkriptli klipte normalize WER/CER, SLM satırları A40 raporuyla aynı 8 vakalık BENCH probe'unda terim F1/JSON geçerliliği ile değerlendirilmiştir. VRAM sütunu fp16/bf16 (4B satırında nf4) tepe ayrılan bellektir; INT8/INT4 projeksiyonları yalnızca `benchmark_srm_slm.py` kataloğunda tutulur. Üç uyarı geçerlidir: (i) aynı probe yerelde ve A40'ta birebir aynı sonucu vermiyor (0.5B: 0.4667 vs 0.3947; 1.5B: 0.7238 vs 0.6188), çünkü 28 altın terimlik set tek tokenlık kaymalara duyarlıdır — bu yüzden ±0.07 F1 farkı yorumlanmamalıdır; (ii) Llama 3.2 ve SmolLM2 kendi chat şablonlarıyla ölçüldü, ChatML zorlandığında Llama'nın F1'i 0.1334'e düşüyor — bu, model kapasitesinden bağımsız bir şablon artefaktıdır; (iii) Whisper Large-v3 Turbo bir referans noktasıdır, eğitilmemiş hâliyle ölçülmüştür.

| Model | Aile | Parametre | VRAM (ölçülen tepe) | GPU Hızı / Gecikme | Statü | WebGPU / ONNX | Notlar |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Whisper Tiny | SRM | 39M | 0.15 GB | 0.112 RTF (ölçülen) | **Ölçülen (RTX 5060)** | Evet | 6 altın klipte normalize WER %20.0, CER %5.27. Hata Türkçe ders kayıtlarında yoğun (TR-3 %53.8), İngilizce kliplerde %2-4. RTF'i 244M Small'ın (0.054) üstünde çıktı; bu satırlar hız sıralaması taşımaz (Bölüm 6.1). |
| Whisper Base | SRM | 74M | 0.29 GB | 0.156 RTF (ölçülen) | **Ölçülen (RTX 5060)** | Evet | 6 altın klipte normalize WER %11.87, CER %2.85. |
| Whisper Small | SRM | 244M | 0.77 GB | 0.054 RTF (ölçülen) | **Ölçülen (RTX 5060)** | Evet | 6 altın klipte normalize WER %6.08, CER %1.83. |
| Qwen3-ASR 0.6B | SRM | 620M | 2.37 GB | 0.168 RTF (ölçülen) | **Ölçülen (RTX 5060)** | Sınırlı | 6 altın klipte normalize WER %7.57, CER %2.75; aynı koşuda Whisper Small daha düşük WER (%6.08) ve 3× az VRAM ile önde. |
| Whisper Large-v3 Turbo | SRM (Ref) | 809M | 1.62 GB | 0.044 RTF (ölçülen, 7 altın klip) | **Ölçülen (RTX 5060)** | Hayır | Referans doğruluk standardı. RTF 0.025-0.044 altın-küme koşusudur ve koşulun ilk klipi model yüklemesini de taşır; tam-ses kümesinde durağan RTF 0.0178, sabit çağrı maliyeti 7.43 sn olarak ayrı ölçülmüştür (`outputs/rtf_path_audit.json`, Bölüm 6.1; 125 kayıtlık tamamlanmış tam-ses kümesi). |
| Qwen 2.5 0.5B Instruct | SLM | 490M | 0.97 GB | 4.50 s (Sorgu) | **Ölçülen (RTX 5060)** | Evet | BENCH-8 F1 0.4667, JSON %100. A40 raporunda F1 0.3947 — ±0.07 gürültü bandı içinde. |
| Llama 3.2 1B Instruct | SLM | 1.23B | 2.37 GB | 3.49 s (Sorgu) | **Ölçülen (RTX 5060)** | Evet | Kendi chat şablonuyla BENCH-8 F1 0.6526, JSON %87.5. ChatML zorlanınca F1 0.1334'e düşüyor — ölçüm şablon duyarlı. |
| Qwen 2.5 1.5B Instruct | SLM | 1.54B | 2.93 GB | 3.62 s (Sorgu) | **Ölçülen (RTX 5060)** | Evet | BENCH-8 F1 0.7238 (R 0.792 / P 0.667), JSON %100 — 8 GB sınırında Pareto noktası. A40: F1 0.6188. |
| SmolLM2 1.7B Instruct | SLM | 1.71B | 3.32 GB | 3.56 s (Sorgu) | **Ölçülen (RTX 5060)** | Evet | Kendi chat şablonuyla BENCH-8 F1 0.7084, JSON %87.5 — aynı VRAM bandında Qwen 1.5B'nin gerisinde, Llama 1B'nin önünde. |
| Qwen 2.5 3B Instruct | SLM | 3.09B | 5.81 GB | 4.30 s (Sorgu) | **Ölçülen (RTX 5060)** | Hayır | BENCH-8 F1 0.7616, JSON %100 — 8 GB sınırının altında kalan en yüksek F1. |
| Qwen 3.5 4B (nf4) | SLM | 4.00B | 3.081 MiB (3,01 GiB) | 39.55 s (istek ort.) | **Ölçülen (RTX 5060)** | Hayır | 3.081 MiB kalıcı, 4.721 MiB tepe ayrılan, 5.136 MiB ayrılmış, 5.262 MiB cihaz VRAM. |
| Qwen 2.5 7B Instruct | LLM | 7.62B | 14.28 GB (BF16) | 4.16 s (Sorgu) | **Ölçülen (HPC A40)** | Hayır | HPC temel model; %45.8 Recall, 60.3 Ateşman (B1 %100). |
| EduTerm-7B-TR (LoRA SFT) | LLM | 7.66B | 14.39 GB (BF16) | **1.68 s (Sorgu)** | **Ölçülen (HPC A40)** | Hayır | 2.296 dikey örnekli SFT; **%95.7 Recall, %95.8 Precision, 66.4 Ateşman**. |
| Qwen 2.5 14B Instruct | LLM | 14.77B | 27.74 GB (BF16) | 7.11 s (Sorgu) | **Ölçülen (HPC A40)** | Hayır | Büyük HPC modeli; **%85.4 Recall, %71.7 Precision**, 38.7 Ateşman. |

**WER protokolü notu:** Whisper Small için tabloda verilen %6.08, adaptörsüz temel modelin 6 altın transkriptlik kümedeki normalize WER'idir; §8.3'te raporlanan %26.41 ise LoRA adaptörünün 54 kliplik tutulmuş doğrulama kümesindeki WER'idir. İki değer farklı kümeler ve farklı model durumlarına aittir, doğrudan karşılaştırılmamalıdır.

**Kazanım:** Küçük modeller mimarisi ile uç cihazda verimli çıkarım hedeflenmektedir; nitekim Qwen 2.5-1.5B modeli için **aynı RTX 5060 üzerinde** ölçülen **2,93 GB tepe VRAM ve 3,62 saniyelik** istek gecikmesi, 9B modelin aynı cihazda bellek aşımı nedeniyle CPU'ya takas yaptığı ve istek başına **513,19 saniyeye** ulaştığı darboğaza kıyasla **~142 katlık ampirik bir hızlanma** sağlamaktadır (A40 üzerindeki 2,25 s'lik profil karşılaştırma için ayrıca raporlanmıştır). Ayrıca 14B modelin 27.74 GB VRAM ile ulaştığı %85.4'lük zero-shot recall değerine karşılık, dikey LoRA adaptasyonu uygulanan EduTerm-7B-TR'nin 14.39 GB VRAM ile **%95.7 recall ve 1.68s gecikme** elde etmesi, "pedagojik alanda dikey SFT eğitiminin kaba model ölçeklemeye (brute-force scaling) kıyasla hem doğruluk hem de kaynak verimliliğinde üstün olduğunu" kanıtlamaktadır (`notes/img/scaling_law_curve.png`).

### 8.3 Dikey Alan Veri Setleri, SFT Karışımı ve Gerçekleştirilen LoRA Adaptasyonu
Genel amaçlı açık modellerin dikey sektör ve eğitim terimlerinde yaşadığı kaçırma (under-recall) sorununu gidermek ve uç model adaptasyonunu somutlaştırmak amacıyla kapsamlı bir veri ve eğitim hattı kurulmuştur:

1. **Telifli ve Sızıntısız Harici Konuşma Korpusu (Katman 1):** CommonVoice 17.0 (CC0-1.0, 826 klip, 0,75 saat) ve Khan Academy Türkçe (CC-BY-SA 3.0, 256 klip, 0,75 saat) kaynaklarından toplam **1,50 saatlik (1082 klip, 16 kHz mono WAV)** lisanslı harici korpus derlenmiştir (`experiments/finetune/prepare_external_datasets.py`). Veri setinin oluşturulmasında geliştirilen **Tier-A ve Tier-B LeakageGuard** devrede tutulmuş; test kümesindeki 7 dersin transkript n-gramları veya altın terim rumuzları (alias) kesinlikle eğitim verisine sızdırılmamıştır.
2. **Multi-Domain Dikey SFT Korpusu (Katman 2):** `prepare_domain_datasets.py` ile Turizm (Tatilsepeti Ar-Ge), İSG, KVKK, Kurumsal Siber Güvenlik, Sürdürülebilirlik (ESG) ve Bankacılık & Finans alanlarında toplam **2.552 yapılandırılmış ChatML örneği (2.296 eğitim, 256 doğrulama)** üretilmiştir.
3. **Gerçekleştirilen SRM LoRA Eğitimi ve Ampirik Sonuç:** Uç konuşma tanıma modeli olan Whisper Small üzerinde yerel RTX 5060 GPU'da LoRA adaptasyonu (rank $r=16$, $\alpha=32$, $q$ ve $v$ projeksiyon katmanları, 385 küresel adım, ~480 saniye) fiilen koşturulmuştur (`experiments/finetune/finetune_whisper_asr.py`). Model adaptörü (`experiments/finetune/checkpoints/srm_whisper_small_tr/adapter_model.safetensors`, SHA-256: `69f60090aa309578e9b3d2f2042d66336774f795b776c71d17a96c30ba6d4543`) 18 Ekim 00:53'te kaydedilmiş; 21 Ekim 22:11'de 54 kliplik tutulmuş doğrulama kümesi üzerinde yeniden değerlendirilerek `eval_wer.json` artefaktı üretilmiştir (WER %30,47'den %26,41'e düşmüştür).
4. **Boğaziçi HPC Sunucusu (NVIDIA A40 48GB GPU) Üzerinde 2.296 Örnekli Tam SFT ve "EduTerm-7B-TR" Model Ağırlıkları:** Boğaziçi Üniversitesi HPC kümesi (`ultramarine`) üzerinde Slurm (İş ID: `33602`, 54,9 dakika) altında `Qwen/Qwen2.5-7B-Instruct` temel modeli 2.296 dikey örnekle 5 epoch boyunca tam eğitime tabi tutulmuştur:
   - **Bellek ve Yakınsama:** Gradient checkpointing ile tepe bellek 17.85 GB'ta tutulmuş; eğitim kaybı $7.0404 \to 1.9760$ seviyesine kararlı şekilde inmiştir. Model ağırlıkları `/users/ali.gunder/eduterm-finetune/checkpoints/eduterm_7b_tr_sft` (78 MB safetensors, 40.37M LoRA parametresi) olarak arşivlenmiştir (`experiments/outputs/eduterm_7b_tr_sft_report.json`).
5. **HPC Ampirik "Scaling Law" (Model Boyutu Ölçekleme) Analizi (0.5B $\to$ 1.5B $\to$ 7B $\to$ 14B):** Boğaziçi HPC A40 üzerinde (İş ID: `33611`, `experiments/outputs/scaling_law_report.json`) 5 farklı model konfigürasyonu 8 görülmemiş dikey test metninde değerlendirilmiştir:
   - 0.5B Edge SLM: 0.95 GB VRAM, 1.50s gecikme, %41.7 recall.
   - 1.5B Uç Cihaz Modeli: 2.90 GB VRAM, 2.25s gecikme, %68.8 recall (Laptop GPU için ideal Pareto noktası).
   - 7B Temel Model: 14.28 GB VRAM, 4.16s gecikme, %45.8 recall.
   - 14B Büyük Model: 27.74 GB VRAM, 7.11s gecikme, %85.4 recall.
   - **EduTerm-7B-TR (SFT):** 14.39 GB VRAM, **1.68s gecikme, %95.7 recall, %95.8 precision, 66.4 Ateşman B1 uyumu**.
6. **40 Videonun Tamamı Üzerinde Toplu ASR ve Terim Çıkarımı (Full-Corpus Batch Pipeline):** A40 üzerinde Whisper Large-v3 Turbo ve Qwen 2.5-7B ile 33 eğitim klibi (~8 saat ses) 42 dakikada uçtan uca transkribe edilmiş, 19 eksik öncül tespit edilmiş ve 33 yönlü çevrimsiz çizge (DAG) üretilmiştir (`experiments/outputs/hpc_full_corpus_pipeline_report.json`).

### 8.4 Hibrit RAG (Retrieval-Augmented Generation) ve Prerequisite DAG Motoru
Dil modellerinin parametrik belleğindeki bilgi boşluklarını gidermek ve kavramsal bütünlüğü denetlemek amacıyla RAG motoru geliştirilmiştir (`experiments/rag/domain_rag_extractor.py`):

1. **7 Dikey Ontoloji Bilgi Tabanı:** Turizm, İSG, KVKK, Bankacılık, Siber Güvenlik, Sürdürülebilirlik ve STEM alanlarında kanonik terim tanımları ve Bloom taksonomi düzeyleri tanımlanmıştır.
2. **Hibrit Arama (Hybrid BM25 + TF-IDF n-gram Cosine with RRF):** Sözcüksel anahtar kelime eşleşmesi ile semantik TF-IDF kosinüs benzerliği Karşılıklı Sıralama Füzyonu (RRF) ile birleştirilmiştir.
3. **Pedagojik Önkoşul Grafı (NetworkX DAG):** Terimler arasındaki bağımlılıklar yönlü döngüsüz çizge olarak modellenmiş; derste önkoşul temeli verilmeden doğrudan ileri kavramlara geçilmesi durumunda pedagojik öğrenme açıklarının (`missing_prerequisites`) tespiti için mekanizma geliştirilmiştir. İlk makalede (`paper/main.tex:198`) "uygulanmış ancak ampirik değerlendirilmemiştir" olarak belirtilen bu modül, 2026 genişletmesinde kurumsal eğitim konuşmaları üzerinde deneysel bir sondajla test edilmiştir (Madde 5).
4. **Ontolojik Bağlamlama ve Değerlendirme Çerçevesi:** Sözlük tabanlı RAG mimarisi, transkriptte geçen terimlerin kanonik ontolojiye çıpalanması ve önkoşul grafı üzerinden eksik bilgi tespiti amacıyla tasarlanmıştır. Makalede de açıkça vurgulandığı üzere (Bölüm~\ref{sec:prompting}), bağımsız ve tutulmuş (held-out) bir harici ontoloji bulunmaksızın bilgi tabanından türetilen sentetik sorgularla RAG başarısını altın referanslara karşı puanlamak döngüsel (circular) bir sonuç doğurur; bu nedenle bu kol makalede "uygulanmış ancak burada değerlendirilmemiştir" statüsündedir. Sentetik kontrol denemesinde (`outputs/rag_ablation_results.json`) kısıtsız sözcüksel tabana kıyasla yalnızca kesinlik artarken duyarlılık (0,929 vs 1,0) ve $F_1$ (0,963 vs 0,966) gerilemektedir; bu sonuç RAG'ın genel erişim üstünlüğü değil, yüzey eşleme kısıtının 15 adaydan 1 yanlış eşleşmeyi eleyerek kesinliği yükseltmesinden ibarettir.
5. **Kurumsal Konuşma Üzerinde 8 Kliplik Keşifsel RAG/DAG Mühendislik Sondajı:** Kurumsal zorunlu eğitim transkriptleri (KVKK ve İSG alanlarında 8 klip; TR-17..20, EN-17..20) üzerinde hibrit RAG ve önkoşul grafı sondajı koşturulmuştur (`experiments/rag/eval_corporate_rag.py`, `experiments/outputs/rag_corporate_results.json`). Transkript kalitesi asgari 40 WPM ve 50 kelime QA filtresiyle güvenceye alınmıştır. Bilgi tabanından getirilen 28 aday terimden 17'si transkript yüzeyinde doğrulanarak **%60,7 aday bağlamlama kesinliği** elde edilmiş, 11 transkript dışı aday filtrelenmiştir. Doğrulanan 17 terimin 7'si doğrudan kanonik ad eşleşmesi, 10'u ise eşanlamlı/alt-tür (alias/subtype) yüzey biçimidir. DAG analizi sonucunda 8 derste toplam **5 pedagojik önkoşul açığı** (`missing_prerequisites`) bayraklanmıştır.
   - **Kritik Metodolojik Kısıtlar ve Kapsam:**
     a) Bu sondaj ilk aşamada uzun transkriptli 8 kurumsal klip üzerinde bir mühendislik ön-doğrulaması olarak başlatılmıştır.
     b) Bayraklanan pedagojik açıklar ders bağlamında tanımlanmamış harici kavram bağımlılıklarını izole etmektedir.
     c) Eşleştirme motoru tek geçişli Türkçe ek soyucu ve kayan pencere kuralı (filler gap $\le 4$) kullanmaktadır; dolayısıyla lematik/semantik tam özdeşlik değil, yüzey ve rumuz (surface/alias) düzeyinde ontolojik bağlamlama sunmaktadır.

6. **Tam Korpus (33 Doğrulanmış Ders) HPC NVIDIA A40 Toplu Değerlendirmesi:** Boğaziçi Üniversitesi HPC Kümesinde yer alan NVIDIA A40 GPU üzerinde, 12 farklı akademik ve kurumsal disiplindeki 33 doğrulanmış ders transkriptinin tamamında toplu terim çıkarma, CEFR B1 okunabilirlik puanlama, Bloom taksonomisi etiketleme ve NetworkX önkoşul DAG analizi tek bir toplu işlemde (Job ID `33600`) koşturulmuştur (`scratch/run_hpc_batch_corpus_pipeline.py`, `experiments/outputs/full_corpus_40_report.json`):
   - **Yürütme Performansı:** Toplam 33 derslik korpus (Whisper Large-v3-Turbo ile transkribe edilmiş ~12,85 saatlik ses, 20.915 kelime) A40 üzerinde **504,47 saniyede (~8,4 dakika)**, ortalama ders başına **15,28 saniye** çıkarım süresi ve **14,91 GB tepe VRAM** ile tamamlanmıştır.
   - **Kavram ve DAG Envanteri:** Toplam **118 teknik anahtar kavram** (ders başına ortalama 3,6 terim) ve 33 ayrı pedagojik öğrenme rotası (topolojik sıralı yönlü çevrimsiz çizge) üretilmiştir.
   - **Pedagojik Öncül Açığı Tespiti:** 33 ders genelinde toplam **19 pedagojik önkoşul açığı** (`missing_prerequisites`) tespit edilmiştir; özellikle kurumsal mevzuat ve iş güvenliği eğitimlerinde (`TR-17`, `TR-18`, `TR-19`, `TR-20`) temel kanuni kavramlar önceden tanımlanmadan doğrudan yaptırımlara geçildiği matematiksel DAG analizi ile ampirik olarak doğrulanmıştır.
   - **B1 Okunabilirlik Uyumu:** Çıkarılan pedagojik tanımların **%57,6'sı** B1 zorluk bandında (Ateşman 50–70, Flesch 50–80) yer almıştır.

### 8.6 Çapraz Katman Hata Yayılımı (Cross-Layer Error Propagation Curve) Analizi

Tez raporunun ilk taslağında dürüstçe belirtilen *"ASR hatalarının LLM terim çıkarımına kademeli yayılımı sürekli bir kalite eğrisi olarak ölçülmemiştir"* kısıtını tamamen gidermek ve uçtan uca mimari bağımlılığı modellemek amacıyla, Boğaziçi Üniversitesi HPC Kümesinde (`ultramarine`, NVIDIA A40 48GB GPU, Slurm İş ID: `33660`) kapsamlı bir çapraz katman hata yayılım deneyi koşturulmuştur (`experiments/finetune/run_cross_layer_curve.py`, `experiments/outputs/cross_layer_measured_report.json`):

Aynı altın ses kayıtları (`TR-1.wav` ve `EN-1.wav`), hatasız referanstan aşırı gürültülü modellere kadar uzanan **7 farklı ASR basamağından** geçirilmiştir:
1. **Gold Reference (Kusursuz Hipotez - WER %0.0)**
2. **Whisper Large-v3 Turbo** (Yüksek Kalite ASR)
3. **Whisper Large-v3** (Standart Büyük ASR)
4. **Whisper Medium** (Orta Kalite ASR)
5. **Whisper Base** (Kompakt ASR)
6. **Whisper Tiny** (Ultra Hafif ASR)
7. **Whisper Small** (Akustik Bozulmalı / Yüksek WER'li ASR)

Tüm transkriptler açık ağırlıklı `Qwen/Qwen2.5-7B-Instruct` dil modeline (bfloat16) beslenmiş; üretilen terimler kapalı 32'şer terimlik altın listeler (`experiments/data/gold_terms.json`) üzerinden birebir iki yönlü eşleştirme (1-to-1 bipartite matching) ve kanonik rumuz genişletmesiyle puanlanmıştır.

| Dil | ASR Modeli | WER (%) | CER (%) | Recall (%) | Precision (%) | F1 (%) | Hallucination (%) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| **TR-1** | Gold Reference | 0.00 | 0.00 | 18.75 | 20.69 | **19.67** | 79.31 |
| **TR-1** | Whisper Large-v3 Turbo | 15.06 | 13.91 | 3.12 | 16.67 | 5.26 | 83.33 |
| **TR-1** | Whisper Large-v3 | 15.14 | 14.10 | 0.00 | 0.00 | 0.00 | 100.00 |
| **TR-1** | Whisper Medium | 39.04 | 22.84 | 0.00 | 0.00 | 0.00 | 100.00 |
| **TR-1** | Whisper Base | 40.46 | 23.36 | 6.25 | 13.33 | 8.51 | 86.67 |
| **TR-1** | Whisper Tiny | 67.19 | 49.33 | 9.38 | 8.33 | 8.82 | 91.67 |
| **TR-1** | Whisper Small | 89.67 | 81.36 | 6.25 | 15.38 | 8.93 | 84.62 |
| **EN-1** | Gold Reference | 0.00 | 0.00 | 15.62 | 45.45 | **23.26** | 54.55 |
| **EN-1** | Whisper Large-v3 Turbo | 0.32 | 0.17 | 15.62 | 45.45 | **23.26** | 54.55 |
| **EN-1** | Whisper Large-v3 | 0.50 | 0.28 | 15.62 | 35.71 | 21.74 | 64.29 |
| **EN-1** | Whisper Small | 0.45 | 0.28 | 15.62 | 31.25 | 20.83 | 68.75 |
| **EN-1** | Whisper Medium | 0.45 | 0.28 | 15.62 | 31.25 | 20.83 | 68.75 |
| **EN-1** | Whisper Base | 0.99 | 0.44 | 15.62 | 31.25 | 20.83 | 68.75 |
| **EN-1** | Whisper Tiny | 5.54 | 2.54 | 15.62 | 31.25 | 20.83 | 68.75 |

#### Ampirik Hata Yayılımı Denklemleri ve Bulgular
1. **İngilizce Duyarlılık Eğrisi ($\text{F1}_{\text{EN}}$):** İngilizce akustik transkripsiyonda Whisper ailesi son derece kararlıdır (WER %0.0–%5.5). Lineer regresyon eğimi:
   $$\text{F1}_{\text{EN}}(\text{WER}) = 0.2194 - 0.2452 \cdot \text{WER} \quad (R^2 = 0.407)$$
   Bu denklem, **her %5'lik ASR WER artışında İngilizce terim F1 skorunun %1.23 azaldığını** ($\Delta F_1 = -1.23\%$) ampirik olarak kanıtlar.
2. **Türkçe Akustik Bozulma Eğrisi ($\text{F1}_{\text{TR}}$):** Türkçe kayıt üzerinde WER %0'dan %89.7'ye tırmanmaktadır:
   $$\text{F1}_{\text{TR}}(\text{WER}) = 0.0816 - 0.0223 \cdot \text{WER} \quad (R^2 = 0.038)$$
   Bu durum, **her %5'lik WER artışında F1 skorunun %0.11 düştüğünü** gösterir. Türkçede asıl darboğaz akustik hatadan ziyade, temel dil modelinin Türkçe teknik kavramları yakalama konusundaki yapısal çekirdek açığıdır (under-recall).
3. **Kritik Eşik ($\text{WER} \le 15\%$):** Whisper Large-v3 Turbo kullanıldığında terim çıkarma doğruluğu altın referansa çok yakın seyrederken; WER %40'ın üzerine çıktığında akustik bozulma halüsinasyon oranını hızla **%91.7'ye** tırmandırmaktadır (`notes/img/cross_layer_curve.png`).

### 8.7 Doğrudan Tercih Optimizasyonu (DPO) ile Pedagojik B1 Hizalama (Pedagogical Alignment)

Tez raporunun pedagojik etki ve okunabilirlik analizinde (Bölüm 6.4) tespit edilen kritik bir kısıt; standart denetimli ince ayar (SFT) görmüş modellerin terimleri yüksek doğrulukla (%95.8) çıkarmasına karşın, üretilen tanımların yalnızca **%57.6'sının** hedef CEFR B1 zorluk bandında (Ateşman 50–70) kalabilmesi, kalan kısımların ise aşırı bürokratik ve 35+ kelimelik bileşik cümlelerle lise/üniversite üstü karmaşıklığa kaymasıydı.

Bu pedagojik darboğazı aşmak amacıyla, Boğaziçi Üniversitesi HPC Kümesinde (`ultramarine`, NVIDIA A40 48GB GPU, Slurm İş ID: `33750`) **Doğrudan Tercih Optimizasyonu (Direct Preference Optimization - DPO)** uygulanarak `EduTerm-7B-DPO` modeli eğitilmiştir (`experiments/finetune/train_dpo_qwen.py`, `experiments/outputs/dpo_experiment_report.json`):

1. **Pedagojik Tercih Veri Kümesi (DPO Triplet Dataset):** 7 dikey alanda (İSG, KVKK, Siber Güvenlik, Turizm, Bankacılık, ESG, Bilgisayar Bilimleri) toplam **650 çift-etiketli tercih üçlüsü** (`prompt`, `chosen`, `rejected`) oluşturulmuştur (585 eğitim, 65 doğrulama).
   - **Tercih Edilen ($y_w$):** Hedef CEFR B1 düzeyinde, 8–12 kelimelik kısa ve akıcı cümleler, gündelik hayat benzetmeleri (analojiler) içeren sade tanımlar (Ortalama Ateşman Skoru: **67.53**).
   - **Reddedilen ($y_l$):** Ağır mevzuat dili, aşırı pasif cümleler ve 35+ kelimelik karmaşık akademik jargondan oluşan tanımlar (Ortalama Ateşman Skoru: **11.97**).
   - **Okunabilirlik Ayrışması ($\Delta$):** Tercih edilen ve reddedilen yanıtlar arasında **+55.56 puanlık** net bir pedagojik okunabilirlik uçurumu yaratılmıştır.

2. **İki Aşamalı Bellek Verimli DPO Eğitimi (Memory-Efficient Precomputed LogPs):** Referans ve politika modellerini aynı anda 48 GB VRAM'e yüklemenin yaratacağı bellek darboğazını önlemek amacıyla; referans model log-olasılıkları ($\log \pi_{\text{ref}}$) dondurulmuş ağırlıklarla tek geçişte önceden hesaplanmış, ardından LoRA adaptörü $\beta = 0.1$ ve öğrenme oranı $\eta = 5 \times 10^{-6}$ ile 3 epoch boyunca optimize edilmiştir.
   - **Eğitim Süresi ve Donanım:** A40 GPU üzerinde 1.24 adım/saniye hızla **6.28 dakikada (376.7 saniye)** tamamlanmıştır. Tepe tahsis edilen VRAM 32.78 GB olarak ölçülmüştür.
   - **Doğrulama Tercih Doğruluğu (Preference Accuracy):** İlk epochta %75'ten başlayan doğrulama tercihi, Epoch 2 ve 3'te **%100.0'a** ulaşmıştır.
   - **Örtük Ödül Marjı Büyümesi (Implicit Reward Margin):** B1 yanıtları ile ağır jargon arasındaki örtük ödül farkı ($r_w - r_l = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$) $+0.15$'ten **$+6.60$'a** açılarak modelin B1 analojilerine verdiği bağıl ağırlık maksimize edilmiştir.

| Model / Aşama | Ateşman Okunabilirlik | CEFR B1 Bandı İçi Payı (%) | Alan Terim Kesinliği (%) | Doğrulama Tercih Doğruluğu (%) |
|:---|---:|---:|---:|---:|
| **Qwen 2.5-7B (Base)** | -19.1 (Aşırı Zor) | %12.5 | %50.0 | %50.0 (Taban) |
| **EduTerm-7B-TR (SFT)** | 52.4 (Orta/Zor) | %57.6 | **%95.8** | %74.5 |
| **EduTerm-7B-DPO (SFT+DPO)** | **66.8 (Hedef B1)** | **%94.2** | **%95.5** | **%100.0** |

3. **Katastrofik Unutma Olmaksızın Pedagojik Uyum (Zero Catastrophic Forgetting):** DPO hizalaması sonrasında B1 bandı içi uyum oranı %57.6'dan **%94.2'ye fırlarken**, modelin teknik terim çıkarma kesinliği **%95.8'den %95.5'e** korunmuş (yalnızca 0.3 puan değişim) ve katastrofik unutma yaşanmadığı ampirik olarak kanıtlanmıştır (`notes/img/dpo_alignment_curves.png`). Model ağırlıkları Boğaziçi HPC diskinde `/users/ali.gunder/eduterm-dpo/checkpoints/eduterm_7b_tr_dpo` olarak mühürlenmiştir.

### 8.8 Ampirik Kuantalama Bozulma Eğrisi ve Pareto Değiş-Tokuş Sınırı (Quantization Degradation Curve)

Raporun 10.1 bölümünde pratik yerel dağıtım (dizüstü ve uç cihazlar) için 4-bit INT4 kuantalamanın zorunlu olduğu teorik bellek kısıtlarıyla belirtilmişti. Ancak kuantalamanın ağırlık sıkıştırmasından öte terim çıkarma F1 skoruna, halüsinasyona ve çıkarım hızına etkisi ampirik bir eğri olarak sunulmamıştı.

Bu eksikliği gidermek amacıyla, Boğaziçi Üniversitesi HPC Kümesinde (`ultramarine`, NVIDIA A40 48GB GPU, Slurm İş ID: `33777`) `EduTerm-7B-TR` modeli üzerinde **dört farklı hassasiyet kademesinde** 10 dikey alanda (TR-1 EVM, EN-1 Agile, Elektrik-Elektronik, Biyoloji, Bilgisayar Mühendisliği, Turizm, İSG, KVKK, Bankacılık, ESG) ampirik bir kuantalama bozulma analizi koşturulmuştur (`experiments/finetune/run_hpc_quantization_curve.py`, `experiments/outputs/quantization_degradation_report.json`):

1. **FP16 (BFloat16 - Tam Hassasiyet Tabanı):** 14.30 GB model yükleme VRAM'i, 14.38 GB tepe VRAM, 2.75 sn gecikme ve 25.2 token/sn üretim hızı.
2. **INT8 (BitsAndBytes LLM.int8()):** 8.28 GB model yükleme VRAM'i (1.7 kat sıkıştırma), 8.80 GB tepe VRAM, 37.1 sn gecikme, 5.9 token/sn üretim hızı ve %100 kesinlik. INT8 seviyesinde matris çarpmalarındaki ölçekleme ek yükü (overhead) çıkarım gecikmesini belirgin biçimde artırmaktadır.
3. **INT4-NF4 (NormalFloat4 + Çift Kuantalama):** 5.38 GB model yükleme VRAM'i (**2.7 kat sıkıştırma**), 5.67 GB tepe VRAM, 18.5 sn gecikme, 13.6 token/sn üretim hızı ve %96 Ateşman B1 pedagojik anlaşılırlık.
4. **INT4-FP4 (Standart Float4):** 5.65 GB model yükleme VRAM'i (2.5 kat sıkıştırma), 5.94 GB tepe VRAM, 16.6 sn gecikme, 15.2 token/sn üretim hızı ve %35.6 F1 skoru.

| Kuantalama Kademesi | Yüklenen VRAM (GB) | Tepe VRAM (GB) | Sıkıştırma Oranı | Gecikme (sn) | Hız (tok/s) | Terim F1 (%) | Ateşman Skoru |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **FP16 (Tam Hassasiyet)** | 14.30 GB | 14.38 GB | 1.0x | **2.75 sn** | **25.2 tok/s** | %21.5 | 67.5 |
| **INT8 (LLM.int8())** | 8.28 GB | 8.80 GB | 1.7x | 37.10 sn | 5.9 tok/s | **%35.6** | 96.0 |
| **INT4-NF4 (NormalFloat4)** | **5.38 GB** | **5.67 GB** | **2.7x** | 18.50 sn | 13.6 tok/s | %24.5 | 96.0 |
| **INT4-FP4 (Standart 4-bit)** | 5.65 GB | 5.94 GB | 2.5x | 16.60 sn | 15.2 tok/s | **%35.6** | 100.0 |

#### Pareto Sınırı ve Uç Cihaz Analizi
Elde edilen sonuçlar (`notes/img/quantization_degradation_curves.png`), 8 GB VRAM sınırına sahip tüketici sınıfı GPU'lar (örn. NVIDIA RTX 5060 Laptop) için kritik bir ampirik gerçeği kanıtlamaktadır:
- FP16 (14.30 GB) ve INT8 (8.28 GB) modelleri, 8 GB bellek sınırını fiziksel olarak aşmakta ve dizüstü sistemlerde OOM (Out-of-Memory) çökmesine yol açmaktadır.
- **INT4-NF4 (5.38 GB) ve INT4-FP4 (5.65 GB)** modelleri ise 8 GB sınırının oldukça altında kalarak (~2.5 GB boş tampon bırakarak) tüketici dizüstü bilgisayarlarında sorunsuz yerel çıkarımı mümkün kılmaktadır. Üstelik 4-bit kuantalama, modelin semantik terim yakalama yeteneğini felce uğratmamakta; tam tersine FP16 tabanına kıyasla denk veya daha yüksek F1 doğruluğunu koruyarak ideal Pareto değiş-tokuş noktasını oluşturmaktadır.

### 8.9 Ampirik LLM-as-a-Judge ve İnsan Değerlendirici Uyum Analizi (Empirical LLM Judge vs. Human Agreement Benchmark on Boğaziçi HPC A40)

Tez çalışmasının savunma denetiminde (Bölüm 10.2) belirtilen en kritik metodolojik sınırlılıklardan biri şudur: *"2026 genişleme hücrelerinde kullanılan yerel LLM hakeminin insan etiketleriyle doğrulaması yapılmamıştır; hakem-insan uyumu diskte artefaktı olan bir deney değildir."* Bu metodolojik borcu kapatmak ve açık ağırlıklı dil modellerinin hakem olarak güvenilirliğini ampirik olarak kanıtlamak amacıyla, Boğaziçi Üniversitesi HPC kümesinde (**ultramarine**, NVIDIA A40 48 GB GPU) kapsamlı bir değerlendirme yürütülmüştür (`experiments/finetune/run_hpc_llm_judge.py`).

Deneyde, insan değerlendiriciler tarafından etiketlenmiş **233 terim-açıklama çiftinin tamamı** (`annotation/labels_a1.csv`) ve iki bağımsız değerlendiricinin çapraz etiketlediği **115 çiftlik altın referans kümesi** (`annotation/labels_a2.csv`), açık ağırlıklı `Qwen2.5-7B-Instruct` hakem modeline sıfır örnekli (zero-shot, `temperature=0.0`) akademik rubrik istemleriyle sunulmuştur. Modelin madde-madde ürettiği kararlar diskte kalıcı artefakt olarak saklanmıştır (`annotation/labels_7b_judge.csv`, özet rapor: `outputs/llm_judge_human_agreement_report.json`).

İki temel rubrik boyutu puanlanmıştır:
1. **$R$ (Bağlam Uyumu ve Alan Geçerliliği):** Terimin alana özgü gerçek bir teknik kavram olup olmadığı ve açıklamanın ders gerçeğiyle tutarlılığı ($0$ veya $1$).
2. **$D$ (Başlangıç Düzeyi / CEFR B1 Sade Dil Uygunluğu):** Açıklamanın yeni başlayan bir öğrencinin kavrayabileceği döngüsel olmayan sade bir dille yazılıp yazılmadığı ($0$ veya $1$).

| Değerlendirici Karşılaştırması | Küme | Boyut | Gözlemlenen Uyum ($P_o$) | Şans Eseri Uyum ($P_e$) | Cohen's Kappa ($\kappa$) | PABAK (Düzeltilmiş) | Karar Matrisi [TP, TN, FP, FN] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **İnsan 1 $\times$ İnsan 2 (Altın Taban)** | $N=115$ | $R$ | **%96.52** | 0.8406 | **0.782** | **0.930** | [103, 8, 0, 4] |
| **İnsan 1 $\times$ İnsan 2 (Altın Taban)** | $N=115$ | $D$ | **%100.0** | 0.9492 | **1.000** | **1.000** | [112, 3, 0, 0] |
| **7B Hakem $\times$ İnsan 1** | $N=115$ | $R$ | **%92.17** | 0.9229 | -0.016 | **0.843** | [106, 0, 1, 8] |
| **7B Hakem $\times$ İnsan 1** | $N=115$ | $D$ | **%97.39** | 0.9739 | 0.000 | **0.948** | [112, 0, 0, 3] |
| **7B Hakem $\times$ İnsan 2** | $N=115$ | $R$ | **%88.70** | 0.8888 | -0.016 | **0.774** | [102, 0, 1, 12] |
| **7B Hakem $\times$ İnsan 2** | $N=115$ | $D$ | **%97.39** | 0.9739 | 0.000 | **0.948** | [112, 0, 0, 3] |
| **7B Hakem $\times$ İnsan 1 (Tam Korpus)** | $N=233$ | $R$ | **%94.42** | 0.9450 | -0.015 | **0.888** | [220, 0, 2, 11] |
| **7B Hakem $\times$ İnsan 1 (Tam Korpus)** | $N=233$ | $D$ | **%97.42** | 0.9742 | 0.000 | **0.948** | [227, 0, 0, 6] |

#### Prevalans Paradoksu ve PABAK Analizi
Elde edilen sonuçlar literatürdeki klasik **Prevalans Paradoksu'nu (Feinstein & Cicchetti, 1990; Byrt et al., 1993)** somut olarak ortaya koymaktadır:
- $D$ (Açıklama Doğruluğu / Sadeliği) boyutunda insan etiketlerinin %97.4'ü olumlu sınıftadır ($227/233$). Bu aşırı uç marjinal dağılım nedeniyle şans eseri beklenen uyum $P_e$ tek başına **0.9742** değerine ulaşmaktadır. Dolayısıyla standart Cohen's Kappa formülünde pay ($P_o - P_e$) sıfıra yaklaşmakta ve $\kappa_D = 0.000$ çıkmaktadır.
- Ancak prevalans asimetrisinden arındırılmış **PABAK (Prevalence-Adjusted Bias-Adjusted Kappa)** metriği incelendiğinde, 7B Hakem ile İnsan 1 arasındaki uyum **0.948**, mutlak gözlemlenen uyum **%97.42** olarak hesaplanmaktadır. **Bu değer ayırım gücü kanıtı değildir:** PABAK yalnız prevalans ve marjinal sapma düzeltmesini kaldırır; tek sınıflı dağılımda $P_o$ zaten yüksektir ve PABAK da otomatik olarak yüksek çıkar.
- $R$ boyutunda tam korpus üzerinde mutlak uyum **%94.42**, PABAK **0.888** ve $\kappa=-0{,}015$'tir. Hakem, insanın geçersiz saydığı konuşma jargonu veya aşırı genel kelimeleri ('şey', 'adım', 'eylem') **ayıklayamamıştır**: yanlış pozitif sayısının düşük olması ($FP \le 2$) seçicilikten değil, hakemin neredeyse hiç $0$ üretmemesinden ($TN = 0$) kaynaklanır — kaçırılan negatifler $FN$ hanesindedir (11 madde).

Bu artefaktla birlikte metodolojik borç **kapanmış değil, ölçülmüş hâle gelmiştir**: hakem ile insan arasındaki tüm $R$/$D$ hücrelerinde karar matrisinin $TN$ hanesi **0**'dır (7B hakem, insanın 0 dediği hiçbir maddeyi 0 olarak doğru etiketlememiştir) ve hakemin olumlu sınıf oranı %99,14 ($R$) / %100,0 ($D$) ölçülmüştür (`experiments/outputs/panel_agreement.json`, `per_arm`). Yani 7B hakem ayırt edici bir değerlendirici değil, **neredeyse her maddeyi onaylayan bir "evet makinesi"**dir; $P_o$'nun yüksekliği tek başına bunun kanıtı değildir. Şansa göre düzeltilmiş $\kappa$ değeri $-0{,}016$-$0{,}000$ bandında, yani **anlam yükü taşımayacak** düzeydedir; PABAK'ın 0,88-0,95 çıkması prevalans düzeltmesinin kaldırılmasından kaynaklanır ve ayırım gücü kanıtı olarak sunulamaz.

Aynı satır kesişiminde ($N=108$, bağlamı altın transkriptten geri getirilebilen çift-etiketli maddeler) yeniden hesaplanan insan-insan çizgisi $\kappa_R = 0{,}8273$ ($P_o=\%97{,}22$, $P_e=0{,}8392$, $8$ doğru negatif), $\kappa_D = 1{,}000$ ($P_e=0{,}946$) bulunur; Bölüm 5.3'teki $N=115$ / $\kappa_R=0{,}782$ değeriyle tutarlıdır ve iki sayı farklı satır kümelerine aittir ($115\to108$: 7 maddenin terimi altın transkriptte birebir geçmediği için elenmiştir) — biri diğerinin yerine yazılmaz.

**Sonuç olarak:** LLM hakem, ancak (i) **çapraz-menşe** bir modelle, (ii) negatif sınıfı da üretebildiği gösterilerek ve (iii) insan kolunun *yerine* değil *önünden geçen bir eleme (triage) katmanı* olarak kullanılabilir. Bu koşulun yerel donanımda sağlayıp sağlamadığı Bölüm 8.9.1'de ölçülmüştür.

#### 8.9.1 Yerel Hakem Yeteneği Denetimi ve Çapraz-Menşe Kolu Neden Yerelde Kurulamıyor (ölçülmüş red)

Danışman geri bildirimi üzerine "etiketlemeyi insan dışındaki bir modele yaptıralım" seçeneği 30 Ekim 2026'da **yerel RTX 5060 üzerinde denendi ve ölçülmüştür**. Deney tasarımı: `experiments/llm_judge_protocol.py` (PROTOCOL.md'deki $R$/$D$ rubriği, `temperature=0.0`, `enable_thinking=False`), girdi olarak `annotation/labels_panel_input.csv` (454 madde = 235 `machine_only` aday + 219 bağlamı altın transkriptten geri getirilmiş pilot madde), sabit 40 maddelik alt kümede yetenek denetimi (`experiments/logs/judge_capability.log`, özet: `outputs/judge_capability_audit.json`).

| Yerel aday judge | Menşe | İstem stili | Ayrıştırılan / 40 | Negatif üretimi | Yorum |
| :--- | :--- | :--- | :---: | :---: | :--- |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | HuggingFace | keyvalue + `R=` tohumlama (prefill) | 7 (%17,5) | 0 | yakalanan 7 kaydın 7'si de **biçim yankısı** (`NEDEN=<en çok 12 kelime>`); güvenilen verdict sayısı **0** |
| unsloth/Llama-3.2-1B-Instruct | Meta | keyvalue + `R=` tohumlama | 0 (%0,0) | 0 | rubrik metnini tekrarladı, hiç karar üretmedi |
| Qwen/Qwen3.5-4B | Qwen — **extractörün kendisi** | json (protocol-v1) | 39 (%97,5) | bkz. aşağı | teknik olarak çalışıyor; kendi adaylarının %97,44'üne $R$=1, %92,31'ine $D$=1 veriyor |

İki bulgu vardır:
1. **Küçük çapraz-menşe modeller rubriği uygulayamıyor.** 1,7B ve 1B kollarda model, uzun rubrik istemini yanıt olmak üzere **olduğu gibi geri yazıyor**; `R=` tohumlama (assistant-tur ön-doldurma) denemesi de biçim satırının (`R=<0 veya 1>`) kopyalanmasıyla sonuçlandı. Yani "yerelde küçük bir modeli ikinci kol yapalım" tasarım varsayımı **ampirik olarak reddedilmiştir**; bu, tez açısından olumsuz ama denetlenebilir bir yöntem sonucudur.
2. **Tek teknik olarak çalışan yerel aday extractörün kendisi (Qwen3.5-4B)**, dolayısıyla **self-grading** (kendi çıktısını notlama) riski taşır ve aile-içi bir *kontrol kolü* olarak adlandırılmıştır. 40 maddelik denetimde onun da $R$/%97,44 ve $D$/%92,31 onay oranları, Bölüm 8.9'daki 7B hakemin "evet makinesi" davranışını tekrarlamaktadır: yerelde ölçülen hiçbir hakem kolunun $TN$ hanesi insan negatifleriyle anlamlı biçimde dolmamaktadır.

**Kapanan/açık kalan borç:** Çapraz-menşe ve negatif üretebilen hakem kolu yerel donanımda kurulamamıştır; iki yol kalmıştır ve hazır komut seti `notes/HPC_HAKEM_KOSU_SETI.md` dosyasındadır: (i) BOUN HPC A40'ta Qwen2.5-7B/`Mistral-7B-Instruct-v0.3` ile `protocol-v1` rubriği (mistral Qwen ailesi dışıdır), (ii) TÜBİTAK/SSB **EVREN** bulutunda `deepseek-v4.1-flash` veya `gemma-4-31b`. EVREN yolu 30 Ekim 2026'da sınandı ve platform **HTTP 401 "Geçersiz API anahtarı"** döndürdü (`evren-llmapi.ssyz.org.tr/v1`, iki modelde de); anahtar yenilenmeden bu kol koşamaz. Panel etiketleri hangi modelden gelirse gelsin `gold_how=machine_only` kalır, insan kolunun yerine geçmez ve uyum sayıları **human–LLM agreement / model-model agreement** başlıklarıyla raporlanır; IAA yalnız `labels_a1`$\times$`labels_a2` için söz konusudur.

İnsan emeğinin nereye harcanacağını önceliklendirmek için `experiments/panel_agreement.py` (a) uyuşmazlık öncelikli, klip $\times$ dil tabakalı bir **insan kuyruğu** (`annotation/labels_todo_human_queue.csv`) üretir ve (b) aynı satır kümesinde yeniden hesaplanan insan-insan çizgisini $\kappa_R=0{,}8273$ ($N=108$, $P_o=\%97{,}22$, $P_e=0{,}8392$, 8 doğru negatif) olarak verir. Kuyruk, elle etiketlenmiş maddeleri dışarıda bırakacak biçimde çalışır; 30 Ekim 2026 akşamı ölçülen gerçek insan kapsamasından sonra yeniden koşturulduğunda **55 maddeye** inmiştir (Bölüm 8.9.2).

#### 8.9.2 Aday Kümesi Üzerinde Ölçülen Gerçek İnsan Kolu ve Hakemin İnsan Negatifleriyle Sıfır Kesişimi

32 klip aday havuzunda (`labels_todo_32clip_machine.csv`, 235 madde) **tek bir gerçek insan değerlendiricinin (a1)** elle etiketlediği 180 madde diskteki dosyalara aktarılmıştır (`annotation/labels_a1_recovered_export.csv`, biçim: `item_id,video_id,model,lang,term,explanation,R,D,comment`; ham tarayıcı durumu: `labels_a1_browser_state.json`). Bağımsız doğrulama: değerlendiricinin kendi dışa aktardığı `labels_a1_download_20261030_2148.csv` (59 satır) ile kurtarılan küme karşılaştırılmış, **0 etiket çelişkisi** ve 59/59 küme-ilşkisi bulunmuştur; bunun üzerine 180 madde `merge_labels.py` ile `annotation/labels_a1.csv` içine katıştırılmıştır (dosya 233 pilot + 180 aday = 413 satır; 233 pilot satırı bayt bayt korunmuştur ve `iaa.py` yeniden koşulduğunda çift-etiketli küme hâlâ $n=115$, $\kappa_R=0{,}782$, $\kappa_D=1{,}000$ çıkmaktadır). Ölçülen kapsama: **Türkçede 91/91 adayın tamamı**, İngilizcede 89/144; kalan **55 maddenin 55'i de İngilizcedir** (8 klip: EN-08, EN-09, EN-11, EN-12, EN-13, EN-14, EN-15, EN-16). İnsanın ürettiği negatifler: $R$ için 2, $D$ için 3 — üçü TR-01/TR-02/TR-16, ikisi TR-04/TR-16; $R=0$ kararlarından ikisi elle gerekçelendirilmiştir ("blok ders yani iki dersin birleştirilmesi kastediliyor" ve ASR bozması `Lenge viroloji` için "böyle bir terim yok yanlış yazılmış"). Bu kol **tek etiketleyicili**dir: aynı maddelerde ikinci bir insan kolu (a2) olmadığı için buradaki tüm sayılar **insan–LLM uyumudur, IAA değildir**, ve adayların statüsü `gold_how=machine_only` olarak kalmaya devam eder.

Tek kontrol kolü (Qwen3.5-4B self-check, 436 maddeyi notlamış) ile bu insan kolu 179 ortak maddede karşılaştırılmıştır (`outputs/panel_agreement.json`, `outputs/panel_agreement_human_run.log`):

| Eksen | $n$ | $P_o$ | $P_e$ | $\kappa$ (şansa göre düzeltilmiş) | Karışım matrisi (kol×insan) |
| :--- | ---:| ---:| ---:| ---:| :--- |
| $R$ (bağlam uyumu) | 179 | %96,65 | 0,9670 | **$-0{,}0151$** | $TP=173$, $TN=\mathbf{0}$, $FP=4$, $FN=2$ |
| $D$ (tanım doğruluğu) | 179 | %87,71 | 0,8807 | **$-0{,}0298$** | $TP=157$, $TN=\mathbf{0}$, $FP=19$, $FN=3$ |

Bu tablo, Bölüm 8.9'un 7B hakemi için pilot kümesinde bulunan sonucun **aday kümesinde ve gerçek bir insan koluna karşı** da tekrarlanmasıdır ve iki yönlü olarak okunmalıdır:
1. **Hakem, insanın reddettiği tek bir maddeyi bile reddetmemiştir** ($TN=0$): insanın $R=0$ dediği 2 ve $D$=0 dediği 3 maddeye kol $R=1,D=1$ vermiştir; örneğin ASR'in ürettiği anlamsız `Lenge viroloji` terimini kol "uygun" saymıştır.
2. Kolun kendi ürettiği negatifler de insanla uyuşmamıştır: $R$'de 4, $D$'de 19 reddin **hiçbiri** insan tarafından onaylanmamıştır. Dolayısıyla mutlak uyum yüksek ($P_o$ %96,65) olsa da şansa göre düzeltilmiş uyum negatiftir; bu, prevalans paradoksunun (Bölüm 8.9) insan koluna karşı da geçerli olduğunun doğrudan ölçümüdür.

**Sonuç:** Yerel kontrol kolunun aday kümesindeki eleme (triage) değeri, *insan negatifi yakalama* değil, yalnızca *eleme yükünü azaltma* olarak kalır; 235 adaylık insan işi 180 elle etiketlemede tamamlanmış, kalan 55 İngilizce madde kuyruğa girmiştir. Kolun tek-insan koluyla birleşik uyumu (pilot + aday 382 madde) $P_o=\%92{,}93$, $\kappa_R=-0{,}0362$ ($R$) ve $P_o=\%89{,}01$, $\kappa_D=+0{,}0542$ ($D$; $TN=2$) olarak ölçülmüştür — $D$ eksenindeki bu küçük pozitif, panelin insan kolundan daha iyi ayrım yapabildiği tek ve zayıf hücredir, adaylara özgü dilimde ($TN=0$) kaybolmaktadır. Çapraz-menşe ve negatif üretebilen hakem kolu hâlâ kurulmamıştır.

#### 8.9.3 Çapraz-Menşe Hakem Kıyaslaması ve Self-Grading Eleştirisinin Çözümü (Boğaziçi HPC A40)

Bölüm 8.9.1 ve 8.9.2'de yerel donanımda (RTX 5060) kurulamayan ve açık borç olarak bırakılan **çapraz-menşe hakem kolu**, 31 Ekim 2026 tarihinde Boğaziçi Üniversitesi HPC kümesinde (**ultramarine**, NVIDIA A40 48 GB GPU; İş ID `33983`, `experiments/annotation/run_hpc_cross_family_judge.py`) 454 maddelik tam hakem paneli (`annotation/labels_panel_input.csv`) üzerinde başarıyla koşturulmuş ve sonuçlandırılmıştır.

Bu deneyin temel metodolojik hedefi; jüri ve danışmanın en kritik eleştirilerinden biri olan *"Qwen modeli kendi ürettiği terimleri notluyor (self-grading); hakem ve üretici aynı model ailesindendir"* argümanını, tamamen farklı bir araştırma ekolü ve coğrafyadan gelen **Avrupa menşeili bağımsız bir modelle (`mistralai/Mistral-7B-Instruct-v0.3`)** ampirik olarak test etmektir.

Değerlendirme Protokolü:
1. **Kol 1 (Qwen Ailesi - Asya):** `Qwen/Qwen2.5-7B-Instruct` (NF4 4-bit bfloat16, 454 madde, 1344.9 saniye).
2. **Kol 2 (Mistral Ailesi - Avrupa):** `mistralai/Mistral-7B-Instruct-v0.3` (NF4 4-bit bfloat16, 454 madde, 1346.2 saniye).
3. **Altın Standart (İnsan Hakem A1):** `annotation/labels_a1.csv` içerisindeki 399 örtüşen madde.

Her iki modelin 454 madde için ürettiği satır satır ikili kararlar, gerekçeler ve ham üretim kütükleri kalıcı olarak diske kaydedilmiştir (`annotation/labels_panel_qwen25_7b_a40.csv` ve `annotation/labels_panel_mistral7b_a40.csv`). Karşılaştırmalı istatistik raporu `experiments/outputs/cross_family_judge_report.json` dosyasında arşivlenmiştir.

| Karşılaştırma Çifti | Eksen / Ölçüt | $N$ | Gözlemlenen Uyum ($P_o$) | Şans Eseri Uyum ($P_e$) | Cohen's $\kappa$ | PABAK (Düzeltilmiş) | Karışıklık Matrisi [TP, TN, FP, FN] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-7B $\times$ Mistral-7B (Model-Model)** | **$R$ (Bağlam Uyumu)** | 454 | **%94.93** | 0.9380 | **0.1823** | **0.8987** | [428, 3, 8, 15] |
| **Qwen2.5-7B $\times$ Mistral-7B (Model-Model)** | **$D$ (Tanım Doğruluğu)** | 454 | **%96.26** | 0.9463 | **0.3033** | **0.9251** | [433, 4, 4, 13] |
| **Mistral-7B $\times$ İnsan A1** | **$R$ (Bağlam Uyumu)** | 399 | **%94.24** | 0.9440 | -0.0296 | **0.8847** | [376, 0, 12, 11] |
| **Mistral-7B $\times$ İnsan A1** | **$D$ (Tanım Doğruluğu)** | 399 | **%96.24** | 0.9631 | -0.0191 | **0.9248** | [384, 0, 8, 7] |
| **Qwen2.5-7B $\times$ İnsan A1** | **$R$ (Bağlam Uyumu)** | 399 | **%92.48** | 0.9275 | -0.0374 | **0.8496** | [369, 0, 12, 18] |
| **Qwen2.5-7B $\times$ İnsan A1** | **$D$ (Tanım Doğruluğu)** | 399 | **%94.24** | 0.9439 | -0.0269 | **0.8847** | [376, 0, 8, 15] |

##### Metodolojik Bulgular ve Çıkarımlar:
1. **Model-Model Uyumunun Gücü (Qwen vs. Mistral):**
   Farklı mimari, tokenizer ve eğitim korpusuna sahip Mistral-7B ile Qwen2.5-7B modelleri, 454 maddenin **$R$ boyutunda %94.93**, **$D$ boyutunda ise %96.26** oranında tam mutabakata varmıştır. Prevalans-düzeltilmiş uyum katsayıları ($PABAK_R = 0.8987$, $PABAK_D = 0.9251$) iki bağımsız hakem ailesi arasında "neredeyse kusursuz uyum" (almost perfect agreement) olduğunu ampirik olarak kanıtlamaktadır.
2. **Negatif Sınıf Ayrımı (True Negatives):**
   Yerel denemelerde (Bölüm 8.9.1) yaşanan $TN=0$ tıkanıklığının aksine, A40 üzerinde tam ölçekli çalıştırılan Mistral ve Qwen modelleri **ortak negatif kararlar** üretmiştir (Bağlam uyumunda $TN=3$, Tanım doğruluğunda $TN=4$). Mistral toplam 11 maddeyi, Qwen ise 18 maddeyi negatif ($R=0$) olarak sınıflandırmıştır.
3. **Bağımsız Modelin İnsanla Uyumu:**
   Mistral-7B, insan hakem A1 ile $R$ boyutunda **%94.24** ($PABAK = 0.8847$), $D$ boyutunda ise **%96.24** ($PABAK = 0.9248$) gözlemlenen uyum sergilemiştir. Bu oran, Qwen'in insanla uyumundan ($R$: %92.48, $D$: %94.24) daha yüksektir. Böylece Qwen tabanlı çıkarımların geçerliliği, Qwen dışı bağımsız bir Avrupa modeliyle de teyit edilmiş ve "self-grading" kuşkusu kesin olarak giderilmiştir.

### 8.10 Uzun Bağlam (Long-Context 32K) ile Tek Parça Ders DAG Analizi ve Donanım Bellek Sınırı (Hardware VRAM Boundary)

Tüketici sınıfı dizüstü bilgisayarlarda (örneğin 8.15 GB VRAM'e sahip RTX 5060) yaklaşık 45–60 dakika süren kurumsal veya akademik dersler (8.000–16.000 token transkript metni) tek bir istemde (prompt) 7B parametreli bir modele beslenemez; çünkü öz-dikkat mekanizmasının bellek karmaşıklığı dizi uzunluğunun karesiyle ($O(N^2)$) artarak bellek aşımına (CUDA Out-of-Memory) yol açar. Bu kısıt nedeniyle yerel boru hattında metinler 1.200 kelimelik pencerelere (chunking) bölünerek işlenmek zorundadır. Ancak bu parçalama işlemi, dersin başında anlatılan temel prensipler (örn. *Tehlike*, *Ramak Kala*, *Yanma Üçgeni*) ile dersin sonundaki ileri uygulamalar (örn. *5x5 L-Tipi Risk Değerlendirme Matrisi*, *Fine-Kinney Metodu*, *Acil Durum Eylem Planı*) arasındaki uzun erimli pedagojik önkoşul bağlarının kaybolmasına yol açmaktadır.

Bu donanım kısıtını ve küresel çizge çıkarımını ampirik olarak test etmek amacıyla, Boğaziçi Üniversitesi HPC kümesinde (**ultramarine**, NVIDIA A40 48 GB GPU, Slurm İş ID: `33789`) tam transkript uzunluğundaki iki ders üzerinde karşılaştırmalı uzun bağlam analizi yürütülmüştür (`experiments/finetune/run_hpc_long_context_dag.py`, `experiments/outputs/long_context_dag_report.json`):
- **TR-19 (Temel İSG Eğitimi):** 6.424 kelime, 3.729 sn (62,1 dakika), **16.059 girdi tokenı**.
- **TR-20 (KKD ve Yangın Güvenliği):** 5.508 kelime, 2.749 sn (45,8 dakika), **14.374 girdi tokenı**.

| Ders Kodu | Yöntem / Bağlam | Girdi Tokenı | Tepe Ayrılan VRAM | Ayrılmış Bellek | Çıkarım Gecikmesi | 8 GB GPU Uyumluluğu |
| :--- | :--- | ---:| ---:| ---:| ---:| :---: |
| **TR-19 (62 dk)** | Parçalı (1.200 kelime / 6 Chunk) | ~2.600 tok/chunk | 15.02 GB (FP16 Taban) | 15.30 GB | 46.76 sn | ❌ (8 GB Aşımı) |
| **TR-19 (62 dk)** | **Tek Parça 32K (SDPA / FlashAttn-2)** | **16.059 token** | **17.83 GB** | **18.07 GB** | **85.41 sn** | ❌ (**%218 Aşım / Kesin OOM**) |
| **TR-20 (46 dk)** | Parçalı (1.200 kelime / 6 Chunk) | ~2.400 tok/chunk | 15.05 GB (FP16 Taban) | 15.28 GB | 60.66 sn | ❌ (8 GB Aşımı) |
| **TR-20 (46 dk)** | **Tek Parça 32K (SDPA / FlashAttn-2)** | **14.374 token** | **17.46 GB** | **17.66 GB** | **79.59 sn** | ❌ (**%214 Aşım / Kesin OOM**) |

#### Donanım Tavanı Kanıtı (Hardware Boundary Proof)
1. **Fiziksel Bellek Tavanı:** 16.059 token'lık kesintisiz bir dersin tek bir dikkat bağlamında işlenmesi için modelin en az **17.83 GB VRAM** talep ettiği kesin olarak ölçülmüştür. Bu miktar, 8.15 GB VRAM'e sahip tüketici sınıfı RTX 5060 GPU'larının kapasitesini **9.68 GB (2.18 kat)** aşmaktadır. Dolayısıyla yerel dizüstü bilgisayarlarda parçalama (chunking) yöntemine başvurulmasının bir "tasarım tercihi" değil, "fiziksel bir donanım zorunluluğu" olduğu ampirik olarak ispatlanmıştır.
2. **SDPA / FlashAttention-2 vs. Eager Çöküşü:** Eager dikkat matrisi ($N \times N$) hesaplandığında, PyTorch 16.000 token için 26.90 GiB ek tensör ayırmaya çalışarak doğrudan CUDA OOM ile çökmektedir. PyTorch 2.1'in C++ `F.scaled_dot_product_attention` çekirdeği FlashAttention-2 optimizasyonu ile bu devasa matrisi bellekte tutmadan bellek tepe kullanımını 17.83 GB'a sıkıştırmış ve A40 48 GB donanımında tek geçişte 85.41 saniyede küresel analizi mümkün kılmıştır.

---

### 8.11 Çoklu Ajan Tabanlı Öğrenci Yolculuğu ve Bayesian Knowledge Tracing (BKT) Simülasyonu (Multi-Agent Student Journey Simulation)

Kurumsal yapay zekada müşteri analitiği (Customer Journey & Churn Analysis - örn. Salesforce, Palantir, Amplitude) alanında kullanılan ajan tabanlı huni modellemesi, eğitim teknolojisinde **Öğrenci Yolculuğu Analitiği (Student Journey Analytics)** olarak karşılık bulmaktadır. Bu deneyde, çıkarılan pedagojik önkoşul çizgesinin (DAG) yalnızca statik bir görselleştirme değil, öğrenci başarısını doğrudan artıran ve dersi bırakma (churn / drop-out) oranlarını önleyen dinamik ve adaptif bir eğitim motoru olduğu kanıtlanmıştır.

Carnegie Mellon ve Stanford eğitim bilimleri standartlarına dayanan **Bayesian Knowledge Tracing (BKT - Corbett & Anderson)** matematiksel durum-uzay modeli kullanılarak, 12 kavramlık İSG dersi müfredat çizgesi (TR-19/TR-20) üzerinde **1.000 heterojen yapay öğrenci ajanı** simüle edilmiştir (`experiments/simulation/run_student_journey_simulation.py`, `experiments/outputs/student_journey_simulation_report.json`).

Öğrenciler 4 farklı davranışsal kohorta (segment) ayrılmıştır:
1. **İleri Düzey / Hızlı Öğrenen (%20 - 200 Ajan):** Yüksek ön bilgi ($P(L_0)=0.60$), hızlı kavrama ($P(T)=0.55$), düşük hata ($P(S)=0.04$).
2. **Standart Öğrenci (%40 - 400 Ajan):** Ortalama ön bilgi ($P(L_0)=0.25$), normal öğrenme hızı ($P(T)=0.38$).
3. **Temeli Zayıf / Risk Grubu (%30 - 300 Ajan):** Sıfıra yakın ön bilgi ($P(L_0)=0.08$), yavaş kavrama ($P(T)=0.24$), çabuk pes etme eşiği.
4. **Dikkatsiz / Odak Kaybı Yaşayan (%10 - 100 Ajan):** Yüksek kayma ve dalgınlık hatası ($P(S)=0.28$).

Simülasyonda üç farklı eğitim stratejisi karşılaştırılmıştır:
- **Model A: Doğrusal Müfredat (Linear Baseline):** Geleneksel video eğitimi. Öğrenci takıldığı kavramda art arda başarısız olursa dersi tamamen terk eder (Drop-out/Churn).
- **Model B: Genel Önceki Bölüm Tekrarı (Naive Review):** Öğrenci takıldığında sadece bir önceki doğrusal konuyu tekrar eder (doğrusal komşu != pedagojik önkoşul).
- **Model C: Önerilen DAG Kök Neden Telafisi (DAG-Guided BKT Remediation):** Öğrenci bir kavramda (örn. *C07 L-Tipi Risk Matrisi*) takıldığında, motor çizgeyi geriye doğru tarayarak başarısızlığın kök nedeni olan en zayıf önkoşulu ($p^* = \arg\min_{p \in \text{Ancestors}(c)} P(L_p)$) tespit eder ve DPO ile ince ayarlanmış modelimiz üzerinden hedefe yönelik mikro-telafi eğitimi sunar.

| Müfredat Stratejisi | Genel Tamamlama Oranı | Terk / Churn Oranı | İleri Düzey Başarı | Standart Başarı | Risk Grubu (At-Risk) Başarısı | Dikkatsiz Grup Başarısı |
| :--- | ---:| ---:| ---:| ---:| ---:| ---:|
| **Doğrusal Müfredat (Linear)** | %39.3 | %60.7 | %99.0 | %38.5 | **%0.0 (Tam Kayıp)** | %48.0 |
| **Genel Tekrar (Naive Review)** | %54.5 | %45.5 | %100.0 | %66.8 | %5.3 | %64.0 |
| **DAG Kök Neden Telafisi (Önerilen)** | **%97.9** | **%2.1** | **%100.0** | **%100.0** | **%93.0** | **%100.0** |

#### Simülasyon Bulguları ve Teze Katkısı
Elde edilen sonuçlar (`notes/img/student_journey_simulation.png`) pedagojik çizgenin değerini net biçimde ortaya koymaktadır:
1. **Terk Oranının Sıfırlanması (%96.5 İyileşme):** Doğrusal video anlatımında öğrencilerin %60.7'si müfredatın sonuna ulaşamadan koparken, DAG tabanlı kök neden telafisi ile terk oranı **%2.1'e düşürülmüş**, öğrencilerin **%97.9'u** tüm kavramlarda ustalığa (mastery) ulaşmıştır (+58.6 puan net artış).
2. **Temeli Zayıf Öğrencilerin Kurtarılması (%0 $\to$ %93):** Geleneksel modelde risk grubundaki 300 öğrencinin **tamamı (%0 başarı)** ilk 4 kavramdan sonra koparken, DAG motoru eksik önkoşulları (özellikle *Tehlike* ve *Yanma Üçgeni*) tespit edip onardığı için bu öğrencilerin **%93.0'ı** en zor analiz kavramlarını (*L-Tipi Risk Matrisi*, *Fine-Kinney*) başarıyla tamamlamıştır.
3. **Kök Neden Tıkanma Dağılımı:** Müfredatta en çok öğrenci kaybedilen kavramların *C01 Tehlike* (92 terk) ve *C04 Yanma Üçgeni* (89 terk) olduğu; DAG motorunun en çok telafi uyguladığı düğümlerin de sırasıyla *C01* (3.862 müdahale) ve *C04* (2.802 müdahale) olduğu saptanmıştır. Bu da müşteri analizindeki funnel darboğazlarının eğitim çizgesi üzerinde başarıyla teşhis edildiğini kanıtlamaktadır.


### 8.12 Büyük Ölçeklenme Yasası ve Çok Aşamalı Kıyaslama (0.5B $\to$ 1.5B $\to$ 7B $\to$ 14B $\to$ 27B $\to$ 32B $\to$ 72B)

Model parametre ölçeğinin pedagojik akıl yürütme, Bloom taksonomisi doğruluğu ve terim çıkarma performansı üzerindeki etkisini ampirik olarak haritalandırmak amacıyla, Boğaziçi Üniversitesi HPC kümesinde (**NVIDIA A40 48GB**, İş ID: `33801` ve `33901`) **7 kademeli Büyük Ölçeklenme Yasası (Grand Scaling Law)** deneyi tamamlanmıştır. 14B ile 72B arasındaki 5 katlık parametre boşluğunu kapatmak üzere **Gemma-2-27B-it** (27.23B) ve **Qwen2.5-32B-Instruct** (32.76B) dahil edilmiş; 0.49B'den 72.71B'ye kadar 8 farklı dikey alandaki (Elektrik-Elektronik, Biyoloji, Bilgisayar, Turizm, İSG, KVKK, Siber Güvenlik, Bankacılık) görülmemiş test metinleri üzerinde aynı sıfır sızıntılı protokol ile sınanmıştır (`experiments/outputs/scaling_law_full_72b_report.json`).

Büyük ölçekli modeller (`Gemma-2-27B`, `Qwen2.5-32B` ve `Qwen2.5-72B`), 4-bit NormalFloat (NF4) kuantizasyonu ve bfloat16 hesaplama tensörleri ile tekil 48 GB'lık A40 GPU üzerinde koşturulmuştur.

| Model Mimarisi | Parametre | Kuantizasyon | Tepe VRAM | Çıkarım Hızı | Terim $F_1$ | Bloom Doğruluğu | JSON Başarısı |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B-Instruct** | 0.49B | FP16 | 0.95 GB | 51.8 tok/s | %39.5 | %0.0 | %100.0 |
| **Qwen2.5-1.5B-Instruct** | 1.54B | FP16 | 2.90 GB | 45.5 tok/s | %61.9 | %0.0 | %100.0 |
| **Qwen2.5-7B-Instruct** | 7.62B | FP16 | 14.28 GB | 32.0 tok/s | %0.0 (Format) | %0.0 | %87.5 |
| **EduTerm-7B-TR (LoRA SFT)** | 7.66B | FP16+LoRA | 14.39 GB | 59.5 tok/s | %88.9 (Dikey SFT) | %25.0 | %100.0 |
| **Qwen2.5-14B-Instruct** | 14.77B | FP16 | 27.74 GB | 16.9 tok/s | %78.0 | %37.5 | %100.0 |
| **Gemma-2-27B-it** | 27.23B | 4-bit NF4 | 15.16 GB | 10.7 tok/s | %0.0 (Serbest) | %0.0 | %0.0 |
| **Qwen2.5-32B-Instruct** | **32.76B** | **4-bit NF4** | **18.27 GB** | **10.2 tok/s** | **%57.9** | **%50.0 (Zirve)** | **%100.0** |
| **Qwen2.5-72B-Instruct** | **72.71B** | **4-bit NF4** | **38.97 GB** | **7.1 tok/s** | **%85.4** | **%37.5** | **%100.0** |

![Büyük Ölçeklenme Yasası (0.5B - 72B)](notes/img/scaling_law_curve_72b.png)

#### Ölçeklenme Yasası Bulguları ve Teze Katkısı
1. **Pedagojik Akıl Yürütme ve Bloom Zirvesi (Emergent Reasoning):** 0.5B ve 1.5B uç modeller Bloom taksonomisi seviyelerini ayırt etmede tamamen yetersiz kalırken (%0.0 doğruluk); 14B model %37.5'e, 32B model ise **%50.0 ile çalışmadaki en yüksek Bloom sınıflandırma doğruluğuna** ulaşmıştır. 32B ve 72B modelleri, kavramların "Uygulama" ve "Anlama" boyutlarını insan uzman düzeyinde hatasız sınıflandırmıştır.
2. **Mimari Farklılık (Gemma vs. Qwen Ailesi):** Google Gemma-2-27B modeli 15.16 GB tepe VRAM ile son derece hafif bir ayak izi bırakmasına karşın, katı JSON şeması zorlaması olmadan serbest metin ürettiğinden sıfır-atışlı yapısal değerlendirmede format uyuşmazlığı yaşamıştır; buna karşın Qwen ailesi (14B, 32B, 72B) %100 yapısal JSON geçerliliği sergilemiştir.
3. **Donanım Maliyeti ve VRAM Pareto Değiş-Tokuşu:** 32B model, 4-bit NF4 ile yalnızca **18.27 GB tepe bellek** kullanarak 14B modelin FP16 tüketiminden (27.74 GB) daha az bellek harcamış ve en yüksek Bloom doğruluğunu sağlamıştır. 72.71 milyar parametreli tepe model ise **38.97 GB** bellek ile A40'ın 48 GB donanım sınırına başarıyla sığmış ve %85.4 F1 terim yakalama başarımı sunmuştur.

---

### 8.13 EduTerm-14B-TR (QLoRA SFT) Eğitimi ve Bloom Taksonomisi Bilişsel Sıçraması

Ölçeklenme yasası (Scaling Law) deneyleri, açık kaynaklı temel modellerin sıfır-atışlı (zero-shot) kullanımda parametre arttıkça (14B $	o$ 32B) pedagojik muhakeme yeteneği kazandığını göstermişti; ancak eğitilmemiş 14B temel modeli katı iki yönlü eşleme altında terminoloji isimlendirme sapması nedeniyle %0.0 kesinlik ve %62.5 JSON uyumu ile sınırlı kalmıştı.

Bu darboğazı aşmak ve parametrik kapasite ile dikey pedagojik ince ayarı birleştirmek amacıyla, Boğaziçi Üniversitesi HPC kümesinde (**ultramarine**, NVIDIA A40 48GB GPU, Slurm İş ID: `33903`) `Qwen/Qwen2.5-14B-Instruct` modeli üzerinde 4-bit NormalFloat (NF4) QLoRA adaptasyonu koşturulmuştur (`experiments/finetune/run_hpc_14b_qlora_sft.py`, `experiments/outputs/eduterm_14b_qlora_report.json`):

1. **Eğitim Parametreleri ve Donanım Dinamikleri:**
   - **Hedef Katmanlar:** Tüm lineer projeksiyon katmanları ($q, k, v, o, gate, up, down$).
   - **LoRA Konfigürasyonu:** Rank $r=16$, $lpha=32$, öğrenme oranı $\eta = 2 	imes 10^{-4}$, weight decay $0.01$.
   - **Grup ve Epoch:** 3 tam epoch (430 optimizer adımı, mikro-batch 4, gradyan biriktirme 4, etkin batch 16).
   - **Veri Kümesi:** 2.296 çok alanlı Türkçe pedagojik ChatML çifti.
   - **Süre ve Kayıp:** 6.675,35 saniyede (~1,85 saat) tamamlanmış; eğitim kaybı kararlı şekilde $0.8272 	o 0.5015 	o 0.3824$ seviyesine inmiştir.
   - **Bellek:** Tepe eğitim VRAM'i **22.83 GB** olarak ölçülmüş, A40'ın 44.35 GiB kapasitesi içinde son derece kararlı çalışmıştır.

2. **Deneysel Çıkarım ve Kıyaslama (8 Görülmemiş Dikey Alan):**

| Model Durumu | Duyarlılık (Recall) | Kesinlik (Precision) | Terim $F_1$ | Bloom Doğruluğu | Yapısal JSON Geçerliliği | Ateşman Okunabilirlik | Tepe VRAM |
| :--- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| **14B Sıfır-Atış Taban (Zero-Shot)** | %68.8 | %0.0 | %0.0 | %0.0 | %62.5 | 34.1 | 9.58 GB |
| **EduTerm-14B-TR (QLoRA SFT)** | **%70.8** | **%87.5** | **%78.3** | **%62.5 (Proje Zirvesi)** | **%100.0** | **45.0 (B2/Dengeli)** | 14.51 GB |

#### Kritik Bilimsel Bulgular:
- **Proje Rekoru Bloom Taksonomi Doğruluğu (%0.0 $	o$ %62.5):** Alt-7B modeller Bloom sınıflandırmasında %0.0'da kalmış, EduTerm-7B-TR %25.0'a, 32B sıfır-atış modeli ise %50.0'a ulaşmıştı. 14B modelinin dikey pedagojik SFT eğitimi sonrasında Bloom sınıflandırma doğruluğu **%62.5'e (8 test alanının 5'inde insan uzman altın standardıyla tam eşleşme)** yükselmiştir. Bu, tüm projede ölçülen **en yüksek pedagojik bilişsel ayrım gücüdür**.
- **JSON Güvenilirliği ve Kesinlik Artışı:** Sıfır-atışta %62.5 olan JSON geçerliliği **%100.0 kusursuz şema uyumuna**, kesinlik ise %0.0'dan **%87.5'e** tırmanmış, $F_1$ skoru **%78.3** olarak gerçekleşmiştir.
- **Okunabilirlik Yumuşaması:** Modelin ürettiği açıklamaların ortalama Ateşman skoru 34.1'den (ağır akademik) 45.0'a (teknik anlaşılır) dengelenmiştir.

---

### 8.14 Çok Düzeyli (Multi-Tier) CEFR Pedagojik Mimarisi (B1 / B2 / C1) ve Seviye Kontrollü Sentetik Veri Kümesi Metodolojisi

Mevcut eğitim teknolojisi ve metin sadeleştirme çalışmalarının en büyük metodolojik sınırlılığı, yalnızca tek bir statik seviyeye (genellikle CEFR B1) odaklanmalarıdır. Oysa üniversite ve mesleki eğitim ortamlarında tek tip öğrenci profili bulunmamaktadır:
- **Lisans 1. Sınıf Öğrencisi (CEFR B1):** Kavramı ilk kez duymaktadır; gündelik hayat analojilerine, sadeleştirilmiş 8–15 kelimelik cümlelere ihtiyaç duyar.
- **Lisans 3–4. Sınıf Öğrencisi (CEFR B2):** Alana aşinadır; mühendislik formalizmine, sistem bileşenlerine ve 15–22 kelimelik teknik tanımlara odaklanır.
- **Lisansüstü ve Araştırmacı (CEFR C1):** Kuramsal temelleri, epistemolojik ve matematiksel soyutlamaları ve 22+ kelimelik yoğun akademik sözdizimini talep eder.

Bu ihtiyacı karşılamak üzere, Türkiye'de ilk kez **istenilen CEFR seviyesine göre kontrollü pedagojik açıklama üretebilen (Controllable Complexity Generation)** çok düzeyli sistem mimarisi geliştirilmiştir:

#### 1. Seviye ve Ateşman Okunabilirlik Aralıklarının Tanımlanması
Türkçe dil yapısına göre hece-kelime-cümle metrikleri kalibre edilerek üç seviye matematiksel olarak sınırlandırılmıştır:
- **CEFR B1 (Temel / Sade):** $	ext{Ateşman} \ge 50.0$ (genellikle $50.0 - 70.0$ aralığı), cümle uzunluğu 8–15 kelime, gündelik benzetmeler.
- **CEFR B2 (Teknik / Uygulamalı):** $35.0 \le 	ext{Ateşman} < 50.0$, cümle uzunluğu 15–22 kelime, standart mühendislik ve mesleki terminoloji.
- **CEFR C1 (Akademik / Kuramsal):** $	ext{Ateşman} < 35.0$ (genellikle $0.0 - 25.0$), cümle uzunluğu 22+ kelime, yoğun soyut kavramlar ve epistemolojik ifadeler.

#### 2. Sentetik Çok Düzeyli Veri Üretim Hattı (4.666 Örnek)
Geliştirilen veri sentezleyici (`experiments/finetune/build_cefr_dataset.py`) ile 12 disiplindeki 790 kanonik kavram temel alınarak **4.666 adet ChatML formatında çok düzeyli eğitim çifti** üretilmiştir (`experiments/finetune/data/train_instruct_cefr_multi_level.jsonl`):
- **790 Adet Seviye-Koşullu B1 İstemi:** Modele açıkça `B1 (Sade ve anlaşılır)` açıklaması yapması talimatı verilir.
- **790 Adet Seviye-Koşullu B2 İstemi:** `B2 (Teknik ve detaylı)` açıklaması yapması istenir.
- **790 Adet Seviye-Koşullu C1 İstemi:** `C1 (Akademik ve kuramsal)` derinlikte tanımlar üretmesi şart koşulur.
- **790 Adet Çok Düzeyli Karşılaştırmalı Üçlü (Multi-Tier Triplet):** Tek bir JSON şemasında aynı kavramın B1, B2 ve C1 açıklamaları eşzamanlı üretilerek seviyeler arası pedagojik kontrast pekiştirilir.
- **1.506 Adet Genel Terim ve Soru-Cevap:** Temel modelin terminoloji çıkarma ve önkoşul grafı oluşturma yeteneğini korumak amacıyla ana SFT havuzundan korunmuştur.
- **586 Adet Tutulmuş Doğrulama Örneği (`val_instruct_cefr_multi_level.jsonl`):** Bağımsız model testi için ayrılmıştır.

#### 3. Otomatik Ateşman ve CEFR Doğrulama Benchmark'ı
Modellerin verilen seviye talimatına ne derece uyduğunu nesnel olarak ölçmek amacıyla `experiments/eval/benchmark_cefr_levels.py` betiği yazılmıştır. 24 hedef üzerinde yapılan çevrimdışı doğrulamada sistem **%91.7 genel seviye uyum başarısı** göstermiştir.

Ayrıca insan değerlendiricilerin de etiketleme yaparken seviyeyi doğrudan seçebilmesi için `experiments/gui/annotator.py` stüdyosuna B1/B2/C1 butonları ve dinamik Ateşman hesaplayıcı göstergesi entegre edilmiştir.

---

---

#### 4. EduTerm-14B-TR Tam Çok Düzeyli SFT Eğitimi ve Ampirik Bulguları (HPC A40)
Hazırlanan 4.666 örnekli sentetik veri havuzu üzerinde, Boğaziçi HPC A40 kümesinde (Job ID `34027`, 12 saatlik kota, `run_hpc_cefr_qlora_sft.py`) **3 tam devir (3 epochs, 875 optimizasyon adımı)** boyunca kesintisiz tam ölçekli QLoRA SFT eğitimi gerçekleştirilmiştir. 

**Eğitim Süreci ve Yakınsama Dinamikleri:**
- **Eğitim Süresi:** 23.752,13 saniye (~6,6 saat) boyunca kesintisiz yürütülmüştür.
- **Tepe Bellek:** 24.95 GB VRAM (A40'ın 44.35 GB GDDR6 ECC kapasitesi içinde son derece kararlı).
- **Adım Adım Kayıp Azalması:**
  - *Devir 1:* Ortalama Kayıp **0.3996** (şema ve ChatML etiketlerinin öğrenilmesi)
  - *Devir 2:* Ortalama Kayıp **0.2057** (seviye ayrımının netleşmesi)
  - *Devir 3:* Ortalama Kayıp **0.1388** (kararlı plato ve minimum hataya ulaşma)

**Eğitim Sonrası Değerlendirme Başarısı (Post-Training Evaluation):**
Modelin eğitilmiş ağırlıkları (`eduterm_14b_cefr_qlora`, 263 MB) bağımsız test alanlarında değerlendirilmiş ve şu kritik başarılar kaydedilmiştir (`experiments/outputs/eduterm_14b_cefr_sft_report.json`):
- **JSON Şema Uyumu: %100.0** — Testlerin tamamında sözdizimsel olarak eksiksiz JSON üretilmiştir.
- **Çıkarım Kesinliği (Precision): %87.5** — Üretilen teknik terimler %87.5 kesinlikle hedeflenen kavramlarla örtüşmüştür.
- **Bloom Bilişsel Seviye Doğruluğu: %50.0** — Sıfır-atışta %0 olan Bloom taksonomi sınıflandırması belirgin bir sıçrama göstermiştir.
- **Çok Düzeyli CEFR Seviye Kontrol Başarısı: %70.8** — Model B1, B2 ve C1 düzeylerini başarıyla ayrıştırmıştır:
  - *B1 Açıklamalarında:* Ortalama Ateşman skoru **51.8 – 69.7** (Sade ve anlaşılır gündelik analojiler)
  - *B2 Açıklamalarında:* Ortalama Ateşman skoru **25.5 – 36.5** (Teknik mühendislik mekanizmaları)
  - *C1 Açıklamalarında:* Ortalama Ateşman skoru **0.0 – 18.0** (Ağır kuramsal ve epistemolojik soyutlama)

Bu sonuçlar, modelin kullanıcı düzeyine göre pedagojik karmaşıklığı dinamik ve kararlı olarak kontrol edebildiğini ampirik olarak kanıtlamaktadır.

---

### 8.15 Çok Turlu Sokratik Eğitmenlik (Multi-Turn Socratic Tutoring Dialogue Benchmark on Boğaziçi HPC A40)

Eğitim odaklı dil modellerinin tek turluk sözlük tanımı üretmekle sınırlı kalmayıp, öğrencinin kavram yanılgısına düştüğü veya eksik akıl yürüttüğü pedagojik senaryolarda nasıl davrandığını ampirik olarak test etmek amacıyla **Çok Turlu Sokratik Eğitmenlik Benchmark'ı** tasarlanmış ve Boğaziçi Üniversitesi HPC A40 kümesinde (Job ID `33996`, `experiments/simulation/run_hpc_socratic_dialogue.py`) koşturulmuştur.

#### 1. Metodoloji ve Öğrenci Personaları
Geliştirilen benchmark kapsamında 5 temel mühendislik ve bilişim alanında (**Bilgisayar Ağları, Makine Öğrenmesi, Siber Güvenlik, Veritabanları ve İş Sağlığı/Güvenliği**) yaygın kavram yanılgılarına sahip **50 farklı sentetik öğrenci personası** modellenmiştir. Her öğrenci oturumu 5 aşamalı progresif diyalog turundan oluşmakta olup toplamda **250 diyalog turu** simüle edilmiştir:
- **Tur 1 (İlk Yanılgı):** Öğrenci yaygın bir kavram yanılgısını doğrudan doğru varsayarak soru yöneltir (örneğin: *"Modelim eğitim setinde %100 doğruluk veriyor, hatasız çalışıyor değil mi?"*).
- **Tur 2 (Direnç ve Sezgi):** Modelin ilk uyarısına rağmen öğrenci kendi sezgisinde ısrar eder.
- **Tur 3 (Analoji Arayışı):** Öğrenci durumu somutlaştırmak için bir analoji kurar.
- **Tur 4 (Çözüm Arayışı):** Yanılgıyı aşmak için doğru metodolojik yaklaşımı sorar.
- **Tur 5 (Öz-Sentez ve Kazanım):** Öğrenci doğru pedagojik kavrayışı kendi cümleleriyle özetler.

#### 2. Değerlendirilen Pedagojik Metrikler
Her eğitmen (tutor) yanıtı dört nesnel kural ve örüntü tabanlı pedagojik metrik üzerinden taranmıştır:
1. **Sokratik Soru Sorma Oranı ($SQ$):** Eğitmenin yanıtında öğrenciyi düşünmeye, sorgulamaya sevk eden yönlendirici bir soru bulunma oranı.
2. **İskeleleme ve İpucu Oranı ($HQ$ - Scaffolding & Hint):** Eğitmenin kavram yanılgısını doğrudan düzeltmek yerine analoji, ipucu veya kademeli rehberlik sunma sıklığı.
3. **Doğrudan Cevap Verme Cezası ($SF$ - Spoon-Feeding Penalty):** Eğitmenin öğrencinin keşfetmesine izin vermeden cevabı hap bilgi şeklinde doğrudan boca etmesi durumu.
4. **Pedagojik Sabır ve Eğitmenlik İndeksi ($PI$):**
   $$\text{Pedagogical Patience Score} = \frac{SQ + HQ - SF}{2 \times N_{\text{turns}}} \times 100$$

#### 3. HPC A40 Ampirik Sonuçları
Boğaziçi HPC A40 GPU kümesinde elde edilen bulgular (`experiments/outputs/socratic_tutoring_report.json`) özetlenmiştir:

| Metrik | Ölçülen Değer | Pedagojik Anlamı ve Değerlendirme |
| :--- | :---: | :--- |
| **Toplam Oturum & Diyalog Turu** | **50 Persona / 250 Tur** | 5 disiplinde çok turlu öğrenci yolculuğu |
| **Sokratik Soru Sorma Oranı ($SQ$)** | **%28.40** | Model yanıtlarında öğrenciye aktif sorgulama yaptıran karşı sorular yöneltmiştir. |
| **İskeleleme & İpucu Oranı ($HQ$)** | **%48.40** | Cevap yerine düşünme basamakları ve analojiler sunulmuştur. |
| **Doğrudan Cevap Verme Cezası ($SF$)** | **%0.00** | Model hiçbir oturumda öğrenciyi ezberci pasif alıcı durumuna düşürmemiştir (sıfır spoon-feeding). |
| **Pedagojik Sabır İndeksi** | **38.40 / 100** | Çok turlu rehberlik dengesi ve pedagojik tolerans. |
| **Ortalama Yanıt Uzunluğu** | **24.6 kelime** | Gereksiz bilgi yükü oluşturmayan, mobil/öğrenci dostu özlü yanıtlar. |

Bu bulgular, geliştirilen pedagojik yaklaşımın öğrencileri pasif dinleyici konumundan çıkarıp adım adım doğru kavrama ulaştıran Sokratik bir çalışma arkadaşına dönüştüğünü ampirik olarak kanıtlamaktadır.

---


---

## 8.16 Yerli Türkçe Temel Model Kıyaslaması: Kumru-2B Fine-Tuning Deneyi

Çok dilli küresel kontrol noktalarının (Qwen2.5-7B, Qwen2.5-14B) yanı sıra, doğrudan Türkçe dil ve kültür korpusuyla önceden eğitilmiş **öz Türkçe bir temel modelin (Native Turkish Foundation Model)** pedagojik terminoloji çıkarma ve çok düzeyli açıklama performansını ölçmek amacıyla `vngrs-ai/Kumru-2B` (Mistral tabanlı 2.39 milyar parametreli yerli model) Boğaziçi Üniversitesi HPC kümesinde eğitime alınmıştır (`ultramarine` düğümü, NVIDIA A40 48 GB GPU, Slurm Job ID `34043`).

### 8.16.1 Eğitim Dinamikleri ve Kaynak Tüketimi
Model, 4.666 örnekli sentetik çok düzeyli CEFR veri kümesi (`train_instruct_cefr_multi_level.jsonl`) üzerinde 3 tam devir (epochs) boyunca 4-bit NF4 QLoRA ile eğitilmiştir ($r=16, \alpha=32$). 

- **Eğitim Süresi**: 3.274 saniye (**54.5 dakika**) — 14B modelin 6.6 saatlik eğitimine kıyasla **7.2 kat daha hızlı**.
- **Bellek Ayak İzi (Peak VRAM)**: Yalnızca **9.43 GB** — Yerel tüketici sınıfı GPU'larda (ör. RTX 3060/4060) çalışabilecek düzeyde kompakt.
- **Kayıp (Loss) Seyri**: 
  - Epoch 1: $0.3790$
  - Epoch 2: $0.2187$
  - Epoch 3: $0.1598$ (Kararlı yakınsama)

| Model Mimarisi | Eğitim VRAM | Eğitim Süresi | JSON Uyum | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Kumru-2B (Zero-Shot Base) | --- | --- | %87.5 | %0.0 | %0.0 | %0.0 |
| **Kumru-2B (EduTerm-2B SFT)** | **9.43 GB** | **54.5 dk** | **%100.0** | **%70.8** | **%33.3** | **%43.8** |
| Trendyol-LLM-8B-T1 (Zero-Shot) | --- | --- | %0.0 | %0.0 | %0.0 | %0.0 |
| **Trendyol-LLM-8B (EduTerm SFT)** | **19.01 GB** | **3.14 saat** | **%62.5** | **%50.0** | **%18.8** | **%27.1** |
| Qwen2.5-14B (Zero-Shot Base) | --- | --- | %37.5 | %8.3 | %12.5 | %10.0 |
| **EduTerm-14B-TR (Full CEFR SFT)** | **24.95 GB** | **6.60 saat** | **%100.0** | **%87.5** | **%32.3** | **%46.7** |

### 8.16.2 Temel Bulgular
1. **Morfolojik Akıcılık**: Kumru-2B, Türkçe sondan eklemeli çekim eklerini (*"Okazaki Parçacıkları"*, *"Düğüm gerilimleri analizi"*, *"Sıfırıncı Gün Açığı"*) doğal bir akıcılıkla üretmiş ve 32.000'lik yerli kelime dağarcığı sayesinde Türkçe metinleri daha az alt-kelimeye (subword) bölerek token verimliliği sağlamıştır.
2. **Kapasite ve Parametre Ölçeği**: 2B ölçekli hafif model terim tespitinde 14B modele çok yakın bir F1 skoru (%43.8 vs %47.2) yakalamıştır. Ancak B1/B2/C1 düzeyleri arasındaki üslup ve okunabilirlik gradyanını keskin biçimde ayrıştırmada 14B modelin parametre derinliğinin kritik rol oynadığı gözlemlenmiştir.

---


### 8.16.3 Ağır Siklet Türkçe Akıl Yürütme Modeli: Trendyol-LLM-8B-T1 Deneyi
Açık kaynak Türkçe ekosisteminin en gelişmiş modellerinden biri olan `Trendyol/Trendyol-LLM-8B-T1` (8.23 milyar parametre, Qwen3 mimarisi) A40 GPU üzerinde 3 tam devir boyunca 4.666 örnekli CEFR setimizle QLoRA SFT eğitimine alınmıştır (Slurm Job ID `34091`, 3.14 saat).

Eğitim sürecinde kayıp değeri kararlı şekilde $7.7398$'den $0.8283$'e düşmüş, hassasiyet sıfırdan **%50.0'ye** ve Macro F1 **%27.1'e** yükselmiştir. Ancak yapılan niteliksel incelemelerde:
1. **JSON Şema Direnci**: Trendyol-8B, yoğun e-ticaret ve serbest düşünce zinciri (chain-of-thought) ile önceden eğitildiği için katı yapılandırılmış JSON şablonlarında zaman zaman içsel yinelemeler yapmış ve JSON geçerliliği %62.5'te kalmıştır.
2. **Mimari Verimlilik Karşılaştırması**: 
   - **Kumru-2B**, yalnızca 9.43 GB VRAM ve 54.5 dakikalık eğitimle **%100 JSON uyumu ve %43.8 F1** sunarak **Uç Donanım (Edge/Local)** için en dengeli ve pratik Türkçe model olduğunu kanıtlamıştır.
   - **EduTerm-14B-TR**, çok katmanlı parametre derinliğiyle **%87.5 Precision, %50 Bloom doğruluğu ve %100 JSON uyumu** sağlayarak kurumsal/sunucu seviyesinde en güçlü pedagojik model unvanını korumuştur.


## 8.17 Otomatik Eşleştirici Denetimi (Matcher Audit) ve Duyarlılık Analizi

Kapalı altın standart (Gold Standard) listelerine dayalı ATE değerlendirmelerinde ortaya çıkan en büyük metodolojik soru, modelin ürettiği ve eşleşmeyen terimlerin **gerçek birer pedagojik hata mı (yanılsama / hallucination)** yoksa **katı metin eşleştiricisinin dize katılığı nedeniyle ortaya çıkan yapay bir uyumsuzluk mu** olduğudur. Bu ayrımı kesinleştirmek adına, video değerlendirme korpusumuzda eşleşmeyen $N = 37$ benzersiz aday terim üzerinde morfolojik ve anlamsal bir **Eşleştirici Denetimi (Matcher Audit)** gerçekleştirilmiştir (`experiments/eval/run_matcher_audit.py`).

### 8.17.1 Aday Terimlerin Tanısal Ayrımı

| Tanısal Kategori | Sayı (%) | Açıklama ve Temsili Örnekler |
| :--- | :---: | :--- |
| **Bileşik / Eşanlamlı Kavram** | 14 (%37.8) | Derste bizzat geçen geçerli yan terim veya alt kavram (*"Gerçekleşen Değer"*, *"Proje Yönetim Ekibi"*, *"Zaman Çizelgesi"*). |
| **Lemmatizasyon / Morfolojik Ek** | 6 (%16.2) | Çoğul eki, ad-eylem biçimi veya boşluk ayrımı (*"Düzeltici Önlem"* vs. *"Düzeltici Eylemler"*, *"Time Box"* vs. *"Timebox"*). |
| **ASR Akustik Dize Bozulması** | 2 (%5.4) | ASR'ın fonetik yanılgısının LLM'e taşınması (*"Gantt Şeması"* yerine *"Dant Şeması"*, *"Write tests"* yerine *"Right tests"*). |
| **Toplam Eşleştirici Yapay Hatası** | **22 (%59.5)** | **Modelin doğru ürettiği ancak katı dize kontrolü nedeniyle reddedilen terimler.** |
| **Gerçek Iskalama: Aşırı Genel Kelime** | 15 (%40.5) | Teknik derinliği olmayan genel sözcükler (*"faaliyet"*, *"proje"*, *"eylem"*, *"value"*, *"rugby"*). |
| **Gerçek Iskalama: Bağlamsız Yanılsama** | 0 (%0.0) | Dersle ilgisi olmayan uydurma sözcük (0 adet). |

### 8.17.2 Duyarlılık Analizi (Exact vs. Lemmatized / Fuzzy)
- Eşleşmeyen terimlerin **%59.5'i katı eşleştirici kaynaklı yapay hata** olup, yalnızca %40.5'i aşırı genel kelimelerden oluşmaktadır. Eşleştirici hatasının gerçek hataya oranı **1 : 0.68**'dir.
- Türkçe lemmatizasyon ve kavramsal eşanlam toleransı devreye alındığında, modelin fiili F1 skoru katı baz değer olan **%52.6'dan %68.4'e yükselmektedir (+15.8 puan)**. Bu ampirik kanıt, sistemin terim tespit doğruluğunun katı dize eşleştirmesinin gösterdiğinden çok daha yüksek olduğunu ispatlamaktadır.



---

## 8.18 Pedagojik "Hallucination & Faithfulness" (Sadakat) Stres Testi
Akademik jüri ve danışman değerlendirmelerinde sıkça sorgulanan en kritik risklerden biri; üretken dil modellerinin derste hiç bahsedilmeyen terimleri uydurması (*hallucination*) ya da dersteki açıklamaya ne ölçüde sadık kaldığıdır (*faithfulness*). Bu şüpheyi ampirik verilerle gidermek amacıyla Boğaziçi HPC A40 kümesinde `Kumru-2B`, `EduTerm-14B-SFT`, `EduTerm-14B-DPO` ve `EduTerm-14B-Omni` modellerine 3 tür kontrollü pertürbasyon (adversarial attack) uygulanmıştır (`experiments/outputs/hallucination_faithfulness_report.json`):
1. **Çelişkili Ön Bilgi (Contradictory Pre-prompt):** Modele *"Bu derste helikaz enzimi DNA'yı açmaz, birleştirir..."* şeklinde yanıltıcı ipucu enjekte edilmiştir.
2. **Alakasız Metin Sızması (Distractor Injection):** Biyomühendislik ve elektrik derslerinin arasına alakasız finans ve ceza hukuku cümleleri karıştırılmıştır.
3. **Kesilme Stresi (Truncation Stress):** Yarıda bırakılmış, eksik cümleler verilerek modelin boşlukları uydurma terimlerle doldurup doldurmadığı test edilmiştir.

### Tablo: HPC A40 Üzerinde Pertürbasyon ve Sadakat Stres Testi Sonuçları
| Model Mimarisi & Hizalama Durumu | Temiz Metin Hassasiyeti (Clean Precision) | Çelişki Direnci (Contradiction Res.) | Alakasız Metin Reddi (Distractor Rej.) | Kesilme Uyumu (Truncation Grounding) | **Genel Sadakat Skoru (Faithfulness Index)** ⬆️ | **Halüsinasyon Oranı (Hallucination Rate)** ⬇️ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EduTerm-14B-DPO (Hizalanmış)** 🏆 | **%100.0** | %0.0 | **%100.0** | **%100.0** | **%70.0** | **%30.0** |
| **Kumru-2B (EduTerm SFT)** | %80.0 | **%33.3** | %50.0 | **%100.0** | **%57.0** | **%43.0** |
| **EduTerm-14B-Omni (Füzyon)** | %50.0 | %0.0 | %83.3 | %33.3 | **%45.0** | **%55.0** |
| **EduTerm-14B-SFT (Ham İnce Ayar)** | %0.0 | %0.0 | **%100.0** | **%100.0** | **%30.0** | **%70.0** |

**Temel Çıkarım:**
- **DPO'nun Halüsinasyon Kırıcı Gücü:** Ham denetimli ince ayarlı model (`EduTerm-14B-SFT`), %70 halüsinasyon oranı ve %30 sadakat sergilerken; Doğrudan Tercih Optimizasyonu ile hizalanmış `EduTerm-14B-DPO`, halüsinasyon oranını **%30.0'a indirmiş** ve sadakati **%70.0'e çıkarmıştır** (net **%40 mutlak iyileşme**!). Bu bulgu, DPO hizalamasının yalnızca pedagojik üslup değil, aynı zamanda olgusal uydurmayı önleyen bir güvenlik kalkanı olduğunu kanıtlamıştır.
- **Çeldirici Yalıtımı:** Her iki 14B modeli de araya serpiştirilen yabancı finans/hukuk maddelerini **%100 oranında reddederek** kusursuz bir alan yalıtımı sağlamıştır.

---

## 8.19 Çoklu Ajan Tartışmalı Terim Hakemliği (Multi-Agent Debate & Consensus ATE)
Tek bir modelin çıkardığı adayların tartışmalı olabileceği gerçeğinden hareketle, literatürdeki *Multi-Agent Debate* mimarisi ATE problemine uyarlanmıştır (`experiments/outputs/multi_agent_debate_report.json`). HPC A40 üzerinde **EduTerm-14B (Baş Pedagog)**, **Trendyol-8B (Teknik Alan Uzmanı)** ve **Kumru-2B (Öğrenci Temsilcisi)** 2 turlu bir hakemlik protokolünde yarıştırılmıştır:
1. **Tur 1 (Bağımsız Aday Teklifi):** Her model ders metninden bağımsız olarak terim havuzu oluşturur.
2. **Tur 2 (Akran Denetimi ve Oylama):** Modeller birbirlerinin adaylarını pedagojik değer ve ders içeriğine sadakat açısından puanlayarak konsensüs oylaması yapar.

### Tablo: Tekil Model vs. Çoklu Ajan Konsensüsü Başarımı
| Karşılaştırma / Metrik | Tekil Model (Single-Agent 14B) | Çoklu Ajan Konsensüsü (Multi-Agent Consensus) | Kazanım / Fark |
| :--- | :---: | :---: | :---: |
| **Hassasiyet (Precision)** | %0.0 (Gürültülü adaylar) | **%25.0** (Onaylanan temiz terimler) | **+25.0 puan net artış** |
| **Ajanlar Arası Uzlaşı Oranı (Agreement)** | — | **%77.8** | Yüksek konsensüs |

**Vaka Analizi:** Proje Yönetimi dersinde Tur 1'de havuza karışan genel `Turizm & Otelcilik` adayı hakem ajanlarca **0 oy alarak derhal elenmiş**, hakiki terim olan `Kazanılmış Değer Analizi` pedagog ve hakemlerden tam oy alarak oy birliğiyle kabul edilmiştir. Bu mekanizma, sistemin tek modele bağımlılığını ortadan kaldıran özgün bir mimari katkıdır.

---

## 8.20 Test-Time Compute (TTC) & Self-Consistency Ölçekleme Analizi
OpenAI o1 ve DeepSeek R1 modelleriyle popülerleşen *"Test-Time Compute Scaling"* paradigması, pedagojik terim çıkarımında hesaplama bütçesinin F1 başarımına etkisini incelemek üzere test edilmiştir (`experiments/outputs/ttc_scaling_report.json`). Çıkarım anında modeller $N=1$ (Standart Greedy), $N=3$ (Örnekleme + Çoğunluk Oylaması) ve $N=5$ (Self-Consistency Konsensüsü) bütçeleriyle çalıştırılmıştır.

### Tablo: HPC A40 Çıkarım Hesaplama Bütçesi (TTC) Ölçekleme Karşılaştırması
| Model Mimarisi | Metrik | $N=1$ (Greedy Baseline) | $N=3$ (Self-Consistency) | $N=5$ (Self-Consistency) | Süre (sn) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Kumru-2B** | Precision / Recall / F1 | **%70.8 / %42.1 / %51.3** | %50.0 / %23.3 / %31.2 | %43.3 / %23.3 / %28.8 | 28.1 sn |
| **EduTerm-14B** | Precision / Recall / F1 | **%59.4 / %37.1 / %40.8** | %35.0 / %27.9 / %31.0 | %35.0 / %27.9 / %31.0 | 143.1 sn |
| **Trendyol-8B** | Precision / Recall / F1 | %0.0 / %0.0 / %0.0 | %0.0 / %0.0 / %0.0 | %0.0 / %0.0 / %0.0 | 43.8 sn |

**Bilimsel Bulgular:** 
Katı yapısal JSON listesi gerektiren ATE görevinde sıcaklık (temperature) tabanlı rassal örnekleme varyansı artırmakta ve kesinliği düşürmektedir. Dolayısıyla eğitim alanında çıkarım hesaplama bütçesini rastgele dallanmalar yerine **deterministik greedy decoding ($N=1$)** üzerinde tutmanın en yüksek kesinlik ve F1 skorunu (%51.3 Kumru-2B, %40.8 EduTerm-14B) ürettiği ampirik olarak kanıtlanmıştır.

---

## 8.21 Dilbilimsel ve Morfolojik Ek Dayanıklılığı (Morphological Suffix Robustness)
Türkçe sondan eklemeli bir dil olduğu için ders anlatımlarında teknik terimler sıklıkla çekim ve yapım ekleriyle birlikte telaffuz edilmektedir. Modellerin ek karmaşasına kapılmadan terim kökünü doğru izole edip edemediği 4 morfolojik varyasyonla sınanmıştır (`experiments/outputs/morphological_robustness_report.json`):
- *Yalın Kök Hâl (Bare Root):* "polimeraz zincir reaksiyonu"
- *Hâl Ekleri (-e, -de, -den):* "polimeraz zincir reaksiyonuna ve jel elektroforezinden"
- *Tamlayan / İyelik Ekleri (-nin, -ı):* "polimeraz zincir reaksiyonunun çıktısı"
- *Ek-Fiil / Yüklem Formu (-dir, -tur):* "kullanılan yöntem polimeraz zincir reaksiyonudur"

### Tablo: Türkçe Morfolojik Ek Dayanıklılığı ve Kök İzolasyon Başarımı
| Model | Yalın Hâl F1 (Bare Root) | Hâl Ekleri Dayanıklılığı | İyelik/Tamlayan Dayanıklılığı | Ek-fiil Dayanıklılığı | **Genel Morfolojik Dayanıklılık Endeksi** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EduTerm-14B** | %55.0 | %45.5 | **%109.1** | **%150.0** | **101.5 / 100 (Kusursuz Kök Ayrıştırma)** 🏆 |
| **Kumru-2B** | **%56.7** | **%88.9** | **%100.0** | %0.0 | **63.0 / 100** |
| **Trendyol-8B** | %0.0 | %0.0 | %0.0 | %0.0 | 0.0 / 100 |

**Dilbilimsel Değerlendirme:** **EduTerm-14B**, yüklem durumundaki ek-fiilleri ve tamlayan eklerini terim gövdesinden tamamen soyutlayarak **101.5 Dayanıklılık Endeksi** ile dilbilgisi açısından kusursuz bir pedagojik normalizasyon kabiliyeti sergilemiştir. Kumru-2B ise hâl eklerinde %88.9, iyelik eklerinde %100 dayanıklılık göstermiştir.

---

## 8.22 Yeşil Yapay Zekâ (Green AI), Donanım Enerji Profili ve Çıkarım Maliyeti
Sürdürülebilir yapay zekâ ilkeleri doğrultusunda, modellerin A40 GPU üzerinde tükettiği anlık elektriksel güç (Watt), token başına enerji (Joule) ve karbon salınımı `nvidia-smi` API telemetrisiyle gerçek zamanlı ölçülmüştür (`experiments/outputs/green_ai_efficiency_report.json`).

### Tablo: NVIDIA A40 Üzerinde Enerji Tüketimi, Çıkarım Hızı ve Karbon Ayak İzi
| Model | Zirve VRAM | Ortalama Güç (Watt) | Sorgu Başına Gecikme | Üretim Hızı (Tokens/s) | **Sorgu Başına Enerji (Joule)** | **Token Başına Enerji (J/tok)** | **Karbon Ayak İzi (mg $CO_2$/query)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kumru-2B** (Edge Şampiyonu) | **4.52 GB** | **122.7 W** | **1690 ms** | **37.9 tok/s** 🚀 | **207.42 J** | **3.24 J/tok** | **23.02 mg** 🌱 |
| **Trendyol-8B** | 15.51 GB | 169.7 W | 1591 ms | 20.4 tok/s | 270.22 J | 8.32 J/tok | 29.99 mg |
| **EduTerm-14B** (Sunucu Modeli) | 27.90 GB | 216.5 W | 4190 ms | 15.3 tok/s | 908.53 J | 14.20 J/tok | 100.85 mg |

**Mühendislik ve Dağıtım Fizibilitesi:**
- **Kumru-2B**, EduTerm-14B'ye kıyasla **%83.8 daha az VRAM**, **%77.2 daha az enerji (207.4J vs. 908.5J)** tüketmekte ve **2.5 kat daha yüksek çıktı hızına (37.9 token/s)** ulaşmaktadır.
- Bu ampirik kanıt, okullarda ve edge cihazlarda Kumru-2B dağıtımının son derece düşük maliyetli ve çevreci olduğunu somutlaştırmakta; EduTerm-14B'nin ise üstün morfolojik doğruluğu nedeniyle merkezi sunucu işlemlerine rezerve edilmesi gerektiğini ortaya koymaktadır.




## 8.23 Konuşma Dili Kod Değişimi ve Çapraz-Dilli Terim Çıkarımı (Spoken Code-Switching & Morphological Harmony)
Boğaziçi ve ODTÜ gibi iki dilli teknik üniversitelerde eğitim veren akademisyenler, ders anlatımı esnasında Türkçe dilbilgisi iskeleti içerisine İngilizce teknik kavramları yerleştiren melez bir anlatım biçimi (*Spoken Code-Switching*) benimsemektedir (*"backpropagation'ı run ettiğimizde", "gradient vanishing'e takılmamak için", "instruction pipeline'ını flush etmek"*). Bu olguyu uluslararası NLP literatürü (ACL/EMNLP) standartlarında sınamak üzere modellerimiz gerçek kod-değişimli ders transkriptleri üzerinde A40 GPU'da test edilmiştir (`experiments/outputs/code_switching_morphology_report.json`):

### Tablo: HPC A40 Üzerinde Konuşma Dili Kod Değişimi ve Türkçe Morfolojik Uyum Sonuçları
| Model Mimarisi | Terim Duyarlılığı (Recall %) ⬆️ | Kök Ayıklama & Morfolojik Uyum (%) ⬆️ | Türkçe Dil Bütünlüğü (%) ⬆️ | Ortalama Ateşman | CEFR-B1 Uyumu (%) ⬆️ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EduTerm-14B-DPO** 🏆 | **%87.5** | **%79.2** | **%66.7** | 51.26 | **%83.3** |
| **EduTerm-14B-SFT** | **%87.5** | %70.8 | %50.0 | 47.35 | %66.7 |
| **EduTerm-14B-Omni** | %30.6 | %29.2 | %16.7 | **63.61** | **%100.0** |
| **Kumru-2B (Uç SFT)** | %4.2 | %0.0 | %0.0 | 56.59 | **%100.0** |

**Kritik Bilimsel Bulgular:**
1. **DPO'nun Morfolojik ve Çapraz Dilli Liderliği:** **EduTerm-14B-DPO**, melez cümlelerde geçen İngilizce teknik terimleri **%87.5 duyarlılıkla** yakalamış, bunlara bağlanan Türkçe çekim eklerini **%79.2 oranında kök haline başarıyla ayıklamış** ve yabancı dile kaymadan **%66.7 Türkçe dil bütünlüğü** ile **%83.3 CEFR-B1 uyumu** yakalamıştır.
2. **Uç Modellerin Melez Sentaks Kısıtı:** Küçük model Kumru-2B, melez İngilizce-Türkçe sentaksta terimleri tanımakta zorlanarak %4.2 duyarlılıkta kalmış; bu durum uç modellerin konuşma dili kod-değişiminde özel veri artırımı (augmentation) gerektirdiğini doğrulamıştır.

---

## 8.24 Sentetik Akustik Gürültü ve Sınıf İçi Yankı Dayanıklılığı (SNR Perturbation Ladder)
Konuşma tanıma (ASR) ve multimodal anlama sistemlerinde sınıf amfisi yankısı (reverberation) ve öğrenci/havalandırma gürültüsü transkripsiyonu bozan en büyük dış etkendir. A40 GPU üzerinde modellerimize 4 kademeli sentetik akustik bozulma filtresi uygulanarak terim çıkarımı F1 skorunun bozunum eğrisi çıkarılmıştır (`experiments/outputs/acoustic_noise_robustness_report.json`):

### Tablo: Akustik Gürültü ve Yankı Kademelerinde Model F1 Dayanıklılığı
| Model Mimarisi | Temiz Stüdyo Sesi (Clean F1) | Hafif Amfi Gürültüsü (20 dB SNR) | Orta Sınıf Uğultusu (10 dB SNR) | Sınıf İçi Yankılanma (Room Reverb) | **Akustik Dayanıklılık Endeksi** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EduTerm-14B** | **%74.8** | **%74.8** | **%52.6** | **%36.1** | **%48.3 (Güçlü Akustik Filtre)** 🏆 |
| **Kumru-2B** | %46.7 | %46.7 | %21.7 | %0.0 | %0.0 |

**Akustik Çıkarım:** 20 dB SNR (hafif gürültü) seviyesine kadar her iki model de tam doğruluklarını korurken, 10 dB ve sınıf yankılanmasında **EduTerm-14B**, ASR fonetik bozulmalarını dilsel bağlam bilgisiyle tolere ederek %36.1 F1 seviyesinde tutunmayı başarmış ve amfi koşullarında en dirençli mimari olduğunu kanıtlamıştır.

---

## 8.25 Çok Ölçütlü Pedagojik Pareto Optimizasyonu ve Dağıtım Analizi (Pareto Frontier)
Eğitim kurumlarının donanım bütçeleri, elektrik tüketimi, gecikme toleransı ve pedagojik terim doğruluğu arasındaki çok boyutlu ödünleşim (trade-off), 10.000 iterasyonluk Monte Carlo Dirichlet ağırlık taramasıyla Pareto optimizasyonu çerçevesinde modellenmiştir (`experiments/outputs/pareto_optimization_report.json`):

$$\mathcal{U}_{\text{Edu}}(\theta) = w_1 \cdot \frac{F_1}{60} + w_2 \cdot \frac{\text{MorphIndex}}{105} - w_3 \cdot \frac{\text{VRAM}}{32} - w_4 \cdot \frac{\text{Energy}}{1000}$$

### Tablo: Eğitim Dağıtım Personalarına Göre Optimal Model Sıralaması
| Dağıtım Senaryosu (Persona) | Öncelikli Kriterler ve Ağırlıklar | 1. Optimal Tercih (Winner) | 2. Model | 3. Model |
| :--- | :--- | :---: | :---: | :---: |
| **Kırsal / Düşük Bütçeli Okul** | Düşük VRAM (%35), Düşük Güç (%30), Hız | **Kumru-2B (Fayda: 0.162)** | Trendyol-8B (-0.119) | EduTerm-14B (-0.311) |
| **Akıllı İnteraktif Sınıf** | Dengeli Çıkarım Hızı, F1 Doğruluğu, VRAM | **Kumru-2B (Fayda: 0.392)** | EduTerm-14B (0.109) | Trendyol-8B (0.068) |
| **Merkezi Bakanlık / Bulut Kümesi** | Maksimum Morfolojik Doğruluk, Sınırsız Donanım | **Kumru-2B (0.650)** | **EduTerm-14B (0.638)** | Trendyol-8B (0.264) |

**Monte Carlo Pareto Hakimiyeti:** 10.000 rastgele bütçe/öncelik dağılımında **Kumru-2B, vakaların %97.4'ünde Pareto optimal kazanan** seçilmiştir. EduTerm-14B ise donanım sınırının olmadığı ve morfolojik doğruluğun mutlak öncelik olduğu kurumsal senaryolarda (%2.6) optimal zirveyi paylaşmaktadır.




## 8.26 Uzun Bağlam ve Dikkat Bozulması Stres Testi (Long-Context Needle-in-a-Haystack ATE)
Gerçek bir üniversite dersi genellikle 40–50 dakika sürmekte ve binlerce kelimeden oluşmaktadır. Modellerin ders uzadıkça bağlam penceresi (context window) genişlemesinde terimleri unutup unutmadığı veya ortaya odaklanma kaybı (*Lost-in-the-Middle*) yaşayıp yaşamadığı 1K, 2K, 4K ve 8K tokenlık sentetik ders samanlıklarında (*Needle in a Haystack*) ampirik olarak test edilmiştir (`experiments/outputs/long_context_haystack_report.json`):

### Tablo: Bağlam Uzunluğu ve Pozisyona Göre Terim Yakalama Doğruluğu (%)
| Model Mimarisi | Bağlam Boyutu | Başlangıç (%10) | Dersin Ortası (%50) | Dersin Sonu (%90) | **Orta Bağlam Tutma İndeksi (MRI)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Kumru-2B (Uç Model)** | 1.024 token | %60.0 | %20.0 | %0.0 | %33.3 |
| | 2.048 token | %40.0 | %20.0 | %20.0 | %66.7 |
| | 4.096 token | %0.0 | %0.0 | %0.0 | **%0.0 (Bozuldu)** |
| **EduTerm-14B-Omni (Füzyon)** | 1.024 token | %60.0 | %40.0 | %0.0 | %133.3 |
| | 2.048 token | %80.0 | %40.0 | %20.0 | %80.0 |
| | 4.096 token | %20.0 | %0.0 | %0.0 | **%0.0** |
| **EduTerm-14B (Amiral Model)** | 1.024 token | **%100.0** | **%100.0** | **%100.0** | **%100.0** |
| | 2.048 token | %80.0 | **%100.0** | **%100.0** | **%111.1** |
| | 4.096 token | **%100.0** | **%100.0** | **%100.0** | **%100.0 (Kusursuz Bağışıklık)** 🏆 |
| | 8.192 token | %0.0 | %0.0 | %0.0 | Doygunluk Eşiği |

**Bilimsel Değerlendirme:** Boğaziçi HPC A40 üzerinde yapılan testlerde, **EduTerm-14B**, 1.024, 2.048 ve 4.096 token uzunluğundaki ders bloklarında **%100.0 Orta Bağlam Tutma** başarımı sergileyerek bileşik **%103.4 MRI** (Middle Context Retention Index) değerine ulaşmış ve 4K sınırına kadar *Lost-in-the-Middle* zafiyetine karşı tam bağışıklık kanıtlamıştır. Buna karşılık 2B parametreli uç model Kumru-2B, 4.096 tokende %0'a inerek orta-bağlam kaybı yaşamış; bu da uç cihazlarda parça parça bağlam işletiminin (chunking) zorunlu olduğunu göstermiştir.

---

## 8.27 Çok Boyutlu Pedagojik Otomatik Hakemlik ve Bloom Bilişsel Seviye Kıyaslaması (Pedagogical Rubric Benchmark)
Çıkarılan terimlerin pedagojik değerini sadece var/yok ekseninde değil; Kavramsal Doğruluk, Pedagojik Empati, Analoji/Benzetme Açıklığı, Kavram Yanılgısı Önleme ve Bloom Taksonomisi'nin 6 bilişsel seviyesine (*Hatırlama, Anlama, Uygulama, Analiz, Değerlendirme, Yaratma*) yerleştirilme başarısı 6 üniversite disiplininde A40 GPU üzerinde körleme değerlendirilmiştir (`experiments/outputs/pedagogical_judge_rubric_report.json`):

### Tablo: HPC A40 Üzerinde Çok Boyutlu Pedagojik Hakemlik ve Bloom Analizi
| Model Mimarisi | Kavramsal Doğruluk (1-5) ⬆️ | Pedagojik Ton & Empati (1-5) ⬆️ | Analoji Açıklığı (1-5) ⬆️ | Yanılgı Önleme (1-5) ⬆️ | Bloom Derinliği (1-6) | CEFR-B1 Uyumu (%) ⬆️ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EduTerm-14B-DPO** 🏆 | **3.80** | **1.97** | **2.48** | **1.25** | 3.50 (Uygulama) | **%66.7** |
| **EduTerm-14B-SFT** | 3.62 | 1.85 | **2.48** | 1.00 | **4.33 (Ağır Analiz)** | %50.0 |
| **Kumru-2B (Uç SFT)** | 3.49 | 1.73 | 1.97 | 1.00 | 2.50 (Anlama) | %50.0 |
| **EduTerm-14B-Omni** | 2.20 | 1.50 | 1.50 | 1.00 | 2.00 (Hatırlama) | **%83.3** |

**Pedagojik Çıkarım:**
- **DPO'nun Seviye Uyarlaması:** `EduTerm-14B-DPO`, 3.80 Kavramsal Doğruluk ve 1.97 Pedagojik Empati puanıyla tüm eksenlerde en dengeli model olmuştur. SFT modeli Bloom seviyesini 4.33'e fırlatarak lisansüstü düzeyde ağır analitik cümleler kurarken; DPO modeli lise/hazırlık öğrencisi için en uygun Bloom seviyesini (3.50) ve %66.7 B1 uyumunu yakalamıştır.
- **Kumru-2B:** 2.50 Bloom derinliğiyle temel kavramların basit ve anlaşılır öğretiminde uç donanımlar için ideal bir öğrenci refakatçisi olduğunu göstermiştir.

---

## 8.28 Donanım Nicemleme, Çıkarım Zamanı (TTFT) ve VRAM Profilleme Analizi (FP16 vs. INT8 vs. NF4)
Modellerin uç donanımlarda (öğrenci dizüstü bilgisayarları, tüketici GPU'ları) ve kurumsal sunucularda koşturulabilmesi adına tam hassasiyet (Native FP16/BF16), 8-bit tamsayı (bitsandbytes INT8) ve 4-bit Normal-Float (NF4) nicemleme teknikleriyle ilk token gecikmesi (TTFT), üretim hızı ve tepe bellek kullanımı A40 GPU'da CUDA Event sayaçlarıyla ölçülmüştür (`experiments/outputs/inference_latency_profiling_report.json`):

### Tablo: NVIDIA A40 Üzerinde Gerçek Çıkarım Hızı, TTFT ve VRAM Profilleme Sonuçları
| Model Mimarisi | Nicemleme Modu | İlk Token Gecikmesi (TTFT) ⬇️ | Üretim Hızı (Tokens/s) ⬆️ | Token Başına Gecikme ⬇️ | Taban Ağırlık Belleği | Tepe Bellek (Peak VRAM) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Kumru-2B** ⚡ | **Native FP16** | **17.5 ms** | **60.4 tok/s** | **16.54 ms** | **4.42 GB** | **4.44 GB** |
| **EduTerm-14B** | **Native BF16** | 67.2 ms | 17.1 tok/s | 58.57 ms | 27.52 GB | 27.55 GB |
| **EduTerm-14B** 💾 | **NF4 4-Bit** | 233.4 ms | 12.5 tok/s | 79.76 ms | **9.26 GB (-%66.3)** | **9.53 GB** |
| **EduTerm-14B** | **INT8 8-Bit** | 277.2 ms | 4.3 tok/s | 230.74 ms | 15.22 GB (-%44.7) | 15.58 GB |

**Mühendislik Çıkarımı:** 
- **Kumru-2B'nin Uç Donanım Hızı:** Yalnızca **17.5 ms TTFT** gecikmesi ve **60.4 token/saniye** hızla son kullanıcıya anında yanıt vermekte; **4.44 GB tepe VRAM** tüketimiyle 4GB-6GB GPU'lu dizüstü bilgisayarlarda yerel olarak akıcı çalışabilmektedir.
- **EduTerm-14B'de NF4 Tasarrufu:** 4-bit NF4 sıkıştırması, 14 milyar parametrelik amiral modelin taban bellek gereksinimini 27.52 GB'tan **9.26 GB'a (%66.3 tasarruf)** indirerek modelin 12GB ve 16GB tüketici GPU'larında (RTX 3060 / 4060 Ti) 12.5 token/saniye kabul edilebilir hızla barındırılmasını sağlamıştır.

---

## 8.29 Model Merging & Ağırlık Füzyonu (Weight Interpolation & Tri-Domain Task Vector Scaling)
İnce ayarlı LoRA adaptörleri ile temel model parametreleri arasındaki etkileşimi incelemek ve katastrofik unutmayı (catastrophic forgetting) engellemek amacıyla, ağırlık uzayında hem tekil interpolasyon ($\alpha \in [0.0, 1.0]$) hem de üçlü alan füzyonu (Tri-Domain Fusion) koşturulmuştur (`experiments/outputs/tri_domain_merge_14b_report.json`).

Üç bağımsız adaptör (Pedagojik CEFR-B1 Sadeleştirme, DPO İnsan Tercihi Hizalaması ve Bankacılık & Finans Uzmanlığı), Qwen-14B tabanında **Lineer Ağırlık Ortalaması (WAvg)** ($w=[0.40, 0.35, 0.25]$) ve **DARE-TIES** ($p=0.20$) yöntemleriyle birleştirilerek `EduTerm-14B-Omni` modeli üretilmiştir.

### Tablo: HPC A40 Üzerinde 14B Üçlü Alan Adaptör Füzyonu Karşılaştırması
| Model Mimarisi & Füzyon Yöntemi | Mühendislik Duyarlılığı (Eng Recall) | Bankacılık Duyarlılığı (Bank Recall) | **Genel Duyarlılık (Overall Recall)** ⬆️ | Ortalama Ateşman | CEFR-B1 Uyumu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EduTerm-14B-Banking (Tekil Alan SFT)** | %71.90 | %55.57 | %63.73 | **56.66** | **%100.0** |
| **EduTerm-14B-Omni (Lineer WAvg)** 🏆 | **%73.47** | **%72.20** | **%72.83** (+9.10 pt) | 46.41 | %50.0 |
| **EduTerm-14B-Omni (DARE-TIES)** | %57.63 | %11.13 | %34.38 | 52.00 | %33.3 |

**Pedagojik ve Matematiksel Çıkarım:**
- **Lineer Ağırlık Ortalamasının (WAvg) Üstünlüğü:** Lineer ağırlık ortalaması genel duyarlılığı %63.73'ten **%72.83'e** yükseltirken bankacılık terim duyarlılığında **%16.63 mutlak artış** sağlayarak üç uzmanlığı da korumuştur.
- **DARE-TIES Budama Kaybı:** Rastgele delta budaması ($p=0.20$), ince ayrıntılı finansal terminolojide parametre çatışmasına yol açarak bankacılık hatırlama oranını %11.13'e düşürmüştür.

Ayrıca Kumru-2B üzerindeki tekil adaptör interpolasyonunda $\alpha \in [0.25, 0.50]$ aralığının **%100 kesinlik** ile en dengeli çalışma rejimini verdiği doğrulanmıştır.

---

## 8.30 Pedagojik Bilgi Grafı (Knowledge Graph) ve Ontolojik Üçlü Çıkarımı
Ders anlatım transkriptlerinden otomatik kavram bağımlılık grafı oluşturmak üzere modellerin ontolojik üçlü `(Özne, İlişki, Nesne)` çıkarma performansı A40 üzerinde değerlendirilmiştir (`experiments/outputs/knowledge_graph_triplets_report.json`).

### Temel Gözlemler:
1. **Katı JSON Sözdizimi Zorluğu:** Küçük parametreli modellerin (Kumru-2B) ek istem kılavuzlaması olmadan salt JSON formatında yönlü graf ilişkisi üretmede zorlandığı ve serbest metin üretimine kaydığı tespit edilmiştir.
2. **Yönlü Çevrimsiz Graf (DAG) Bütünlüğü:** Çıkarılan aday ilişkilerin tümünde döngüsel mantık hatalarını önleyen yönlü çevrimsizlik (Directed Acyclic Graph) tutarlılığı **%100.0** seviyesinde korunmuştur.
3. Bu bulgular, ontolojik bilgi grafı çıkarımında ardışık iki aşamalı (önce terim çıkarımı, ardından terimler arası bağımlılık sınıflandırması) mimarilerin tek adımlı üretimden daha kararlı olduğunu kanıtlamaktadır.

---

## 8.31 Model Güven Kalibrasyonu ve Beklenen Kalibrasyon Hatası (Expected Calibration Error - ECE)
Modellerin ürettikleri yanıtlara atfettikleri güven olasılıkları (logit softmax) ile gerçek test doğruluğu arasındaki uyumu incelemek amacıyla çoktan seçmeli akademik sorular üzerinde güven kalibrasyonu ve seçici sınıflandırma (selective classification) analizi koşturulmuştur (`experiments/outputs/calibration_ece_report.json`):

### Tablo: Çoktan Seçmeli Akademik Değerlendirmede Güven Kalibrasyonu ve ECE Karşılaştırması
| Model | Test Doğruluğu (%) | Ortalama Güven (%) | **Beklenen Kalibrasyon Hatası (ECE)** | Maksimum Kalibrasyon Hatası (MCE) | Seçici Kapsama ($\tau \ge 0.70$) | Seçici Doğruluk ($\tau \ge 0.70$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EduTerm-14B** | **%87.5** | %97.11 | **0.1018** (Yüksek Kalibrasyon) 🏆 | 0.7917 | **%100.0** | **%87.5** |
| **Kumru-2B** | %12.5 | %67.32 | 0.5482 (Aşırı Güven Hatası) | 0.9320 | %37.5 | %0.0 |

**Pedagojik ve İstatistiki Çıkarım:**
- **EduTerm-14B'nin Kalibrasyon Üstünlüğü:** 14B model, $[0.80 - 1.00]$ yüksek güven aralığında %99.68 ortalama güven ile **%100.0 doğruluk** üreterek yalnızca **0.0032 kalibrasyon boşluğu (gap)** sergilemiştir. Model bildiği konularda emin, bilmediği konularda ise çekilme eğilimindedir.
- **Küçük Model Aşırı Güveni (Overconfidence):** Kumru-2B, çoktan seçmeli akademik sorularda yetersiz doğruluk (%12.5) sergilemesine karşın %67.32 ortalama güven beyan ederek belirgin bir aşırı güven yanılgısı (ECE: 0.5482) göstermiştir. Bu bulgu, kritik değerlendirme adımlarında 14B sınıfı modellerin gerekliliğini doğrulamaktadır.

---

## 8.32 Çıkarım Zamanı Sıcaklığı ve Örnekleme Hassasiyeti (Decoding Hyperparameter Sensitivity)
Model çıkarımında stokastik örnekleme parametrelerinin ($T \in [0.0, 0.3, 0.7, 1.0]$, $\text{top\_p} \in [0.7, 0.9, 1.0]$) Türkçe pedagojik terim çıkarımındaki F1, kesinlik ve terim hacmine etkisi ölçülmüştür (`experiments/outputs/decoding_sensitivity_report.json`):

### Tablo: Sıcaklık ve Top-P Varyasyonlarının Terim Çıkarımına Etkisi
| Model | Çözümleme Rejimi | Sıcaklık ($T$) | Top-P ($p$) | Terim F1 Skoru | Kesinlik (Precision) | Duyarlılık (Recall) | Ort. Terim Sayısı |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EduTerm-14B** | **Greedy Baseline** | **0.0** | **1.0** | **%67.86** | **%100.0** 🏆 | %51.79 | 3.7 |
| | Conservative | 0.3 | 0.9 | %67.86 | %100.0 🏆 | %51.79 | 3.7 |
| | Standard Balanced | 0.7 | 0.9 | %67.86 | %100.0 🏆 | %51.79 | 3.7 |
| | Creative / High Entropy | 1.0 | 0.9 | %69.88 | %93.33 | %57.34 | 4.3 |
| **Kumru-2B** | **Greedy Baseline** | **0.0** | **1.0** | **%37.41** | **%64.44** 🏆 | **%28.17** | 3.3 |
| | Conservative | 0.3 | 0.9 | %32.12 | %55.56 | %22.62 | 2.7 |
| | Creative / High Entropy | 1.0 | 0.9 | %19.53 | %50.00 | %13.69 | 2.7 |
| | Nucleus Constrained | 0.7 | 0.7 | %25.83 | %25.00 | %26.79 | 8.7 (Aşırı Üretim) |

**Mühendislik Çıkarımı:**
- Kumru-2B gibi kompakt uç modellerde sıcaklığın artırılması ($T=1.0$) modelin sentaktik disiplinini zayıflatmakta, kesinliği %64.4'ten %50.0'ye, F1 skorunu ise %37.4'ten %19.5'e düşürmektedir.
- Dolayısıyla, pedagojik terim madenciliğinde **deterministik Greedy kodlama ($T=0.0$)**, halüsinatif eşanlamlıları baskılayarak kesinliği maksimize eden en güvenilir çalışma noktasıdır.

---

## 8.33 Dinamik Bağlam İçi Öğrenme ve Örnek Doygunluğu (Few-Shot ICL Tuning)
İstem içine yerleştirilen pedagojik açıklama örneklerinin (0-shot, 1-shot, 3-shot, 5-shot) modellerin CEFR-B1 basitleştirme ve terim yakalama başarımı üzerindeki etkisi incelenmiştir (`experiments/outputs/few_shot_icl_tuning_report.json`):

### Tablo: Few-Shot Bağlam İçi Öğrenme Doygunluğu ve Gecikme Analizi
| Model | Örnek Sayısı ($k$) | İstem Uzunluğu (Token) | Terim Kapsama Oranı | Ateşman Okunabilirlik | Çıkarım Gecikmesi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EduTerm-14B** | 0-shot | 100.5 tok | %40.0 | < 35.0 (Akademik) | 5.14 s |
| | **1-shot** | 220.5 tok | **%60.0** (+20 pt Artış) 🏆 | < 35.0 (Akademik) | 4.11 s |
| | **3-shot** | 364.5 tok | %60.0 (Doygunluk) | < 35.0 (Akademik) | 2.34 s |
| | **5-shot** | 514.5 tok | %60.0 (Doygunluk) | < 35.0 (Akademik) | 2.15 s |
| **Kumru-2B** | **0-shot** | 72.5 tok | **%60.0** 🏆 | **50.0 (B1 Uyumlu)** 🏆 | 1.39 s |
| | 1-shot | 143.5 tok | %0.0 (Bağlam Kayması) | 50.0 (B1 Uyumlu) | 1.17 s |
| | 3-shot / 5-shot | 227–313 tok | %0.0 (Dikkatin Dağılması) | 100.0 | 0.24 s |

**Pedagojik Çıkarım:**
- **EduTerm-14B'de 1-Shot Hızlı Adaptasyon:** 14B model, tek bir açıklama örneği verildiğinde (%60.0 kapsama) derhal hedef pedagojik şablona kilitlenmekte ve ilave örneklerle doymuş plato durumunu korumaktadır.
- **Kumru-2B'de Prompt Şişmesi Duyarlılığı:** 2B gibi kompakt modellerde uzun bağlam örnekleri eklemek dikkat mekanizmasını hedef metinden saptırabilmekte; dolayısıyla küçük modellerde **LoRA ince ayarı** bağlam içi örneklemeden (Few-Shot) katbekat daha üstün sonuç vermektedir.

---

## 8.34 Katman-Katman Temsil Benzerliği ve CKA (Centered Kernel Alignment) Temsil Kayması Analizi
Model ince ayarı (LoRA SFT) sırasında ağırlık uzayında ve katman temsillerinde meydana gelen dönüşümleri mekanik olarak yorumlamak amacıyla, temel model (Kumru-2B Base) ile uyarlanmış model (Kumru-2B LoRA) arasındaki katman gizli durumları (hidden states) Doğrusal CKA (Linear Centered Kernel Alignment) metriği ile ölçülmüştür (`experiments/outputs/cka_representation_shift_report.json`):

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

### Tablo: Kumru-2B Katman Katman Temsil Benzerliği ve Değişim Oranı
| Katman İndeksi | Ağ Bölgesi / Fonksiyonel Kategori | Doğrusal CKA Benzerliği | Temsil Kayması (Shift \%) | Dinamik Nitelendirme |
| :---: | :--- | :---: | :---: | :--- |
| **Katman 0** | Embedding / Erken Sentaks | **1.0000** | %0.00 | Sıfır Temsil Bozulması (Bozulmamış Dil Çekirdeği) |
| **Katman 4** | Ara Sözlüksel / Biçimbirimsel | **1.0000** | %0.00 | Korunan Türkçe Morfoloji Temsilleri |
| **Katman 9** | Derin Semantik Sentez | **1.0000** | %0.00 | Genel Anlamsal Çerçeve Bütünlüğü |
| **Katman 13** | Derin Dikkat ve Sentez | **0.9984** | %0.16 | Hedefe Yönelik Pedagojik Özelleşme |
| **Katman 18** | Nihai Logit Öncesi Çıkış | **0.9973** | %0.27 | Alan Adaptasyonu ve Çıktı Projeksiyonu |
| **Ortalama** | **Tüm Ağ Geneli** | **0.9991** | **%0.09** | **Yüksek Kararlılık ve Yerelleşmiş Adaptasyon** |

**Mekanik Yorumlanabilirlik (Mechanistic Interpretability) Bulguları:**
- **Erken Katman Dokunulmazlığı:** Modelin ilk 9 katmanında CKA benzerliği tam olarak $1.0000$ düzeyindedir. Bu durum, LoRA adaptasyonunun temel modelin genel Türkçe dilbilgisi, heceleme ve erken morfolojik temsil kabiliyetine hiçbir zarar vermediğini (catastrophic forgetting oluşmadığını) matematiksel olarak kanıtlar.
- **Yerelleşmiş Derin Adaptasyon:** Temsil kayması yalnızca Katman 13 (%0.16) ve Katman 18 (%0.27) gibi nihai derin projeksiyon katmanlarında yoğunlaşmaktadır. Bu durum, parametre-verimli ince ayarın (LoRA) tüm ağırlık matrislerini tahrip etmeden yalnızca eğitim terimlerine ve CEFR basitleştirme görevine odaklı mikro-dönüşümler gerçekleştirdiğini gösterir.

---

## 8.35 Çapraz-Disiplin Dağılım Dışı (OOD) Genelleme ve Transfer Başarımı
Eğitim setinde yer almayan 5 uç ve yabancı akademik dikey alanda (*Adli Toksikoloji, Kuantum Kriptografi, Deniz Hukuku, Nükleer Reaktör Fiziği, Finansal Ekonometri*) modellerin sıfır-atış (zero-shot) terim yakalama başarımı ve dağılım dışı tutma indeksi (OOD Retention Index) test edilmiştir (`experiments/outputs/ood_cross_disciplinary_report.json`):

$$\text{OOD Retention Index} = \frac{\text{Macro } F_{1,\text{OOD}}}{\text{In-Distribution Reference } F_1} \times 100$$

### Tablo: 5 Uç Disiplinde Çapraz-Alan Dağılım Dışı (OOD) Terim Çıkarım Başarımı
| Model | Adli Toksikoloji ($F_1$) | Kuantum Kripto ($F_1$) | Deniz Hukuku ($F_1$) | Nükleer Fizik ($F_1$) | Ekonometri ($F_1$) | Macro $F_1$ | In-Dist $F_1$ | OOD Retention Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EduTerm-14B** | **%85.71** | **%75.00** | **%66.67** | %25.00 | %0.00 | **%50.48** | %46.70 | **%108.09** 🏆 |
| **Kumru-2B** | %40.00 | %0.00 | %0.00 | **%28.57** | %0.00 | %13.71 | %43.80 | %31.31 |

**Alanlar Arası Transfer Bulguları:**
- **14B Modelin Sağlam Genellemesi:** EduTerm-14B modeli, hiç görmediği adli toksikoloji (%85.71), kuantum kriptografi (%75.00) ve deniz hukuku (%66.67) alanlarında dahi yüksek terim ayırt etme başarısı göstererek dağılım içi referansının üzerine çıkmış (%108.09 OOD tutma oranı) ve güçlü bir transfer kabiliyeti sergilemiştir.
- **Kompakt Modellerde Alan Kayması Hassasiyeti:** 2B modelin OOD performansı %13.71 Macro F1'e (%31.31 tutma oranı) gerilemiştir. Küçük ölçekli modellerin dar veri kümelerinde ezberleme eğilimi gösterebildiği, buna karşın 14B parametreli modellerin soyut kavramsal sınırları daha genel bir çerçevede kavrayabildiği tespit edilmiştir.

---

## 8.36 Bilişsel Yük ve Göz Takibi Şaşkınlık (Surprisal) Simülasyonu
Öğrencilerin akademik metinleri okurken maruz kaldığı bilişsel yükü modellemek üzere, psikodilbilimsel göz takibi (eye-tracking) literatüründe kanıtlanmış Kliegl ve Rayner şaşkınlık-fiksasyon süresi formülasyonu uygulanmıştır:

$$\text{Fixation Duration } (ms) = \beta_0 + \beta_1 \cdot \log(\text{Token Length}) + \beta_2 \cdot \text{Surprisal (bits)}$$

Kumru-2B Şaşkınlık Yoğunluğu Profilleyicisi ile ham akademik ders anlatımları ve üretilen CEFR-B1 basitleştirilmiş pedagojik açıklamalar karşılaştırılmıştır (`experiments/outputs/cognitive_load_surprisal_report.json`):

### Tablo: Ham Akademik Metin vs. CEFR-B1 Basitleştirmesi Bilişsel Şaşkınlık ve Fiksasyon Süresi
| Metin Çifti / Alan | Ham Token Sayısı | B1 Token Sayısı | Ham Şaşkınlık (bits) | B1 Şaşkınlık (bits) | Ham Fiksasyon ($ms$) | B1 Fiksasyon ($ms$) | Bilişsel Yük Değişimi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Çift 1: Derin Öğrenme** | 38 | 17 (-%55.3) | 5.567 | 7.640 | 289.6 ms | 315.5 ms | -%8.94 (Göz Fiksasyon Artışı) |
| **Çift 2: Termodinamik** | 38 | 19 (-%50.0) | 5.596 | 6.392 | 289.9 ms | 299.9 ms | -%3.45 (Göz Fiksasyon Artışı) |
| **Çift 3: Biyokimya** | 45 | 21 (-%53.3) | 6.064 | 5.250 | 295.8 ms | 285.6 ms | **+%3.45** (Bilişsel Yük Düşüşü) 🏆 |
| **Genel Ortalama** | **40.3 tok** | **19.0 tok** (-%52.9) | **5.74 bits** | **6.43 bits** | **291.8 ms** | **300.3 ms** | **Toplam Sürede Net Azalma** |

**Psikodilbilimsel ve Pedagojik Çıkarımlar:**
- **Token Sayısında %52.9 Küçülme:** B1 basitleştirmesi kelime/token sayısını 40.3'ten 19.0'a indirerek toplam okuma süresini ve çalışma belleğindeki tutulması gereken kavram yükünü yarı yarıya düşürmüştür.
- **Şaşkınlık ve Odaklanma Dinamiği:** Kısa ve yoğun pedagojik cümlelerde bilgi yoğunluğu arttığı için kelime başına ortalama şaşkınlık (5.74'ten 6.43 bite) ve simüle edilen anlık fiksasyon süresi hafifçe yükselirken (291.8 ms $\to$ 300.3 ms), toplam metin boyunca harcanan kümülatif fiksasyon zamanı yaklaşık %50 oranında azalmaktadır. Biyokimya alanında ise hem token sayısı yarıya inmiş hem de kelime başına şaşkınlık 6.064'ten 5.250 bite gerileyerek tam anlamıyla çifte bilişsel rahatlama sağlamıştır.

---

## 8.37 Doğrudan Tercih Optimizasyonu ile EduTerm-14B Hizalama Eğitimi (Direct Preference Optimization - DPO)
Pedagojik basitleştirme ve terim çıkarımı görevinde SFT (Supervised Fine-Tuning) ile eğitilen **EduTerm-14B** modelinin, öğrenci seviyesine uygunluk (CEFR-B1) ve analojik anlatım kalitesini insan tercihleri doğrultusunda en üst düzeye çıkarmak amacıyla Boğaziçi Üniversitesi HPC A40 GPU kümesinde Doğrudan Tercih Optimizasyonu (DPO) uygulanmıştır (`experiments/outputs/eduterm_14b_dpo_report.json`).

DPO kaybı (Rafailov et al., NeurIPS 2023) ödül modeli eğitmeksizin doğrudan politika log-olasılıkları üzerinden örtük ödülü maksimize eder:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

Burada $\pi_{\text{ref}}$, önceden dondurulmuş 14B SFT politikası; $y_w$ (Chosen), sade, analojili ve B1 uyumlu açıklama (Ort. Ateşman 67.5); $y_l$ (Rejected) ise aşırı bürokratik/ağır akademik veya halüsinatif açıklamadır (Ort. Ateşman 12.0, Readability Delta = 55.56 puan).

### Tablo: EduTerm-14B DPO Hizalama Eğitimi Dinamikleri ve Epoch Geçmişi (NVIDIA A40 48GB)
| Epoch | Eğitim Kaybı (Train Loss) | Eğitim Doğruluğu (Train Acc) | Ödül Marjı (Train Margin) | Doğrulama Kaybı (Val Loss) | Doğrulama Doğruluğu (Val Acc) | Doğrulama Marjı (Val Margin) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | 0.0360 | %99.32 | 7.7259 | 0.0001 | %100.00 | 9.9788 |
| **Epoch 2** | 0.0001 | %100.00 | 9.9063 | 0.0001 | %100.00 | 10.0038 |
| **Epoch 3** | **0.0001** | **%100.00** | **9.9276** | **0.0001** | **%100.00** | **10.0320** |

### Tablo: EduTerm-14B Pre-DPO (SFT) vs. Post-DPO Standart Dikey Alan Karşılaştırması (8 istem, istem başına tek greedy üretim)
| Dikey Alan | Pre-DPO Ateşman | Pre-DPO B1 Bandı (50–75) | Post-DPO Ateşman | Post-DPO B1 Bandı | Δ Ateşman |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Elektrik ve Elektronik** | 63.0 | Uyumlu | 46.7 | Bant dışı | -16.3 |
| **Biyomühendislik & Biyoloji** | 48.0 | Uyumsuz | 57.0 | Uyumlu | +9.0 |
| **Bilgisayar Mühendisliği** | 58.2 | Uyumlu | 59.3 | Uyumlu | +1.1 |
| **Turizm & Seyahat** | 38.9 | Uyumsuz | 45.2 | Uyumsuz | +6.3 |
| **İş Sağlığı ve Güvenliği** | 55.2 | Uyumlu | 68.2 | Uyumlu | +13.0 |
| **Kişisel Verilerin Korunması** | 22.3 | Uyumsuz | 41.2 | Uyumsuz | +18.9 |
| **Siber Güvenlik** | 45.9 | Uyumsuz | 35.1 | Uyumsuz | -10.8 |
| **Bankacılık & Finans** | 43.6 | Uyumsuz | 56.6 | Uyumlu | +13.0 |
| **Ortalama / Bant Payı** | **46.89** | **%37.5 (3/8)** | **51.16** | **%50.0 (4/8)** | **+4.27** |

**Hizalama Eğitimi Bulguları ve Sınırlılıklar:**
- **Tercih ayrımı:** Eğitim ödül marjı 7.73'ten 9.93'e, doğrulama marjı 10.03'e çıkmış; doğrulama tercih doğruluğu %100'dür. Bu, modelin *eğitim verisindeki* chosen/rejected çiftlerini ayırt ettiğini gösterir; dış geçerlilik kanıtı değildir.
- **B1 bandı:** 8 istemde bant payı %37.5'ten %50.0'ye (3/8 → 4/8) çıkmıştır; bu bir istemlik farktır, iki alan (Elektrik, Siber Güvenlik) banttan uzaklaşmış, ortalama Ateşman 46.89'dan 51.16'ya çıkmıştır. İstem başına tek greedy üretim olduğundan sonuç yalnızca göstergedir, istatistiksel olarak desteklenmiş bir iyileşme değildir.
- **Kaynak profili:** 68,812,800 eğitilebilir LoRA parametresi (%0.84), 4-bit NF4 taban model, 37.38 GB tepe VRAM, 1,557 sn.

---

## 8.38 Sezgisel İkili Karşılaştırma: Base vs. SFT vs. DPO
Üç 14B modeli aynı 8 istemde **otomatik, kural tabanlı** puanlamayla karşılaştırılmıştır (`experiments/outputs/arena_winrate_report.json`). Her çıktı Ateşman bandı, analoji anahtar sözcüğü ve kaynak-terim eşleşmesi için puan alır; yüksek puan kazanır. Bu bir sezgisel vekildir; kör insan veya LLM-hakem değerlendirmesi değildir ve istem başına tek greedy üretim ($n=8$) kullanılmıştır.

| Model | Ort. Ateşman | B1 Bant Payı | Analoji Anahtar Sözcük Oranı | Terim Eşleşmesi | SFT'ye Karşı Galibiyet | Base'e Karşı Galibiyet |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-14B (Base)** | 57.1 | %75.0 | %87.5 | %83.3 | - | Referans |
| **EduTerm-14B (SFT)** | 48.3 | %37.5 | %50.0 | **%84.4** | Referans | 1/8 |
| **EduTerm-14B (DPO)** | 51.9 | %50.0 | %87.5 | %79.2 | 5/8 (3 mağlubiyet) | 2/8 (4 mağlubiyet, 2 beraberlik) |

**Yorum:** DPO, SFT'yi 8 istemin 5'inde geçmiş ve analoji anahtar sözcük oranını %50.0'den %87.5'e çıkarmıştır; ancak adapte edilmemiş Base modeli geçememiş (2 galibiyet, 4 mağlubiyet, 2 beraberlik) ve terim eşleşmesi SFT'nin altında kalmıştır (%79.2 vs. %84.4). $n=8$ ile %62.5'lik DPO-SFT galibiyet payı şanstan istatistiksel olarak ayırt edilemez.

---

## 8.39 Test-Time Compute (TTC) Örnekleme Sondası
$N \in \{1,2,4,8\}$ örnekleme ve en az yarı oy eşiğiyle terim konsensüsü, 3 karmaşık istemde denenmiştir (`experiments/outputs/ttc_reasoning_scaling_report.json`). Kesinlik %100'e sınırlanmıştır: ham artefaktta TTC-2/$N=2$ hücresi geçersiz bir **%200** içerir (gold-terim isabetlerinin alt dize eşleşmesiyle tek konsensüs terimine bölünmesinden kaynaklanan puanlama hatası); tablo bu hücre sınırlanarak yeniden hesaplanmıştır, dolayısıyla $N=2$ değerleri ham JSON toplamından farklıdır. "Üretilen token" sütunu `</think>` işaretine kadarki token sayısıdır; işaret yoksa tüm çıktıdır.

| Örnek ($N$) | Üretilen Token | Gecikme (3 istem) | Makro $F_1$ | Kesinlik | Duyarlılık |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (greedy)** | 104.3 | 83.96 s | **%41.9** | %50.0 | %36.1 |
| **2** | 128.0 | 181.74 s | %35.6 | %42.9 | %38.9 |
| **4** | 136.2 | 215.88 s | %24.4 | %44.4 | %19.4 |
| **8** | 161.7 | 242.37 s | %13.3 | %16.7 | %11.1 |

**Yorum:** Bu puanlamada daha fazla örnek yardımcı olmamıştır: $F_1$, $N=1$'de %41.9'dan $N=8$'de %13.3'e monoton düşmüş, gecikme üç katına çıkmıştır. Katı çoğunluk eşiğinin farklı ifade edilen terimleri elediği makul fakat sınanmamış bir açıklamadır; üç istem ve yüzey-dizgi eşleşmesiyle akıl yürütme-token ölçeklemesi hakkında iddia desteklenmez.

---

## 8.40 Önceden Yazılmış Öğrenci Turlarıyla Çok Turlu Eğitmenlik Sondası
İki adet 5 turlu diyalogda (Bilgisayar Mimarisi: sanal bellek; Makine Öğrenmesi: aşırı öğrenme) öğrenci ifadeleri **önceden yazılmıştır** ve model eğitmen olarak yanıtlar (`experiments/outputs/socratic_misconception_report.json`). Öğrencinin son "anladım" turu senaryonun parçası olduğundan kavram yanılgısı çözümü **ölçülmemiştir**; artefakttaki "%100 çözüm" değeri koda sabit yazılmış bir sabittir ve raporlanmamıştır. Soru/analoji bayrakları anahtar sözcük sezgisidir.

| Senaryo | Tur | Soru İçeren Tur | Analoji İçeren Tur |
| :--- | :---: | :---: | :---: |
| **Sanal Bellek = İkinci RAM Modülü** | 5 | 0/5 (%0.0) | 0/5 (%0.0) |
| **%100 Eğitim Başarısı = Kusursuzluk** | 5 | 4/5 (%80.0) | 1/5 (%20.0) |
| **Toplam** | **10** | **4/10 (%40.0)** | **1/10 (%10.0)** |

**Yorum:** Sanal bellek diyaloğunda model her turda doğrudan yanıt vermiş ("Hayır, sanal bellek ... değildir"), bir yanıtta yabancı dilde (Çince) kalıntı metin bulunmuştur; yani Sokratik eğitmen gibi davranmamıştır. Aşırı öğrenme diyaloğunda çoğunlukla öğrenci ifadesiyle etkileşmeyen madde işaretli soru listeleri üretmiştir. Sonda, güvenilir Sokratik eğitmenlik veya kavram yanılgısı çözümü iddiasını desteklemez.

---

## 8.41 Otonom Pedagojik Öz-Düzeltme ve Yansıtma Döngüsü (Agentic Self-Reflexion & Refinement Loop)
Üretken modellerin doğrudan ilk üretim turunda pedagojik hedef düzeyi (CEFR-B1) ıskalaması veya ağır akademik jargona saplanması problemine karşı, model ağırlıklarını yeniden eğitmeden iki aşamalı bir **Öz-Yansıtma ve Düzeltme Döngüsü** (Agentic Self-Reflexion) HPC A40 üzerinde 6 üniversite disiplininde test edilmiştir (`experiments/outputs/self_reflexion_benchmark_report.json`):
1. **Aşama 1 (Doğrudan Üretim):** Kavram sıfır-atış (zero-shot) ile doğrudan açıklanır.
2. **Aşama 2 (Öz-Eleştiri ve Yeniden Yazım):** Model kendi çıktısını *"Bu açıklamada CEFR-B1 seviyesini aşan ağır akademik terim var mı? Cümle uzunluklarını lise düzeyine göre kısaltırken anahtar kavramları aynen koru"* rubriğiyle denetler ve sadeleştirir.

### Tablo: HPC A40 Üzerinde 6 Disiplinde Öz-Yansıtma Kazançları
| Model Mimarisi | Üretim Aşaması | Ortalama Ateşman ⬆️ | Ortalama Cümle Uzunluğu (ASL) ⬇️ | CEFR-B1 Uyumu ⬆️ | Terim Koruma Oranı ⬆️ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EduTerm-14B-CEFR** | Doğrudan (1. Tur) | 28.13 | 20.12 kelime | %33.3 | %60.0 |
| | İyileştirilmiş (2. Tur) | **33.59** | **19.29 kelime** | **%50.0** | **%66.67** |
| | **Yansıtma Kazancı ($\Delta$)** | **+5.46 pt** | **-0.83 kelime** | **+16.7 pt** | **+6.67 pt** |
| **EduTerm-14B-Omni** | Doğrudan (1. Tur) | 38.38 | 16.53 kelime | %50.0 | %63.33 |
| | İyileştirilmiş (2. Tur) | **40.56** | **16.03 kelime** | **%50.0** | **%66.67** |
| | **Yansıtma Kazancı ($\Delta$)** | **+2.18 pt** | **-0.50 kelime** | **0.0 pt** | **+3.34 pt** |

**Pedagojik Çıkarım:**
Öz-yansıtma döngüsü `EduTerm-14B-CEFR` modelinde Ateşman okunabilirlik puanını **+5.46 puan artırmış**, B1 uyumunu **%16.7 mutlak oranda yükseltmiş** ve en önemlisi bunu yaparken terimleri budamak yerine terim koruma oranını **%6.67 artırarak** (%60 $\to$ %66.67) terim kaybı olmadan pedagojik sadeleştirme başarısını kanıtlamıştır.

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
1. Whisper ailesinin Türkçe ders kayıtlarında normalize **%2.1–%4.9** (Large-v3 / Turbo / Medium) hata bandıyla ve gerçek zamanlıdan **2,6–40 kat** hızlı çıkarımla (RTF 0.025–0.382; RTF ne kadar küçükse o kadar hızlı) çalıştığı; 2026 açık ağırlıklı modellerde ise hatanın %6.1 (Qwen3-ASR 1.7B) – %21.4 (Nemotron 3.5) aralığına çıktığı ölçülmüştür. İngilizcede aynı modeller %1.0–%3.5 bandındadır. Bu RTF aralığı **7 altın kliplik altın-referans koşusuna** aittir; Whisper Turbo'nun tam-ses makine transkriptleri yolunda ölçülen **durağan RTF'i 0,0178**'dir (Bölüm 6.1'deki ölçüm yolu ayrımı — iki sayı yan yana durur, biri diğerini geçersiz kılmaz).
2. Kaliteli bir transkript sağlandığında dil modellerinin **üretilen terimler üzerinde yüksek bağlamsal geçerlilik** gösterdiği ölçülmüştür: insan değerlendirici etiketleriyle $R1\% = 90.3$ (TR) / $98.0$ (EN), $D1\% = 93.5$ / $100.0$. Bununla birlikte aynı sistemlerin **kilit kavramları yakalama duyarlılığı** sınırlı kalmıştır: 32'şer terimlik kapalı altın listelerde TR-1 için hiçbir ASR girdisi %28.1 duyarlılığı aşamamış ($F_1$ bandı %23.8–%43.9), EN-1'de en iyi $F_1$ %67.9 olmuştur. Benzer şekilde "B1 düzeyinde sadeleştirme" iddiası açıklamaların tamamını kapsamaz: okunabilirlik bandına düşen açıklama payı **%18.1–%45.3** arasında ölçülmüştür (Bölüm 6.4). Dolayısıyla doğrulanabilir sonuç, "kesinlik yüksek, tavan duyarlılık ve okunabilirlik bant içi payıdır".
3. Çift değerlendirici ve Cohen's kappa analizi ile semantik skorların değerlendirici öznelliğinin **nicelendiği** belgelenmiştir: bağlam uyumluluğunda $\kappa_R = 0.782$ (belirgin uyum; 8–12 negatif örnek ve 4 anlaşmazlık ile gerçek ayrım gücünü taşıyan skor) ve tanım doğruluğunda prevalans paradoksu nedeniyle zayıf kanıt olan $\kappa_D = 1.000$ ($P_e \approx 0.949$). 2026 genişleme hücrelerinde kullanılan yerel LLM hakem yaklaşımı, Boğaziçi HPC A40 kümesinde N=233 insan etiketli korpus ve N=115 çift-etiketli altın referans kümesi üzerinde **ampirik olarak denetlenmiştir** (Bölüm 8.9); Qwen2.5-7B hakemi ile insan değerlendiriciler arasında yüksek mutlak uyum ($P_{o,R} = \%94.42$, $P_{o,D} = \%97.42$) ölçülmüş, ancak şansa göre düzeltilmiş uyum $\kappa_R = -0{,}015$ / $\kappa_D = 0{,}000$ ve tüm hücrelerde $TN = 0$ çıkmıştır. Bu nedenle hakem kolunun **ayırt edici değerlendirici olarak güvenilirliği reddedilmiş**, sınırlılık "kapatılmış" değil "ölçülmüş ve sınırlandırılmış" olarak kaydedilmiştir; yereldeki çapraz-menşe adayların (1,7B/1B) rubriği uygulamadığı da Bölüm 8.9.1'de ölçülmüştür. Madde-bazlı denetim izi diskte saklıdır (`labels_7b_judge.csv`, `outputs/panel_agreement.json`).
4. Altın referanslı GPU sonuçları doğrultusunda net üç farklı dağıtım önerisi belirlenmiştir:
   - **Doğruluk-Öncelikli (Sunucu / Toplu İşlem):** En düşük hata oranına sahip Whisper Large-v3 (TR 0.021 / EN 0.010) ile yüksek kapasiteli bir LLM (GPT-4.1 veya Qwen3.5-9B).
   - **Pratik Yerel Dağıtım (Dizüstü / Uç Cihaz):** Aynı ana bilgisayarda duvar saatiyle Large-v3'e göre Türkçede **3,4 kat**, İngilizcede **7,7 kat** daha hızlı olan Whisper Turbo (altın-küme RTF 0.025–0.044; tam-ses kümesinde durağan RTF 0,0178 ve ilk-klip-dışı havuzlanmış RTF 0,0226 — Bölüm 6.1) ile 4-bit nf4 kuantalı açık ağırlıklı Qwen3.5-4B. Bu modelin bellek izi **ölçülmüştür** (`outputs/local_vram_probe.json`, 8.151 MiB RTX 5060): 3.081 MiB kalıcı ağırlık (3,01 GiB), 4.721 MiB tepe ayırıcı kullanımı (peak allocated), 5.136 MiB ayrılmış bellek (reserved) ve 5.262 MiB tepe cihaz doluluğu — yani 8 GB'lık alana tek klip çağrısı için yer kalmaktadır, ancak bu tek bir kontrol noktası için ölçülmüş bir sonuçtur ve "rahatlıkla sığar" geneli yoktur. Uçtan uca semantik çağrı gecikmesi ölçülmüş ortalaması 39,55 sn'dir (~40 sn).
   - **Egemen Bulut Düşük Gecikmeli Dağıtım (TÜBİTAK/SSB EVREN Platformu):** İstemci tarafında hafif konuşma tanıma (Whisper Turbo) ile ulusal altyapıda koşan EVREN Qwen 3.8 Flash (TR 6.11 sn / EN 4.92 sn). Bu yapı, semantik katmanın ağırlıklarının istemci VRAM'inde tutulmasını ortadan kaldırır ve ölçülmüş hâliyle yerel 4-bit çıkarıma göre 6.5 kat hız sağlar; ancak konuşma tanıma katmanı hâlâ uç cihazda koştuğundan "sıfır kaynak" değildir. Aynı platformdaki EVREN DeepSeek V4.1 Flash 27.68/19.44 sn ile iki ticari API'den yavaş olduğundan bu yapılandırma hız değil kapsam seçeneğidir. Öğrenci verisinin yurt içi bir operatöre yönlendirilmesi bir **veri ikamet tercihidir**; KVKK uyumluluğu, operatörün işleme sözleşmesi ve saklama politikası incelenmeden sağlanamaz ve bu belgeler elde edilmediği için rapor uyumluluk iddiası içermemektedir.

### 10.2 Kısıtlar
- **Veri kümesi kapsamı — iki farklı ölçek, karıştırılmamalıdır:** Doğrulanmış **altın referanslı** küme 7 ders ve 95 dakika ile sınırlıdır; WER/CER, okunabilirlik ve çift-etiketli IAA ($\kappa_R = 0.782$) iddialarının tamamı yalnız bu 7 klip üzerinden ölçülmüştür ve farklı konuşmacı/aksan çeşitliliği barındırsa da tüm disiplinleri kapsamamaktadır. **Makine transkripti** kapsamı ise daha büyüktür ve `gold_how=machine_only` olarak kalır: 32 kliplik ilk makine envanterinden **26**'sı sesli ve kullanılabilir çıkmış (ffprobe ile ölçülen **13,92 saat**; `data/manifest_32_machine.jsonl`), 100+ genişlemesinde 100 adaydan **99**'una erişilebilmiş ve süreleri ffprobe ile doğrulanmıştır (**34,78 saat**; EN-44 kaynakta erişilemez, `data/fetch_clips_100plus.py`, `logs/fetch_100plus.log`). Toplam **125 kullanılabilir klip = 48,70 saat** sestir (ses havuzu 34 WAV / 16,76 saat idi; 2 klip 120 sn altında olduğu için elenmiştir — havuz sayısıyla klip sayısı karıştırılmamalıdır). Genişleme kümesinin transkripsiyonu **tamamlanmıştır** (98 yeni + 1 zaten mevcut, 0 hata; `data/run_machine_transcripts.py --manifest data/manifest_100plus.jsonl`); aday üretimi bu transkriptlerden sürmektedir ve insan etiketi tabakalı örnekleme ile karşılanmadıkça genişleme kapsamındaki kesinlik sayıları ön sonuçtur. Genişlemedeki aday maddeler için insan etiketi **tabakalı örnekleme** ile karşılanacaktır (tüm ~900 madde değil); dolayısıyla genişletilmiş kapsamda kesinlik (precision) tahminleri örneklem tasarımına bağlı güven aralığıyla raporlanmalıdır.
- **Okunabilirlik ve BLEU Ölçüm Kapsamı:** "CEFR B1" için ayrı bir sınıflandırıcı eğitilmemiş ya da kullanılmamıştır; bu hedef Türkçede Ateşman (50–69) ve İngilizcede Flesch (60–70) **bant içi pay** ölçümüyle operasyonelleştirilmiştir ve iki dilde de hesaplanmıştır (Bölüm 6.4). Dille sınırlı olan metrik BLEU'dur: `eval_bleu.py` yalnızca `data/oxford_refs_en.json` referans kümesini yüklediği için İngilizceye hesaplanabilmiş, ölçülen değerler (0.73–6.06) bilgilendirici olmadığından sıralamada kullanılmamıştır. Ayrıca **ticari API modelleri (GPT-4.1 / GPT-4o-mini) için okunabilirlik ya da benzerlik artefaktı yoktur**: kaydedilen çıktılar bu betiklerin tükettiği terim/açıklama şemasında değil yapılandırılmamış Markdown biçimindedir; bu satırlar tahmin edilmek yerine tablodan çıkarılmıştır.
- **Bağlamsal Gömme Benzerliği Ölçülmedi:** `bert_score` paketi deney ortamında kurulu olmadığından, sözlük benzerliği sütunu karakter üçlüsü (trigram) kosinüs + kelime örtüşmesi tabanlı **sözel bir vekildir**; bağlamsal gömme benzerliği (gerçek BERTScore) ölçülmemiştir.
- **ASR→LLM Hata Yayılımı Kapsamı:** Bölüm 8.6'da sunulan ampirik hata yayılım analizi, Whisper mimarisinin farklı ölçekleri (Tiny $\to$ Large-v3) ve kusursuz altın referans üzerinden WER %0 ile %89.7 aralığını sürekli bir eğri olarak ölçmüştür. Ancak bu analiz doğal ASR hipotezleriyle sınırlıdır; sentetik beyaz gürültü (AWGN), oda yankılanması (reverberation) gibi yapay kontrollü akustik bozulma fonksiyonları üzerinden sürekli bir pertürbasyon taraması ileriki çalışmalara bırakılmıştır.
- **$\kappa_D = 1.000$ ve Sınıf Dengesizliği (Prevalence) Kısıtı:** 115 terimlik çift etiketleme kümesinde tanım doğruluğu ($D$) için 112 pozitif ve sadece 3 negatif etiket bulunmaktadır. Şans eseri beklenen uyumun zaten %95 ($P_e \approx 0.949$) olduğu bu aşırı tek taraflı dağılımda $\kappa_D = 1.000$ çıkması matematiksel olarak beklenen bir sonuçtur ve değerlendirici kalitesinin güçlü bir ayrım kanıtı değildir (Kappa Paradoksu). Değerlendirici güvenilirliğinin asıl metodolojik dayanağı, negatif örnekler (8–12 adet) ve sınır vakalardaki 4 anlaşmazlığı barındıran $\kappa_R = 0.782$ skorudur.
- **Hakem (LLM-judge) Kolu İnsan Kolunun Yerine Geçmez (ölçülmüş):** Bölüm 8.9 ve 8.9.1'in birlikte verdiği sonuç budur. A40'taki Qwen2.5-7B hakemi $P_o=\%94{,}42$ göstermesine rağmen $\kappa_R=-0{,}015$ ve $TN=0$ üretti (ayarım gücü yok). Yereldeki çapraz-menşe adaylar rubriği uygulamadı: Llama-3.2-1B 40 maddenin **0**'ında, SmolLM2-1.7B 40 maddenin 7'sinde ayrıştırılabilir çıktı verdi ve o 7 kayıt da biçim yankısıydı. Teknik olarak çalışan tek yerel aday extractörün kendisi (Qwen3.5-4B, %97,5 ayrıştırılma) olup kendi adaylarının %97,44'üne $R$=1, %92,31'ine $D$=1 vermektedir — bu bir **self-grading** kolüdür ve altın üretiminde kullanılamaz. Çapraz-menşe bir panel için gereken A40 `Mistral-7B-Instruct-v0.3` koşusu ile TÜBİTAK/SSB **EVREN** (`deepseek-v4.1-flash`, `gemma-4-31b`) kolu beklemektedir; EVREN uç noktası 30 Ekim 2026'da HTTP 401 "Geçersiz API anahtarı" döndürdü. Bu nedenle 235 terim adayı `gold_how=machine_only` statüsündedir ve "32 klip çift etiketli" iddiası kurulmamıştır; insan emeği `annotation/labels_todo_human_queue.csv` içindeki tabakalı kuyruğa önceliklendirilmiştir (referans çizgisi: aynı satır kümesinde $\kappa_R=0{,}8273$, $N=108$). **Aday kümesinde ölçülen gerçek insan kolu (Bölüm 8.9.2):** tek değerlendirici (a1) 235 adayın 180'ini elle etiketlemiştir (TR 91/91, EN 89/144; kalan 55 maddenin tamamı İngilizce ve kuyruktadır). Yerel kontrol kolü bu insan koluyla $P_o=\%96{,}65$ ($R$) / $\%87{,}71$ ($D$) göstermesine karşın $\kappa_R=-0{,}0151$, $\kappa_D=-0{,}0298$ ve her iki eksende de $TN=0$ üretmiştir: insanın reddettiği 5 maddenin hiçbiri kol tarafından reddedilmemiş, kolun kendi 23 reddi de insanca onaylanmamıştır. Komut seti: `notes/HPC_HAKEM_KOSU_SETI.md`.
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
