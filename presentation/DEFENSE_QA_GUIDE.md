# EduTerm AI Bitirme Projesi Savunma Rehberi (Defense Q&A Guide)

**Proje:** Çok Dilli Eğitim İçin Konuşma Tanıma (ASR) ve Büyük Dil Modeli (LLM) Tabanlı Terminoloji Çıkarma ve Pedagojik Açıklama Sistemlerinin İki Katmanlı Değerlendirilmesi  
**Yazarlar:** Çakıl Yılmaz (Koç Üniv.); Berat Enes Dadak, Efe Batur Giritli, Yekta Said Can (Tatilsepeti Ar-Ge)  
**Kurumlar:** Koç Üniversitesi & Tatilsepeti Ar-Ge Merkezi  
**Tarih:** Eylül 2026  

---

## 1. Savunma Zaman Yönetimi Stratejisi (20 Dakika Sunum + 10 Dakika Soru-Cevap)

| Bölüm | Slaytlar | Hedef Süre | Ana Vurgu |
|---|---|---|---|
| **Giriş & Motivasyon** | 1 – 3 | 3 dk | Bloom 2-Sigma, Bilişsel Yük, Araştırma Soruları (RQ1-3) |
| **Sistem Mimarisi & Veri Kümesi** | 4 – 5 | 3 dk | İki katmanlı bağımsız boru hattı, 7 ders (95 dk), Altın Referans kökeni |
| **Katman 1: ASR Değerlendirmesi** | 6 – 7 | 4 dk | 2026 modelleri, Macro/Micro WER, RTF, Turbo vs Large-v3 dengesi |
| **Katman 2: LLM & Pedagojik Semantik** | 8 – 9 | 3 dk | Prompt stratejileri (RAG F1 %76.9), CEFR B1 okunabilirlik (Ateşman/Flesch) |
| **Güvenilirlik: IAA & Hata Yayılımı** | 10 – 11 | 3.5 dk | Cohen's Kappa, Prevalans Paradoksu, Sentetik Hata Eğrisi |
| **EduTerm AI Studio & Test Paketi** | 12 – 13 | 2.5 dk | Canlı PoC mimarisi, Uyarlanabilir öğrenci, Sınav motoru, 213 test |
| **Dağıtım Önerileri, Kısıtlar & Sonuç** | 14 – 16 | 2 dk | Doğruluk vs Hız matrisi, 7 Guardrail, Gelecek çalışmalar |

---

## 2. Jüriye Karşı 7 Altın İlke (Scientific Guardrails)

Sunum ve soru-cevap sırasında bu 7 prensip asla ihlal edilmemeli, tam tersine **savunmacı tarafından proaktif olarak** dile getirilmelidir:

1. **Guardrail 1 (Altın Referans Kökeni - Gold Provenance):** Referanslar sıfırdan çift-kör transkripsiyon değildir. TR-1, TR-2, EN-2, EN-3'te otomatik altyazı ve ASR taslakları insan denetimiyle düzeltilmiştir (post-editing). Whisper stiline hafif bir biçimsel yakınlık olabilir.
2. **Guardrail 2 (Prevalans Paradoksu - $\kappa_D = 1.000$):** Tanım doğruluğunda 112 pozitif / 3 negatif vardır. Şans eseri beklenen uyum $P_e \approx 0.95$'tir. Mükemmel kappa yüksek sınıf dengesizliğinin matematiksel sonucudur. Güvenilirliğin gerçek göstergesi sınır vakalardaki $\kappa_R = 0.782$'dir.
3. **Guardrail 3 (Semantik Recall Eksikliği):** Semantik metriklerde kapalı sözlük yoktur. $R1\%$ bir kesinlik (precision) metriğidir. GPT-4.1'in 105 terim çıkarması bir çıkarım hacmi/kapsam göstergesidir, kanıtlanmış tam recall değildir.
4. **Guardrail 4 (Örneklem Büyüklüğü ve Dil İddiaları):** Kıyaslama $n=7$ video (3 TR, 4 EN) üzerinden alınan ortalamalardır. "Türkçe ASR için İngilizceden daha zordur" gibi genel mimari/linguistik iddialardan kaçınılmalıdır.
5. **Guardrail 5 (Hata Yayılım Eğrisi):** Ablasyon tablosunda LLM'e yalnızca düşük WER'li (%1-5) Whisper çıktıları verilmiştir. Bu çalışma "düşük WER rejiminde stabilite" ölçümüdür. Geniş hata yayılımı ise kontrollü sentetik gürültü deneyiyle gösterilmiştir.
6. **Guardrail 6 (Web GUI Çerçeveleme):** Geliştirilen FastAPI arayüzü canlıya alınmış dağıtık bir ürün değil, uçtan uca boru hattını sergileyen yerel bir Ar-Ge Konsept Kanıtlama (PoC) demo stüdyosudur.
7. **Guardrail 7 (Dürüst Boş Hücreler):** GPU altın referansında yeniden koşturulmayan modeller (Nemotron 3.5, Wav2Vec 2.0, SpeechBrain) uydurma verilerle değil, dürüstçe `---` olarak bırakılmıştır.

---

## 3. Sıkça Sorulan 15 Zor Jüri Sorusu ve Model Yanıtlar

### Soru 1: $\kappa_D = 1.000$ çıkmış. İki değerlendiricinin hiç hata yapmadan yüzde yüz uyumlu olması gerçekçi mi?
> **Model Yanıt:**  
> "Hocam çok haklısınız; biz de raporumuzda ve makalemizde bunu özellikle **Prevalans Paradoksu (Prevalence Paradox)** olarak ele aldık. 115 terimlik çift etiketleme kümesinde, dil modelinin ürettiği tanımların 112'si her iki değerlendirici tarafından da doğru (1), yalnızca 3'ü yanlış (0) bulunmuştur. Sınıf aşırı dengesiz olduğundan, rastlantısal beklenen uyum $P_e$ zaten %94.9'dur. Dolayısıyla $\kappa_D = 1.000$ çıkması insanüstü bir başarıdan ziyade formülün uç marjinal dağılımından kaynaklanmaktadır. Değerlendirici güvenilirliğini kanıtlayan asıl ayırt edici metriğimiz, 4 uyuşmazlığın ve negatif örneklerin bulunduğu bağlam uyumluluğundaki **$\kappa_R = 0.782$ (Belirgin Uyum)** skorudur."

---

### Soru 2: Altın standart transkriptleri nasıl oluşturdunuz? Whisper'a karşı yanlı (biased) olabilir mi?
> **Model Yanıt:**  
> "Evet, bu önemli bir metodolojik kısıttır ve makalemizin 4. Bölümünde bunu açıkça belirttik. 7 dersimizin tamamı sıfırdan çift-kör insan transkripsiyonu değildir. 4 videoda (TR-1, TR-2, EN-2, EN-3) YouTube otomatik altyazıları veya ASR taslakları insan uzmanlar tarafından dinlenerek terimler, teknik kısaltmalar ve noktalama hataları düzeltilmiştir. Bu işlem insan emeği maliyetini düşürmekle birlikte, referans metinleri Whisper veya altyazı segmentasyon stiline hafifçe yakınlaştırabilir. Bu sebeple elde ettiğimiz %1-2'lik çok düşük WER skorlarını mutlak bir tavan skordan ziyade post-edit referansına karşı göreli bir kıyaslama olarak sunmaktayız."

---

### Soru 3: Tablolarda neden Wav2Vec 2.0, SpeechBrain ve Nemotron 3.5'in bazı hücreleri boş (---)?
> **Model Yanıt:**  
> "Akademik dürüstlük prensibimiz gereği koşturmadığımız hiçbir testi tahmini olarak doldurmadık. Wav2Vec 2.0 ve SpeechBrain modelleri Türkçe desteği sunmayan eski nesil modellerdir; CPU pilot çalışmasında yer almış ancak RTX 5060 GPU altın kümesinde çalıştırılmamıştır. NVIDIA Nemotron 3.5 ASR ise protokol havuzuna dahil edilmiş ancak geliştirme döneminde resmi TR ağırlıkları GPU ortamında tam altın yeniden puanlamaya tabi tutulmamıştır. Makalede ve raporumuzda spekülatif sayılar yerine dürüstçe `---` işaretini korumayı tercih ettik."

---

### Soru 4: LLM terim çıkarımında Recall (Duyarlılık) neden raporlanmadı? R1% nedir?
> **Model Yanıt:**  
> "Gerçek bir Duyarlılık (Recall) hesaplayabilmek için bir eğitim videosunda geçebilecek tüm potansiyel terimlerin eksiksiz bir 'kapalı altın sözlüğüne' (closed gold vocabulary) sahip olmamız gerekir. Oysa eğitim konuşmalarında terimsellik sürekli bir spektrumdur. Bu nedenle çalışmamızda $R1\%$ metriği, dil modelinin ürettiği terimlerin video bağlamında gerçekten teknik bir kavram olup olmadığını ölçen bir **Kesinlik (Precision)** metriğidir. Modelin terim yakalama kapasitesini ise GPT-4.1 (105 terim) ve GPT-4o-mini (56 terim) gibi modellerin çıkardığı toplam terim hacmi ve TR-1/EN-1 için oluşturduğumuz 32 terimlik alt kümedeki kapalı F1 analiziyle dolaylı olarak gösterdik."

---

### Soru 5: "Türkçe ASR modelleri için İngilizceden zordur" sonucuna varabilir miyiz?
> **Model Yanıt:**  
> "Böyle genel bir dilbilimsel veya mimari iddiada bulunmaktan özenle kaçınıyoruz. Veri kümemiz toplam 7 videodan (3 Türkçe, 4 İngilizce) oluşmaktadır. Türkçe videolardaki konuşmacı temposu, oda yankısı, mikrofon kalitesi veya YouTube ses sıkıştırması İngilizce profesyonel derslere (Grow with Google, Efficient Engineer) göre farklılık göstermektedir. Dolayısıyla tablodaki WER farkı dilin morfolojik yapısından kaynaklanabileceği gibi tamamen kayıt akustiğinden de kaynaklanmış olabilir. İstatistiksel anlamlılık testi ve güven aralıkları için 30+ saatlik geniş bir korpus gerekmektedir."

---

### Soru 6: ASR hatası arttıkça LLM'in terim çıkarma başarısı nasıl etkileniyor? Uçtan uca hata yayılımı ölçtünüz mü?
> **Model Yanıt:**  
> "İlk ablasyon çalışmamızda sabit bir LLM'e (Qwen3.5-4B) yalnızca Whisper Medium, Large-v3 ve Turbo transkriptleri verilmiştir. Bu modellerin tümü %1-5 gibi çok düşük WER rejiminde çalıştığı için LLM terim çıkarımında belirgin bir bozulma görülmemiştir (R1 %89-98 arasında sabit kalmıştır). Bu nedenle makalemizde bunu 'düşük WER rejiminde semantik kararlılık' olarak tanımladık. Gerçek hata yayılımını (error propagation) modellemek için ise sisteme %0'dan %50'ye kadar kontrollü sentetik fonetik gürültü enjekte ettiğimiz bir deney tasarladık. Bu deneyde WER %20'yi aştığında F1 skorunun %70'ten %40'ın altına dramatik şekilde düştüğünü gösterdik."

---

### Soru 7: Çıkarılan açıklamaların CEFR B1 düzeyinde olduğunu nasıl kanıtladınız?
> **Model Yanıt:**  
> "Açıklamaların pedagojik seviyesini iki bağımsız formülle ölçtük: Türkçe çıktılar için hece ve cümle uzunluğunu esas alan **Ateşman Okunabilirlik İndeksi** (elde edilen skor: 58-61, 'Orta Güçlükte / B1'), İngilizce çıktılar için ise uluslararası standart olan **Flesch Reading Ease** formülü (elde edilen skor: 64-66, 'Standard / B1'). Ayrıca terimlerin sözlük tanımlarına anlamsal yakınlığını denetlemek için TDK ve Oxford referanslarına karşı BERTScore hesapladık ve tüm modellerde 0.82'nin üzerinde anlamsal örtüşme yakaladık."

---

### Soru 8: Prompt stratejilerinde RAG neden Chain-of-Thought (CoT)'tan daha başarılı oldu?
> **Model Yanıt:**  
> "Zero-Shot taban çizgisi F1'de %47.6'da kalırken, CoT %60.9'a, Sözlük Destekli RAG (Glossary-based RAG) ise %76.9'a ulaşmıştır. CoT düşünme adımları ekleyerek muhakemeyi artırsa da gecikmeyi 6.85 saniyeye çıkarmakta ve bazen aşırı genel kelimeleri terim sanabilmektedir. RAG yaklaşımında ise alana özgü kavram çekirdeği isteme bağlam olarak verildiğinde model hem halüsinasyonu engellemekte hem de gereksiz terim kaçırmamaktadır. Üstelik 4.15 saniye gecikmeyle CoT'a göre çok daha dengelidir."

---

### Soru 9: Neden Whisper Large-v3 yerine bazı durumlarda Whisper Large-v3 Turbo öneriyorsunuz?
> **Model Yanıt:**  
> "Whisper Large-v3 mutlak doğrulukta liderdir (TR WER %2.1, EN WER %1.0). Ancak kod çözücüsü 32 katmandır ve RTF değeri 0.15-0.21 düzeyindedir. Whisper Large-v3 Turbo ise 4 katmanlı budanmış kod çözücüsü sayesinde RTX 5060 üzerinde 0.025-0.044 RTF sunmaktadır; yani sesi gerçek zamandan 25 ila 40 kat daha hızlı işlemektedir! Üstelik WER kaybı Türkçede yalnızca %1.6, İngilizcede ise neredeyse sıfırdır (%1.1). Canlı sınıf altyazılama veya etkileşimli eğitim arayüzlerinde Turbo modeli devasa bir donanım maliyeti ve gecikme avantajı sağlamaktadır."

---

### Soru 10: EduTerm AI Studio'nun mimarisi nedir? Üretim ortamına hazır mı?
> **Model Yanıt:**  
> "EduTerm AI Studio; Python 3.12, FastAPI, yt-dlp, ffmpeg ve Vanilla HTML/CSS/JS ile inşa edilmiş bir yerel araştırma stüdyosudur. İçerisinde dinamik DAG görselleştirici, aralıklı tekrar (spaced-repetition) algoritmalı uyarlanabilir öğrenci profili ve otomatik çoktan seçmeli sınav motoru barındırmaktadır. Ancak savunmada açıkça belirttiğimiz üzere, bu sistem şu anda bir **yerel Konsept Kanıtlama (PoC) demo aracıdır**. İş kuyruğu bellek içi sözlükte (in-memory dict) tutulmakta, indirmeler geçici WAV dosyaları olarak saklanmaktadır. Üretime geçiş için Docker, Redis iş kuyruğu ve PostgreSQL entegrasyonu gelecek çalışma olarak planlanmıştır."

---

### Soru 11: Sistemin birim test kapsamı ve yazılım mühendisliği güvenilirliği ne durumdadır?
> **Model Yanıt:**  
> "Projemiz yalnızca teorik bir araştırma değil, titiz bir yazılım mühendisliği çalışmasıdır. `experiments/tests/` dizininde toplam **213 birim test** bulunmaktadır ve test paketinin geçiş oranı %100'dür. Testler; GPU gerektirmeyen mocked API çağrıları, port çakışmalarında otomatik dinamik port bulma (port hunting failover), dış ağ bağlamalarında güvenlik belirteci (auth token) doğrulama, aralıklı tekrar ustalık formülleri ve sınav soru üretim mantığını kapsamaktadır."

---

### Soru 12: LLM-as-a-Judge doğrulamasında Cohen's Kappa neden PABAK'tan düşüktür?
> **Model Yanıt:**  
> "233 çift üzerinde yaptığımız LLM-as-a-Judge testinde, yapay zekâ değerlendirici ile insan etiketleyici arasında başlangıç seviyesine uygunluk (D) için ham kappa 0.490 iken PABAK 0.948 çıkmıştır. Gözlemlenen gerçek uyum %97.42'dir. Kappa'nın düşük görünmesi, veri kümesindeki pozitif örnek oranının aşırı yüksek olmasından (%97.4 true) kaynaklanan klasik marjinal asimetridir. Prevalans ve Asimetri Düzeltmeli Kappa (PABAK), bu önyargıyı nötrleyerek gerçek uzlaşmanın %94.8 olduğunu kanıtlamaktadır."

---

### Soru 13: Ticari modeller (GPT-4.1) ile açık ağırlıklı modeller (Qwen3.5) arasındaki fark nedir?
> **Model Yanıt:**  
> "GPT-4.1, pilot çalışmamızda 105 terim çıkararak en geniş terminolojik kapsama ulaşmıştır ve 'tavan başarım' (ceiling performance) sınırımızı temsil eder. Ancak eğitim kurumlarının veri gizliliği, çevrimdışı çalışma gereksinimi ve API maliyetleri göz önüne alındığında açık ağırlıklı Qwen3.5-4B modeli yerel RTX 5060 GPU'da 4-bit nicemleme ile sıfır API maliyetiyle saniyeler içinde çalışabilmekte ve %90'ın üzerinde bağlam uyumluluğu sağlamaktadır."

---

### Soru 14: Silero VAD tabanlı parçalama ile 30 saniyelik sabit pencereleme arasında ne fark gördünüz?
> **Model Yanıt:**  
> "Whisper doğal olarak 30 saniyelik pencerelerle eğitilmiştir. Ancak eğitim konuşmalarında hoca cümlenin ortasında durakladığında veya slayt değiştirirken sessizlik olduğunda sabit 30 saniye pencereleme, kelimeleri ortadan bölebilmekte ve halüsinasyon riskini artırmaktadır. Silero VAD (Ses Aktivite Algılama) ile konuşmanın doğal duraklama noktalarından (enerji vadileri) dinamik segmentasyon yapıldığında transkript sınırları çok daha temiz olmakta ve LLM'e gönderilen bağlam bütünlüğü korunmaktadır."

---

### Soru 15: Bu projeyi bir eğitim kurumuna (örneğin Koç Üniversitesi'ne) entegre etmek istesek ilk adım ne olmalıdır?
> **Model Yanıt:**  
> "İlk adım, sistemin LTI (Learning Tools Interoperability) standardı üzerinden Moodle, Canvas veya Blackboard gibi mevcut Öğrenme Yönetim Sistemlerine (LMS) bir eklenti olarak bağlanmasıdır. Ders kaydı bittiğinde ASR katmanı arka planda Whisper Turbo ile transkripti çıkaracak, Qwen3.5 terminoloji kartlarını ve sınav sorularını üretecek ve öğrencinin video izleme ekranına senkronize terim sözlüğü olarak yerleştirecektir."
