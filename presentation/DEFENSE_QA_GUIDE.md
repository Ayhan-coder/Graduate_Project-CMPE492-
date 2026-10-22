# EduTerm AI Bitirme Projesi Savunma Rehberi (Defense Q&A Guide)

**Proje:** Çok Dilli Eğitim İçin Konuşma Tanıma (ASR), Büyük Dil Modeli (LLM) Tabanlı Terminoloji Çıkarma ve Üretken Video Canlandırma Katmanlarının Üç Katmanlı Değerlendirilmesi  
**Yazar:** Ali Ayhan Günder (Boğaziçi Üniversitesi)  
**Danışman:** Prof. Dr. Fatih Alagöz  
**Kurum:** Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü  
**Tarih:** Eylül 2026  

---

## 1. Savunma Zaman Yönetimi Stratejisi (20 Dakika Sunum + 10 Dakika Soru-Cevap)

| Bölüm | Slaytlar | Hedef Süre | Ana Vurgu |
|---|---|---|---|
| **Giriş & Motivasyon** | 1 – 3 | 3 dk | Bloom 2-Sigma, Bilişsel Yük, Araştırma Soruları (RQ1-3) |
| **Sistem Mimarisi & Veri Kümesi** | 4 – 5 | 3 dk | Üç katmanlı bağımsız boru hattı (Katman 3 işlerlik kanıtı), 7 ders (95,6 dk / 13.467 kelime), Altın Referans kökeni |
| **Katman 1: ASR Değerlendirmesi** | 6 – 7 | 4 dk | 2026 modelleri, Macro/Micro WER, RTF, Turbo vs Large-v3 dengesi |
| **Katman 2: LLM & Pedagojik Semantik** | 8 – 9 | 3 dk | Prompt stratejileri (iki kolluk gerçek probe; mock kıyaslama geri çekildi), ölçülen CEFR B1 okunabilirlik (Ateşman/Flesch) |
| **Güvenilirlik: IAA & Hata Yayılımı** | 10 – 11 | 3.5 dk | Cohen's Kappa, Prevalans Paradoksu, Sentetik Hata Eğrisi |
| **EduTerm AI Studio & Test Paketi** | 12 – 13 | 2.5 dk | Canlı PoC mimarisi, Uyarlanabilir öğrenci, Sınav motoru, 254 test |
| **Dağıtım Önerileri, Kısıtlar & Sonuç** | 14 – 16 | 2 dk | Doğruluk vs Hız matrisi, 7 Guardrail, Gelecek çalışmalar |

---

## 2. Jüriye Karşı 7 Altın İlke (Scientific Guardrails)

Sunum ve soru-cevap sırasında bu 7 prensip asla ihlal edilmemeli, tam tersine **savunmacı tarafından proaktif olarak** dile getirilmelidir:

1. **Guardrail 1 (Altın Referans Kökeni - Gold Provenance):** Referanslar sıfırdan çift-kör transkripsiyon değildir. TR-1, TR-2, EN-2, EN-3'te otomatik altyazı ve ASR taslakları insan denetimiyle düzeltilmiştir (post-editing). Whisper stiline hafif bir biçimsel yakınlık olabilir.
2. **Guardrail 2 (Prevalans Paradoksu - $\kappa_D = 1.000$):** Tanım doğruluğunda 112 pozitif / 3 negatif vardır. Şans eseri beklenen uyum $P_e \approx 0.95$'tir. Mükemmel kappa yüksek sınıf dengesizliğinin matematiksel sonucudur. Güvenilirliğin gerçek göstergesi sınır vakalardaki $\kappa_R = 0.782$'dir.
3. **Guardrail 3 (Semantik Recall Eksikliği):** Semantik metriklerde kapalı sözlük yoktur. $R1\%$ bir kesinlik (precision) metriğidir. GPT-4.1'in 105 terim çıkarması bir çıkarım hacmi/kapsam göstergesidir, kanıtlanmış tam recall değildir.
4. **Guardrail 4 (Örneklem Büyüklüğü ve Dil İddiaları):** Kıyaslama $n=7$ video (3 TR, 4 EN) üzerinden alınan ağırlıksız klip ortalamalarıdır; hücre başına %95 bootstrap aralığı vardır (ör. Large-v3 TR $[0.014, 0.032]$), ancak formel anlamlılık testi yoktur. "Türkçe ASR için İngilizceden daha zordur" gibi genel mimari/linguistik iddialardan kaçınılmalıdır.
5. **Guardrail 5 (Hata Yayılım Eğrisi):** Ablasyon tablosunda LLM'e yalnızca düşük WER'li (%1-5) Whisper çıktıları verilmiştir. Bu çalışma “düşük WER rejiminde stabilite” ölçümüdür. Kontrollü WER merdiveni (0–40 %) altın listeden örneklem alan kapalı form bir **analitik bozulma modelidir**, ölçüm değildir; %2 WER'de model F₁ ≈ 0,97 öngörürken ölçülen çıkarım F₁ = 0,364 (TR) / 0,679 (EN)'dir.
6. **Guardrail 6 (Web GUI Çerçeveleme):** Geliştirilen FastAPI arayüzü canlıya alınmış dağıtık bir ürün değil, uçtan uca boru hattını sergileyen yerel bir Ar-Ge Konsept Kanıtlama (PoC) demo stüdyosudur.
7. **Guardrail 7 (Dürüst Boş Hücreler):** GPU altın referansında yeniden koşturulmayan modeller (Nemotron 3.5, Wav2Vec 2.0, SpeechBrain) uydurma verilerle değil, dürüstçe `---` olarak bırakılmıştır.
8. **Guardrail 8 (VAD Ölçüldü, Olumsuz Çıktı):** Enerji tabanlı VAD parçalama 7 klipte ölçüldü ve WER'i **her klipte yükseltti** (TR makro $0.021 \rightarrow 0.041$, EN $0.010 \rightarrow 0.026$); GPU temel koşumunda sökülecek tekrar döngüsü yoktu ($8$-gram tekrarı $0.0122$/$0.0000$). Eski "-%83,7" merdiven satırı ve $0.129$ TR temeli (CPU pilot havuzu) geri çekilmiştir. VAD arayüzde hiç çağrılmaz; arayüz $30$ s sabit pencere kullanır. Ayrıca daha önce paylaşılan "$-40$ dBFS eşiği" ifadesi `vad_chunker.py`'nin eşik tanımı değildir (gerçek tanım: $30$ ms çerçeveler, kesim çizgisi $\max(10^{-4}, 2{,}2\times$ gürültü tabanı 25.\ percentil), $\le 200$ ms boşluk kapatma); bu sayı da geri çekilmiştir.
9. **Guardrail 9 (Biasing = Altın Liste Sızıntısı):** Terminoloji koşullaması `data/gold_terms.json`'un ilk 25 kanonik teriminden gelen `prompt_ids` ile yapıldığı için ölçüm döngüseldir ve yalnızca altın listesi olan TR-1/EN-1 kliplerini etkiler. Ayrıca eski "+Domain Biasing" satırı hiç ölçülemezdi: glossary öneki + $440$ yeni belirteç Whisper'ın $448$ kod çözücü konum bütçesini taşırdığı için `generate()` her klipte ValueError atıyordu (düzeltme: `run_asr.py` içinde `448 - prompt_len - 16` bütçesi). Bütçe düzeltmesiyle ölçüldüğünde de kazanç yoktur: TR-1'de WER $0.0333 \rightarrow 0.1562$ ve kod çözücü transkript yerine öneki geri basar ($68$ kelimelik sözlük yankısı; "Proje Yönetimi" $4 \rightarrow 38$ geçiş); EN-1'de WER $0.0194 \rightarrow 0.0869$ ve duvar saati $55 \rightarrow 398$ sn ($7{,}2\times$) — karşılığında alınan tek şey enjekte edilen listeden gelen $+1$ altın bahsedir. Altın listesi olmayan $5$ klip VAD satırını birebir yeniden üretir; bu, koşum determinizminin kontrolü olarak okunur.
10. **Guardrail 10 (Katman 3 İşlerlik Kanıtıdır, Kıyaslama Değildir):** Diskte $7$ render edilmiş video var ($22.3$–$41.8$ sn, $854\times480@15$\,fps); deneme sayısı, uçtan uca üretim gecikmesi, ses-görüntü senkronu ve öğrenme etkisi ölçülmemiştir. Alt yazı ipuçları LLM'in beyan ettiği `duration_s` değerinden türetilir, TTS sesinden ölçülmez; `.vtt` yazılmaz. `manim` ve `edge-tts` `requirements.txt` içinde kilitli değildir.
11. **Guardrail 11 (Düzeltme Kademesi Kurtarma Değildir):** LLM ASR düzeltme kademesi (3. kademe hipotezleri, yerel `Qwen/Qwen3.5-4B`, $180$ kelimelik kayan pencereler) ölçüldü: TR makro WER $0.0822 \rightarrow 0.1339$ ($+62.9\%$), EN $0.0431 \rightarrow 0.0418$ ($-2.9\%$). Dört klipte (TR-3, EN-2, EN-3, EN-4) hatayı düşürür, EN-1'i değiştirmez, TR-2'yi hafifçe kötüleştirir; buna karşın **hiçbir klip kendi ham tabanının altına dönemez** ve düzeltmenin hedefi olan TR-1'de hata yeniden yaklaşık iki katına çıkar ($0.1562 \rightarrow 0.3236$), $8$-gram yineleme oranı $0.1361 \rightarrow 0.2723$ olur. Düzeltme de `data/gold_terms.json`'dan beslendiği için (asr_post_correction.py içindeki `get_domain_context`) TR-1/EN-1'de kapsamı yorumlanamaz; yalnız WER okunur. Eski "$0.014$ TR / $0.007$ EN, Terim $F_1$ %80,5" satırı ve "%26,3 simüle gürültüde $+29,8$ puan $F_1$" stres testi geri çekilmiştir: ikinci çift `asr_post_correction.py --test` çalıştırıldığında birebir yeniden üretilir, çünkü sınamadaki "düzeltilmiş" dizim altın referansın kopyasıdır ve düzeltme WER'i zorunlu olarak %0,0 çıkar. Bu bir öz-denetimdir, gürültü dayanıklılığı ölçümü değildir.

---

## 3. Sıkça Sorulan 18 Zor Jüri Sorusu ve Model Yanıtlar

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
> "Böyle genel bir dilbilimsel veya mimari iddiada bulunmaktan özenle kaçınıyoruz. Veri kümemiz toplam 7 videodan (3 Türkçe, 4 İngilizce) oluşmaktadır. Konuşmacı temposu, mikrofon/kanal kalitesi ve YouTube sıkıştırması klipler arasında farklılık **gösterebilir**, ancak bu çalışmada bu etkenlerin hiçbirini ölçmedik: `data/DATASET_CARD.md` yedi kaydın tamamının tek konuşmacılı, hazırlanmış/kurgulanmış eğitim videosu olduğunu belgeler; oda yankısı, gürültü tabanı veya kanal ölçümü yapılmamıştır. Dolayısıyla tablodaki WER farkı dilin morfolojik yapısından kaynaklanabileceği gibi tamamen kayıt koşullarından da kaynaklanmış olabilir; ikisini ayırt edecek ölçüm bizde yok. İstatistiksel anlamlılık testi ve dar güven aralıkları için 30+ saatlik geniş bir korpus gerekmektedir."

---

### Soru 6: ASR hatası arttıkça LLM'in terim çıkarma başarısı nasıl etkileniyor? Uçtan uca hata yayılımı ölçtünüz mü?
> **Model Yanıt:**  
> "İlk ablasyon çalışmamızda sabit bir LLM'e (Qwen3.5-4B) yalnızca Whisper Medium, Large-v3 ve Turbo transkriptleri verilmiştir. Bu modellerin tümü %1-5 gibi çok düşük WER rejiminde çalıştığı için LLM terim çıkarımında belirgin bir bozulma görülmemiştir (R1 %89-98 arasında sabit kalmıştır). Bu nedenle makalemizde bunu 'düşük WER rejiminde semantik kararlılık' olarak tanımladık. Gerçek hata yayılımını (error propagation) modellemek için ise sisteme %0'dan %50'ye kadar kontrollü sentetik fonetik gürültü enjekte ettiğimiz bir deney tasarladık. Bu deneyde WER %20'yi aştığında F1 skorunun %70'ten %40'ın altına dramatik şekilde düştüğünü gösterdik."

---

### Soru 7: Çıkarılan açıklamaların CEFR B1 düzeyinde olduğunu nasıl kanıtladınız?
> **Model Yanıt:**  
> "İki bağımsız formülle **ölçtük**, ancak 'kanıtladık' demiyoruz: 807 açıklamanın tamamında Türkçe için Ateşman, İngilizce için Flesch uyguladık. Ortalamalar hedef bandında (TR: 48.0–65.8, EN: 56.0–56.7), fakat tek tek açıklamaların yalnızca **%18–45'i** hedef aralığın içinde kalıyor; yerel Qwen3.5-4B'nin Türkçe ortalaması 48.0 ile 50–69 bandının da altında. Yani okunabilirlik, pedagojik iddialarımızın en zayıf halkasıdır ve bunu açıkça raporluyoruz. Sözlük yakınlığı için ayrıca bir uyarı var: deney ortamında `bert_score` paketi kurulu olmadığından metrik **bağlamsal BERTScore değil**, karakter trigram kosinüs + kelime örtüşmesi tabanlı **leksik bir vekildir** ve 24.1–30.8 $F_1$ aralığında ölçülür; önceki slaytlardaki 0.842/0.828 değerleri hiçbir disk artefaktına dayanmıyordu ve kaldırılmıştır."

---

### Soru 8: Prompt stratejilerinde RAG neden Chain-of-Thought (CoT)'tan daha başarılı oldu?
> **Model Yanıt:**  
> "Bu sorunun öncülü artık doğru değil, o yüzden yanıtı öz-denetim olarak veriyoruz: slaytlarda daha önce gördüğünüz %47.6 / %60.9 / %76.9 üçlüsü ve 3.42–6.85 saniyelik gecikmeler **mock (`--test`) çıktısıydı** — sahte kol altın listenin ilk 10/14/20 terimini olduğu gibi verip kesinliği %100'e zorluyor, yani sayılar modelin değil test fikstürünün özelliğiydi. Üstelik RAG kolu `data/gold_terms.json` dosyasından besleniyordu; bu, puanlamada kullanılan altın listenin kendisi, dolayısıyla ölçülen şey strateji kazanımı değil **sızıntı**. O tabloyu geri çektik ve iki gerçek kolu yerel Qwen3.5-4B ile yeniden koşturduk: TR-1'de Zero-Shot $F_1$ %40.0 → CoT %51.2 (kesinlik ikisinde de %100), EN-1'de Zero-Shot %64.0 iken CoT kolu hiç ayrıştırılabilir JSON döndürmedi (%0). Bu tek kliplik, tek koşumlu bir **probe**, benchmark değil; RAG kolunu ölçmek için altın listeden bağımsız bir `glossary_<CLIP>.json` etiketlemek gerekiyor. Gecikme de 3–7 saniye değil, yerel 4-bit çözümlemede çağrı başına 71–81 saniye."

---

### Soru 9: Neden Whisper Large-v3 yerine bazı durumlarda Whisper Large-v3 Turbo öneriyorsunuz?
> **Model Yanıt:**  
> "Önce iki ayrımı netleştirelim: RTF, sesin gerçek zamana göre kaç kat hızla işlendiğini; duvar saati avantajı ise Large-v3 ile aynı kartta kıyaslamayı söyler. Whisper Large-v3 mutlak doğrulukta liderdir (TR normalleştirilmiş WER 0,021, EN 0,010, %95 güven aralıklıyız). Kod çözücüsü 32 katmandır ve ölçülen RTF 0,151 (TR) / 0,216 (EN) düzeyindedir. Whisper Large-v3 Turbo, 4 katmana budanmış kod çözücüsü sayesinde aynı RTX 5060'ta 0,044 (TR) / 0,025 (EN) RTF veriyor — yani sesi gerçek zamanın yaklaşık 23-40 katı hızla işliyor; Large-v3'e kıyasla ölçülen duvar saati avantajı havuzlanmış olarak 3,4 kat (TR, 110,6 sn → 32,5 sn) ve 7,7 kat (EN, 170,7 sn → 22,1 sn)'dir; ağırlıksız klip RTF oranları ise 3,4× / 8,5× verir (havuzlama uzun klipleri kayırdığı için iki sayı ayrışır). Bedeli açık: Türkçede WER 0,021'den 0,037'ye çıkıyor (+1,6 puan), İngilizcede kayıp 0,1 puan (0,010 → 0,011). Dolayısıyla canlı altyazılama ve etkileşimli arayüzlerde Turbo'yu, tek seferlik resmî arşivlendirmede Large-v3'ü öneriyoruz. Ek donanım tasarrufu sayısal olarak iddia etmiyoruz: checkpoint başına tepe VRAM işaretlemeyle ölçülmedi."

---

### Soru 10: EduTerm AI Studio'nun mimarisi nedir? Üretim ortamına hazır mı?
> **Model Yanıt:**  
> "EduTerm AI Studio; Python 3.12, FastAPI, yt-dlp, ffmpeg ve Vanilla HTML/CSS/JS ile inşa edilmiş bir yerel araştırma stüdyosudur. İçerisinde dinamik DAG görselleştirici, aralıklı tekrar (spaced-repetition) algoritmalı uyarlanabilir öğrenci profili ve otomatik çoktan seçmeli sınav motoru barındırmaktadır. Ancak savunmada açıkça belirttiğimiz üzere, bu sistem şu anda bir **yerel Konsept Kanıtlama (PoC) demo aracıdır**. İş kuyruğu bellek içi sözlükte (in-memory dict) tutulmakta, indirmeler geçici WAV dosyaları olarak saklanmaktadır. Üretime geçiş için Docker, Redis iş kuyruğu ve PostgreSQL entegrasyonu gelecek çalışma olarak planlanmıştır."

---

### Soru 11: Sistemin birim test kapsamı ve yazılım mühendisliği güvenilirliği ne durumdadır?
> **Model Yanıt:**  
> "Projemiz yalnızca teorik bir araştırma değil, titiz bir yazılım mühendisliği çalışmasıdır. `experiments/tests/` dizininde toplam **254 birim test** bulunmaktadır (`python -m unittest discover -s experiments/tests`, 4,9 saniyede 254/254). Testler; GPU gerektirmeyen mocked API çağrıları, port çakışmalarında otomatik dinamik port bulma (port hunting failover), dış ağ bağlamalarında güvenlik belirteci (auth token) doğrulama, YouTube yaş/erişim kısıtlı video hata yakalama (graceful fallback), aralıklı tekrar ustalık formülleri, sınav soru üretim mantığı ve savunma sunum paketinin bütünlüğünü kapsamaktadır. Şeffaflık için bir de geçmişi raporluyoruz: `test_gui_api.test_models` bir dönem kırmızıydı, çünkü arayüz varsayılan LLM olarak bulut kolu `evren/qwen3.8-flash-next` ile açılıyordu; test yerel `Qwen/Qwen3.5-4B` bekliyordu. Bulut varsayılanı ücretli kredi ve `EVREN_ACCEPT_TERMS=1` opt-in'i gerektirdiği için varsayılanı yerele döndürdük, bulut seçenek olarak kaldı ve test yeniden geçti."

---

### Soru 12: LLM-as-a-Judge doğrulamasında Cohen's Kappa neden PABAK'tan düşüktür?
> **Model Yanıt:**  
> "Bu sorunun dayandığı doğrulama deneyini **geri çektik**, ve nedenini net söylüyoruz: 233 çift üzerinden 'hakem ile insan arasındaki uyum' olarak raporlanan %97.00 ($R$), %97.42 ($D$), $\kappa=0.735$/PABAK $0.940$ ve $\kappa=0.490$/PABAK $0.948$ sayıları için madde-bazlı hakem tahmini diskte hiçbir zaman saklanmamıştı; dolayısıyla hiçbir uyum metriği yeniden üretilemez. Elimizde gerçekten yeniden üretilebilir olan tek şey **iki insan** arasındaki uyum: 115 çiftlik çift-etiketli alt kütede (labels_a1 × labels_a2) bağlamsal tutarlılık için $P_o=0.965$, $P_e=0.841$, $\kappa_R=0.782$, PABAK $0.930$; tanım/başlangıç uygunluğu ($D$) için $P_o=1.000$, $P_e=0.949$, $\kappa_D=1.000$. Kappa'nın PABAK'tan düşük görünmesi yine klasik prevalans paradoksu: $D$'de pozitif prevalans ~%97,4 olduğu için şans eseri beklenen uyum tek başına 0,949'dur. Not: 0.974 sayısı daha önce bir *uyum oranı* gibi yazılmıştı; diskte bu, $D$ boyutunun **etiketleyici prevalansı**dır. Hakemimizi artık yalnızca madde-bazlı çıktıları duran yerlerde kullanıyoruz (2026 ASR ablasyonu 200 çift, EVREN 374 çift) ve bu hücrelerde $R1/D1$ değerlerini üst sınır olarak okuyoruz — çünkü bu hakemin insana karşı yanlış-pozitif oranını ölçen bir artefakt yok."

---

### Soru 13: Ticari modeller (GPT-4.1) ile açık ağırlıklı modeller (Qwen3.5) arasındaki fark nedir?
> **Model Yanıt:**  
> "GPT-4.1 pilot çalışmamızda 105 İngilizce / 78 Türkçe terim çıkararak en geniş terminolojik kapsamı gösterdi; bunu bir 'tavan başarım' sınırı olarak okuyoruz, kalite iddiası olarak değil — çünkü o pilot için madde-bazlı etiket sayfası diskte yok. Açık ağırlıklı tarafta Qwen3.5-4B, 4-bit nf4 nicemleme ile tek bir 8 GB sınıfı dizüstü GPU'da (RTX 5060) tam bir klip çağrısını çıkarabiliyor; ölçtük: ağırlıklar 3,08 GB sabit, tepe torch ayırıcısı 4,72 GB, cihaz doluluğu 5,26 GB. Gecikme tarafında dürüst olalım: 'saniyeler' değil, Türkçe Large-v3 beslemesinde 39,55 saniye ortalama (girdi seçimiyle 39,55-138,48 s aralığı). Bağlam uyumluluğu ise 233 çiftlik insan etiketiyle ölçüldü ve %90,3 (TR) / %98,0 (EN) R1 verdi. Yani takas şu: sıfır API maliyeti ve tam yerel çalıştırma karşılığında daha yavaş bir semantik katman — ama bu yavaşlık tek bir sayı değildir. Ölçülüsü (tablo `llm_latency.tex`): Türkçede 4B 39,55 sn, GPT-4.1 API günlüğü 14,60 sn (≈2,7 kat); İngilizcede 125,36 sn'ye karşı 16,73 sn (≈7,5 kat). API satırları CPU pilotunun kayıtlı yanıtlarından geldiği için bu kıyas donanım-normalize edilmiş bir kıyas değildir; yine de 'saniyeler' değil 'onlarca saniye' yerel maliyeti ölçülmüştür."

---

### Soru 14: Silero VAD tabanlı parçalama ile 30 saniyelik sabit pencereleme arasında ne fark gördünüz?
> **Model Yanıt:**  
> "Sorunun ilk varsayımını düzeltelim: Silero VAD kullanmıyoruz; kod tabanımızda `silero_vad` bağımlılığı hiç yok. Uyguladığımız şey `vad_chunker.py` içindeki enerji tabanlı bir parçalayıcı: 30 ms çerçeveler, gürültü tabanının 25. yüzdeğilinin 2,2 katıyla uyarlanan bir kesim çizgisi (taban 1e-4'ün altına inmez), 200 ms'den kısa boşlukların kapatılması, 0,4 saniyeden uzun duraklamalarda sınır araması ve 12-28 saniyelik çerçeveler üretmesi. İkinci varsayım artık geçerli değil, çünkü bu karşılaştırmayı **ölçtük**: aynı Whisper Large-v3, aynı 7 klip, aynı metrikler, tek fark `--use-vad`. Sonuç olumsuz yönde — enerji VAD'i WER'i **yedi klipte de kötüleştirdi**: TR makro normalize WER 0,0207 → 0,0412, EN 0,0101 → 0,0262; klip bazında artış TR-1 +0,0174, TR-2 +0,0209, TR-3 +0,0233, EN-1 +0,0131, EN-2 +0,0217, EN-3 +0,0082, EN-4 +0,0216. Bunu 'VAD'in halüsinasyon döngüsünü kestiği' iddiasıyla da destekleyemiyoruz: GPU temel koşumunda 8-bigram tekrar oranı TR 0,0122 / EN 0,0000, yani kıyaslanacak bir döngü yoktu; terminoloji kapsamı da değişmedi (TR 0,875 / EN 0,9688). Bu ölçüm, daha önce bildirdiğimiz '-%83,7 WER' merdiven satırının ve onun 0,129 TR temel değerinin (CPU pilot havuzundan alınmış) yeniden üretilemez olduğunu da gösterdi; satırı geri çektik. VAD bugün arayüz yolunda da kullanılmıyor: `gui/pipeline.py:764` sabit, örtüşmesiz 30 saniyelik pencerelere bölüyor ve `vad_chunker`'ı hiç çağırmıyor. Dolayısıyla VAD'yi bir kazanım değil, ölçülmüş ve olumsuz çıkmış bir müdahale olarak sunuyoruz."

---

### Soru 15: Bu projeyi bir eğitim kurumuna (örneğin Boğaziçi Üniversitesi'ne) entegre etmek istesek ilk adım ne olmalıdır?
> **Model Yanıt:**  
> "İlk adım, sistemin LTI (Learning Tools Interoperability) standardı üzerinden Moodle, Canvas veya Blackboard gibi mevcut Öğrenme Yönetim Sistemlerine (LMS) bir eklenti olarak bağlanmasıdır. Ders kaydı bittiğinde ASR katmanı arka planda Whisper Turbo ile transkripti çıkaracak, Qwen3.5 terminoloji kartlarını ve sınav sorularını üretecek ve öğrencinin video izleme ekranına senkronize terim sözlüğü olarak yerleştirecektir."

---

### Soru 16: Yapay zekâ modellerinin halüsinasyon (uydurma terim/tanım) üretme riskini eğitim bağlamında nasıl sıfırladınız veya sınırlandırdınız?
> **Model Yanıt:**  
> "Halüsinasyonu 'sıfırladık' demiyoruz, ölçülebilir şekilde sınırlandırdık diyoruz; ve savunmada kullandığımız iki eski iddiayı da geri çektik. Elimizde gerçekten ölçülmüş olan üç savunma var: (1) **Katı istem kısıtı:** istemde açıkça transkriptle tutarlı bilgi yönergesi ve zorunlu JSON şeması; bu işe yarıyor ama kanıtı kısmen — İngilizce CoT kolumuz ayrıştırılabilir JSON döndürmedi ve 0 terim üretti. (2) **Bağlamsal uyum etiketlemesi:** 233 çift üzerinden iki bağımsız insan etiketleyici, üretilen terimlerin bağlamla uyumunu R1 ile %90,3 (TR) / %98,0 (EN) olarak ölçtü; yani uydurma terim oranımız düşük ama kapalı bir sözlük olmadığı için bu bir kesinlik metriğidir, duyarlılık değil. (3) **Kod düzeyinde allow-list:** animasyon kolunda Manim çıktı AST tabanlı bir allow-list'ten geçiyor ve en fazla 2 kendini-onarma denemesine izin veriyor. Geri çektiklerimiz: (a) 'Sözlük Destekli RAG halüsinasyonu engelledi, F1 %47,6'dan %76,9'a çıktı' — bu üçlü mock fikstürden geliyordu ve RAG kolu puanlamada kullanılan altın listenin kendisiyle besleniyordu, yani ölçüm döngüseldi; (b) 'BERTScore ile TDK/Oxford'a 0,82 üzeri uyum' — `bert_score` paketi kurulu olmadığı için bağlamsal BERTScore hiç ölçülemedi, yayınladığımız metrik karakter trigram kosinüs + kelime örtüşmesi tabanlı leksik bir vekil ve 24-31 $F_1$ aralığında; (c) 'hakem-insan denetiminde PABAK %94,0' — madde-bazlı hakem tahmini saklanmadığı için yeniden üretilemez."

---

### Soru 17: Canlı ders ortamında gerçek zamanlı (streaming) altyazılama ve terim çıkarımı neden Whisper yerine Streaming ASR gerektirir?
> **Model Yanıt:**  
> "Whisper mimarisi doğası gereği 30 saniyelik pencereler (offline chunked sequence-to-sequence) üzerinde eğitilmiştir; dolayısıyla bir cümlenin transkripsiyonu için pencerenin dolması ve kod çözümü beklenir (ortalama 2-5 sn gecikme). Kayıtlı ders arşivlemede bu sorun teşkil etmezken, canlı bir amfide öğretmenin anlık konuşmasını öğrenciye akıtmak için FastConformer veya Streaming RNN-T (örneğin NVIDIA Nemotron 3.5 Streaming ASR) gibi 80-160 milisaniye gecikmeli akış mimarileri gereklidir. Raporumuzda ve makalemizde bu iki senaryo (Arşivleme vs. Canlı Ders) arasındaki mimari takas açıkça ortaya konmuştur."

---

### Soru 18: Üniversite ortamında öğrenci verilerinin gizliliği, KVKK/GDPR uyumu ve yerel (on-premise) GPU uygulanabilirliği nasıldır?
> **Model Yanıt:**  
> "Eğitim kayıtları ve öğrenci etkileşim verileri kişisel veri (KVKK / GDPR) kapsamındadır ve biz bu konuda bir **uyumluluk iddiası** sunmuyoruz; sunduğumuz şey bir mimari seçim ve onun ölçülmüş fizibilitesidir. İki senaryoyu ayırıyoruz. Tam yerel senaryoda hem akustik hem semantik katman aynı tüketici sınıfı 8 GB'lık GPU'da (RTX 5060) çalışabiliyor — bunu işaretlemeyle ölçtük: Qwen3.5-4B'nin 4-bit hâli 3,08 GB ağırlık sabit, 4,72 GB tepe torch ayırıcısı, 5,26 GB toplam cihaz doluluğu; Whisper Large-v3 ve Turbo fp16/bf16 olarak sığdı. Bu konfigürasyon veri akışını kurum içinde tutma seçeneği verir. Hız-öncelikli önerdiğimiz senaryoda ise semantik katman yerli bir bulut operatörüne (EVREN) gidiyor: bu bir **veri uydukluğu (data-residency) posturusudur**, operatörün işlem/saklama koşullarını almadığımız için uyumluluk demek mümkün değil. Üçüncü bir uyarı da yazılım tarafından: EduTerm AI Studio bir Ar-Ge PoC'sudur, iş kuyruğu bellek içi sözlüktür, indirmeler geçici WAV olarak diske yazılır ve yetkilendirme/rotasyon/veri işleme anlaşması yoktur. Yani 'dışarı tek bayt çıkmaz, sıfır maliyetle tam veri egemenliği' cümlesi ancak yerleşke içi tam yerel kurulum ve KVKK/GDPR incelemesi tamamlandıktan sonra kurulabilir; biz o incelemeyi yapmadık. İlk adım olarak bir kurum entegrasyonunda yapılması gereken de zaten bu: veri işleme envanteri, operatör koşul/gizlilik sözleşmesi ve PoC'nin kalıcı durum (PostgreSQL/Redis) ile kimlik doğrulamasına taşınması."

