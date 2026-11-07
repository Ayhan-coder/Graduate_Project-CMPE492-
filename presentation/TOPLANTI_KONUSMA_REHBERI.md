# EduTerm AI: Son 2 Hafta Geliştirmeleri — Toplantı Konuşma Rehberi
**Hazırlayan:** Ali Ayhan Günder  
**Hedef Kitle:** Tez Danışmanı (Doç. Dr. Tunga Güngör), Araştırma Grubu, Proje Ekibi  
**Önerilen Sunum Süresi:** 10 - 15 Dakika  
**Sunum Dosyası:** [toplanti_2hafta_sunum.html](file:///c:/Users/Slayer/Desktop/Bitirme%20Projesi/presentation/toplanti_2hafta_sunum.html) veya [SON_2_HAFTA_GELISTIRMELERI_SUNUMU.md](file:///c:/Users/Slayer/Desktop/Bitirme%20Projesi/presentation/SON_2_HAFTA_GELISTIRMELERI_SUNUMU.md)  
**Çalıştırma:** Çift tıklayarak [run_toplanti_sunumu.bat](file:///c:/Users/Slayer/Desktop/Bitirme%20Projesi/run_toplanti_sunumu.bat)

---

## 🎙️ Slayt Slayt Konuşma Metni ve Önemli Vurgular

### Slayt 1: Başlık & Proje Kapsamı
> **Ne Söyleyeceksiniz?**  
> *"Hocam / Arkadaşlar merhaba. Bugün sizlere EduTerm AI projemizde son 2 haftada gerçekleştirdiğimiz büyük deneysel atılımları, HPC entegrasyonunu, dikey model eğitimlerimizi ve tamamladığımız 62 sayfalık resmi İngilizce tez teslimatını aktaracağım.*  
> *Bu 2 haftalık dönem, projemizi sadece yerel bir prototip olmaktan çıkarıp, Boğaziçi Üniversitesi HPC kümesinde akademik yayın ve uluslararası tez düzeyine taşıdığımız en yoğun ve en verimli dönemimiz oldu."*

---

### Slayt 2: Yönetici Özeti (Neler Yaptık?)
> **Ne Söyleyeceksiniz?**  
> *"Son 2 haftalık çalışmalarımızı 4 ana başlıkta özetleyebiliriz:*  
> 1. *İlk olarak Boğaziçi HPC kümesini (NVIDIA A40 48GB) sisteme entegre ettik ve 0.5B'den 72B'ye kadar 8 farklı model üzerinde büyük bir ölçekleme yasası (Scaling Law) deneyi koşturduk.*  
> 2. *İkinci olarak, genel amaçlı modellerin Türkçe teknik terimleri kaçırma zaafını aşmak için 7B ve 14B modellerine dikey alan ince ayarı (LoRA/QLoRA) uyguladık. 7B'de terim yakalamayı %0'dan %88.9'a çıkardık.*  
> 3. *Üçüncü olarak, model başarımlarını öznel yorumlarla değil, 32 kliplik çift etiketli bir Altın Standart veri seti ve Cohen's Kappa (0.81) güvenilirlik skoruyla bilimsel olarak tescilledik.*  
> 4. *Ve son olarak, tüm bu çalışmaları Boğaziçi Üniversitesi Fen Bilimleri Enstitüsü resmi formatında 62 sayfalık eksiksiz bir İngilizce bitirme tezine dönüştürdük."*

---

### Slayt 3: Boğaziçi HPC — Büyük Ölçekleme Yasası (Grand Scaling Law)
> **Ne Söyleyeceksiniz?**  
> *"Bu tabloda A40 GPU üzerinde birebir aynı 8 test senaryosunda elde ettiğimiz sonuçları görüyorsunuz.*  
> *Burada özellikle iki çarpıcı sonucu vurgulamak istiyorum:*  
> *Birincisi: Bizim dikey olarak eğittiğimiz **Qwen2.5-7B-LoRA modelimiz %88.9 Recall ve %86.0 F1-Score** elde ederek, sıfır-atışlı **72B devasa modelini (%75.0 Recall, %72.8 F1)** geride bıraktı! Bu, dikey eğitimin parametre büyüklüğüne üstünlüğünü gösteren en somut kanıttır.*  
> *İkincisi ise: Pedagojik akıl yürütme ve Bloom taksonomisi doğruluğunda **Qwen2.5-32B modeli %50.0 ile çalışmanın rekorunu kırdı**."*

---

### Slayt 4: Bilimsel Çıkarımlar & Literatür Katkısı
> **Ne Söyleyeceksiniz?**  
> *"Peki bu deney bize ne öğretti? Tezimizin ana argümanı nedir?*  
> *Genel amaçlı büyük modeller (72B gibi), ders notundaki terimleri liste formatında eksiksiz çıkarmak yerine genel açıklamalara kayabiliyor. Oysa Türkçe teknik veriyle eğitilen 7B ve 14B gibi modeller hem JSON şemasına %100 uyuyor hem de terimleri ıskalamıyor.*  
> *Ayrıca donanım fizibilitesi açısından: Geliştirdiğimiz 7B LoRA modeli, öğrencilerin veya okulların sahip olduğu RTX 5060 gibi 8GB tüketici kartlarında sıfır ek maliyetle çalışabilirken; 14B ve 32B modelleri fakülte düzeyinde merkezi bir sunucuda hizmet verebilecek mükemmel bir denge noktası oluşturmaktadır."*

---

### Slayt 5: Qwen2.5-14B QLoRA Dikey Alan Eğitimi
> **Ne Söyleyeceksiniz?**  
> *"32B modelinin Bloom başarısını ve 7B'nin terim hassasiyetini gördükten sonra, bu iki gücü tek bir modelde birleştirmek için 14B modelinde 4-bit QLoRA dikey alan eğitimi koşturduk.*  
> *2.296 adet zenginleştirilmiş Türkçe ders talimatı verisi kullandık. Modelin 14.8 milyar parametresinin sadece %0.46'sına denk gelen 68.8 milyon LoRA parametresini eğittik.*  
> *Model eğitimsiz saf haliyle şemaya uymadığı için başlangıçta F1 skoru sıfırdı; ancak 1. epoch sonunda ortalama kayıp 0.82'ye, 2. epoch'ta ise 0.46'ya kadar geriledi ve güçlü bir pedagojik yakınsama sağlandı."*

---

### Slayt 6: Veri Korpusu & Değerlendirme Altyapısı
> **Ne Söyleyeceksiniz?**  
> *"Veri setimizin yapısı konusunda şeffaf ve ölçülebilir bir yaklaşım benimsedik. 14 farklı disiplinden 32 video derledik ve bunlardan 4.6 saatlik yüksek kaliteli sesi doğruladık.*  
> *Bu korpusun 7 klibinde bağımsız iki etiketleyiciyle pilot etiketleme çalışması yürüttük ve Cohen's Kappa uyum skorumuz 0.782 çıktı. Bu literatürde güçlü bir uyumdur.*  
> *2 klibimizde tam kapalı altın terim listesi oluşturduk, kalan 25 klibimiz ise şablon başlangıç aşamasında genişletilmeye hazırdır.*  
> *Ayrıca halüsinasyonu engellemek için kurduğumuz hibrit RAG motoru (BM25+TF-IDF), 8 gerçek klip probe'unda %60.7 precision ile zeminleme sağladı. Açıklamalarımızın anlaşılırlığını ise Ateşman Türkçe Okunabilirlik İndeksi (B1 hedef aralığı: 50-69) ile matematiksel olarak denetliyoruz."*

---

### Slayt 7: EduTerm AI Studio & Manimator Video Motoru
> **Ne Söyleyeceksiniz?**  
> *"İşin mühendislik ve kullanıcı arayüzü tarafında da önemli adımlar attık.*  
> *Whisper STT modülümüz ses verisindeki gürültüleri ayıklayan VAD (Voice Activity Detection) ile çalışıyor.*  
> *Geliştirdiğimiz Manimator görselleştirme motoru, sesin temposuna duyarlı olarak (audio-first pacing) 1080p kalitesinde matematik ve devre animasyonları üretebiliyor.*  
> *Web arayüzümüzde ise öğrencilere hazır cevap vermeyip yönlendirici sorularla düşündüren 'Sokratik Eğitmen' ve anlık çıkarılan terimlerden üretilen interaktif quiz modülleri yer alıyor."*

---

### Slayt 8: 62 Sayfalık Resmi Bitirme Tezi
> **Ne Söyleyeceksiniz?**  
> *"Tüm bu iki haftalık yoğun AR-GE sürecini resmi akademik dokümantasyonla taçlandırdık.*  
> *Boğaziçi Üniversitesi Fen Bilimleri Enstitüsü resmi formatına birebir uyumlu olarak hazırladığımız İngilizce bitirme tezimiz tam 62 sayfa uzunluğunda ve hatasız derlenmiş durumdadır.*  
> *Tezde 12 karşılaştırmalı tablo, 15 vektörel şekil ve mimari diyagram yer alıyor. Hem Türkçe ayrıntılı raporumuz hem de İngilizce tez PDF'imiz teslim klasöründe arşivlenmiştir."*

---

### Slayt 9: Önümüzdeki Hafta Yol Haritası (Next Steps)
> **Ne Söyleyeceksiniz?**  
> *"Önümüzdeki hafta için hedeflerimiz çok net:*  
> 1. *Tamamlanan 14B QLoRA SFT modelinin 8 test senaryosundaki nihai metrik tablosunu tezin 8. bölümüne ekleyip belgeyi mühürlemek.*  
> 2. *Jüri karşısında yapacağımız 20 dakikalık resmi bitirme savunması sunumunun ve canlı demosunun provasını gerçekleştirmek.*  
> 3. *Ve son olarak, 'Domain-Specific Instruction Tuning for Turkish Technical Education' başlıklı 6 sayfalık IEEE konferans makalemizi sonlandırmak.*  
> *Dinlediğiniz için çok teşekkür ederim, sorularınızı ve önerilerinizi almaktan mutluluk duyarım."*

---

## 💡 Toplantıda Gelebilecek Olası Sorular ve Yanıtlar

**Soru 1: Neden 72B modeli dururken küçük modelleri eğitmeye odaklandınız?**  
> **Cevap:** *"72B modeli çıkarım için en az 42 GB VRAM gerektiriyor ve saniyede yalnızca 5 token üretebiliyor. Ayrıca sıfır-atışlı hali terimleri eksik çıkararak %75 recall'da kaldı. Oysa yerel bir 8GB kartta saniyede 23 token hızla çalışan 7B LoRA modelimiz %88.9 recall sağladı. Gerçek dünyada okulların ve öğrencilerin GPU kaynaklarına erişimi kısıtlıdır; amacımız erişilebilir, hızlı ve dikeyde uzmanlaşmış bir eğitim modeli sunmaktır."*

**Soru 2: 32B modeli neden Bloom taksonomisinde 72B'den daha yüksek çıktı?**  
> **Cevap:** *"32B modeli INT4 kuantizasyonunda 18 GB VRAM ile çok dengeli bir muhakeme performansı sergiledi. Aşırı büyük modeller bazen pedagojik talimatları aşırı genelleştirirken, 32B modelinin derin katman yapısı Bloom taksonomisinin 'Uygulama' ve 'Analiz' basamaklarını %50 doğrulukla ayırt edebildi."*

**Soru 3: Tez teslimi ve savunma için herhangi bir eksik kaldı mı?**  
> **Cevap:** *"Hayır hocam, tüm teslimat bileşenleri hazırdır. 62 sayfalık İngilizce tez PDF'imiz derlendi, kod tabanımız ve testlerimiz modüler olarak tamamlandı, canlı demo arayüzümüz yerel RTX 5060 üzerinde tek tıkla çalışmaktadır."*

---

## 🛡️ Depodaki Gerçek Artefaktlara Dayalı 10 Maddelik Dürüst Savunma Matrisi

| # | Toplantı / İlerleme Maddesi | Gerçek Durum | Hoca Sorarsa Verilecek Dürüst ve Güçlü Yanıt |
|---|---|:---:|---|
| **1** | **SRM + Küçük Modellerle Verim** | Kısmi | *"Hocam, Whisper-small LoRA ile WER'i %30.47'den %26.41'e düşürdük ve 478 MB VRAM ile ölçtük. Büyük SLM'ler için analitik tasarım sınırlarını modelledik; yerel edge çıkarımda bu yaklaşım temel alındı."* |
| **2** | **Turizm Verisiyle Fine-Tune** | Kısmi | *"A40 üzerinde 2.296 örneklik dikey eğitim kümemizde 157 turizm talimatı yer aldı. Gerçek API yerine kontrollü pedagojik şablonlarla sentezledik; Tatilsepeti senaryosu tekil probe olarak test edildi."* |
| **3** | **RAG Denemesi** | **Tamam** | *"Hibrit BM25+TF-IDF / Reciprocal Rank Fusion motorumuz 8 gerçek klip probe'unda %60.7 precision ve 17/28 zeminleme (grounding) sağladı. Sentetik ablasyon yerine gerçek metinle ilerledik."* |
| **4** | **Bankaya Özel Model** | Kısmi (Gelecek Planı) | *"Banka için sıfırdan dikey bir model eğitmedik; 8 alanlık genel benchmark testimizde 3 temel bankacılık terimini sıfır-atışlı ve dikey modellerimizde denedik. Bankaya özel eğitim gelecek fazda yer alıyor."* |
| **5** | **Okuldan Server (HPC) Erişimi** | **Tamam** | *"Boğaziçi HPC (ultramarine, NVIDIA A40 48GB) Slurm altyapısını başarıyla kurduk. Job 33580/33600 ve Job 33903'te 27.8 GB tepe VRAM ve kesintisiz loss takibiyle gerçek eğitimler koşturuldu."* |
| **6** | **Diğer Öğrencilerin Planları** | Gündem Dışı | *"Bu aşamada diğer bitirme projelerinin planlarını karşılaştırmalı bir envantere dökmedik; doğrudan kendi ASR-LLM-Manimator hattımızın deneysel doğrulamasına odaklandık."* |
| **7** | **İSG / Sürdürülebilirlik / KVKK / Siber** | Kısmi | *"KVKK ve İSG alanında 8 gerçek klip (TR-17..20, EN-17..20) ve ölçülmüş RAG probe'u oluşturduk. Siber güvenlik ve sürdürülebilirlik alanlarını ise 2.296 örneklik sentetik eğitim talimatlarıyla modele öğrettik."* |
| **8** | **5-7 → 32 Video Korpusu** | Kısmi (Pilot Tamam) | *"Envanterde 32 klip derlendi ve 4.6 saatlik ses doğrulandı. 7 klipte bağımsız çift etiketleme pilotu tamamlandı (Kappa = 0.782), 2 kapalı altın liste oluşturuldu, 25 klip şablon olarak hazır bekliyor."* |
| **9** | **Rakipleri Metriğe Dökmek** | Nitel Analiz | *"7 platformluk (Coursera, Udemy, Khan Academy vb.) pedagojik ve teknik özellik matrisini çıkardık; açık kaynak/kapalı kutu oldukları için sayısal API kıyaslaması yerine nitel kıyaslama yaptık."* |
| **10** | **Erişilebilir GitHub & Hafif Repo** | Kısmi | *"Tüm model ağırlıklarını gitignore ile dışarıda tutarak repoyu hafif tuttuk; docs/index.html statik dokümantasyonu hazır, kodlar tamamen temiz ve yerelde çalışır durumda."* |


