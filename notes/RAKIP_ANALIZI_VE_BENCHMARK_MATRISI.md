# RAKİP UYGULAMALAR BENCHMARK VE METRİK ANALİZ MATRİSİ

**Proje:** Çok Dilli Eğitim Teknolojisi (ASR $\to$ LLM Terminoloji & Pedagojik DAG)  
**Kurum:** Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü  
**Referans:** Bitirme Projesi Toplantı Notları - Aksiyon 6  

---

## 1. Yönetici Özeti (Executive Summary)

Bu doküman, bitirme projesinde geliştirilen **EduTerm AI** mimarisini küresel ve yerel 6 rakip platformla nitel ve nicel metrikler ekseninde kıyaslamaktadır.

Piyasadaki mevcut çözümler ya **yalnızca akustik transkripsiyona** (Otter.ai, WhisperX) ya da **kapalı bulut ortamında metin tabanlı sohbete** (Coursera AI, Khanmigo, NotebookLM) odaklanmaktadır. **EduTerm AI**, ham ders sesinden başlayarak anlık kavram önkoşul grafiği (DAG) çıkaran, çok dilli (Türkçe-İngilizce), dikey sektörlere (Turizm, İSG, KVKK, Bankacılık) uyarlanabilen ve hem **Millî EVREN H200 Bulutu** hem de **uç cihazda (Edge/Local)** çalışabilen tek bütünleşik sistemdir.

---

## 2. Karşılaştırmalı Özellik ve Metrik Matrisi

| Metrik / Özellik | Coursera AI Coach | Otter.ai / Fireflies | Khanmigo (Khan Acad.) | YouTube Auto-Sub | WhisperX Toolkit | Google NotebookLM | **EduTerm AI (Bizim)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Girdi Türü** | Metin / Video | Toplantı Sesi | Metin / Soru | Video | Ses (WAV/MP3) | PDF / Metin / Ses | **16 kHz Ses + Video** |
| **Türkçe ASR Doğruluğu (WER)** | $\approx \%12-18$ | $\approx \%15-22$ | Yok (Sessiz) | $\approx \%8-14$ | $\%2.5 - 4.5$ | Yok (Doğrudan) | **%2.1 (Whisper Large)** |
| **Çok Dillilik Desteği (TR/EN)** | Sınırlı (EN ağırlıklı) | Sınırlı | EN ağırlıklı | Çok dilli | Çok dilli | Çok dilli | **Eşit Değerlendirilmiş** |
| **Kavram Önkoşul Grafiği (DAG)** | ❌ Yok | ❌ Yok | ❌ Yok | ❌ Yok | ❌ Yok | ❌ Yok | **✅ Var (Bloom Taksonomisi)** |
| **Dikey Alan Desteği (Turizm, İSG)** | ❌ Yok (Genel) | ❌ Yok (İş/Ofis) | ❌ Yok (K12) | ❌ Yok | ❌ Yok | Kısmi (Prompt) | **✅ Var (LoRA + RAG)** |
| **Halüsinasyon Önleme (RAG)** | Kısmi | ❌ Yok | Kısmi | ❌ Yok | ❌ Yok | Vektör RAG | **Ontoloji RAG (%0 hedef)** |
| **Uç Cihaz / Yerel Çalışma (Offline)**| ❌ Kapalı Bulut | ❌ SaaS Bulut | ❌ Kapalı Bulut | ❌ Bulut | ✅ Yerel CLI | ❌ Google Bulut | **✅ Var (RTX 5060 / CPU)** |
| **Egemen Bulut Entegrasyonu** | ❌ (ABD) | ❌ (ABD) | ❌ (ABD) | ❌ (ABD) | ❌ Yok | ❌ (ABD) | **✅ Var (Millî EVREN H200)** |
| **Veri Gizliliği & KVKK Uyumu** | Düşük (3. parti) | Düşük | Düşük | Düşük | Yüksek (Yerel) | Orta | **Tam Uyumlu (On-Premise)** |
| **Açık Kaynak Kod & Veri** | ❌ Kapalı | ❌ Ticari | ❌ Kapalı | ❌ Kapalı | ✅ Açık Kaynak | ❌ Kapalı | **✅ Tam Açık Kaynak** |
| **Uç Çıkarım Gecikmesi** | Bulut (2-5s) | Bulut (3-10s) | Bulut (4-8s) | Asenkron | GPU (1-3s) | Bulut (15-30s) | **6.11s (EVREN) / 3.9s (SLM)** |

---

## 3. Rakipler Karşısında Güçlü ve Zayıf Yönler Analizi (SWOT)

```mermaid
quadrantChart
    title Güçlü ve Zayıf Yönler Matrisi
    x-axis Düşük Yerel Kontrol --> Yüksek Yerel Kontrol
    y-axis Düşük Pedagojik Derinlik --> Yüksek Pedagojik Derinlik
    quadrant-1 Öncü Alan (EduTerm AI)
    quadrant-2 Akademik / Pedagojik (Khanmigo, NotebookLM)
    quadrant-3 Genel Tüketici (YouTube, Otter.ai)
    quadrant-4 Akustik Araçlar (WhisperX)
    "EduTerm AI": [0.85, 0.88]
    "Coursera AI": [0.25, 0.65]
    "Khanmigo": [0.20, 0.78]
    "Otter.ai": [0.30, 0.35]
    "YouTube": [0.15, 0.20]
    "WhisperX": [0.80, 0.30]
    "NotebookLM": [0.35, 0.60]
```

### 3.1. Güçlü Yönlerimiz (Strengths & Value Proposition)
1. **Bütünleşik Katmanlı Mimari:** Akustik ASR ve semantik LLM katmanlarını kopuk bırakmayıp, çapraz katman hata yayılımını ampirik olarak ölçen tek çalışmadır.
2. **Pedagojik DAG ve Bloom Seviyeleri:** Diğer tüm rakipler düz metin veya madde işaretli özet sunarken, sistemimiz kavramlar arası önkoşul ilişkilerini yönlü döngüsüz çizge ($DAG$) olarak çıkarır.
3. **KVKK ve Veri Egemenliği:** Bankacılık, şirket içi İSG/KVKK ve kurum içi eğitim videolarının üçüncü taraf yabancı sunuculara (OpenAI, Google, Otter) gönderilmesini engeller; kurum içi (On-Prem) veya Millî EVREN H200 kümesinde çalışır.
4. **Dikey Alan LoRA/RAG Uyarlaması:** Turizm (Tatilsepeti) ve kurumsal eğitim terimlerinde kaçırma (under-recall) oranını düşüren dikey adaptasyon hattına sahiptir.

### 3.2. Zayıf Yönlerimiz ve İyileştirme Planı (Weaknesses & Mitigation)
1. **Kullanıcı Sayısı ve Veri Hacmi:** Coursera ve YouTube milyonlarca saatlik veriye sahipken, projemiz 32 video ve 7 altın klip ile başlamıştır.  
   *Aksiyon:* 32 videonun tamamı indirilip otomatik etiketleme hattına bağlanmıştır.
2. **Model Parametre Boyutu (Yerelde):** 9B modeller yerel 8GB GPU'da bellek darboğazı yaşamaktadır.  
   *Aksiyon:* Toplantı kararı doğrultusunda Qwen 2.5 (0.5B-1.5B) ve Llama 3.2-1B gibi Small Language Model (SLM) LoRA hattı devreye alınmıştır.
3. **Mobil ve Canlı Arayüz:** Otter.ai mobil uygulamaya sahiptir.  
   *Aksiyon:* Proje web tabanlı responsive FastAPI ve erişilebilir GitHub Pages arayüzüne kavuşturulmaktadır.

---

## 4. Akademik Savunma İçin Anahtar Cevaplar

- **Soru:** *"Neden Otter.ai varken bu projeyi yapıyoruz?"*  
  **Cevap:** *"Otter.ai eğitimsel bir kavram hiyerarşisi çıkarmaz; yalnızca ham transkript ve toplantı maddeleri üretir. Ayrıca Türkçe dil doğruluğu düşüktür (%15+ WER) ve kurum verilerini yurt dışı bulutuna taşır. Biz Türkçe'de %2.1 WER ile Bloom taksonomisine dayalı öğrenme rotası (DAG) oluşturuyoruz."*

- **Soru:** *"NotebookLM veya Khanmigo neden yetersiz?"*  
  **Cevap:** *"NotebookLM video içindeki anlık zaman damgalı akustik hataları denetleyemez; kapalı bir LLM'dir ve kurum içi veri güvenliği sağlamaz. Sistemimiz açık ağırlıklı modellerle sıfır veri sızıntısı güvencesi sunar."*
