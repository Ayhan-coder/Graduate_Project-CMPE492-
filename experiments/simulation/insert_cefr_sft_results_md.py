import io

with open('notes/Bitirme_Projesi_Raporu.md', 'r', encoding='utf-8') as fh:
    text = fh.read()

cefr_results_md = """#### 4. EduTerm-14B-TR Tam Çok Düzeyli SFT Eğitimi ve Ampirik Bulguları (HPC A40)
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
"""

target = "### 8.15 Çok Turlu Sokratik Eğitmenlik (Multi-Turn Socratic Tutoring Dialogue Benchmark on Boğaziçi HPC A40)"
if target in text:
    text = text.replace(target, cefr_results_md + "\n---\n\n" + target)
    with open('notes/Bitirme_Projesi_Raporu.md', 'w', encoding='utf-8') as fh:
        fh.write(text)
    print("SUCCESS: Added CEFR 14B SFT results to Bitirme_Projesi_Raporu.md!")
else:
    print("ERROR: Target not found in md report")

