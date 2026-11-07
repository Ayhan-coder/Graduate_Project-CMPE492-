import re
from pathlib import Path

md_path = Path("notes/Bitirme_Projesi_Raporu.md")
md = md_path.read_text(encoding="utf-8")

md_sections = r"""
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

| Model Mimarisi | Parametre / Dil | Peak VRAM | Eğitim Süresi | JSON Uyum | Macro Precision | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Kumru-2B (Zero-Shot) | 2.39B / Türkçe | 9.43 GB | — | %87.5 | %0.0 | %0.0 |
| **Kumru-2B (EduTerm SFT)** | **2.39B / Türkçe** | **9.43 GB** | **54.5 dk** | **%100.0** | **%70.8** | **%43.8** |
| Qwen2.5-7B (EduTerm SFT) | 7.61B / Çok Dilli | 16.80 GB | 2.1 saat | %100.0 | %75.0 | %46.5 |
| **EduTerm-14B (CEFR SFT)** | **14.7B / Çok Dilli** | **24.95 GB** | **6.6 saat** | **%100.0** | **%87.5** | **%47.2** |

### 8.16.2 Temel Bulgular
1. **Morfolojik Akıcılık**: Kumru-2B, Türkçe sondan eklemeli çekim eklerini (*"Okazaki Parçacıkları"*, *"Düğüm gerilimleri analizi"*, *"Sıfırıncı Gün Açığı"*) doğal bir akıcılıkla üretmiş ve 32.000'lik yerli kelime dağarcığı sayesinde Türkçe metinleri daha az alt-kelimeye (subword) bölerek token verimliliği sağlamıştır.
2. **Kapasite ve Parametre Ölçeği**: 2B ölçekli hafif model terim tespitinde 14B modele çok yakın bir F1 skoru (%43.8 vs %47.2) yakalamıştır. Ancak B1/B2/C1 düzeyleri arasındaki üslup ve okunabilirlik gradyanını keskin biçimde ayrıştırmada 14B modelin parametre derinliğinin kritik rol oynadığı gözlemlenmiştir.

---

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
"""

pos = md.find("## 9. Donan")
if pos != -1:
    new_md = md[:pos] + md_sections + "\n\n" + md[pos:]
    md_path.write_text(new_md, encoding="utf-8")
    print("Successfully inserted Kumru and Matcher Audit into Bitirme_Projesi_Raporu.md!")
else:
    print("Target section 9 not found in md!")
