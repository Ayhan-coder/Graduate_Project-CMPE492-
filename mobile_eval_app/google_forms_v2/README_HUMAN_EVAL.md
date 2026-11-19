# EduTerm CMPE 492 — İnsan Değerlendirme V2 Kurulum ve Uygulama Rehberi

## 1. Etik ve İdari İzin Hatırlatması
Bu anket ve dokümantasyon kurumsal etik kurul (IRB / Etik Komisyonu) onayı alındığını **iddia etmez**. Katılımcılara anketi dağıtmadan önce Boğaziçi Üniversitesi veya ilgili bölümün insan katılımcılı çalışma izin kurallarını kontrol ediniz.

## 2. Eksik Gerçek Model Çıktıları ve Güvenlik Kilidi
`eduterm_human_eval_v2.gs` varsayılan olarak **`STRICT_REQUIRE_ALL_STIMULI = true`** ile çalışır.  
Depo denetiminde şu modüller için henüz ham, sentetik-olmayan (non-mock) gerçek model çıktısı bulunmadığı tespit edilmiştir:
- **Bölüm 2 (Açıklama Kalitesi):** İSG, KVKK, CS, Finans için eşleşmiş `Qwen2.5-14B-Instruct` Base vs. EduTerm ham açıklama metinleri (`hpc_alignment_tax_cefr_report.json` yalnızca özet metrik kaydetmiştir).
- **Bölüm 4 (CEFR B1/B2/C1):** `cefr_multi_level_benchmark_report.json` dosyası `model_backend="mock"` içerdiği için hariç tutulmuştur.
- **Bölüm 6 (Alan Dışı):** CS/KVKK/Banking/EE alanları `EduTerm-14B-CEFR` SFT kümesiyle çakıştığı için saf OOD olarak işaretlenmemiştir.

Bu nedenle `createEduTermHumanEvaluationForms()` çalıştırıldığında eksik çıktıları listeleyerek durur.  
- **Seçenek A (Önerilen):** Önce HPC üzerinde ilgili gerçek çıktıları üretip `human_eval_manifest.json` ve `.gs` dosyasına ekleyin.
- **Seçenek B (Doğrulanmış Core — Bölüm 5 & 7):** Yalnızca doğrulanmış gerçek çıktılarla (`SOC_01`–`SOC_03` ve `SAFE_01`–`SAFE_03`) ilerlemek için `.gs` dosyasında `STRICT_REQUIRE_ALL_STIMULI = false` yapın.

## 3. Google Forms Üzerinde Çalıştırma
1. `https://script.google.com` adresinde **New project** açın (mevcut V1 projenizi silmeyin).
2. `eduterm_human_eval_v2.gs` içeriğini yapıştırıp kaydedin.
3. Üstten `createEduTermHumanEvaluationForms` fonksiyonunu seçip **Run** butonuna basın.
4. `Execution log` çıktısından **Form V1** ve **Form V2** bağlantılarını alın. (Onay almadan katılımcılara dağıtmayın.)

## 4. Dengeli V1 / V2 Dağıtım ve Pilot Çalışma Prosedürü
- **Dengeli Dağıtım:** Katılımcıları sıra etkisini azaltmak için dönüşümlü/dengeli olarak **Form V1** ve **Form V2**'ye yönlendirin (ör. tek numaralı katılımcılar V1, çift numaralı katılımcılar V2 — gerçek rastgele atama uygulanmıyorsa çalışmayı *randomize deney* olarak adlandırmayın).
- **Pilot Çalışma:** Ana veri toplamadan önce **5–10 katılımcı** ile pilot uygulama yapıp gerçek tamamlama süresini ölçün ve mobil okunabilirliği doğrulayın.

## 5. Analiz ve Doğrulama
```powershell
python validate_human_eval.py
python analyze_human_eval.py --manifest human_eval_manifest.json --v1-csv v1.csv --v2-csv v2.csv
```
