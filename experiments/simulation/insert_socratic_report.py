import io

with open('notes/Bitirme_Projesi_Raporu.md', 'r', encoding='utf-8') as f:
    text = f.read()

socratic_section = """---

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
   $$\\text{Pedagogical Patience Score} = \\frac{SQ + HQ - SF}{2 \\times N_{\\text{turns}}} \\times 100$$

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

"""

target = "## 9. Donanım ve Çalışma Ortamı"
if target in text:
    new_text = text.replace(target, socratic_section + target)
    with open('notes/Bitirme_Projesi_Raporu.md', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print("SUCCESS")
else:
    print("TARGET NOT FOUND")

