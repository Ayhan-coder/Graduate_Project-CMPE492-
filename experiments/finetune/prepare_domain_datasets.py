"""Domain-Specific Instruction-Tuning Dataset Generator (2,000–5,000 Samples).

Generates high-quality Turkish instruction datasets across the target vertical domains:
1. Tourism (Tatilsepeti Ar-Ge: acente, dinamik fiyatlandırma, voucher, overbooking)
2. Corporate Occupational Health and Safety (İSG / OSHA: risk, kkd, acil durum)
3. KVKK / GDPR (Veri Koruma Hukuku: veri sorumlusu, açık rıza, ihlal bildirimi)
4. Corporate Cybersecurity (Oltalama, MFA, Zero Trust, fidye yazılımı, SIEM)
5. Sustainability (ESG, Karbon Ayak İzi, Kapsam 1-2-3, Yeşil Mutabakat)
6. Banking & Finance (Kredi, Findeks, MASAK, likidite, Basel, FAST)

Combines:
- Multi-domain terminology extraction with CEFR B1 definitions and Bloom levels (JSON)
- Prerequisite DAG relationship inference & knowledge gap reasoning
- Concept distinction & misconception resolution
- Contextual operational scenario Q&A
- Multi-concept coordination meeting dialogues
- Real academic STEM lecture annotations
- High-quality Turkish general instruction pairs from external corpus (prevents catastrophic forgetting)

Enforces strict zero-contamination LeakageGuard against held-out benchmark evaluation sets.
Outputs standardized JSONL files formatted for ChatML / HuggingFace TRL SFTTrainer.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DomainDatasets")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "experiments" / "finetune" / "data"
EXTERNAL_DIR = DATA_DIR / "external"
DATA_DIR.mkdir(parents=True, exist_ok=True)

from experiments.finetune.prepare_external_datasets import LeakageGuard

# 42 Core Canonical Concepts across 6 Vertical Domains
CONCEPTS_DATA: List[Dict[str, Any]] = [
    # ------------------ 1. TURİZM & TATİLSEPETİ AR-GE ------------------
    {
        "domain": "tourism",
        "domain_tr": "Turizm & Otelcilik",
        "canonical_id": "TOUR-01",
        "term": "Dinamik Fiyatlandırma",
        "definition": "Otel doluluk oranı ve talep yoğunluğuna bağlı olarak oda fiyatlarının anlık ve kural bazlı güncellendiği gelir yönetimi sistemi.",
        "bloom": "Uygulama",
        "prereq": "Doluluk Oranı",
        "context": "Yaz sezonunda otel doluluk oranı %85'i aştığında gelir yönetimi yazılımı dinamik fiyatlandırma kuralını devreye sokarak son kalan odaların gecelik ücretini kademeli olarak yükseltir.",
        "distinction_target": "Sabit Fiyatlandırma",
        "distinction": "Sabit fiyatlandırma tüm sezon boyunca önceden belirlenmiş tek bir fiyat tarifesini uygularken; dinamik fiyatlandırma pazar talebi, rakip fiyatları ve kalan oda sayısına göre anlık fiyat değişimi yapar."
    },
    {
        "domain": "tourism",
        "domain_tr": "Turizm & Otelcilik",
        "canonical_id": "TOUR-02",
        "term": "No-Show",
        "definition": "Rezervasyonu kesinleşmiş bir müşterinin haber vermeksizin ve iptal süresini kaçırarak otele giriş yapmaması durumu.",
        "bloom": "Hatırlama",
        "prereq": "Voucher",
        "context": "Misafir haber vermeksizin otele giriş yapmadığında resepsiyon sistemi kaydı no-show olarak işaretler ve ilk gece konaklama bedeli garanti kartından tahsil edilir.",
        "distinction_target": "Geç İptal (Late Cancellation)",
        "distinction": "Geç iptalde misafir otele gelmeyeceğini bildirmesine rağmen cezasız iptal süresini kaçırmıştır; no-show durumunda ise misafir hiçbir bildirimde bulunmadan doğrudan otele gelmemiştir."
    },
    {
        "domain": "tourism",
        "domain_tr": "Turizm & Otelcilik",
        "canonical_id": "TOUR-03",
        "term": "Voucher",
        "definition": "Müşterinin seyahat acentesi aracılığıyla satın aldığı otel, transfer ve tur hizmetlerinin ödendiğini kanıtlayan resmi rezervasyon belgesi.",
        "bloom": "Anlama",
        "prereq": "Merkezi Rezervasyon Sistemi",
        "context": "Acente üzerinden paket tur satın alan yolculara check-in sırasında ibraz edilmek üzere otel adı, oda tipi ve yemek planını belirten onaylı bir voucher belgesi gönderilir.",
        "distinction_target": "Fatura",
        "distinction": "Fatura mali ve vergi amaçlı resmi bir muhasebe belgesi iken, voucher acente ile hizmet sağlayıcı otel arasındaki hizmet hakkını belgeleyen kupondur."
    },
    {
        "domain": "tourism",
        "domain_tr": "Turizm & Otelcilik",
        "canonical_id": "TOUR-04",
        "term": "Doluluk Oranı",
        "definition": "Belirli bir zaman diliminde satılan oda sayısının tesisteki satılabilir toplam oda sayısına bölünmesiyle elde edilen yüzde oranı.",
        "bloom": "Anlama",
        "prereq": "Oda Envanteri",
        "context": "Tesis müdürü haftalık operasyon toplantısında 200 odalı otelin 160 odasının dolu olduğunu ve doluluk oranının %80 olarak gerçekleştiğini açıkladı.",
        "distinction_target": "RevPAR",
        "distinction": "Doluluk oranı sadece kaç odanın satıldığının yüzdesini gösterirken, RevPAR satılan odaların toplam oda başına düşen parasal gelir verimini ölçer."
    },
    {
        "domain": "tourism",
        "domain_tr": "Turizm & Otelcilik",
        "canonical_id": "TOUR-05",
        "term": "Çifte Rezervasyon (Overbooking)",
        "definition": "Olası son dakika iptalleri ve no-show kayıplarını dengelemek amacıyla otelin mevcut oda kapasitesinden daha fazla rezervasyon kabul etmesi.",
        "bloom": "Analiz",
        "prereq": "Dinamik Fiyatlandırma",
        "context": "Bayram tatilinde geçmiş no-show istatistiklerine güvenilerek %10 overbooking uygulandı; ancak tüm konuklar otele gelince fazla misafirler muadil 5 yıldızlı otele aktarıldı.",
        "distinction_target": "Sistem Hatası",
        "distinction": "Sistem hatası istenmeyen teknik bir aksaklıkken, overbooking gelir maksimizasyonu amacıyla istatistiki olasılık hesaplarına dayalı olarak bilinçli uygulanan bir gelir yönetimi stratejisidir."
    },
    {
        "domain": "tourism",
        "domain_tr": "Turizm & Otelcilik",
        "canonical_id": "TOUR-06",
        "term": "Merkezi Rezervasyon Sistemi (CRS)",
        "definition": "Bir otel zincirinin veya acentenin tüm oda envanterini, fiyat tarifelerini ve rezervasyon verilerini tek merkezden yöneten yazılım platformu.",
        "bloom": "Uygulama",
        "prereq": "Otel Yönetim Sistemi (PMS)",
        "context": "Tatilsepeti Ar-Ge ekibi, otel zincirlerinin oda müsaitliklerini tek API üzerinden anlık sorgulamak için yeni nesil CRS entegrasyon servisini yayına aldı.",
        "distinction_target": "GDS (Global Dağıtım Sistemi)",
        "distinction": "CRS tek bir otel zincirinin iç rezervasyon yönetimini sağlarken, GDS dünya çapında havayolu, otel ve araç kiralama şirketlerini seyahat acentelerine bağlayan küresel ağdır."
    },
    {
        "domain": "tourism",
        "domain_tr": "Turizm & Otelcilik",
        "canonical_id": "TOUR-07",
        "term": "Kanal Yöneticisi (Channel Manager)",
        "definition": "Otelin oda fiyatı ve kontenjan bilgilerini Booking, Tatilsepeti, Expedia gibi tüm online satış kanallarına eşzamanlı aktaran entegrasyon yazılımı.",
        "bloom": "Uygulama",
        "prereq": "Merkezi Rezervasyon Sistemi (CRS)",
        "context": "Kanal yöneticisi sayesinde bir acenteden oda satıldığında diğer tüm online portallardaki oda sayısı saniyeler içinde otomatik olarak bir eksiltilir.",
        "distinction_target": "PMS (Ön Büro Sistemi)",
        "distinction": "PMS otel içindeki konaklama, temizlik ve resepsiyon operasyonlarını yönetirken, kanal yöneticisi otelin dış dünyadaki online satış acenteleriyle olan bağlantısını senkronize eder."
    },

    # ------------------ 2. İŞ SAĞLIĞI VE GÜVENLİĞİ (İSG / OSHA) ------------------
    {
        "domain": "isg",
        "domain_tr": "İş Sağlığı ve Güvenliği",
        "canonical_id": "ISG-01",
        "term": "Ramak Kala Olayı",
        "definition": "İşyerinde meydana gelen, çalışan veya ekipmana zarar verme potansiyeli taşıdığı halde tesadüfen yaralanma veya hasara yol açmadan atlatılan tehlikeli durum.",
        "bloom": "Anlama",
        "prereq": "İş Güvenliği Kültürü",
        "context": "Depoda paletlerin istiften forkliftin hemen yanına devrilmesi ancak kimsenin yaralanmaması üzerine ramak kala formu doldurularak kök neden araştırması başlatıldı.",
        "distinction_target": "İş Kazası",
        "distinction": "İş kazasında çalışanın bedensel veya ruhsal bir zarara uğraması ya da ekipman hasarı söz konusudur; ramak kala olayında ise herhangi bir hasar veya yaralanma meydana gelmemiştir."
    },
    {
        "domain": "isg",
        "domain_tr": "İş Sağlığı ve Güvenliği",
        "canonical_id": "ISG-02",
        "term": "Risk Değerlendirmesi",
        "definition": "İşyerindeki mevcut tehlikelerin belirlenmesi, gerçekleşme olasılığı ile oluşturacağı şiddetin puanlanarak öncelikli tedbirlerin planlanması süreci.",
        "bloom": "Analiz",
        "prereq": "Ramak Kala Olayı",
        "context": "Üretim tesisinde kullanılan pres makineleri için 5x5 L tipi matris yöntemiyle risk değerlendirmesi yapılmış ve acil durdurma butonlarının yenilenmesine karar verilmiştir.",
        "distinction_target": "Tehlike Tespiti",
        "distinction": "Tehlike zarar verme potansiyeli olan unsurun kendisidir (örn. ıslak zemin); risk ise bu tehlikenin yol açabileceği olumsuz sonucun olasılığı ve şiddetinin bileşkesidir."
    },
    {
        "domain": "isg",
        "domain_tr": "İş Sağlığı ve Güvenliği",
        "canonical_id": "ISG-03",
        "term": "Kişisel Koruyucu Donanım (KKD)",
        "definition": "Çalışanı yürütülen işten kaynaklanan sağlık ve güvenlik tehlikelerine karşı korumak amacıyla tasarlanmış baret, maske, gözlük ve güvenlik ayakkabısı gibi ekipmanlar.",
        "bloom": "Hatırlama",
        "prereq": "Risk Değerlendirmesi",
        "context": "Şantiye girişinde tüm taşeron personele CE sertifikalı baret, reflektörlü yelek ve çelik burunlu iş ayakkabısından oluşan KKD seti teslim edildi.",
        "distinction_target": "Toplu Koruma Önlemi",
        "distinction": "Toplu koruma önlemleri (korkuluk, havalandırma) tehlikeyi kaynağında keserek tüm çalışanları korur; KKD ise risk kaynağında yok edilemediğinde sadece onu giyen kişiyi koruyan son savunma hattıdır."
    },
    {
        "domain": "isg",
        "domain_tr": "İş Sağlığı ve Güvenliği",
        "canonical_id": "ISG-04",
        "term": "Acil Durum Eylem Planı",
        "definition": "İşyerinde yangın, patlama, deprem veya kimyasal sızıntı gibi kriz durumlarında tahliye, ilkyardım ve müdahale adımlarını tanımlayan yazılı prosedür.",
        "bloom": "Uygulama",
        "prereq": "Risk Değerlendirmesi",
        "context": "Fabrikada yılda en az bir kez yapılan genel yangın tatbikatında acil durum eylem planına uygun olarak tüm personel 3 dakika içinde toplanma bölgesine tahliye edildi.",
        "distinction_target": "Günlük Çalışma Talimatı",
        "distinction": "Günlük talimat rutin iş adımlarını belirlerken; acil durum eylem planı olağanüstü kriz ve afet anlarında can güvenliğini sağlamak için uygulanır."
    },
    {
        "domain": "isg",
        "domain_tr": "İş Sağlığı ve Güvenliği",
        "canonical_id": "ISG-05",
        "term": "LOTO (Kilitleme ve Etiketleme)",
        "definition": "Makinelerin bakım ve onarımı sırasında elektrik, hidrolik veya basınçlı hava hatlarının fiziksel kilit ve etiket takılarak yetkisiz çalıştırılmasının önlenmesi sistemi.",
        "bloom": "Uygulama",
        "prereq": "Risk Değerlendirmesi",
        "context": "Bant konveyörünün tambur bakımına başlamadan önce teknisyen ana şaltere kişisel asma kilidini ve uyarı etiketini takarak LOTO prosedürünü uyguladı.",
        "distinction_target": "Sadece Şalteri Kapatma",
        "distinction": "Yalnızca şalteri kapatmak başkasının farkında olmadan makineyi açmasına yol açabilir; LOTO ise şalteri fiziksel anahtarlı kilitle kilitler ve enerji boşaltımını garanti eder."
    },
    {
        "domain": "isg",
        "domain_tr": "İş Sağlığı ve Güvenliği",
        "canonical_id": "ISG-06",
        "term": "Ortam Ölçümü",
        "definition": "İşyeri ortamındaki gürültü, toz, aydınlatma, kimyasal buhar ve termal konfor düzeylerinin akredite cihazlarla ölçülüp yasal sınırlarla karşılaştırılması.",
        "bloom": "Analiz",
        "prereq": "Risk Değerlendirmesi",
        "context": "Dokuma atölyesinde yapılan kişisel gürültü maruziyet ölçümünde ses düzeyinin 87 desibeli aştığı tespit edilerek kulaklık kullanımı zorunlu hale getirildi.",
        "distinction_target": "Periyodik Sağlık Muayenesi",
        "distinction": "Ortam ölçümü çalışma alanının fiziksel koşullarını ölçerken; sağlık muayenesi bu koşulların çalışanın vücudundaki etkilerini (işitme testi vb.) takip eder."
    },
    {
        "domain": "isg",
        "domain_tr": "İş Sağlığı ve Güvenliği",
        "canonical_id": "ISG-07",
        "term": "Güvenlik Bilgi Formu (GBF / SDS)",
        "definition": "Kimyasal maddelerin tehlike sınıflandırmasını, ilk yardım önlemlerini, yangınla mücadele yöntemlerini ve güvenli depolama şartlarını içeren 16 maddelik standart belge.",
        "bloom": "Anlama",
        "prereq": "KKD",
        "context": "Laboratuvarda yeni kullanılacak solvent için GBF belgesi incelenerek uygun nitril eldiven tipi ve organik buhar filtreli maske temin edildi.",
        "distinction_target": "Ürün Kullanım Kılavuzu",
        "distinction": "Kullanım kılavuzu ürünün nasıl verimli çalışacağını açıklarken; GBF ürünün insan sağlığına ve çevreye yönelik kimyasal toksisite ve güvenlik risklerini belgeler."
    },

    # ------------------ 3. KİŞİSEL VERİLERİN KORUNMASI (KVKK / GDPR) ------------------
    {
        "domain": "kvkk",
        "domain_tr": "Kişisel Verilerin Korunması",
        "canonical_id": "KVKK-01",
        "term": "Veri Sorumlusu",
        "definition": "Kişisel verilerin işleme amaçlarını ve vasıtalarını belirleyen, veri kayıt sisteminin kurulmasından ve yönetilmesinden hukuken sorumlu olan gerçek veya tüzel kişi.",
        "bloom": "Anlama",
        "prereq": "Kişisel Veri",
        "context": "Şirketimiz müşterilerin kredi kartı ve kimlik bilgilerini işlerken 6698 sayılı kanun kapsamında veri sorumlusu sıfatıyla hareket etmektedir.",
        "distinction_target": "Veri İşleyen",
        "distinction": "Veri sorumlusu verinin neden ve nasıl işleneceğine karar veren asıl karar vericiyken; veri işleyen veri sorumlusunun verdiği yetkiye dayanarak teknik operasyonu yürüten taraftır (örn. bulut sunucu sağlayıcısı)."
    },
    {
        "domain": "kvkk",
        "domain_tr": "Kişisel Verilerin Korunması",
        "canonical_id": "KVKK-02",
        "term": "Açık Rıza",
        "definition": "Belirli bir konuya ilişkin, bilgilendirilmeye dayanan ve kişinin özgür iradesiyle açıklanan tereddütsüz onay beyanı.",
        "bloom": "Uygulama",
        "prereq": "Aydınlatma Yükümlülüğü",
        "context": "Mobil uygulamaya üye olurken pazarlama amaçlı SMS ve bildirim gönderimi için kullanıcıya önceden işaretlenmemiş ayrı bir açık rıza onay kutucuğu sunuldu.",
        "distinction_target": "Örtülü Rıza (Zımni Kabul)",
        "distinction": "Örtülü rıza sessiz kalma veya hareketsizlikle varsayılan kabuldür ve KVKK'da geçersizdir; açık rıza ise mutlaka aktif ve özgür bir irade beyanı gerektirir."
    },
    {
        "domain": "kvkk",
        "domain_tr": "Kişisel Verilerin Korunması",
        "canonical_id": "KVKK-03",
        "term": "Özel Nitelikli Kişisel Veri",
        "definition": "Başkaları tarafından öğrenildiği takdirde ilgili kişi hakkında ayrımcılık yapılmasına yol açabilecek sağlık, biyometrik, ceza mahkumiyeti ve sendika üyeliği gibi hassas veriler.",
        "bloom": "Hatırlama",
        "prereq": "Kişisel Veri",
        "context": "İşe alım sürecinde personelin sabıka kaydı ve sağlık raporları özel nitelikli kişisel veri sınıfına girdiğinden şifreli klasörlerde kilitli tutulmalıdır.",
        "distinction_target": "Genel Kişisel Veri",
        "distinction": "Ad, soyad ve e-posta gibi genel veriler kanuni şartlarda işlenebilirken; özel nitelikli veriler için kanunda açıkça sayılan istisnalar veya ayrı açık rıza şartı aranır."
    },
    {
        "domain": "kvkk",
        "domain_tr": "Kişisel Verilerin Korunması",
        "canonical_id": "KVKK-04",
        "term": "Aydınlatma Yükümlülüğü",
        "definition": "Veri sorumlusunun kişisel verilerin hangi amaçla, kimlere aktarılabileceğini, toplama yöntemini ve ilgili kişinin haklarını veri sahibine bildirme zorunluluğu.",
        "bloom": "Anlama",
        "prereq": "Veri Sorumlusu",
        "context": "Müşteri çağrı merkezini aradığında ses kaydına başlanmadan önce aydınlatma yükümlülüğü kapsamında otomatik yasal bilgilendirme anonsu dinletilir.",
        "distinction_target": "Açık Rıza Alımı",
        "distinction": "Aydınlatma yükümlülüğü tek taraflı bir bilgilendirmedir ve veri sahibinin onayına bağlı olmaksızın her durumda zorunludur; açık rıza ise veri sahibinin rızasını gösteren iki taraflı iradedir."
    },
    {
        "domain": "kvkk",
        "domain_tr": "Kişisel Verilerin Korunması",
        "canonical_id": "KVKK-05",
        "term": "VERBİS (Veri Sorumluları Sicil Bilgi Sistemi)",
        "definition": "Veri sorumlularının işledikleri kişisel veri kategorilerini, saklama sürelerini ve veri güvenliği tedbirlerini kamuya açık şekilde kaydettikleri merkezi sicil sistemi.",
        "bloom": "Uygulama",
        "prereq": "Veri Sorumlusu",
        "context": "Yıllık çalışan sayısı 50'den fazla olan şirketimiz VERBİS kaydını tamamlayarak veri envanterini Kişisel Verileri Koruma Kurumu'na beyan etmiştir.",
        "distinction_target": "Şirket İçi Veri Envanteri",
        "distinction": "Şirket içi veri envanteri tüm ayrıntıları içeren gizli bir iç çalışma belgesiyken; VERBİS bu envanterin özet başlıklar halinde resmi sicile kamuya açık kaydedilmesidir."
    },
    {
        "domain": "kvkk",
        "domain_tr": "Kişisel Verilerin Korunması",
        "canonical_id": "KVKK-06",
        "term": "Veri İhlali Bildirimi (72 Saat Kuralı)",
        "definition": "İşlenen kişisel verilerin kanuni olmayan yollarla başkaları tarafından elde edilmesi halinde durumun en geç 72 saat içinde Kurul'a ve ilgililere bildirilmesi zorunluluğu.",
        "bloom": "Uygulama",
        "prereq": "Veri Güvenliği",
        "context": "E-ticaret veritabanında şüpheli bir sızıntı tespit edilir edilmez siber olay müdahale ekibi rapor hazırladı ve 72 saat dolmadan KVKK Kuruluna resmi ihlal bildirimi yapıldı.",
        "distinction_target": "Genel Bilgi Güvenliği Vakası",
        "distinction": "İç sistem çökmesi gibi olaylar kişisel veri sızmadıkça genel bir BT arızasıdır; veri ihlali ise kişisel verilerin yetkisiz kişilerin eline geçmesi durumudur ve yasal bildirim gerektirir."
    },
    {
        "domain": "kvkk",
        "domain_tr": "Kişisel Verilerin Korunması",
        "canonical_id": "KVKK-07",
        "term": "Kişisel Verilerin Anonim Hale Getirilmesi",
        "definition": "Kişisel verilerin başka verilerle eşleştirilse dahi hiçbir surette kimliği belirli veya belirlenebilir bir gerçek kişiyle ilişkilendirilemeyecek hale getirilmesi.",
        "bloom": "Analiz",
        "prereq": "Veri Maskeleme",
        "context": "Pazar araştırması raporunda müşteri telefon ve kimlik numaraları K-anonimlik algoritmasıyla anonimleştirilerek istatistiksel analiz yapılmıştır.",
        "distinction_target": "Rumuzlaştırma (Pseudonymization)",
        "distinction": "Rumuzlaştırmada ek bir anahtar tablo yardımıyla verinin kime ait olduğu geri çözülebilir; anonimleştirmede ise geri dönüş matematiksel olarak imkansız hale getirilir."
    },

    # ------------------ 4. KURUMSAL SİBER GÜVENLİK ------------------
    {
        "domain": "cybersecurity",
        "domain_tr": "Siber Güvenlik",
        "canonical_id": "SEC-01",
        "term": "Oltalama (Phishing)",
        "definition": "Kullanıcıları güvenilir bir kurumdan geliyormuş gibi görünen sahte e-posta veya web siteleriyle kandırarak parola ve kredi kartı bilgilerini ele geçirmeyi amaçlayan siber saldırı.",
        "bloom": "Anlama",
        "prereq": "Siber Farkındalık",
        "context": "İnsan kaynakları departmanından gönderilmiş gibi gösterilen prim anketi başlıklı sahte bağlantıya tıklayan personelin hesap şifreleri ele geçirilmeye çalışıldı.",
        "distinction_target": "Zararlı Yazılım (Malware)",
        "distinction": "Zararlı yazılım sisteme sızan teknik bir kod dosyasıdır; oltalama ise insan güvenini suistimal ederek şifre veya bilgiyi kurbana kendi eliyle girdiren sosyal mühendislik yöntemidir."
    },
    {
        "domain": "cybersecurity",
        "domain_tr": "Siber Güvenlik",
        "canonical_id": "SEC-02",
        "term": "Çok Faktörlü Kimlik Doğrulama (MFA)",
        "definition": "Kullanıcının sisteme girişinde parola haricinde SMS kodu, mobil onay uygulaması veya biyometrik parmak izi gibi en az iki bağımsız doğrulama unsuru talep eden güvenlik katmanı.",
        "bloom": "Uygulama",
        "prereq": "Parola Güvenliği",
        "context": "VPN sunucularında MFA zorunlu tutulduğu için şifresi sızdırılan bir yöneticinin hesabına saldırganlar cep telefonuna gelen onay kodunu geçemedikleri için giriş yapamadı.",
        "distinction_target": "Güçlü Parola",
        "distinction": "Güçlü parola yalnızca 'bildiğiniz' tek bir faktördür ve çalınabilir; MFA ise parolanın yanında 'sahip olduğunuz' fiziksel cihazı da şart koşarak ek güvenlik sağlar."
    },
    {
        "domain": "cybersecurity",
        "domain_tr": "Siber Güvenlik",
        "canonical_id": "SEC-03",
        "term": "Sıfır Güven Mimarisi (Zero Trust)",
        "definition": "Kurumsal ağın içinde veya dışında bulunan hiçbir kullanıcı veya cihaza varsayılan olarak güvenmeyip, her erişim isteğini sürekli doğrulayan güvenlik yaklaşımı.",
        "bloom": "Analiz",
        "prereq": "MFA",
        "context": "Şirket genelinde sıfır güven mimarisine geçilerek şirket içi yerel ağda dahi olsa kullanıcıların veritabanına erişirken mikro-bölümleme kurallarıyla tekrar doğrulanması sağlandı.",
        "distinction_target": "Geleneksel Çevre Güvenliği (Kale-Hendek Modeli)",
        "distinction": "Kale-hendek modelinde iç ağdaki herkese güvenilirken; sıfır güven yaklaşımında 'asla güvenme, daima doğrula' ilkesi gereği iç ağdaki herkes potansiyel tehdit kabul edilir."
    },
    {
        "domain": "cybersecurity",
        "domain_tr": "Siber Güvenlik",
        "canonical_id": "SEC-04",
        "term": "Fidye Yazılımı (Ransomware)",
        "definition": "Kurbanın bilgisayarındaki dosyaları güçlü algoritmalarla şifreleyerek erişimi engelleyen ve şifre çözme anahtarı karşılığında kripto para talep eden zararlı yazılım türü.",
        "bloom": "Anlama",
        "prereq": "Zararlı Yazılım",
        "context": "Muhasebe sunucusuna sızan fidye yazılımı tüm yedekleri şifrelemeye çalışırken EDR sistemi şüpheli dosya uzantısı değişimlerini fark edip sunucuyu ağdan izole etti.",
        "distinction_target": "Casus Yazılım (Spyware)",
        "distinction": "Casus yazılım fark ettirmeden gizlice klavye vuruşlarını ve bilgileri dinleyip merkeze aktarırken; fidye yazılımı doğrudan dosyaları kilitleyerek açıkça şantaj yapar."
    },
    {
        "domain": "cybersecurity",
        "domain_tr": "Siber Güvenlik",
        "canonical_id": "SEC-05",
        "term": "Sızma Testi (Pentest)",
        "definition": "Kurumun bilişim sistemlerindeki güvenlik açıklarını kötü niyetli saldırganlardan önce bulmak amacıyla yetkili etik hackerlar tarafından gerçekleştirilen kontrollü saldırı simülasyonu.",
        "bloom": "Uygulama",
        "prereq": "Zafiyet Yönetimi",
        "context": "Yıllık sızma testi kapsamında web uygulamasının yetkilendirme katmanında IDOR açığı tespit edilerek canlı ortama çıkmadan önce yazılım ekibince kapatıldı.",
        "distinction_target": "Zafiyet Tarama (Vulnerability Scan)",
        "distinction": "Zafiyet tarama otomatik araçlarla bilinen açıkları listeleyen pasif bir taramadır; sızma testi ise bulunan açıkları aktif olarak istismar edip sisteme sızmayı deneyen manuel uzmanlık sürecidir."
    },
    {
        "domain": "cybersecurity",
        "domain_tr": "Siber Güvenlik",
        "canonical_id": "SEC-06",
        "term": "SIEM (Güvenlik Bilgi ve Olay Yönetimi)",
        "definition": "Ağ cihazları, sunucular ve uygulamalardan gelen log kayıtlarını tek bir havuzda toplayıp korelasyon kurarak şüpheli siber saldırıları tespit eden izleme platformu.",
        "bloom": "Analiz",
        "prereq": "Log Yönetimi",
        "context": "SIEM sistemi aynı IP adresinden 5 dakika içinde 50 başarısız oturum açma denemesini korele ederek güvenlik operasyon merkezine (SOC) yüksek öncelikli alarm üretti.",
        "distinction_target": "Basit Log Sunucusu",
        "distinction": "Basit log sunucusu sadece verileri depolar; SIEM ise farklı kaynaklardan gelen logları çapraz eşleştirerek ve davranış kurallarını işleterek gerçek zamanlı tehdit tespiti yapar."
    },
    {
        "domain": "cybersecurity",
        "domain_tr": "Siber Güvenlik",
        "canonical_id": "SEC-07",
        "term": "Sıfır Gün Zafiyeti (Zero-Day)",
        "definition": "Yazılım geliştiricisi tarafından henüz fark edilmemiş veya bilindiği halde resmi bir güvenlik yaması yayınlanmamış olan kritik güvenlik açığı.",
        "bloom": "Analiz",
        "prereq": "Sızma Testi",
        "context": "Kullanılan tarayıcı çekirdeğinde tespit edilen sıfır gün zafiyeti nedeniyle acil durum eylem grubu oluşturuldu ve geçici koruyucu IPS imzaları devreye alındı.",
        "distinction_target": "Bilinen Zafiyet (CVE)",
        "distinction": "Bilinen zafiyet için üretici tarafından yama veya çözüm duyurulmuştur; sıfır gün açığında ise henüz hiçbir resmi yama bulunmadığından savunma çok daha güçtür."
    },

    # ------------------ 5. KURUMSAL SÜRDÜRÜLEBİLİRLİK (ESG) ------------------
    {
        "domain": "sustainability",
        "domain_tr": "Kurumsal Sürdürülebilirlik",
        "canonical_id": "ESG-01",
        "term": "Karbon Ayak İzi",
        "definition": "Bir şirketin veya bireyin faaliyetleri sonucunda doğrudan veya dolaylı olarak atmosfere salınan toplam sera gazı miktarının karbondioksit eşdeğeri (CO2e) cinsinden ölçümü.",
        "bloom": "Anlama",
        "prereq": "Sürdürülebilirlik",
        "context": "Lojistik operasyonlarımızda elektrikli araçlara geçilmesi sayesinde yıllık kurumsal karbon ayak izi değerimiz %18 oranında düşürüldü.",
        "distinction_target": "Su Ayak İzi",
        "distinction": "Karbon ayak izi iklim değişikliğine yol açan sera gazı salımlarını ölçerken; su ayak izi mal veya hizmet üretimi boyunca tüketilen tatlı su miktarını hesaplar."
    },
    {
        "domain": "sustainability",
        "domain_tr": "Kurumsal Sürdürülebilirlik",
        "canonical_id": "ESG-02",
        "term": "Kapsam 1-2-3 Emisyonları",
        "definition": "Şirketin kendi tesislerindeki doğrudan salımları (Kapsam 1), satın aldığı elektrikten kaynaklanan salımları (Kapsam 2) ve tedarik zincirindeki dolaylı salımları (Kapsam 3) sınıflandıran sera gazı protokolü.",
        "bloom": "Analiz",
        "prereq": "Karbon Ayak İzi",
        "context": "Şirketimizin toplam karbon salımının %75'inin tedarikçilerden satın alınan hammaddelerin taşınmasından kaynaklanan Kapsam 3 emisyonları olduğu raporlandı.",
        "distinction_target": "Yalnızca Şirket İçi Fosil Yakıt Tüketimi",
        "distinction": "Fosil yakıt tüketimi sadece Kapsam 1'e girerken; Kapsam 2 ve 3 satın alınan enerjiyi ve tüm tedarik zinciri ortaklarının dolaylı etkilerini de dahil eder."
    },
    {
        "domain": "sustainability",
        "domain_tr": "Kurumsal Sürdürülebilirlik",
        "canonical_id": "ESG-03",
        "term": "Yeşil Mutabakat (European Green Deal)",
        "definition": "Avrupa Birliği'nin 2050 yılına kadar kıtayı net sıfır karbonlu ilk kıta yapmayı hedefleyen, sınırda karbon vergisi ve döngüsel ekonomiyi şart koşan kapsamlı stratejisi.",
        "bloom": "Anlama",
        "prereq": "Karbon Ayak İzi",
        "context": "Avrupa'ya ihracat yapan tekstil üreticilerimiz Yeşil Mutabakat uyum kriterlerini karşılamak için yenilenebilir enerji sertifikası almaya başladı.",
        "distinction_target": "Gönüllü Sosyal Sorumluluk Projesi",
        "distinction": "Gönüllü projeler şirket isteğine bağlı reklam faaliyetleriyken; Yeşil Mutabakat sınırda karbon vergisi gibi somut ticari ve hukuki yaptırımları olan yasal bir çerçevedir."
    },
    {
        "domain": "sustainability",
        "domain_tr": "Kurumsal Sürdürülebilirlik",
        "canonical_id": "ESG-04",
        "term": "Döngüsel Ekonomi (Circular Economy)",
        "definition": "Kaynakların 'al-yap-at' doğrusal modeli yerine; ürünlerin tamir, geri dönüşüm ve yeniden kullanım yoluyla ekonomik sistemde mümkün olan en uzun süre tutulması modeli.",
        "bloom": "Uygulama",
        "prereq": "Sürdürülebilirlik",
        "context": "Fabrikamızda üretim fireleri tekrar eritilerek hammaddeye dönüştürülmüş ve döngüsel ekonomi prensibiyle atık miktarı sıfıra yaklaştırılmıştır.",
        "distinction_target": "Geleneksel Doğrusal Ekonomi",
        "distinction": "Doğrusal ekonomide hammadde tüketilip kullanım sonrası çöpe atılırken; döngüsel ekonomide atık yeni bir ürünün girdisi olarak sürekli dolaşımda tutulur."
    },
    {
        "domain": "sustainability",
        "domain_tr": "Kurumsal Sürdürülebilirlik",
        "canonical_id": "ESG-05",
        "term": "GRI Standartları (Küresel Raporlama Girişimi)",
        "definition": "Şirketlerin çevresel, sosyal ve ekonomik etkilerini şeffaf, karşılaştırılabilir ve denetlenebilir biçimde kamuoyuna açıklamalarını sağlayan evrensel sürdürülebilirlik raporlama çerçevesi.",
        "bloom": "Uygulama",
        "prereq": "ESG Kriterleri",
        "context": "Yıllık faaliyet raporumuz GRI Standartları Temel (Core) seçeneğine göre hazırlanarak bağımsız denetimden geçmiş ve Borsa İstanbul sürdürülebilirlik endeksine girmiştir.",
        "distinction_target": "Mali Gelir Tablosu",
        "distinction": "Mali tablo yalnızca şirketin parasal kar-zarar durumunu gösterirken; GRI raporu şirketin doğaya, çalışan haklarına ve topluma olan çok boyutlu etkilerini raporlar."
    },
    {
        "domain": "sustainability",
        "domain_tr": "Kurumsal Sürdürülebilirlik",
        "canonical_id": "ESG-06",
        "term": "Sınırda Karbon Düzenleme Mekanizması (SKDM / CBAM)",
        "definition": "AB dışından ithal edilen demir-çelik, çimento, alüminyum gibi yüksek karbonlu ürünlerden üretim esnasındaki sera gazı emisyonlarına göre gümrükte karbon vergisi alınması mekanizması.",
        "bloom": "Analiz",
        "prereq": "Yeşil Mutabakat",
        "context": "Çimento fabrikamız SKDM kapsamında gümrükte ek karbon vergisi ödememek için fırınlarda atık ısı geri kazanımı yatırımı gerçekleştirdi.",
        "distinction_target": "Geleneksel Gümrük Tarifesi",
        "distinction": "Geleneksel tarifeler malın türü ve fiyatı üzerinden sabit oranda alınırken; SKDM doğrudan üretimin doğaya saldığı sera gazı tonajı üzerinden hesaplanan bir iklim vergisidir."
    },

    # ------------------ 6. BANKACILIK & FİNANS ------------------
    {
        "domain": "banking",
        "domain_tr": "Bankacılık & Finans",
        "canonical_id": "BNK-01",
        "term": "Kredi Kayıt Bürosu (KKB) ve Findeks Notu",
        "definition": "Bireylerin ve işletmelerin geçmiş kredi, kredi kartı ve kredili mevduat ödeme alışkanlıklarına göre 1 ile 1900 puan arasında hesaplanan finansal güvenilirlik skoru.",
        "bloom": "Anlama",
        "prereq": "Kredi Tahsis",
        "context": "Ticari kredi başvurusunda bulunan şirketin Findeks kredi notu 1750 olarak tespit edildiğinden kredi komitesi faiz indirimi onayladı.",
        "distinction_target": "Aylık Net Maaş / Gelir",
        "distinction": "Aylık gelir ne kadar para kazanıldığını gösterir; Findeks notu ise kazanılan parayla geçmiş borçların ne kadar düzenli ödendiğinin ahlaki ve istatistiki göstergesidir."
    },
    {
        "domain": "banking",
        "domain_tr": "Bankacılık & Finans",
        "canonical_id": "BNK-02",
        "term": "MASAK Uyum ve Şüpheli İşlem Bildirimi",
        "definition": "Mali Suçları Araştırma Kurulu mevzuatına göre kara para aklama ve terörizmin finansmanını önlemek amacıyla olağandışı para hareketlerinin resmi makamlara zorunlu bildirimi.",
        "bloom": "Uygulama",
        "prereq": "KYC (Müşterini Tanı)",
        "context": "Düzenli geliri olmayan bir hesaba birdenbire yurt dışından yüklü miktarda parçalı para transferi gelmesi üzerine MASAK uyum görevlisi şüpheli işlem bildirimi düzenledi.",
        "distinction_target": "Normal Havale Dekontu",
        "distinction": "Havale dekontu müşteri için yapılan işlemin sıradan bir makbuzudur; şüpheli işlem bildirimi ise müşteriye hissettirilmeden gizlice devlete iletilen denetim raporudur."
    },
    {
        "domain": "banking",
        "domain_tr": "Bankacılık & Finans",
        "canonical_id": "BNK-03",
        "term": "Likidite Karşılama Oranı (LCR)",
        "definition": "Bankanın 30 günlük ciddi bir kriz ve stres senaryosunda ortaya çıkabilecek net nakit çıkışlarını karşılamaya yetecek yüksek kaliteli likit varlık stoğuna sahip olma zorunluluğu rasyosu.",
        "bloom": "Analiz",
        "prereq": "Aktif-Pasif Yönetimi",
        "context": "BDDK denetiminde bankanın konsolide likidite karşılama oranının %120 düzeyinde kalarak yasal asgari sınırın üzerinde olduğu onaylandı.",
        "distinction_target": "Kredi Mevduat Oranı",
        "distinction": "Kredi-mevduat oranı mevduatın ne kadarının krediye dönüştüğünü ölçer; LCR ise acil kriz anında bankanın batmasını engelleyecek nakit tamponunun yeterliliğini test eder."
    },
    {
        "domain": "banking",
        "domain_tr": "Bankacılık & Finans",
        "canonical_id": "BNK-04",
        "term": "Munzam Karşılık (Zorunlu Karşılık)",
        "definition": "Bankaların topladıkları mevduatların Merkez Bankası tarafından belirlenen belirli bir yüzdesini TCMB nezdindeki hesaplarında nakden bloke tutma yükümlülüğü.",
        "bloom": "Anlama",
        "prereq": "Mevduat Yönetimi",
        "context": "Merkez Bankası piyasadaki fazla likiditeyi çekmek amacıyla Türk lirası mevduatlar için munzam karşılık oranlarını 200 baz puan artırdı.",
        "distinction_target": "Banka Sermayesi",
        "distinction": "Sermaye banka ortaklarının koyduğu özkaynaktır; munzam karşılık ise müşterilerden toplanan mevduatlardan zorunlu olarak Merkez Bankası'na yatırılan güvence payıdır."
    },
    {
        "domain": "banking",
        "domain_tr": "Bankacılık & Finans",
        "canonical_id": "BNK-05",
        "term": "Sermaye Yeterlilik Standart Rasyosu (SYR)",
        "definition": "Bankanın sahip olduğu özkaynakların, maruz kaldığı kredi, piyasa ve operasyonel risklerin toplamına oranını ifade eden temel sağlamlık göstergesi.",
        "bloom": "Analiz",
        "prereq": "Risk Ağırlıklı Varlıklar",
        "context": "Basel standartlarına göre asgari %8 olması gereken SYR oranı bankamızda %17.4 olarak gerçekleşerek güçlü bilanço yapısını kanıtlamıştır.",
        "distinction_target": "Dönem Net Kârı",
        "distinction": "Kâr o yıl kazanılan gelirin giderden farkıdır; sermaye yeterlilik rasyosu ise gelecekteki batık kredilere ve şoklara karşı bankanın ne kadar dayanıklı olduğunu ölçer."
    },
    {
        "domain": "banking",
        "domain_tr": "Bankacılık & Finans",
        "canonical_id": "BNK-06",
        "term": "FAST (Fonların Anlık ve Sürekli Transferi)",
        "definition": "Farklı bankalardaki hesaplar arasında 7/24 kesintisiz, saniyeler içinde para transferi sağlayan ve Kolay Adresleme Sistemiyle çalışan anlık ödeme altyapısı.",
        "bloom": "Uygulama",
        "prereq": "EFT Sistemi",
        "context": "Müşterilerimiz mesai saatleri ve hafta sonu kısıtı olmaksızın cep telefonu numarasına tanımlı FAST sistemiyle anında para transferi yapabilmektedir.",
        "distinction_target": "Geleneksel EFT",
        "distinction": "EFT yalnızca hafta içi mesai saatlerinde çalışır ve seans aralıklarıyla gecikmeli iletilir; FAST ise 7/24 saniyeler içinde anında gerçekleşir."
    },
    {
        "domain": "banking",
        "domain_tr": "Bankacılık & Finans",
        "canonical_id": "BNK-07",
        "term": "Müşterini Tanı (KYC - Know Your Customer)",
        "definition": "Bankaların hesap açılışı veya finansal işlem öncesinde müşterinin kimlik bilgilerini, iş adresini, imza sirkülerini ve vergi kaydını resmi belgelerle doğrulama süreci.",
        "bloom": "Uygulama",
        "prereq": "MASAK Mevzuatı",
        "context": "Görüntülü görüşmeyle dijital hesap açılışında çipli kimlik kartı NFC okuyucuyla taranarak KYC kimlik teyidi tamamlandı.",
        "distinction_target": "Kredi İstihbaratı",
        "distinction": "KYC kişinin gerçekte kim olduğunu ve yasal varlığını teyit eder; kredi istihbaratı ise kimliği doğrulanan kişinin borcunu ödeme gücünü ve finansal geçmişini inceler."
    },
    # --- EK TURİZM KONSEPTLERİ (Tatilsepeti Ar-Ge) ---
    {
        "domain": "tourism",
        "domain_tr": "Turizm & Otelcilik",
        "canonical_id": "TOUR-08",
        "term": "RevPAR (Mevcut Oda Başına Gelir)",
        "definition": "Toplam oda gelirinin tesisteki toplam satılabilir oda sayısına bölünmesiyle elde edilen ana gelir yönetimi göstergesi.",
        "bloom": "Analiz",
        "prereq": "Doluluk Oranı",
        "context": "Otel yönetim kurulu toplantısında doluluk oranı %90'a çıksa da ortalama oda fiyatı düştüğü için RevPAR değerinin hedefin altında kaldığı vurgulandı.",
        "distinction_target": "ADR (Ortalama Günlük Fiyat)",
        "distinction": "ADR sadece fiilen satılan odaların ortalamasıdır; RevPAR ise boş kalan odaları da dahil ederek tüm tesis kapasitesinin getirisini ölçer."
    },
    {
        "domain": "tourism",
        "domain_tr": "Turizm & Otelcilik",
        "canonical_id": "TOUR-09",
        "term": "Her Şey Dahil (All-Inclusive)",
        "definition": "Konaklama, ana ve ara öğünler ile tesis içi temel içecek ve aktivitelerin tek bir paket fiyata dahil olduğu konaklama modeli.",
        "bloom": "Anlama",
        "prereq": "Voucher",
        "context": "Tatilsepeti acente portalında filtreleme yapan ailelerin %70'inin tatil bütçesini sabitlemek için ultra her şey dahil resort otelleri tercih ettiği belirlendi.",
        "distinction_target": "Yarım Pansiyon (Half Board)",
        "distinction": "Yarım pansiyon yalnızca sabah kahvaltısı ve akşam yemeğini kapsar; her şey dahil konseptinde gün boyu yiyecek ve içecekler fiyata dahildir."
    },
    {
        "domain": "tourism",
        "domain_tr": "Turizm & Otelcilik",
        "canonical_id": "TOUR-10",
        "term": "Acente Komisyonu",
        "definition": "Seyahat acentesinin otel veya havayolu ürünlerini müşteriye satması karşılığında satış bedeli üzerinden hak kazandığı sözleşmeye bağlı yüzde gelir payı.",
        "bloom": "Uygulama",
        "prereq": "Voucher",
        "context": "Otel ile yapılan yıllık kontratta acentemize erken rezervasyon satışları için %15 net acente komisyonu tanımlandı.",
        "distinction_target": "Müşteri Hizmet Bedeli",
        "distinction": "Acente komisyonu hizmet sağlayıcı otel tarafından acenteye ödenir; hizmet bedeli ise doğrudan son tüketiciden işlem başına tahsil edilir."
    },
    # --- EK İSG KONSEPTLERİ ---
    {
        "domain": "isg",
        "domain_tr": "İş Sağlığı ve Güvenliği",
        "canonical_id": "ISG-08",
        "term": "İş Kazası Bildirimi",
        "definition": "İşyerinde veya işin yürütümü sırasında meydana gelen ve bedensel veya ruhsal zarara yol açan olayın en geç 3 iş günü içinde SGK'ya resmi sistemden bildirilmesi.",
        "bloom": "Uygulama",
        "prereq": "Ramak Kala Olayı",
        "context": "Bakım atölyesinde parmağı kesilen teknisyene ilk müdahale yapıldıktan sonra İSG uzmanı tarafından tutanak düzenlenip SGK sistemine iş kazası bildirimi yapıldı.",
        "distinction_target": "Ramak Kala Bildirimi",
        "distinction": "Ramak kala iç şirket takibi için formlanırken; iş kazası yasal zorunluluk olarak 3 iş günü içinde doğrudan Sosyal Güvenlik Kurumu'na bildirilmek zorundadır."
    },
    {
        "domain": "isg",
        "domain_tr": "İş Sağlığı ve Güvenliği",
        "canonical_id": "ISG-09",
        "term": "Meslek Hastalığı",
        "definition": "Sigortalının çalıştığı veya yaptığı işin niteliğinden dolayı tekrarlanan bir sebeple veya işin yürütüm şartları yüzünden uğradığı geçici veya sürekli hastalık ve engellilik hali.",
        "bloom": "Analiz",
        "prereq": "Ortam Ölçümü",
        "context": "Boya atölyesinde 10 yıldır çalışan ustada solunum fonksiyon testi bozulması üzerine durum meslek hastalıkları hastanesine sevk edilerek incelendi.",
        "distinction_target": "İş Kazası",
        "distinction": "İş kazası aniden meydana gelen tekil bir travmatik olaydır; meslek hastalığı ise zararlı etkenlere uzun süre maruziyet sonucu sinsi ve zamana yayılarak gelişir."
    },
    {
        "domain": "isg",
        "domain_tr": "İş Sağlığı ve Güvenliği",
        "canonical_id": "ISG-10",
        "term": "İş Güvenliği Kültürü",
        "definition": "Bir kurumdaki tüm çalışanların ve yöneticilerin güvenliği bir kural zorunluluğu olarak değil, ortak bir değer ve davranış biçimi olarak içselleştirmesi durumu.",
        "bloom": "Anlama",
        "prereq": "Risk Değerlendirmesi",
        "context": "Fabrikamızda güvenlik kültürünü geliştirmek amacıyla yöneticilerin de katıldığı haftalık 5 dakikalık iş başı güvenlik konuşmaları (toolbox) başlatıldı.",
        "distinction_target": "Yasal Mevzuat Uyumu",
        "distinction": "Mevzuat uyumu ceza almamak için asgari kanun maddelerini yerine getirmektir; güvenlik kültürü ise kanun zorlamasa bile çalışanların birbirinin canını koruduğu proaktif bilinçtir."
    },
    # --- EK KVKK KONSEPTLERİ ---
    {
        "domain": "kvkk",
        "domain_tr": "Kişisel Verilerin Korunması",
        "canonical_id": "KVKK-08",
        "term": "Veri Maskeleme",
        "definition": "Kişisel verilerin yetkisiz gözlerden saklanması amacıyla kredi kartının ilk 6 ve son 4 hanesi dışındaki rakamların yıldızlanması gibi geri döndürülebilir gizleme tekniği.",
        "bloom": "Uygulama",
        "prereq": "Veri Güvenliği",
        "context": "Çağrı merkezi ekranlarında müşteri temsilcilerinin ekranına gelen T.C. kimlik numaraları veri maskeleme yöntemiyle kısmen yıldızlı olarak gösterilmektedir.",
        "distinction_target": "Kişisel Veriyi Anonimleştirme",
        "distinction": "Maskeleme yetkili sistemlerde orijinal veriyi arkada saklayıp ekranda gizler; anonimleştirmede ise verinin kişiyle olan tüm bağı kalıcı olarak geri dönülemez biçimde koparılır."
    },
    {
        "domain": "kvkk",
        "domain_tr": "Kişisel Verilerin Korunması",
        "canonical_id": "KVKK-09",
        "term": "İlgili Kişi Hakları (Madde 11)",
        "definition": "Verisi işlenen gerçek kişinin verisinin işlenip işlenmediğini öğrenme, düzeltme, silinmesini isteme ve kanuna aykırı işlemede zararın giderilmesini talep etme hakkı.",
        "bloom": "Anlama",
        "prereq": "Veri Sorumlusu",
        "context": "Eski bir müşterimiz KVKK 11. madde kapsamında şirketimize yazılı başvuruda bulunarak kayıtlı e-posta ve adres verilerinin sistemlerimizden silinmesini talep etti.",
        "distinction_target": "Açık Rıza Beyanı",
        "distinction": "Açık rıza veri işlenirken verilir; ilgili kişi hakları ise verisi işlenen kişinin süreç boyunca ve sonrasında verisini denetleme ve sildirme hukuki yetkisidir."
    },
    {
        "domain": "kvkk",
        "domain_tr": "Kişisel Verilerin Korunması",
        "canonical_id": "KVKK-10",
        "term": "Saklama ve İmha Politikası",
        "definition": "Kişisel verilerin hangi yasal sürelerle saklanacağını ve süresi dolan verilerin periyodik olarak hangi teknik yöntemlerle silineceğini belirleyen kurumsal yönetmelik.",
        "bloom": "Uygulama",
        "prereq": "VERBİS Kaydı",
        "context": "Personel özlük dosyaları kanuni saklama süresi olan 10 yıl dolduktan sonra saklama ve imha politikamız uyarınca güvenli şekilde kıyılarak fiziksel imhası yapıldı.",
        "distinction_target": "Aydınlatma Metni",
        "distinction": "Aydınlatma metni dışarıdaki veri sahibine yapılan bilgilendirmedir; imha politikası ise şirket içinde personelin veriyi ne zaman ve nasıl yok edeceğini belirten iç rehberdir."
    },
    # --- EK SİBER GÜVENLİK KONSEPTLERİ ---
    {
        "domain": "cybersecurity",
        "domain_tr": "Siber Güvenlik",
        "canonical_id": "SEC-08",
        "term": "Sosyal Mühendislik",
        "definition": "Teknik açıkları kullanmak yerine; insan psikolojisindeki zaafları, aciliyet hissini veya otoriteye itaati suistimal ederek gizli bilgileri sızdırma sanatı.",
        "bloom": "Analiz",
        "prereq": "Oltalama (Phishing)",
        "context": "Kendisini genel müdür yardımcısı olarak tanıtıp acil şifre sıfırlaması talep eden sahte bir arama karşısında bilgi işlem teknisyeni sosyal mühendislik şüphesiyle teyit mekanizmasını işletti.",
        "distinction_target": "Kaba Kuvvet Saldırısı (Brute Force)",
        "distinction": "Kaba kuvvet saldırısı bilgisayar algoritmalarıyla milyonlarca şifreyi otomatik dener; sosyal mühendislik ise doğrudan insanı kandırarak şifreyi kendi ağzından alır."
    },
    {
        "domain": "cybersecurity",
        "domain_tr": "Siber Güvenlik",
        "canonical_id": "SEC-09",
        "term": "Dağıtık Hizmet Engelleme (DDoS)",
        "definition": "Zombi bilgisayar ağları (botnet) kullanılarak hedeflenen web sunucusuna kapasitesinin üzerinde yapay trafik gönderilip sistemin kilitlenmesi saldırısı.",
        "bloom": "Anlama",
        "prereq": "Ağ Güvenliği",
        "context": "Tatilsepeti kampanya gününde gelen şüpheli yurt dışı istek dalgası cloud DDoS koruma katmanı tarafından filtrelenerek sitenin kesintisiz hizmet vermesi sağlandı.",
        "distinction_target": "Sistem Hackleme / Veri Tabanı Sızıntısı",
        "distinction": "Veri tabanı sızıntısında bilgi çalınır; DDoS saldırısında ise hiçbir bilgi çalınmaz, yalnızca sistemin meşru kullanıcılara hizmet veremeyecek şekilde çökmesi amaçlanır."
    },
    {
        "domain": "cybersecurity",
        "domain_tr": "Siber Güvenlik",
        "canonical_id": "SEC-10",
        "term": "EDR (Uç Nokta Tespiti ve Yanıt)",
        "definition": "Çalışan bilgisayarları ve sunuculardaki dosya, işlem ve bellek hareketlerini sürekli kaydedip gelişmiş tehditleri davranış analitiğiyle engelleyen güvenlik yazılımı.",
        "bloom": "Uygulama",
        "prereq": "Antivirüs",
        "context": "Bellekte sessizce çalışan Powershell komut dosyasını geleneksel antivirüs kaçırırken, EDR yazılımı sıfır gün şüphesiyle süreci dondurup güvenlik merkezine bildirdi.",
        "distinction_target": "Geleneksel İmza Bazlı Antivirüs",
        "distinction": "Antivirüs yalnızca daha önce bilinen virüs dosyalarını tanır; EDR ise dosyasız saldırıları ve anormallikleri şüpheli davranış paterni üzerinden yakalar."
    },
    # --- EK SÜRDÜRÜLEBİLİRLİK KONSEPTLERİ ---
    {
        "domain": "sustainability",
        "domain_tr": "Kurumsal Sürdürülebilirlik",
        "canonical_id": "ESG-07",
        "term": "Su Ayak İzi",
        "definition": "Bir mal veya hizmetin üretiminden tüketimine kadar geçen süreçte doğrudan tüketilen ve kirletilen tatlı su hacminin toplam miktarı.",
        "bloom": "Anlama",
        "prereq": "Karbon Ayak İzi",
        "context": "Tekstil fabrikamızda boyahane atık sularının membran filtreyle geri kazanılması projesi sayesinde ürün başına su ayak izi %40 oranında azaltıldı.",
        "distinction_target": "Karbon Ayak İzi",
        "distinction": "Karbon ayak izi atmosfere salınan sera gazlarını ölçer; su ayak izi ise gezegendeki tatlı su kaynaklarının tüketim ve kirlilik miktarını hesaplar."
    },
    {
        "domain": "sustainability",
        "domain_tr": "Kurumsal Sürdürülebilirlik",
        "canonical_id": "ESG-08",
        "term": "Yaşam Döngüsü Değerlendirmesi (LCA)",
        "definition": "Bir ürünün hammaddeden başlayarak üretim, dağıtım, kullanım ve ömrünü tamamlama aşamalarındaki tüm çevresel etkilerini beşikten mezara ölçen metodoloji.",
        "bloom": "Analiz",
        "prereq": "Döngüsel Ekonomi",
        "context": "Yeni ambalaj tasarımı için ISO 14040 standardına uygun LCA analizi yaptırılarak biyobozunur malzemenin çevresel yükünün plastik muadiline göre yarı yarıya düşük olduğu kanıtlandı.",
        "distinction_target": "Basit Karbon Emisyonu Ölçümü",
        "distinction": "Karbon ölçümü sadece salımı hesaplar; LCA ise asitlenme, ozon incelmesi, su tüketimi ve ötrofikasyon gibi tüm ekolojik etkileri bütüncül modeller."
    },
    {
        "domain": "sustainability",
        "domain_tr": "Kurumsal Sürdürülebilirlik",
        "canonical_id": "ESG-09",
        "term": "Sıfır Atık Hiyerarşisi",
        "definition": "Atık yönetiminde öncelikle atığın oluşmasını önleme, ardından yeniden kullanım, geri dönüşüm ve en son çare olarak bertarafı esas alan piramidal yaklaşım.",
        "bloom": "Uygulama",
        "prereq": "Döngüsel Ekonomi",
        "context": "Ofislerimizde tek kullanımlık plastik bardaklar kaldırılarak cam termos uygulamasına geçilmiş ve sıfır atık hiyerarşisinin en üst basamağı olan kaynağında önleme hayata geçirilmiştir.",
        "distinction_target": "Sadece Çöpleri Ayrıştırma",
        "distinction": "Çöpleri ayrıştırmak atık oluştuktan sonraki geri dönüşümdür; sıfır atık hiyerarşisi ise ilk öncelik olarak atığın hiç ortaya çıkmamasını sağlayacak tasarımı zorunlu kılar."
    },
    # --- EK BANKACILIK KONSEPTLERİ ---
    {
        "domain": "banking",
        "domain_tr": "Bankacılık & Finans",
        "canonical_id": "BNK-08",
        "term": "Swap Sözleşmesi (Takas İşlemi)",
        "definition": "İki tarafın belirli bir süre boyunca farklı para birimlerindeki veya faiz türlerindeki nakit akışlarını karşılıklı değiştirmesini sağlayan türev finansal sözleşme.",
        "bloom": "Analiz",
        "prereq": "Döviz Kuru Riski",
        "context": "Döviz cinsinden borcu olan şirketimiz kur riskinden korunmak amacıyla bankayla 1 yıl vadeli para swapı sözleşmesi imzalayarak ödemelerini TL'ye sabitledi.",
        "distinction_target": "Spot Döviz Alım-Satımı",
        "distinction": "Spot işlemde para hemen ve kesin olarak el değiştirir; swap işleminde ise vadede paraların önceden anlaşılan kurdan geri takası taahhüt edilir."
    },
    {
        "domain": "banking",
        "domain_tr": "Bankacılık & Finans",
        "canonical_id": "BNK-09",
        "term": "Sendikasyon Kredisi",
        "definition": "Büyük tutarlı ve tek bir bankanın risk limitini aşan dev projeler veya dış borç çevrimleri için birden fazla uluslararası bankanın bir araya gelerek açtığı ortak kredi.",
        "bloom": "Uygulama",
        "prereq": "Kredi Tahsis",
        "context": "Bankamız uluslararası 35 bankanın katılımıyla 600 milyon dolarlık sürdürülebilirlik bağlantılı sendikasyon kredisini başarıyla yeniledi.",
        "distinction_target": "Bireysel İhtiyaç Kredisi",
        "distinction": "Bireysel kredi tek bir bankanın küçük tutarda vatandaşa verdiği kredidir; sendikasyon ise yüz milyonlarca dolarlık küresel bir bankalar konsorsiyumu finansmanıdır."
    },
    {
        "domain": "banking",
        "domain_tr": "Bankacılık & Finans",
        "canonical_id": "BNK-10",
        "term": "Donuk Alacaklar (NPL)",
        "definition": "Geri ödeme vadesi üzerinden 90 günden fazla süre geçmiş ve banka tarafından tahsilatı şüpheli veya batık olarak sınıflandırılarak karşılık ayrılan krediler.",
        "bloom": "Analiz",
        "prereq": "Sermaye Yeterlilik Rasyosu",
        "context": "Bankacılık sektöründe takipteki alacaklar oranı (NPL) riskli sektör kredilerinin yapılandırılmasıyla %1.8 seviyesine kadar geriledi.",
        "distinction_target": "Canlı Kredi",
        "distinction": "Canlı krediler anapara ve faiz taksitleri gününde veya en geç 30 gün içinde ödenen sağlıklı kredilerdir; NPL ise 90 günü aşan tahsilatsız borçtur."
    }
]

PROMPT_VARIANTS_EXTRACTION = [
    "Aşağıdaki {domain_tr} eğitim konuşması transkriptinden teknik terimleri tespit et, CEFR B1 düzeyinde sade bir dille açıkla ve Bloom taksonomi düzeyini ata. Sonucu geçerli JSON olarak ver.",
    "Verilen {domain_tr} kurumsal metnini analiz et. İçindeki anahtar terimleri, kanonik kimliklerini, sade açıklamalarını ve öğrenme basamaklarını (Bloom) JSON formatında listele.",
    "Aşağıdaki metinden {domain_tr} alanına özgü kavramları ayıkla ve her biri için başlangıç seviyesindeki bir çalışanın anlayabileceği B1 tanımları ve Bloom seviyelerini içeren JSON nesnesi oluştur.",
    "Kurumsal {domain_tr} transkriptinde geçen uzmanlık terimlerini belirle. Her kavram için geçerli JSON şemasında canonical_id, term, definition ve bloom_level alanlarını doldur."
]

PROMPT_VARIANTS_PREREQ = [
    "Aşağıdaki {domain_tr} kavramı için öğrenme yol haritasında bilinmesi gereken önkoşul (prerequisite) kavramı belirt ve aralarındaki pedagojik ilişkiyi açıkla:\nKavram: {term}",
    "Bir çalışanın '{term}' konusunu kavramsal olarak tam anlayabilmesi için öncelikle hangi temel {domain_tr} kavramını öğrenmiş olması gerekir? Gerekçesiyle açıkla.",
    "{domain_tr} eğitim müfredatında '{term}' konusuna geçilmeden önce anlatılması gereken öncül kavram nedir ve bu iki terim hangi mantıkla birbirine bağlanır?"
]

PROMPT_VARIANTS_DISTINCTION = [
    "Kurumsal {domain_tr} alanında sıklıkla birbirine karıştırılan '{term}' ile '{distinction_target}' arasındaki temel farkları açık ve net biçimde açıkla.",
    "Yeni başlayan bir çalışana '{term}' ve '{distinction_target}' kavramlarının ayrımını en sade şekliyle nasıl izah edersin? Karşılaştırmalı olarak anlat.",
    "Aşağıdaki iki {domain_tr} kavramı arasındaki farkı belirt:\n1. {term}\n2. {distinction_target}"
]

PROMPT_VARIANTS_SCENARIO = [
    "Aşağıdaki kurumsal senaryoda geçen {domain_tr} durumunu incele ve uygulanması gereken standart operasyonel adımı teknik gerekçesiyle açıkla:\nSenaryo: {context}",
    "Bir kurumda şu olay yaşandığında: \"{context}\"\nBurada işletilmesi gereken {domain_tr} kavramı nedir ve hangi kural devreye girer?",
    "Verilen işyeri vaka analizini değerlendir: \"{context}\"\nBu durum hangi {domain_tr} terimiyle ifade edilir ve sürecin doğru yönetimi nasıl olmalıdır?"
]


def build_domain_samples(guard: LeakageGuard) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    for item in CONCEPTS_DATA:
        domain = item["domain"]
        domain_tr = item["domain_tr"]
        term = item["term"]
        cid = item["canonical_id"]
        definition = item["definition"]
        bloom = item["bloom"]
        prereq = item["prereq"]
        context = item["context"]
        dist_target = item["distinction_target"]
        dist_text = item["distinction"]

        if guard.term_is_eval_answer(term) or guard.text_is_eval(context):
            continue

        # 1. Extraction Task
        output_json = json.dumps({
            "domain": domain_tr,
            "terms": [
                {
                    "term": term,
                    "canonical_id": cid,
                    "definition": definition,
                    "bloom_level": bloom
                }
            ]
        }, ensure_ascii=False)

        for template in PROMPT_VARIANTS_EXTRACTION:
            instr = template.format(domain_tr=domain_tr)
            samples.append({
                "instruction": instr,
                "input": context,
                "output": output_json,
                "domain": domain,
                "task_type": "extraction"
            })

        # 2. Prerequisite Reasoning Task
        prereq_answer = (
            f"'{term}' konusunun tam ve doğru anlaşılabilmesi için önkoşul olarak '{prereq}' kavramının bilinmesi gerekir. "
            f"Çünkü '{prereq}' öğrenilmeden '{term}' sürecinin işleyişi ve oluşturduğu sonuçlar kavranamaz. "
            f"Pedagojik DAG hiyerarşisinde: {prereq} -> {term} yönlü bağıntısı bulunmaktadır."
        )
        for p_template in PROMPT_VARIANTS_PREREQ:
            instr = p_template.format(domain_tr=domain_tr, term=term)
            samples.append({
                "instruction": instr,
                "input": "",
                "output": prereq_answer,
                "domain": domain,
                "task_type": "prerequisite"
            })

        # 3. Distinction Task
        dist_answer = (
            f"Kurumsal {domain_tr} kapsamında '{term}' ve '{dist_target}' kavramları sıkça karıştırılsa da aralarında belirgin farklar vardır:\n\n"
            f"- **{term}**: {definition}\n"
            f"- **{dist_target}**: Bu kavram tamamlayıcı veya farklı bir amacı hedefler.\n\n"
            f"Temel Ayrım: {dist_text}"
        )
        for d_template in PROMPT_VARIANTS_DISTINCTION:
            instr = d_template.format(domain_tr=domain_tr, term=term, distinction_target=dist_target)
            samples.append({
                "instruction": instr,
                "input": "",
                "output": dist_answer,
                "domain": domain,
                "task_type": "distinction"
            })

        # 4. Contextual Scenario QA Task
        scenario_answer = (
            f"Bu senaryoda işletilmesi gereken temel kavram **{term}** ({cid}) sürecidir. "
            f"Bilişsel düzeyi '{bloom}' olan bu uygulamada; {definition} kuralı geçerlidir. "
            f"Somut vakada yapılması gereken işlem, ilgili yönerge uyarınca sürecin standart protokole göre kayıt altına alınmasıdır."
        )
        for s_template in PROMPT_VARIANTS_SCENARIO:
            instr = s_template.format(domain_tr=domain_tr, term=term, context=context)
            samples.append({
                "instruction": instr,
                "input": context,
                "output": scenario_answer,
                "domain": domain,
                "task_type": "scenario_qa"
            })

    # Multi-term coordination dialogues
    domain_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in CONCEPTS_DATA:
        domain_groups.setdefault(item["domain"], []).append(item)

    for dom, items in domain_groups.items():
        if len(items) < 2:
            continue
        dom_tr = items[0]["domain_tr"]
        for i in range(len(items)):
            c1 = items[i]
            c2 = items[(i + 1) % len(items)]
            multi_context = (
                f"Şirket içi {dom_tr} koordinasyon toplantısında şu konular görüşüldü: "
                f"Öncelikle {c1['term'].lower()} sürecinin mevcut durumu ele alındı; "
                f"burada {c1['context'].lower()} Ardından gündemin ikinci maddesi olarak "
                f"{c2['term'].lower()} uygulamasına geçildi ve {c2['context'].lower()} "
                f"Tüm yöneticilerin bu iki süreci birbiriyle entegre şekilde yürütmesi kararlaştırıldı."
            )
            multi_output = json.dumps({
                "domain": dom_tr,
                "terms": [
                    {
                        "term": c1["term"],
                        "canonical_id": c1["canonical_id"],
                        "definition": c1["definition"],
                        "bloom_level": c1["bloom"]
                    },
                    {
                        "term": c2["term"],
                        "canonical_id": c2["canonical_id"],
                        "definition": c2["definition"],
                        "bloom_level": c2["bloom"]
                    }
                ]
            }, ensure_ascii=False)

            for tpl in PROMPT_VARIANTS_EXTRACTION[:3]:
                samples.append({
                    "instruction": tpl.format(domain_tr=dom_tr),
                    "input": multi_context,
                    "output": multi_output,
                    "domain": dom,
                    "task_type": "multi_term_extraction"
                })

    logger.info("Generated %d domain-specific instruction samples across 6 sectors.", len(samples))
    return samples


def load_external_turkish_samples(limit: int = 1600) -> List[Dict[str, Any]]:
    ext_file = EXTERNAL_DIR / "train_external_mixed.jsonl"
    if not ext_file.exists():
        logger.warning("External mixed file not found at %s", ext_file)
        return []

    external_samples: List[Dict[str, Any]] = []
    with open(ext_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            instr = row.get("instruction", "")
            out = row.get("output", "")
            if len(instr) > 10 and len(out) > 10:
                external_samples.append({
                    "instruction": instr,
                    "input": row.get("input", ""),
                    "output": out,
                    "domain": "general_turkish",
                    "task_type": "general_qa"
                })
            if len(external_samples) >= limit:
                break

    logger.info("Loaded %d general Turkish instruction samples from external corpus.", len(external_samples))
    return external_samples


def load_lecture_stem_samples() -> List[Dict[str, Any]]:
    lec_file = DATA_DIR / "train_lecture_notes.jsonl"
    if not lec_file.exists():
        return []
    samples: List[Dict[str, Any]] = []
    with open(lec_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["task_type"] = "academic_lecture"
                samples.append(row)
    logger.info("Loaded %d academic lecture note samples.", len(samples))
    return samples


def generate_extended_dataset() -> Tuple[Path, Path]:
    guard = LeakageGuard()

    # 1. Domain samples (6 sectors)
    domain_samples = build_domain_samples(guard)

    # 2. General Turkish instruction samples
    general_samples = load_external_turkish_samples(limit=1600)

    # 3. Academic STEM lecture samples
    stem_samples = load_lecture_stem_samples()

    # Combine all
    all_samples = domain_samples + general_samples + stem_samples
    random.seed(42)
    random.shuffle(all_samples)

    logger.info("Total combined samples before split: %d", len(all_samples))

    # 90% train, 10% validation split
    split_idx = int(len(all_samples) * 0.90)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    out_train = DATA_DIR / "train_instruct_domain_full.jsonl"
    out_val = DATA_DIR / "val_instruct_domain_full.jsonl"

    with open(out_train, "w", encoding="utf-8") as f:
        for item in train_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(out_val, "w", encoding="utf-8") as f:
        for item in val_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("==========================================================")
    logger.info("Domain SFT Full Dataset Generation Complete:")
    logger.info("  Training File:   %s (%d samples)", out_train, len(train_samples))
    logger.info("  Validation File: %s (%d samples)", out_val, len(val_samples))
    logger.info("==========================================================")
    return out_train, out_val


if __name__ == "__main__":
    generate_extended_dataset()
