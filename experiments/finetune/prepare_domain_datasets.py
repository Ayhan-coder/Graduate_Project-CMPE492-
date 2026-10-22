"""Domain-Specific Instruction-Tuning Dataset Generator.

Generates high-quality Turkish & English instruction datasets across target vertical domains
stipulated in the meeting notes:
1. Tourism (Tatilsepeti Ar-Ge: acente, rezervasyon, otelcilik)
2. Corporate Occupational Health and Safety (İSG - İş Sağlığı ve Güvenliği)
3. KVKK / GDPR (Veri Koruma Hukuku ve Aydınlatma)
4. Corporate Cybersecurity (Oltalama, Parola, Sosyal Mühendislik)
5. Sustainability (ESG, Karbon Ayak İzi, Yeşil Mutabakat)
6. Banking & Finance (Kredi, Findeks, Mevduat, MASAK)

Outputs standardized JSONL files formatted for Alpaca / HuggingFace TRL SFTTrainer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_SAMPLES: List[Dict[str, Any]] = [
    # 1. Turizm (Tatilsepeti Ar-Ge)
    {
        "domain": "tourism",
        "instruction": "Aşağıdaki turizm ve acente eğitim konuşma transkriptinden teknik terimleri yakala, CEFR B1 düzeyinde açıkla ve Bloom taksonomi düzeyini ata.",
        "input": "Misafirlerimize erken rezervasyon döneminde sunduğumuz dinamik fiyatlandırma ve no-show durumundaki iptal iade koşullarını çağrı merkezinde doğru aktarmalıyız. Voucher belgesi düzenlenmeden konaklama teyidi verilemez.",
        "output": json.dumps({
            "terms": [
                {
                    "term": "Dinamik Fiyatlandırma",
                    "canonical_id": "TOUR-01",
                    "definition": "Otel doluluğu ve talep yoğunluğuna göre oda fiyatlarının anlık ve otomatik değiştiği gelir yönetimi sistemi.",
                    "bloom_level": "Uygulama"
                },
                {
                    "term": "No-Show",
                    "canonical_id": "TOUR-02",
                    "definition": "Rezervasyon yaptıran müşterinin haber vermeksizin otele giriş yapmaması durumu.",
                    "bloom_level": "Hatırlama"
                },
                {
                    "term": "Voucher",
                    "canonical_id": "TOUR-03",
                    "definition": "Müşterinin satın aldığı otel ve transfer hizmetlerinin ödendiğini kanıtlayan resmi rezervasyon belgesi.",
                    "bloom_level": "Anlama"
                }
            ],
            "domain": "Turizm & Otelcilik"
        }, ensure_ascii=False)
    },
    # 2. İş Sağlığı ve Güvenliği (İSG)
    {
        "domain": "isg",
        "instruction": "Aşağıdaki zorunlu şirket İş Sağlığı ve Güvenliği (İSG) konuşma metninden teknik terimleri tespit et ve yapılandırılmış pedagojik JSON olarak çıkar.",
        "input": "Fabrika ve ofis alanlarında ramak kala olaylarının derhal formla bildirilmesi ve risk değerlendirmesi matrisine işlenmesi esastır. Yüksekte çalışmalarda kişisel koruyucu donanım ve paraşüt tipi emniyet kemeri kullanımı zorunludur.",
        "output": json.dumps({
            "terms": [
                {
                    "term": "Ramak Kala Olayı",
                    "canonical_id": "ISG-01",
                    "definition": "İşyerinde meydana gelen, ancak yaralanma veya maddi hasara yol açmadan son anda atlatılan tehlikeli durum.",
                    "bloom_level": "Anlama"
                },
                {
                    "term": "Risk Değerlendirmesi",
                    "canonical_id": "ISG-02",
                    "definition": "İşyerindeki tehlikelerin olasılık ve şiddetini sistematik puanlayarak öncelikli tedbirleri belirleme süreci.",
                    "bloom_level": "Analiz"
                },
                {
                    "term": "Kişisel Koruyucu Donanım (KKD)",
                    "canonical_id": "ISG-03",
                    "definition": "Çalışanı iş kazası ve meslek hastalığı risklerine karşı koruyan baret, eldiven, yelek gibi teçhizatlar.",
                    "bloom_level": "Hatırlama"
                }
            ],
            "domain": "İş Sağlığı ve Güvenliği"
        }, ensure_ascii=False)
    },
    # 3. KVKK / GDPR
    {
        "domain": "kvkk",
        "instruction": "Aşağıdaki KVKK şirket içi uyum eğitimi metnini analiz ederek terimleri ve pedagojik önkoşul seviyelerini çıkar.",
        "input": "Şirketimizde müşteri verilerini işlerken veri sorumlusu sıfatıyla aydınlatma yükümlülüğümüzü yerine getirmeli, pazarlama amaçlı SMS'ler için mutlaka açık rıza almalıyız. Özel nitelikli kişisel veriler şifrelenmeden saklanamaz.",
        "output": json.dumps({
            "terms": [
                {
                    "term": "Veri Sorumlusu",
                    "canonical_id": "KVKK-01",
                    "definition": "Kişisel verilerin işleme amaçlarını ve vasıtalarını belirleyen, veri kayıt sistemini yöneten gerçek veya tüzel kişi.",
                    "bloom_level": "Anlama"
                },
                {
                    "term": "Açık Rıza",
                    "canonical_id": "KVKK-02",
                    "definition": "Belirli bir konuya ilişkin, bilgilendirilmeye dayanan ve özgür iradeyle açıklanan onay beyanı.",
                    "bloom_level": "Uygulama"
                },
                {
                    "term": "Özel Nitelikli Kişisel Veri",
                    "canonical_id": "KVKK-03",
                    "definition": "Sağlık, biyometrik veri, ırk ve ceza kaydı gibi öğrenilmesi halinde ayrımcılık yaratabilecek hassas veriler.",
                    "bloom_level": "Hatırlama"
                }
            ],
            "domain": "Kişisel Verilerin Korunması Hukuku"
        }, ensure_ascii=False)
    },
    # 4. Siber Güvenlik
    {
        "domain": "cybersecurity",
        "instruction": "Aşağıdaki kurumsal siber güvenlik farkındalık konuşmasından teknik terimleri yakala ve Bloom taksonomi düzeyini ata.",
        "input": "Kurum e-postalarındaki sahte bağlantılarla yapılan oltalama saldırılarına karşı çalışanlarımızın sosyal mühendislik tekniklerine dikkat etmesi ve tüm hesaplarda çok faktörlü kimlik doğrulamayı (MFA) etkinleştirmesi gerekir.",
        "output": json.dumps({
            "terms": [
                {
                    "term": "Oltalama (Phishing)",
                    "canonical_id": "SEC-01",
                    "definition": "Güvenilir bir kurum izlenimi vererek kullanıcıların şifre veya kart bilgilerini çalmayı hedefleyen aldatıcı saldırı türü.",
                    "bloom_level": "Anlama"
                },
                {
                    "term": "Sosyal Mühendislik",
                    "canonical_id": "SEC-02",
                    "definition": "İnsan psikolojisindeki güven, korku veya merak duygularını manipüle ederek gizli bilgilere erişme sanatı.",
                    "bloom_level": "Analiz"
                },
                {
                    "term": "Çok Faktörlü Kimlik Doğrulama (MFA)",
                    "canonical_id": "SEC-03",
                    "definition": "Sisteme girişte parola haricinde telefona gelen SMS kodu veya doğrulama uygulamasıyla sağlanan ek güvenlik katmanı.",
                    "bloom_level": "Uygulama"
                }
            ],
            "domain": "Siber Güvenlik"
        }, ensure_ascii=False)
    },
    # 5. Sürdürülebilirlik (ESG)
    {
        "domain": "sustainability",
        "instruction": "Aşağıdaki kurumsal sürdürülebilirlik konuşmasından kavramları ve yalın tanımlarını çıkar.",
        "input": "Şirketimizin net-sıfır karbon hedefi doğrultusunda doğrudan emisyonlar ile şebeke elektriği tüketimini azaltmalı, tedarik zincirimizdeki dolaylı ayak izini ESG kriterlerine göre raporlamalıyız.",
        "output": json.dumps({
            "terms": [
                {
                    "term": "Karbon Ayak İzi",
                    "canonical_id": "ESG-01",
                    "definition": "Bir kurumun faaliyetleri sonucu atmosfere salınan toplam sera gazı miktarının karbondioksit eşdeğeri cinsinden ifadesi.",
                    "bloom_level": "Anlama"
                },
                {
                    "term": "Sera Gazı Protokol Emisyonları",
                    "canonical_id": "ESG-02",
                    "definition": "Doğrudan şirket içi yakıt kullanımı (Kategori 1), satın alınan elektrik (Kategori 2) ve tedarik zinciri kaynaklı dolaylı salımlar (Kategori 3).",
                    "bloom_level": "Analiz"
                },
                {
                    "term": "ESG Kriterleri",
                    "canonical_id": "ESG-03",
                    "definition": "Çevresel (E), Sosyal (S) ve Kurumsal Yönetim (G) boyutlarında bir şirketin sürdürülebilirlik performansını ölçen standartlar.",
                    "bloom_level": "Uygulama"
                }
            ],
            "domain": "Kurumsal Sürdürülebilirlik"
        }, ensure_ascii=False)
    },
    # 6. Bankacılık & Finans
    {
        "domain": "banking",
        "instruction": "Aşağıdaki banka içi kredi tahsis ve mevzuat eğitim metninden teknik terimleri ve pedagojik hiyerarşiyi çıkar.",
        "input": "Ticari kredi onay sürecinde müşterinin KKB ve Findeks kredi notu risk analitiğiyle incelenir. Şüpheli para transferlerinde ise MASAK mevzuatı gereği zorunlu bildirim formu düzenlenerek likidite rasyoları denetlenir.",
        "output": json.dumps({
            "terms": [
                {
                    "term": "Findeks Kredi Notu",
                    "canonical_id": "BNK-01",
                    "definition": "Kişi ve şirketlerin geçmiş finansal borç ödeme alışkanlıklarına göre 1 ile 1900 arasında hesaplanan kredi güvenilirlik skoru.",
                    "bloom_level": "Uygulama"
                },
                {
                    "term": "MASAK Bildirimi",
                    "canonical_id": "BNK-02",
                    "definition": "Suç gelirlerinin aklanması ve terörizmin finansmanını önlemek amacıyla Mali Suçları Araştırma Kurulu'na yapılan şüpheli işlem bildirimi.",
                    "bloom_level": "Hatırlama"
                },
                {
                    "term": "Likidite Rasyosu",
                    "canonical_id": "BNK-03",
                    "definition": "Bankanın kısa vadeli yükümlülüklerini ve mevduat çıkışlarını nakde çevrilebilir varlıklarıyla karşılama oranı.",
                    "bloom_level": "Analiz"
                }
            ],
            "domain": "Bankacılık & Finans"
        }, ensure_ascii=False)
    }
]


def generate_extended_dataset() -> None:
    from experiments.finetune.prepare_external_datasets import LeakageGuard
    guard = LeakageGuard()

    train_file = DATA_DIR / "train_instruct_domain.jsonl"
    val_file = DATA_DIR / "val_instruct_domain.jsonl"

    train_data: List[Dict[str, Any]] = []
    val_data: List[Dict[str, Any]] = []

    for idx, sample in enumerate(DOMAIN_SAMPLES):
        # Validate through LeakageGuard
        blob = f"{sample['instruction']} {sample['input']} {sample['output']}"
        if guard.text_is_eval(blob):
            continue
        out_terms = json.loads(sample["output"]).get("terms", [])
        if any(guard.term_is_eval_answer(t.get("term", "")) for t in out_terms):
            continue

        if idx % 3 == 0:
            val_data.append(sample)
        else:
            train_data.append(sample)

        var1 = {
            "domain": sample["domain"],
            "instruction": f"Verilen {sample['domain'].upper()} eğitim konuşmasından teknik anahtar kavramları ve açıklamalarını JSON formatında ayıkla.",
            "input": sample["input"],
            "output": sample["output"]
        }
        train_data.append(var1)

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated domain fine-tuning datasets (LeakageGuard validated):")
    print(f"  Training set: {train_file} ({len(train_data)} samples)")
    print(f"  Validation set: {val_file} ({len(val_data)} samples)")


if __name__ == "__main__":
    generate_extended_dataset()
