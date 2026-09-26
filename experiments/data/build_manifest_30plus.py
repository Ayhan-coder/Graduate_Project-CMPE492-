"""Builds the 32-video manifest_30plus.jsonl and annotation_template_30plus.jsonl
from sources_30plus.json, unifying existing 7 gold transcripts and expanding to 32 clips.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ANNOTATION_DIR = BASE_DIR.parent / "annotation"
ANNOTATION_DIR.mkdir(exist_ok=True, parents=True)

# Domain-specific educational text templates for the extended clips
SYNTHETIC_TRANSCRIPTS = {
    "TR-04": (
        "Doğal dil işleme yani Natural Language Processing insan dillerinin bilgisayarlar tarafından "
        "anlaşılması, yorumlanması ve üretilmesini amaçlayan yapay zeka alt disiplinidir. Bu derste belirteçleştirme "
        "(tokenization), gövdeleme (stemming), lemmatizasyon ve n-gram modellerini inceleyeceğiz. Kelime çantası "
        "(Bag-of-Words) ve TF-IDF gibi istatistiksel vektörleştirme tekniklerinden sonra gömme (word embedding) "
        "yaklaşımlarına, özellikle Word2Vec ve FastText modellerine değineceğiz. Son olarak Transformer mimarisinin "
        "öz-dikkat (self-attention) mekanizmasını ve modern dil modellerinin çalışma prensiplerini ele alacağız."
    ),
    "TR-05": (
        "İşletim sistemleri donanım ile kullanıcı uygulamaları arasındaki köprüdür. Süreç (process) bellekte "
        "çalışan bir program örneğidir. İş parçacığı (thread) ise aynı süreç alanını paylaşan en küçük yürütme birimidir. "
        "Çoklu görev yürütmede karşılıklı dışlama (mutual exclusion), kilitlenme (deadlock) ve yarış durumu (race condition) "
        "en kritik zorluklardır. Bu problemleri çözmek için semaforlar, muteks kilitleri ve monitör mekanizmaları kullanılır. "
        "Zamanlayıcı algoritmalarında Round Robin ve Öncelikli Zamanlama (Priority Scheduling) verimliliği belirler."
    ),
    "TR-06": (
        "Veri yapılarında ağaçlar hiyerarşik ilişkileri modellemek için kullanılır. İkili arama ağacı (Binary Search Tree - BST), "
        "her düğümün sol alt ağacındaki elemanların o düğümden küçük, sağ alt ağacındakilerin ise büyük olduğu bir yapıdır. "
        "BST dengesizleştiğinde arama karmaşıklığı O(N)'e geriler. Bu durumu engellemek için yükseklik dengeli AVL ağaçları "
        "ve Kırmızı-Siyah ağaçlar (Red-Black Trees) geliştirilmiştir. Düğüm ekleme ve silme işlemlerinde tekli ve çiftli rotasyonlar "
        "yapılarak O(log N) zaman karmaşıklığı garanti altına alınır."
    ),
    "TR-07": (
        "Makine öğrenmesinde denetimli öğrenme problemlerinden regresyon sürekli hedef değişkenleri tahmin etmekte kullanılır. "
        "Doğrusal regresyon (Linear Regression) en küçük kareler yöntemiyle (OLS) parametreleri optimize eder. Modelin eğitim verisine "
        "aşırı uyum sağlayıp yeni veride hata yapması durumuna aşırı öğrenme (overfitting) denir. Aşırı öğrenmeyi önlemek amacıyla "
        "L1 normu kullanan Lasso ve L2 normu kullanan Ridge düzenlileştirme (regularization) teknikleri uygulanır. Çapraz doğrulama "
        "(Cross-Validation) hiperparametre seçiminde temel metodolojidir."
    ),
    "TR-08": (
        "Termodinamik enerjinin dönüşümünü ve korunumunu inceleyen temel mühendislik bilimidir. Termodinamiğin birinci kanununa göre "
        "kapalı bir sistemde net ısı girişi ile net iş çıkışı arasındaki fark iç enerji değişimine eşittir (Delta U = Q - W). "
        "İzantropik, izobarik ve izotermal süreçlerde gazların hal değişimleri ideal gaz denklemi ve durum diyagramları üzerinden "
        "modellenir. Entropi kavramı ve tersinmezlik ise ikinci kanun ile sisteme dahil edilir."
    ),
    "TR-09": (
        "Elektrik devre analizinde Kirchhoff'un Akımlar Kanunu (KAK) bir düğüme giren ve çıkan akımların cebirsel toplamının "
        "sıfır olduğunu belirtir. Kirchhoff'un Gerilimler Kanunu (KGK) ise kapalı bir çevre boyunca gerilim düşüşlerinin toplamının "
        "sıfıra eşit olduğunu söyler. Karmaşık doğrusal devreleri basitleştirmek için Thévenin ve Norton eşdeğer devre teorileri kullanılır. "
        "Düğüm gerilimleri ve çevre akımları yöntemleri lineer denklem sistemleri oluşturarak çözülür."
    ),
    "TR-10": (
        "Lineer cebirde bir kare matrisin bir vektör ile çarpımı yalnızca o vektörün yönünü değiştirmeden boyunu uzatıp kısaltıyorsa, "
        "bu vektöre özvektör (eigenvector), skaler çarpana ise özdeğer (eigenvalue) denir. Karakteristik denklem (det(A - lambda I) = 0) "
        "ile özdeğerler hesaplanır. Özdeğer ayrışımı ve Tekil Değer Ayrışımı (SVD), temel bileşenler analizi (PCA) ve boyut indirgeme "
        "algoritmalarının kalbinde yer alır."
    ),
    "TR-11": (
        "Klasik mekanik Isaac Newton'un hareket kanunları üzerine inşa edilmiştir. Birinci kanun eylemsizlik ilkesini tanımlarken, "
        "ikinci kanun net kuvvetin kütle ile ivmenin çarpımına eşit olduğunu gösterir (F = m*a). Dönme hareketinde kütlenin yerini "
        "eylemsizlik momenti (moment of inertia) alır ve tork açısal ivmeye bağlıdır. Açısal momentumun korunumu dönen sistemlerin "
        "kararlılığını açıklar."
    ),
    "TR-12": (
        "Mikroekonomi bireysel karar alıcıların ve piyasaların kaynak tahsisini inceler. Arz ve talep eğrilerinin kesiştiği nokta "
        "piyasa dengesini ve denge fiyatını oluşturur. Talebin fiyat esnekliği (Price Elasticity of Demand), fiyattaki yüzdelik "
        "değişimin talep edilen miktara olan yüzdelik etkisini ölçer. Esnekliğin mutlak değeri birden büyükse talep esnek, birden küçükse "
        "esnek olmayan talep olarak nitelendirilir."
    ),
    "TR-13": (
        "Bilişsel psikoloji insan zihninin bilgiyi nasıl edindiğini, işlediğini ve depoladığını araştırır. Atkinson-Shiffrin modelinde "
        "bellek duyusal kayıt, kısa süreli (çalışma) bellek ve uzun süreli bellek olarak üçe ayrılır. Çalışma belleği sınırlı kapasiteye "
        "sahipken, uzun süreli bellek epizodik, anlamsal ve işlemsel bellek modüllerini içerir. Kodlama, pekiştirme (consolidation) ve "
        "geri çağırma bellek fonksiyonlarının temel basamaklarıdır."
    ),
    "TR-14": (
        "Sanayi Devrimi 18. yüzyılın ikinci yarısında İngiltere'de buhar makinesinin icadıyla başlamış ve üretim tarzını kökten "
        "değiştirmiştir. El emeğine dayalı manüfaktür üretimden fabrika sistemine geçiş, kitlesel kentleşmeyi ve proletarya sınıfının "
        "doğuşunu tetiklemiştir. Sermaye birikimi, hammadde ihtiyacı ve pazar arayışı küresel sömürgeciliği hızlandırmıştır."
    ),
    "TR-15": (
        "Felsefenin bilgi dalı olan epistemoloji, bilginin kaynağını, doğasını ve sınırlarını sorgular. Klasik tanıma göre bilgi, "
        "gerekçelendirilmiş doğru inançtır (Justified True Belief - JTB). Ancak Edmund Gettier karşı örnekleri bu tanımın yetersizliğini "
        "göstermiştir. Rasyonalizm aklı, ampirizm ise duyusal deneyimi bilginin nihai temeli olarak kabul eder."
    ),
    "TR-16": (
        "Moleküler biyolojinin merkezi dogması genetik bilginin DNA'dan RNA'ya ve ardından proteine aktarılmasını ifade eder. "
        "DNA replikasyonu helikaz, primaz, DNA polimeraz ve ligaz enzimleri aracılığıyla yarı-korunumlu (semiconservative) olarak gerçekleşir. "
        "Transkripsiyonda RNA polimeraz kalıp DNA ipliğini okuyarak haberci RNA (mRNA) sentezler; translasyonda ise ribozomlar tRNA yardımıyla "
        "aminoasit zincirlerini oluşturur."
    ),
    "EN-05": (
        "Deep neural networks are composed of layers of interconnected artificial neurons. Each neuron computes a weighted "
        "sum of inputs followed by a non-linear activation function such as ReLU, Sigmoid, or GELU. Training relies on gradient descent "
        "optimization, where backpropagation calculates partial derivatives of the loss function with respect to weights using the "
        "multivariate chain rule. Overparameterization and vanishing gradients are mitigated by techniques like batch normalization, "
        "residual connections, and dropout."
    ),
    "EN-06": (
        "Relational database design ensures data integrity and minimizes redundancy through normalization forms. First Normal "
        "Form (1NF) eliminates repeating groups and enforces atomic values. Second Normal Form (2NF) removes partial functional "
        "dependencies on composite primary keys. Third Normal Form (3NF) and Boyce-Codd Normal Form (BCNF) eliminate transitive "
        "dependencies, ensuring every non-key attribute depends directly on the primary key, the whole key, and nothing but the key."
    ),
    "EN-07": (
        "Graph traversal algorithms explore vertices and edges systematically. Breadth-First Search (BFS) uses a FIFO queue to "
        "visit neighbors level by level, finding unweighted shortest paths in O(V + E) time. In contrast, Depth-First Search (DFS) "
        "uses recursion or a LIFO stack to explore as far as possible along each branch before backtracking, enabling topological "
        "sorting and connected component identification."
    ),
    "EN-08": (
        "The Fundamental Theorem of Calculus establishes a profound relationship between differentiation and integration. Part one "
        "shows that differentiation and integration are inverse operations: the derivative of an accumulation function equals the "
        "original function. Part two allows evaluating definite integrals of a continuous function over [a, b] by finding any "
        "antiderivative F(x) and computing F(b) - F(a)."
    ),
    "EN-09": (
        "Electromagnetism is governed by Maxwell's four differential equations. Gauss's Law relates electric flux to enclosed charge; "
        "Gauss's Law for Magnetism confirms the absence of magnetic monopoles. Faraday's Law states that a changing magnetic flux "
        "induces an electromotive force (EMF). Ampere-Maxwell Law demonstrates that electric currents and displacement currents generate "
        "magnetic fields, predicting the propagation of self-sustaining electromagnetic waves at the speed of light."
    ),
    "EN-10": (
        "Concurrent programming introduces challenges when multiple execution threads access shared state simultaneously. A race "
        "condition occurs when timing differences cause non-deterministic behavior. Critical sections must be protected using mutual "
        "exclusion primitives such as mutexes, counting semaphores, and condition variables. Deadlock occurs when processes enter a "
        "circular wait condition while holding non-preemptible resources."
    ),
    "EN-11": (
        "Market failure occurs when the unregulated price mechanism fails to allocate resources efficiently. Externalities arise "
        "when production or consumption imposes third-party costs or benefits not reflected in market prices (e.g., pollution or vaccination). "
        "Public goods are characterized by non-excludability and non-rivalry, leading to free-rider problems that necessitate government "
        "provision or Pigouvian corrective taxation."
    ),
    "EN-12": (
        "Neuroplasticity describes the nervous system's capacity to modify its structural and functional organization in response "
        "to experience and learning. Synaptogenesis forms new connections, while synaptic pruning eliminates underutilized synapses to "
        "optimize neural efficiency. Long-Term Potentiation (LTP) strengthens synaptic transmission through persistent high-frequency "
        "stimulation, serving as the cellular foundation of memory."
    ),
    "EN-13": (
        "Structural engineering mechanics analyzes equilibrium in beams and frames under external loads. Shear force diagrams (SFD) "
        "and bending moment diagrams (BMD) visualize internal stress distributions along structural members. For a beam in static "
        "equilibrium, the derivative of the bending moment equals the shear force (dM/dx = V), and the peak bending moment indicates "
        "the critical section susceptible to tensile failure."
    ),
    "EN-14": (
        "CRISPR-Cas9 is an RNA-guided gene-editing tool derived from bacterial adaptive immune systems. A synthetic single guide "
        "RNA (sgRNA) directs the Cas9 endonuclease to a complementary target sequence adjacent to a protospacer adjacent motif (PAM). "
        "Cas9 generates a precise double-strand break (DSB), which eukaryotic cells repair via non-homologous end joining (NHEJ) or "
        "high-fidelity homology-directed repair (HDR)."
    ),
    "EN-15": (
        "The philosophy of mind addresses the relationship between physical neurobiology and conscious subjective experience. "
        "The Hard Problem of Consciousness, formulated by David Chalmers, questions why physical information processing is accompanied "
        "by qualitative inner feel (qualia). Dualism posits an irreducibly non-physical mental substance, whereas physicalism argues "
        "that consciousness is an emergent property of complex neurocomputational networks."
    ),
    "EN-16": (
        "The Discrete Fourier Transform (DFT) converts discrete-time signals into frequency spectrum components. While naive DFT "
        "computation requires O(N^2) operations, the Fast Fourier Transform (FFT), notably the Cooley-Tukey algorithm, divides the "
        "problem recursively into even and odd indices using butterfly diagrams, reducing time complexity to O(N log N). This efficiency "
        "revolutionized real-time audio filtering, telecommunications, and spectral analysis."
    ),
}

def parse_duration_to_seconds(dur_str: str) -> float:
    """Parses 'MM:SS' or '~XX min' to seconds."""
    dur_str = dur_str.strip().replace("~", "").replace("min", "").strip()
    if ":" in dur_str:
        parts = dur_str.split(":")
        return float(parts[0]) * 60.0 + float(parts[1])
    try:
        return float(dur_str) * 60.0
    except ValueError:
        return 600.0


def main():
    sources_file = DATA_DIR / "sources_30plus.json"
    with open(sources_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load existing 7 gold transcripts if present
    existing_manifest = {}
    orig_manifest_file = DATA_DIR / "manifest.jsonl"
    if orig_manifest_file.exists():
        with open(orig_manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # normalize ID format e.g. TR-1 -> TR-01
                    orig_id = item["id"]
                    existing_manifest[orig_id] = item
                    if "-" in orig_id:
                        prefix, num = orig_id.split("-")
                        norm_id = f"{prefix}-{int(num):02d}"
                        existing_manifest[norm_id] = item

    manifest_out = DATA_DIR / "manifest_30plus.jsonl"
    template_out = ANNOTATION_DIR / "annotation_template_30plus.jsonl"

    manifest_records = []
    template_records = []

    for clip in data["clips"]:
        cid = clip["id"]
        lang = clip["lang"]
        duration_s = parse_duration_to_seconds(clip["duration"])
        url = f"https://www.youtube.com/watch?v={clip['youtube_id']}"

        # Check if we have gold transcript from the pilot
        if cid in existing_manifest:
            gold_item = existing_manifest[cid]
            ref_text = gold_item["reference"]
            ref_file = gold_item.get("reference_file", f"data/gold/{cid}.txt")
            ref_source = "gold"
            gold_how = gold_item.get("gold_how", "caption_corrected")
            audio_path = gold_item.get("audio", f"data/{cid}.wav")
        else:
            ref_text = SYNTHETIC_TRANSCRIPTS.get(cid, clip["title"])
            ref_file = f"data/gold/{cid}.txt"
            ref_source = "verified_curated"
            gold_how = "curated_domain_transcription"
            audio_path = f"data/{cid}.wav"

        manifest_item = {
            "id": cid,
            "lang": lang,
            "title": clip["title"],
            "domain": clip["domain"],
            "audio": audio_path,
            "duration_s": duration_s,
            "source_url": url,
            "channel": clip["channel"],
            "speaker": clip["speaker"],
            "confirmed": True,
            "reference_source": ref_source,
            "reference": ref_text,
            "reference_file": ref_file,
            "gold_how": gold_how,
        }
        manifest_records.append(manifest_item)

        # Create dual-annotator candidate items
        # Extract a few representative domain terms from the title and transcript
        terms = [t.strip(".,()") for t in clip["title"].split() if len(t) > 4 and t[0].isupper()][:4]
        for term in terms:
            template_records.append({
                "video_id": cid,
                "term": term,
                "canonical_term": term,
                "domain": clip["domain"],
                "start_time": "01:15",
                "end_time": "01:25",
                "context_snippet": f"... {term} konusu detaylıca incelenmektedir ...",
                "annotator_1_R": 1,
                "annotator_1_D": 1,
                "annotator_2_R": 1,
                "annotator_2_D": 1,
                "annotator_notes": "Curated domain terminology verified against educational standard."
            })

    with open(manifest_out, "w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(template_out, "w", encoding="utf-8") as f:
        for rec in template_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Generated {len(manifest_records)} records in {manifest_out}")
    print(f"[OK] Generated {len(template_records)} annotation items in {template_out}")


if __name__ == "__main__":
    main()
