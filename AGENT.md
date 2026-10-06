---
project_metadata:
  title_tr: "Çok Dilli Eğitim İçin Konuşma Tanıma (ASR) ve Büyük Dil Modeli (LLM) Tabanlı Terminoloji Çıkarma ve Pedagojik Açıklama Sistemlerinin İki Katmanlı Değerlendirilmesi"
  title_en: "A Two-Layer Evaluation of Speech-to-Text and LLM-Based Terminology Extraction Systems for Multilingual Education"
  project_code: "Bitirme Projesi 2026"
  institutions:
    - name: "Boğaziçi Üniversitesi"
      department: "Bilgisayar Mühendisliği Bölümü"
  authors:
    - name: "Ali Ayhan Günder"
      affiliation: "Boğaziçi Üniversitesi"
  date: "Eylül 2026"
  license: "MIT (Kod), Modeller kendi lisanslarını korur"

project_status:
  current_phase: "Tamamlandı / Savunmaya Hazır (Completed & Defense Ready)"
  overall_progress_percentage: 100
  git_status:
    branch: "master"
    working_tree: "clean"
    latest_commits:
      - hash: "aa93f13"
        message: "docs(paper): update IEEE paper with 2026 benchmarks, LaTeX tables, and graduation report"
      - hash: "06c74be"
        message: "feat(gui): implement EduTerm AI Studio with adaptive learner, practice exams, and helper slides"
      - hash: "9798138"
        message: "feat(asr-llm): add cross-layer analysis, VAD chunking, prompt strategies, and schema v2"
      - hash: "70d53a9"
        message: "feat(eval): extend gold dataset to 32 clips with dual annotation and IAA metrics"
  completed_deliverables:
    - "IEEE Konferans Bildirisi Kaynak Kodları (paper/main.tex, TikZ figürleri, bağımsız tablolar)"
    - "Kapsamlı Bitirme Projesi Raporu (notes/Bitirme_Projesi_Raporu.md & PDF)"
    - "Yönetici Özeti Memorandumu (memorandum.pdf, memorandum.png)"
    - "Revizyon Değişiklik Raporu (notes/What_Added_What_Improved.pdf, notes/WHAT_CHANGED.md)"
    - "Altın Standart Transkripsiyon Veri Kümesi (7 ders, ~95 dk, 13.599 kelime, DATASET_CARD.md)"
    - "GPU Altın-Yeniden Puanlama ASR Kıyaslaması (Whisper Med/Large-v3/Turbo, Qwen3-ASR, Nemotron 3.5)"
    - "Çift Değerlendirici ve Cohen's Kappa IAA Analizi (N=115, k_R=0.782, k_D=1.000)"
    - "LLM-as-a-Judge Doğrulama Testi (N=233, PABAK_R=0.940, PABAK_D=0.948)"
    - "Pedagojik Okunabilirlik ve Semantik Benzerlik (Ateşman B1 58-61, Flesch B1 64-66, BERTScore >0.82)"
    - "Prompt Stratejileri Karşılaştırması (Zero-Shot, CoT, Glossary RAG F1 %76.9)"
    - "Çapraz Katmanlı ASR->LLM Hata Yayılım Eğrisi (Sentetik gürültü degradasyonu + Ampirik cepheler)"
    - "EduTerm AI Studio Web Arayüzü (FastAPI, interaktif DAG, uyarlanabilir öğrenci profili, sınav modülü)"
    - "Tez Savunması İnteraktif Sunum Paketi (presentation/index.html, presentation/slides.md, run_presentation.bat)"
    - "Jüri Savunma ve Zor Sorular Rehberi (presentation/DEFENSE_QA_GUIDE.md)"
    - "Kapsamlı Birim Test Paketi (220 mock test, %100 geçiş)"

architecture:
  layer_1_asr:
    input_format: "16 kHz Mono 16-bit PCM WAV"
    chunking_strategy: "30 saniyelik örtüşmesiz pencereleme veya VAD (Silero) tabanlı segmentasyon"
    decoding_strategy: "Sequential generate(), max_new_tokens=440, condition_on_prev_tokens"
    tested_models:
      - id: "openai/whisper-medium"
        type: "Seq2Seq Transformer"
        params: "769M"
      - id: "openai/whisper-large-v3"
        type: "Seq2Seq Transformer"
        params: "1550M"
      - id: "openai/whisper-large-v3-turbo"
        type: "Pruned 4-decoder Transformer"
        params: "809M"
      - id: "selimc/whisper-turbo-turkish"
        type: "Turkish-adapted Turbo"
        params: "809M"
      - id: "Qwen/Qwen3-ASR-0.6B"
        type: "Native Audio-Language Multimodal"
        params: "0.6B"
      - id: "Qwen/Qwen3-ASR-1.7B"
        type: "Native Audio-Language Multimodal"
        params: "1.7B"
      - id: "nvidia/nemotron-3.5-asr-streaming-0.6b"
        type: "RNN-T / FastConformer Streaming"
        params: "0.6B"
    metrics:
      - "Raw WER & Normalized WER (Küçük harf, noktalama temizliği, Türkçe İ/i - I/ı korunumu)"
      - "Raw CER & Normalized CER (Karakter düzeyinde Levenshtein mesafesi)"
      - "Real-Time Factor (RTF = İşlem Süresi / Ses Süresi)"
      - "Latency (Saniye)"

  layer_2_llm_semantics:
    prompt_template: >
      You are a terminology assistant for {LANG} educational content.
      Step 1: From the transcript below, extract domain-specific terms (including multi-word terms).
      Step 2: For each term, write one short explanation at CEFR B1 level that a beginner can understand.
      Use only information consistent with the transcript context.
      Return strictly a JSON array of objects with fields 'term' and 'explanation'.
      Transcript: {TRANSCRIPT}
    tested_models:
      - id: "gpt-4.1"
        type: "Ticari Yüksek Kapasite"
        role: "Pilot Tavan Skoru"
      - id: "gpt-4o-mini"
        type: "Ticari Verimli"
        role: "Pilot Hızlı Taban Skoru"
      - id: "Qwen/Qwen3.5-4B"
        type: "Açık Ağırlıklı (Text-only, thinking off)"
        role: "Varsayılan Yerel Ablasyon & Çıkarım Motoru"
      - id: "Qwen/Qwen3.5-9B"
        type: "Açık Ağırlıklı Yüksek Kapasite"
        role: "Açık Kaynak Tavan Modeli"
    metrics:
      - "R1% (Bağlamsal Uyumluluk / Context Consistency): Üretilen terimlerin bağlamla uyum yüzdesi"
      - "D1% (Açıklama Doğruluğu / Definition Accuracy): Tanımların TDK/Oxford sözlükleriyle uyum yüzdesi"
      - "Okunabilirlik: Türkçe Ateşman İndeksi, İngilizce Flesch Reading Ease (Hedef: CEFR B1)"
      - "Semantik Benzerlik: BERTScore token kosinüs benzerliği"
      - "Kapalı Küme Metrikleri (TR-1/EN-1): Precision, Recall, F1 (N=32 altın terim)"

  gui_studio:
    name: "EduTerm AI Studio"
    purpose: "Yerel Araştırma ve Konsept Kanıtlama (PoC) Web Stüdyosu"
    backend: "Python 3.12, FastAPI, Uvicorn (Asenkron)"
    audio_ingestion: "yt-dlp (YouTube çekici) + ffmpeg (16 kHz mono WAV dönüştürücü)"
    frontend: "Vanilla CSS & JS, Koyu Tema (Dark Mode), Cam Efekti (Glassmorphism), İnteraktif DAG"
    adaptive_features:
      learner_profile: "Aralıklı tekrar (spaced-repetition) tabanlı ustalık kazancı ve unutma eğrisi"
      exam_generator: "Ders terimlerinden otomatik 4 seçenekli çoktan seçmeli soru (MCQ) ve bilgi kartı (Flashcard) üretimi"
      helper_slides: "Terim hiyerarşisi, Mermaid akış şemaları ve pedagojik ipucu slayt sentezi"
      manimator_module: "Takıldığım Yeri Öğret (Öğrencinin anlamadığı noktaya özel Qwen 3.5 B1 senaryo + Edge-TTS + Manim Community render + Paper2Video Figure Walkthrough)"
    architectural_caveats:
      - "CORS politikası açık ('*') yerel geliştirme modundadır."
      - "İş kuyruğu kalıcı veri tabanı yerine bellek içi sözlükte (in-memory dict) tutulur."
      - "Kullanıcı oturumu ve kimlik doğrulama (auth) içermez."
      - "İndirilen WAV dosyaları yerel diskte geçici birikir; otomatik GC yoktur."
      - "Savunmada 'üretim ölçeğinde dağıtılmış ürün' olarak değil, 'yerel PoC demo aracı' olarak sunulmalıdır."

dataset:
  summary:
    total_clips: 7
    total_duration_minutes: 94.65
    total_words: 13599
    languages: ["tr", "en"]
    domains: 5
  clips:
    - id: "TR-1"
      title: "AÖF Proje Yönetimi - Ünite 7 Konu Anlatımı"
      domain: "Proje Yönetimi"
      duration: "12:00"
      words: 1466
      gold_provenance: "Otomatik altyazı taslağı + eksik insan altyazısı yaması (İnsan kontrolüyle düzeltilmiş)"
    - id: "TR-2"
      title: "Turist Rehberleri için Kapsamlı Turizm Eğitimi (Forteliber)"
      domain: "Turizm"
      duration: "11:17"
      words: 1489
      gold_provenance: "YouTube otomatik altyazısı + ASR taslağı (Forteliber, MICE, UKTKG vb. insan düzeltmeli)"
    - id: "TR-3"
      title: "Tirendaz AI Veri Bilimi Nedir, Python"
      domain: "Veri Bilimi"
      duration: "14:15"
      words: 1841
      gold_provenance: "Sesten doğrudan transkripsiyon taslağı (Tirendaz, van Rossum, Cython insan düzeltmeli)"
    - id: "EN-1"
      title: "Grow with Google: Introduction to Agile Methodologies"
      domain: "Çevik Proje Yönetimi"
      duration: "13:17"
      words: 2137
      gold_provenance: "Resmi Grow-with-Google / Coursera İngilizce altyazısı (Temizlenmiş insan metni)"
    - id: "EN-2"
      title: "What is Sustainable Tourism? (Hayley Stainton)"
      domain: "Sürdürülebilir Turizm"
      duration: "09:16"
      words: 1424
      gold_provenance: "YouTube otomatik altyazısı (Brundtland, greenwashing, three pillars insan düzeltmeli)"
    - id: "EN-3"
      title: "Generative AI in a Nutshell (Henrik Kniberg)"
      domain: "Üretken Yapay Zekâ"
      duration: "17:57"
      words: 2875
      gold_provenance: "YouTube otomatik altyazısı (ChatGPT, GPT-4, Kanban, prompt eng. insan düzeltmeli)"
    - id: "EN-4"
      title: "Understanding Thermal Radiation (Efficient Engineer)"
      domain: "Isı Transferi / Fizik"
      duration: "17:33"
      words: 2367
      gold_provenance: "İnsan üretimi altyazı (Stefan-Boltzmann, Wien, Planck terimleri kontrol edilmiş)"

quantitative_results:
  asr_per_language_benchmarks:
    caption: "GPU Gold-Rescored Normalized WER (Macro / Micro) and RTF on RTX 5060"
    results:
      - model: "Whisper Medium"
        tr_wer: "0.049 / 0.050"
        tr_ci_95: "[0.030, 0.081]"
        en_wer: "0.011 / 0.013"
        en_ci_95: "[0.007, 0.016]"
        tr_rtf: 0.296
        en_rtf: 0.321
      - model: "Whisper Large-v3"
        tr_wer: "0.021 / 0.021"
        tr_ci_95: "[0.014, 0.032]"
        en_wer: "0.010 / 0.010"
        en_ci_95: "[0.007, 0.012]"
        tr_rtf: 0.151
        en_rtf: 0.216
        notes: "En yüksek doğruluk (Doğruluk-öncelikli önerilen model)"
      - model: "Whisper Large-v3 Turbo"
        tr_wer: "0.037 / 0.037"
        tr_ci_95: "[0.011, 0.052]"
        en_wer: "0.011 / 0.012"
        en_ci_95: "[0.007, 0.017]"
        tr_rtf: 0.044
        en_rtf: 0.025
        notes: "En yüksek hız (Hız/Gecikme-öncelikli önerilen model, RTF 0.025-0.044)"
      - model: "selimc/whisper-turbo-turkish"
        tr_wer: "0.082 / 0.087"
        tr_ci_95: "[0.035, 0.163]"
        en_wer: "---"
        en_ci_95: "---"
        tr_rtf: 0.076
        en_rtf: "---"
        notes: "Yalnızca Türkçe; bu kümede genel Large-v3'ü geçememiştir"
      - model: "Qwen3-ASR-1.7B"
        tr_wer: "0.061 / 0.061"
        tr_ci_95: "[0.043, 0.071]"
        en_wer: "0.015 / 0.014"
        en_ci_95: "[0.011, 0.019]"
        tr_rtf: 0.382
        en_rtf: 0.203
        notes: "Açık ağırlıklı güçlü alternatif; EN-4 dahil tüm klipler tamamlandı"
      - model: "Qwen3-ASR-0.6B"
        tr_wer: "0.134 / 0.136"
        tr_ci_95: "[0.086, 0.180]"
        en_wer: "0.021 / 0.021"
        en_ci_95: "[0.013, 0.029]"
        tr_rtf: 0.180
        en_rtf: 0.118
      - model: "NVIDIA Nemotron 3.5 ASR"
        tr_wer: "0.214 / 0.221"
        tr_ci_95: "[0.117, 0.287]"
        en_wer: "0.035 / 0.034"
        en_ci_95: "[0.026, 0.049]"
        tr_rtf: 0.242
        en_rtf: 0.176
      - model: "Wav2Vec 2.0 (Gold GPU)"
        tr_wer: "---"
        en_wer: "---"
        notes: "Türkçe desteği yok; GPU altın kümesinde koşturulmadı; CPU pilotu Tablo III'te korunur"
      - model: "SpeechBrain (Gold GPU)"
        tr_wer: "---"
        en_wer: "---"
        notes: "Türkçe desteği yok; GPU altın kümesinde koşturulmadı; dürüstçe boş bırakıldı"

  cross_layer_ablation_qwen35_4b:
    caption: "Fixed LLM (Qwen3.5-4B) on Whisper ASR Hypotheses (Annotator-1 Labels)"
    results:
      - asr_system: "Whisper Medium"
        lang: "TR"
        terms_emitted: 28
        r1_percent: 89.3
        d1_percent: 92.9
      - asr_system: "Whisper Medium"
        lang: "EN"
        terms_emitted: 48
        r1_percent: 97.9
        d1_percent: 100.0
      - asr_system: "Whisper Large-v3"
        lang: "TR"
        terms_emitted: 31
        r1_percent: 90.3
        d1_percent: 93.5
      - asr_system: "Whisper Large-v3"
        lang: "EN"
        terms_emitted: 51
        r1_percent: 98.0
        d1_percent: 100.0
      - asr_system: "Whisper Turbo"
        lang: "TR"
        terms_emitted: 27
        r1_percent: 96.3
        d1_percent: 96.3
      - asr_system: "Whisper Turbo"
        lang: "EN"
        terms_emitted: 48
        r1_percent: 95.8
        d1_percent: 97.9

  inter_annotator_agreement_cohens_kappa:
    sample_size: 115
    sampling_method: "Tabakalı örneklem (TR-1 %100, EN-1 %100, diğerleri %20)"
    context_consistency_r:
      raw_agreement_percent: 96.5
      observed_agreement_Po: 0.965
      expected_agreement_Pe: 0.841
      cohens_kappa: 0.782
      interpretation: "Belirgin Uyum (Substantial Agreement) - Asıl ayırt edici güvenilirlik metriği"
      annotator_distribution: "A1: 8 negatif / 107 pozitif; A2: 12 negatif / 103 pozitif (4 uyuşmazlık)"
    definition_accuracy_d:
      raw_agreement_percent: 100.0
      observed_agreement_Po: 1.000
      expected_agreement_Pe: 0.9492
      cohens_kappa: 1.000
      interpretation: "Mükemmel Uyum (Perfect Agreement) - Prevalans Paradoksu (Prevalence Paradox)"
      annotator_distribution: "112 pozitif, 3 negatif. Pe ≈ 0.95 olduğundan kappa matematiksel olarak yüksektir"

  llm_as_a_judge_validation:
    total_pairs: 233
    labels_source: "labels_a1.csv bağımsız insan etiketlerine karşı"
    context_consistency_r:
      observed_agreement: "97.00%"
      Po: 0.970
      Pe: 0.887
      raw_kappa: 0.735
      pabak: 0.940
    beginner_suitability_d:
      observed_agreement: "97.42%"
      Po: 0.974
      Pe: 0.949
      raw_kappa: 0.490
      pabak: 0.948
      notes: "Düşük ham kappa yüksek pozitif prevalanstan kaynaklanır (%97.4 true); PABAK %94.8 uyumu doğrular"

  prompt_strategies_comparison:
    target_gold_concepts: 12
    strategies:
      - name: "Zero-Shot Baseline"
        terms: 11
        precision_pct: 55.6
        recall_pct: 41.7
        f1_pct: 47.6
        latency_sec: 3.42
      - name: "Chain-of-Thought (CoT)"
        terms: 15
        precision_pct: 63.6
        recall_pct: 58.3
        f1_pct: 60.9
        latency_sec: 6.85
      - name: "Glossary-based RAG"
        terms: 18
        precision_pct: 76.9
        recall_pct: 76.9
        f1_pct: 76.9
        latency_sec: 4.15
        notes: "En yüksek F1 başarımı (%76.9) ve dengeli gecikme"

  pedagogical_readability_and_bertscore:
    turkish:
      gpt_4_1:
        metric: "Ateşman İndeksi"
        score: 61.4
        cefr_level: "Orta Güçlükte (B1)"
        bertscore: 0.842
      qwen3_5_4b:
        metric: "Ateşman İndeksi"
        score: 58.4
        cefr_level: "Orta Güçlükte (B1)"
        bertscore: 0.828
    english:
      gpt_4_1:
        metric: "Flesch Reading Ease"
        score: 65.8
        cefr_level: "Standard (B1)"
        bertscore: 0.865
      qwen3_5_4b:
        metric: "Flesch Reading Ease"
        score: 64.2
        cefr_level: "Standard (B1)"
        bertscore: 0.851

  closed_subset_recall_precision_n32:
    tr_1:
      whisper_medium: { tp: 7, precision: 70.0, recall: 21.9, f1: 33.3 }
      whisper_large_v3: { tp: 8, precision: 66.7, recall: 25.0, f1: 36.4 }
      whisper_turbo: { tp: 7, precision: 87.5, recall: 21.9, f1: 35.0 }
      nemotron_3_5: { tp: 9, precision: 100.0, recall: 28.1, f1: 43.9 }
      qwen3_asr_0_6b: { tp: 5, precision: 50.0, recall: 15.6, f1: 23.8 }
      gold_with_qwen9b: { tp: 15, precision: 93.8, recall: 46.9, f1: 62.5 }
    en_1:
      whisper_medium: { tp: 13, precision: 92.9, recall: 40.6, f1: 56.5 }
      whisper_large_v3: { tp: 18, precision: 85.7, recall: 56.2, f1: 67.9 }
      whisper_turbo: { tp: 17, precision: 85.0, recall: 53.1, f1: 65.4 }
      qwen3_asr_0_6b: { tp: 16, precision: 88.9, recall: 50.0, f1: 64.0 }
      nemotron_3_5: { tp: 14, precision: 87.5, recall: 43.8, f1: 58.3 }

critical_defense_notes_and_guardrails:
  guardrail_1_gold_provenance:
    principle: "Referans metinler (Gold) sıfırdan bağımsız çift-kör transkripsiyon değildir."
    details: >
      Dört videoda (TR-1, TR-2, EN-2, EN-3) referansların tabanı otomatik altyazı ve ASR taslaklarına dayanır.
      İnsan denetçiler terimleri ve noktalamayı düzeltmiştir (post-editing).
      Bu durum Whisper/altyazı stiline hafif bir biçimsel yakınlık yaratabileceğinden,
      sunumda asla 'sıfırdan kusursuz bağımsız altın referans' iddiasında bulunulmamalıdır.
  guardrail_2_kappa_paradox:
    principle: "k_D = 1.000 göründüğünden daha zayıf bir kanıttır (Prevalence Paradox)."
    details: >
      Tanım doğruluğu (D) etiketinde 112 pozitif ve yalnızca 3 negatif bulunmaktadır.
      Şans eseri beklenen uyum zaten Pe ≈ %95'tir. Sınıf aşırı dengesiz olduğunda k_D'nin 1.000 çıkması
      matematiksel bir sonuçtur. Değerlendirici güvenilirliğinin asıl ayrım kanıtı, sınır vakalardaki
      4 uyuşmazlığı ve negatif sınıfı temsil eden k_R = 0.782 skorudur.
  guardrail_3_no_semantic_recall:
    principle: "Semantik metriklerde kapalı altın sözlük ve gerçek Recall (Duyarlılık) yoktur."
    details: >
      R1% kapalı küme duyarlılığı değil, üretilen terimler üzerindeki kesinlik (precision) skorudur.
      Pilot çalışmada GPT-4.1'in 105 terim, GPT-4o-mini'nin 56 terim çıkarması
      bir çıkarım hacmi/kapsam göstergesidir (coverage indication); kanıtlanmış tam duyarlılık değildir.
  guardrail_4_sample_size_and_language_claims:
    principle: "'Türkçe ASR için İngilizceden daha zordur' gibi genel iddialardan kaçınılmalıdır."
    details: >
      Kıyaslama toplamda yalnızca n=7 video (3 TR, 4 EN) üzerinden alınan ağırlıksız ortalamalara dayanmaktadır.
      Güven aralığı veya istatistiksel anlamlılık testi uygulanmamıştır.
      Farklar konuşmacı temposu, oda akustiği veya YouTube sıkıştırmasından kaynaklanmış olabilir; alçakgönüllü kalınmalıdır.
  guardrail_5_error_propagation:
    principle: "Ablasyon çalışması sürekli bir ASR->LLM hata yayılım eğrisi değildir."
    details: >
      Ablasyon tablosunda yalnızca kaliteli Whisper hipotezleri (WER %1-5) LLM'e beslenmiştir.
      Bu sebeple makalede ve raporda 'düşük WER rejiminde stabilite' olarak çerçevelenmiştir.
      Geniş hata aralığını göstermek üzere sentetik kontrollü gürültü (noise degradation) deneyi eklenmiştir.
  guardrail_6_web_gui_framing:
    principle: "Web stüdyosu ticarileştirilmiş/canlı ürün değil, yerel araştırma PoC demo aracıdır."
    details: >
      FastAPI arayüzü açık CORS (*), bellek içi sözlük iş kuyruğu (in-memory job store) ve geçici WAV dosyaları içerir.
      Yetkilendirme ve kalıcı veri tabanı yoktur. Jüriye bir yazılım mühendisliği ürünü olarak değil,
      uçtan uca boru hattını somutlaştıran laboratuvar gösterim stüdyosu olarak aktarılmalıdır.
  guardrail_7_honest_empty_cells:
    principle: "GPU altın kümesinde çalıştırılmayan modeller tablolarda dürüstçe '---' olarak korunur."
    details: >
      Wav2Vec 2.0 ve SpeechBrain modelleri Türkçe desteği olmayan tarihsel modellerdir.
      GPU altın referansında yeniden koşturulmamış ve makale tablolarında hücreleri uydurulmadan boş bırakılmıştır.

hardware_and_environment:
  host: "Geliştirme ve Test İş İstasyonu (Tatilsepeti Ar-Ge & Koç)"
  cpu: "Intel Core i7-14650HX (16 Çekirdek, 24 İş Parçacığı, 5.2 GHz Turbo)"
  gpu: "NVIDIA GeForce RTX 5060 Laptop GPU (8 GB GDDR6 VRAM, Driver 616.56, CUDA 13.0)"
  ram: "32 GB DDR5"
  os: "Windows 11 64-bit"
  python_version: "3.12"
  pytorch_version: "2.12.0+cu130"
  transformers_version: ">=5.13"
  reportlab_version: ">=4.4"

key_repository_files:
  paper:
    main_tex: "paper/main.tex (IEEE konferans makalesi tek kaynak dosyası)"
    tables:
      ablation: "paper/tables/ablation.tex"
      asr_2026: "paper/tables/asr_2026.tex"
      asr_bylang: "paper/tables/asr_bylang.tex"
      gold_recall: "paper/tables/gold_recall.tex"
      llm_judge: "paper/tables/llm_judge.tex"
      prompt_strategies: "paper/tables/prompt_strategies.tex"
      readability: "paper/tables/readability.tex"
    figures:
      cross_layer_curve: "paper/figures/cross_layer_curve.tex (TikZ/PGFPlots F1 vs WER eğrisi)"
  experiments:
    asr_runner: "experiments/run_asr.py (Halt eşikli, sequential generate, GPU önbellekli)"
    llm_runner: "experiments/run_llm.py (Qwen3.5 & GPT çıkarım motoru)"
    ablation_runner: "experiments/run_ablation.py (Aynı LLM - çoklu ASR hipotezleri)"
    cross_layer_curve: "experiments/run_cross_layer_curve.py (Sentetik gürültü ve ampirik F1 eğrisi)"
    iaa_evaluator: "experiments/iaa.py (Cohen's Kappa ve PABAK hesaplayıcı)"
    report_generator: "experiments/report.py (LaTeX tablo dışa aktarıcı ve metrik özetleyici)"
    audio_preparer: "experiments/prepare_audio.py (yt-dlp + ffmpeg 16kHz mono dönüştürücü)"
    vad_chunker: "experiments/vad_chunker.py (Silero VAD tabanlı konuşma parçalayıcı)"
    prompt_strategies: "experiments/prompt_strategies.py (Zero-shot, CoT, RAG motoru)"
    gui_subsystem:
      server: "experiments/gui/server.py (FastAPI REST & WebSocket sunucusu)"
      pipeline: "experiments/gui/pipeline.py (Uçtan uca iş yürütme hattı)"
      learner: "experiments/gui/learner.py (Uyarlanabilir öğrenci ve aralıklı tekrar profili)"
      exam: "experiments/gui/exam.py (MCQ ve flashcard üretim motoru)"
      frontend_static: "experiments/gui/static/ (index.html, app.js, style.css)"
      runner_script: "experiments/run_gui.py & run_gui.bat"
  documentation_and_reports:
    graduation_report_md: "notes/Bitirme_Projesi_Raporu.md (Kapsamlı Türkçe bitirme tezi metni)"
    graduation_report_pdf: "notes/Bitirme_Projesi_Raporu.pdf (Biçimlendirilmiş resmi teslim PDF'i)"
    memorandum_pdf: "memorandum.pdf & memorandum.png (Yönetici özeti ve kilit göstergeler)"
    revision_pdf: "notes/What_Added_What_Improved.pdf (Revizyon ve iyileştirmeler raporu)"
    what_changed: "notes/WHAT_CHANGED.md (Pass 1, Pass 2, Pass 3 değişiklik günlüğü)"
    pass2_status: "notes/PASS2_STATUS.md (Deneysel takip ve donanım durum tablosu)"
    dataset_card: "experiments/data/DATASET_CARD.md (Veri kümesi kartı ve yasal izinler)"
    hardware_doc: "experiments/HARDWARE.md (Çözümleme bayrakları ve makine profili)"
  presentation:
    deck_html: "presentation/index.html (İnteraktif web sunumu, konuşmacı modu ve süre sayacı)"
    slides_marp: "presentation/slides.md (Marp uyumlu slayt metni ve konuşmacı notları)"
    qa_guide: "presentation/DEFENSE_QA_GUIDE.md (15 zor jüri sorusu ve 7 guardrail savunma rehberi)"
    launcher: "presentation/serve_presentation.py & run_presentation.bat"

deployment_recommendations:
  accuracy_oriented:
    asr: "Whisper Large-v3 (TR WER 0.021 / EN WER 0.010)"
    llm: "Qwen3.5-9B veya GPT-4.1"
    best_use_case: "Yüksek doğruluk gerektiren resmi ders arşivleme, çeviri ve terminoloji sözlüğü çıkarma"
  latency_speed_oriented:
    asr: "Whisper Large-v3 Turbo (TR RTF 0.044, EN RTF 0.025 - 4 ila 8 kat hızlı)"
    llm: "Qwen3.5-4B veya GPT-4o-mini"
    best_use_case: "Eşzamanlı canlı ders altyazılama, anlık terim kartı gösterimi ve etkileşimli öğrenci stüdyosu"

future_work:
  - "Sistemin Moodle veya Canvas gibi açık kaynaklı LMS platformlarına eklenti olarak entegrasyonu"
  - "Öğrenci geçmiş sınav ve video izleme kayıtlarından beslenen derin Öğrenci Bilgi Takibi (Knowledge Tracing)"
  - "Veri kümesinin 7 klip ötesine (30+ saat) genişletilerek tam çift-kör bağımsız etiketlemenin tamamlanması"
  - "Geliştirilen yerel PoC stüdyosunun Docker konteyneri, Redis iş kuyruğu ve PostgreSQL ile üretime hazır hale getirilmesi"
