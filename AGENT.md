---
project_metadata:
  title_tr: "Çok Dilli Eğitim İçin Konuşma Tanıma (ASR), Büyük Dil Modeli (LLM) Tabanlı Terminoloji Çıkarma ve Üretken Video Canlandırma Katmanlarının Üç Katmanlı Değerlendirilmesi"
  title_en: "A Three-Layer Evaluation of Speech-to-Text, LLM-Based Terminology Extraction, and Generative Video-Animation Systems for Multilingual Education"
  project_code: "Bitirme Projesi 2026"
  institutions:
    - name: "Boğaziçi Üniversitesi"
      department: "Bilgisayar Mühendisliği Bölümü"
  authors:
    - name: "Ali Ayhan Günder"
      affiliation: "Boğaziçi Üniversitesi"
  advisor:
    name: "Prof. Dr. Fatih Alagöz"
    affiliation: "Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü"
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
    - "Altın Standart Transkripsiyon Veri Kümesi (7 ders, ~95 dk, 13.467 kelime, DATASET_CARD.md)"
    - "GPU Altın-Yeniden Puanlama ASR Kıyaslaması (Whisper Med/Large-v3/Turbo, Qwen3-ASR, Nemotron 3.5)"
    - "Çift Değerlendirici ve Cohen's Kappa IAA Analizi (N=115, k_R=0.782, k_D=1.000)"
    - "LLM Hakem Etiketlemesi (madde-bazlı çıktılar diskte: ablation 200 çift, EVREN 374 çift). Hakem-insan doğrulaması GERİ ÇEKİLDİ; hakem türevli hücreler üst sınır."
    - "Pedagojik Okunabilirlik ve Leksik Benzerlik Vekili (Ateşman/Flesch bant içi payı %18,1–%45,3; benzerlik leksik vekil, BERTScore ölçülmedi)"
    - "Prompt Stratejileri Kısa Probu (Zero-Shot ile CoT, TR-1/EN-1; RAG kolunun %76,9 F1 iddiası döngüsellik nedeniyle geri çekildi)"
    - "Çapraz Katmanlı ASR->LLM Karşılaştırması (11 ölçülmüş cephe + kontrollü WER merdiveni; merdiven **analitik bozulma modelidir**, ölçüm değildir)"
    - "Katman 3 İşlerlik Kanıtı: diskte 7 render edilmiş açıklayıcı video (22,3–41,8 s, 854×480@15); gecikme/deneme/senkron ölçülmedi"
    - "Ölçülmüş Akustik Müdahale Merdiveni (temel → VAD → VAD+biasing → LLM düzeltme; `outputs/interventions_summary.json`)"
    - "EduTerm AI Studio Web Arayüzü (FastAPI, interaktif DAG, uyarlanabilir öğrenci profili, sınav modülü)"
    - "Tez Savunması İnteraktif Sunum Paketi (presentation/index.html, presentation/slides.md, run_presentation.bat)"
    - "Jüri Savunma ve Zor Sorular Rehberi (presentation/DEFENSE_QA_GUIDE.md)"
    - "Birim Test Paketi: 11 unittest dosyası, 254 test metodu. Yürütme: `cd experiments && python -m unittest discover -s tests -p 'test_*.py'`. 2026-10-12 koşumu: 254/254 OK (test_presentation'daki `presentation` modül yolu hatası düzeltildi). `pytest` kilitli ortamda kurulu değil; eski '220 mock test' ifadesi hiçbir kayıtla desteklenmiyordu."

architecture:
  layer_1_asr:
    input_format: "16 kHz Mono 16-bit PCM WAV"
    chunking_strategy: "Arayüz yolu: sabit, örtüşmesiz 30 s pencere (gui/pipeline.py, CHUNK_S = 30.0). Deney yolu: enerji tabanlı VAD parçalayıcı (vad_chunker.py; Silero DEĞİL), run_asr.py --use-vad ile; ölçümlü etkisi WER'i yükseltiyor. Aile bazlı: Whisper tam ses üzerinden sequential_generate, Qwen3-ASR 30 s, Nemotron 60 s."
    decoding_strategy: "Sequential generate(), max_new_tokens=440; prompt_ids verildiğinde 448 - prompt_len - 16 bütçesine kırpılır (aksi halde generate() ValueError atar)"
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
      - "Semantik Benzerlik: Leksik vekil (karakter trigram kosinüsü + kelime örtüşmesi). Bağlamsal BERTScore ÖlçÜLMEDİ: `bert_score` paketi deney ortamında yok"
      - "BLEU (yalnızca İngilizce, `data/oxford_refs_en.json` referanslarına karşı): 20 hücre / 107 çift, 0,73–6,06, ort. 2,04 — ayırt edici değil"
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
    total_words: 13467  # data/gold/*.txt bosluk ayraclı sayım; normalize N toplamı 13771
    languages: ["tr", "en"]
    domains: 5
  clips:
    - id: "TR-1"
      title: "AÖF Proje Yönetimi - Ünite 7 Konu Anlatımı"
      domain: "Proje Yönetimi"
      duration: "12:00"
      words: 1256
      gold_provenance: "Otomatik altyazı taslağı + eksik insan altyazısı yaması (İnsan kontrolüyle düzeltilmiş)"
    - id: "TR-2"
      title: "Turist Rehberleri için Kapsamlı Turizm Eğitimi (Forteliber)"
      domain: "Turizm"
      duration: "11:17"
      words: 1464
      gold_provenance: "YouTube otomatik altyazısı + ASR taslağı (Forteliber, MICE, UKTKG vb. insan düzeltmeli)"
    - id: "TR-3"
      title: "Tirendaz AI Veri Bilimi Nedir, Python"
      domain: "Veri Bilimi"
      duration: "14:15"
      words: 1583
      gold_provenance: "Sesten doğrudan transkripsiyon taslağı (Tirendaz, van Rossum, Cython insan düzeltmeli)"
    - id: "EN-1"
      title: "Grow with Google: Introduction to Agile Methodologies"
      domain: "Çevik Proje Yönetimi"
      duration: "13:17"
      words: 2168
      gold_provenance: "Resmi Grow-with-Google / Coursera İngilizce altyazısı (Temizlenmiş insan metni)"
    - id: "EN-2"
      title: "What is Sustainable Tourism? (Hayley Stainton)"
      domain: "Sürdürülebilir Turizm"
      duration: "09:16"
      words: 1267
      gold_provenance: "YouTube otomatik altyazısı (Brundtland, greenwashing, three pillars insan düzeltmeli)"
    - id: "EN-3"
      title: "Generative AI in a Nutshell (Henrik Kniberg)"
      domain: "Üretken Yapay Zekâ"
      duration: "17:57"
      words: 3459
      gold_provenance: "YouTube otomatik altyazısı (ChatGPT, GPT-4, Kanban, prompt eng. insan düzeltmeli)"
    - id: "EN-4"
      title: "Understanding Thermal Radiation (Efficient Engineer)"
      domain: "Isı Transferi / Fizik"
      duration: "17:33"
      words: 2270
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
    status: "GERI CEKILDI - madde bazli hakem tahminleri diskte yok, karsilastirma yeniden uretilemez"
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
    replacement_measured:
      source: "labels_a1.csv x labels_a2.csv (115 cift, iki insan) -> outputs/rater_agreement.json"
      R: {Po: 0.965, Pe: 0.841, kappa: 0.782, pabak: 0.930}
      D: {Po: 1.000, Pe: 0.949, kappa: 1.000, pabak: 1.000}
      note: "0.974 degeri D boyutunun olumlu prevalansidir (prevalence_r1 = prevalence_r2 = 0.9739), uyum orani degildir."

  prompt_strategies_comparison:
    status: "MOCK ÇIKTISI - prompt_strategies.py --test ile birebir üretiliyor (P=1.0 zorlanmış, gold listenin ilk 10/14/20 terimi). Yayından kaldırılmıştır."
    mock_strategies_withdrawn:
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
    replacement_measured:
      source: "outputs/prompt_strategies.jsonl (simulated=false) -> paper/tables/prompt_strategies.tex"
      gold_concepts_per_clip: 32
      clips: ["TR-1", "EN-1"]
      runs: "tek geçiş, tek klip/dil; RAG kolu hiç ölçülmedi (gold_terms.json döngüselliği)"
      rows:
        - {clip: TR-1, strategy: zero_shot, terms: 8,  precision_pct: 100.0, recall_pct: 25.0, f1_pct: 40.0, latency_sec: 71.43}
        - {clip: TR-1, strategy: cot,       terms: 11, precision_pct: 100.0, recall_pct: 34.4, f1_pct: 51.2, latency_sec: 75.57}
        - {clip: EN-1, strategy: zero_shot, terms: 18, precision_pct: 88.9,  recall_pct: 50.0, f1_pct: 64.0, latency_sec: 75.38}
        - {clip: EN-1, strategy: cot,       terms: 0,  precision_pct: 0.0,   recall_pct: 0.0,  f1_pct: 0.0,  latency_sec: 80.80, note: "yanıt ayrıştırılamadı, 0 terim"}

  pedagogical_readability_and_bertscore:
    status: "ESKİ HALİ GERİ ÇEKİLDİ - gpt_4_1/qwen3_5_4b için yayılan 61.4/0.842, 58.4/0.828, 65.8/0.865, 64.2/0.851 değerlerinin diskte kaynağı yok ve BERTScore bu ortamda kurulmadı. Aşağıdaki blok outputs/readability_eval.json yeniden hesaplanarak yazıldı."
    source: "outputs/readability_eval.json -> paper/tables/readability.tex"
    similarity_backend: "lexical-proxy::char3gram-cosine+word-overlap (bert_score paketi yok; eval_semantic.compute_bert_score belgeli fallback dalında)"
    turkish:
      metric: "Ateşman İndeksi"
      target_band: "50-69 (Orta Güçlükte)"
      evren_qwen_3_8_flash: {expl: 53, refs: 23, mean: 56.4, in_band_pct: 45.3, lexical_proxy_f1_pct: 30.8}
      evren_deepseek_v4_1_flash: {expl: 107, refs: 32, mean: 65.8, in_band_pct: 39.2, lexical_proxy_f1_pct: 24.1}
      qwen3_5_4b_local_4bit: {expl: 179, refs: 78, mean: 48.0, cell_mean_range: "42.1-53.6", in_band_pct: 41.9, lexical_proxy_f1_pct: 30.5}
    english:
      metric: "Flesch Reading Ease"
      target_band: "60-70 (Standard)"
      evren_qwen_3_8_flash: {expl: 72, refs: 46, mean: 56.7, in_band_pct: 18.1, lexical_proxy_f1_pct: 29.6}
      evren_deepseek_v4_1_flash: {expl: 142, refs: 55, mean: 56.0, in_band_pct: 28.9, lexical_proxy_f1_pct: 26.7}
      qwen3_5_4b_local_4bit: {expl: 254, refs: 141, mean: 56.2, cell_mean_range: "52.4-61.3", in_band_pct: 28.7, lexical_proxy_f1_pct: 28.9}
    notes:
      - "Ortalama bandın içinde olmak her açıklamanın B1 düzeyinde olduğu anlamına gelmez; in_band oranı madde bazında ölçülmüştür."
      - "refs sütunu TDK/Oxford referans tanımına eşleşen açıklamaların sayısıdır; benzerlik sütunu yalnız bu alt küme ortalamasıdır."
      - "Commercial-API modelleri için okunabilirlik/benzerlik artifact'i yok (kayıtlar yapılandırılmamış Markdown); tahmin edilmeyip yayından çıkarıldı."

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

  acoustic_interventions_measured:
    caption: "Aynı 7 klip + aynı openai/whisper-large-v3 ağırlıkları ile ölçülmüş müdahale merdiveni (ağırlıksız normalize WER, makro)"
    artifact: "experiments/outputs/interventions_summary.json"
    regenerators:
      - "python experiments/summarize_interventions.py"
      - "python experiments/make_interventions_table.py -> paper/tables/asr_interventions.tex + asr_interventions_perclip.tex"
    stages:
      - { stage: "(1) Ham taban (GPU)", tr_wer: 0.0207, en_wer: 0.0101, tr_coverage: 0.875, en_coverage: 0.969, tr_rep8: 0.0122, en_rep8: 0.0000 }
      - { stage: "(2) + Enerji VAD", tr_wer: 0.0412, en_wer: 0.0262, rel_delta_tr: "+99.6%", rel_delta_en: "+160.1%" }
      - { stage: "(3) + VAD + Sözlük koşullama", tr_wer: 0.0822, en_wer: 0.0431, rel_delta_tr: "+99.4%", rel_delta_en: "+64.4%" }
      - { stage: "(4) + LLM düzeltme (Qwen3.5-4B)", tr_wer: 0.1339, en_wer: 0.0418, rel_delta_tr: "+62.9%", rel_delta_en: "-2.9%" }
    verdict: "NEGATIVE RESULT: hiçbir kademe ham tabanın altına inmez (0/7 klip). Düzeltme 4/7 klipte hatayı düşürür ama TR-1'de 0,1562→0,3236 ile büyütür."
    per_clip_stage4: { TR-1: 0.3236, TR-2: 0.0358, TR-3: 0.0424, EN-1: 0.0869, EN-2: 0.0310, EN-3: 0.0183, EN-4: 0.0311 }
    leakage: "Koşullama ve düzeltme istemleri data/gold_terms.json'dan üretildiği için TR-1/EN-1'de kapsam metrics'i yorumlanamaz; yalnız WER okunur. Altın listesi olmayan 5 klip (2)->(3) geçişinde bayt düzeyinde değişmez (determinizm kontrolü)."
    withdrawn:
      - "−%83.7 / −%12.4 / −%22.2 bağıl azalma satırları ve 0.129 TR temeli (hiçbir JSONL'den yeniden üretilmiyor)"
      - "'-40 dBFS' eşiği (vad_chunker.py'de hiç kodlanmadı; gerçek eşik 2.2 × gürültü tabanı 25. percentil)"
      - "Terim F1 %74.4 / %76.2 / %80.5 (sütun geçiş kapsamıdır; F1 değildir)"
      - "'%26.3 simüle gürültüde +29.8 F1' stres testi (asr_post_correction.py --test fixture'ı; 'düzeltilmiş' dizim altın referansın kopyası, WER %0.0)"
      - "Eski biasing hücresi ölçülemezdi: prompt + max_new_tokens=440, 448 konumu taşırdı (arşiv: outputs/*.positions-crash-attempt1)"
    cost: "Koşullama duvar saatini TR-1'de 1.8x (129→230 s), EN-1'de 7.2x (55→398 s) katlıyor; düzeltme kademesi 1076.7 s TR + 698.4 s EN ek hesaplama istiyor."
    published_carriers:
      auto_generated: "paper/tables/asr_interventions.tex ve asr_interventions_perclip.tex (yukarıdaki regenerators ile yeniden üretilir)"
      hand_maintained:
        - "experiments/gui/static/poster.html — Kutu 4 hem ölçülmüş merdiveni hem 'Geri çekilen sayılar' kaydını taşır; bu dosyanın regenerator'ı YOKTUR, statik HTML'dir ve elle düzenlenir. Merdiven veya geri çekme kaydı silinirse jüri bulgusu (i) yeniden açılır."
        - "notes/generate_a3_poster_pdf.py -> notes/Bitirme_Projesi_A3_Posteri.pdf (Kutu 4/6/10; script çalıştırılarak yeniden üretilir)"
        - "notes/generate_final_report_pdf.py -> notes/Bitirme_Projesi_Raporu.legacy-reportlab.pdf (YASLANMIŞ/renderer; kanonik rapor md2tex.py + latexmk ile üretilir)"
        - "experiments/generate_memo.py -> memorandum.pdf; presentation/index.html + slides.md + DEFENSE_QA_GUIDE.md"
      rule: "Bu yüzeylerdeki sayılar yalnız interventions_summary.json / asr_by_lang.json / gold/*.txt / ffprobe ölçümlerinden gelmelidir; 'anlık', 'canlı', '4-8 kat', 'akustik ortam', 'altın referansla beslenen 9B' ifadeleri yasaktır."

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
      Kıyaslama toplamda yalnızca n=7 video (3 TR, 4 EN) üzerinden alınan ağırlıksız klip ortalamalarına dayanmaktadır.
      Model-dil hücreleri için %95 bootstrap güven aralığı HESAPLANMIŞTIR (10.000 yeniden örnekleme, `outputs/asr_by_lang.json` → `wer_ci`; ör. Large-v3 TR [0.014, 0.032] / EN [0.008, 0.012], Turbo TR [0.011, 0.052]), ancak formel bir anlamlılık testi (ör. çift taraflı randomizasyon testi) uygulanmamıştır.
      Aralıklar beş çift dilli modelde TR/EN arasında çakışmaz; yalnızca Turbo'da örtüşür (n=3 Türkçe alt kümesinin yayılımı nedeniyle).
      Farklar konuşmacı temposu, oda akustiği veya YouTube sıkıştırmasından kaynaklanmış olabilir; alçakgönüllü kalınmalıdır.
  guardrail_5_error_propagation:
    principle: "Ablasyon çalışması sürekli bir ASR->LLM hata yayılım eğrisi değildir; WER merdiveni de ölçüm değil analitik modeldir."
    details: >
      Ablasyon tablosunda yalnızca kaliteli Whisper hipotezleri (WER %1-5) LLM'e beslenmiştir.
      Bu sebeple makalede ve raporda 'düşük WER rejiminde stabilite' olarak çerçevelenmiştir.
      Kontrollü WER merdiveni (0-40%) altın listeden örneklem alan kapalı form bir bozulma modeli olduğundan
      'ölçülmüş' diye sunulamaz; %2 WER'de model F1 ≈ 0,97 öngörürken ölçülen çıkarım 0,364 (TR) / 0,679 (EN)'dir.
      Yüksek hatalı ASR çıktıları LLM'e beslenmediği için kalite-hata yayılım eğrisi bu çalışmada ölçülmemiştir.
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
  guardrail_8_vad_measured_negative:
    principle: "Enerji tabanlı VAD parçalama ölçüldü ve WER'i KÖTÜLEŞTİRİYOR; 'tekrar döngülerini yok eder' iddiası geri çekilmiştir."
    details: >
      Aynı Whisper Large-v3, aynı 7 klip, tek fark --use-vad: TR makro normalize WER 0,0207 -> 0,0412, EN 0,0101 -> 0,0262.
      Klip bazında artış: TR-1 +0,0174, TR-2 +0,0209, TR-3 +0,0233, EN-1 +0,0131, EN-2 +0,0217, EN-3 +0,0082, EN-4 +0,0216.
      GPU temel koşumunda 8-gram tekrar oranı TR 0,0122 / EN 0,0000 idi; yani sökülecek bir döngü yoktu.
      Terminoloji kapsamı değişmedi (TR 0,875 / EN 0,9688). Eski "-%83,7" satırı ve 0,129 TR temeli (CPU pilot havuzu) yeniden üretilemez.
      VAD arayüzde çağrılmaz: gui/pipeline.py:764 sabit 30 s pencere kullanır.
  guardrail_9_biasing_is_gold_leakage:
    principle: "Terminoloji biasing'i döngüseldir ve eski satırı hiç ölçülemezdi."
    details: >
      prompt_ids, data/gold_terms.json'un ilk 25 kanonik teriminden türetilir; altın liste puanlamada da kullanıldığı için
      ölçüm sızıntıdır ve yalnızca altın listesi bulunan TR-1 / EN-1 kliplerini etkiler. Sütun 'F1' değil 'mention coverage' olarak adlandırılır.
      Yayınlanan "+Domain Biasing 0,018" satırı hiç üretilemezdi: glossary öneki + max_new_tokens=440, Whisper'ın 448 kod çözücü
      konum bütçesini taşırdığı için generate() her klipte ValueError atıyordu (Arşiv: outputs/*.positions-crash-attempt1).
      Düzeltme run_asr.py içinde bütçe kırpma: max_new_tokens = max(64, min(440, 448 - prompt_len - 16)).
  guardrail_10_layer3_is_functional_proof:
    principle: "Katman 3 işlerlik kanıtıdır, kıyaslama değildir."
    details: >
      Diskte 7 render edilmiş video var (22,3-41,8 sn; 854x480@15 fps + AAC 24 kHz; ffprobe ile ölçüldü, bkz. rapor Bölüm 7.4).
      Ölçülmemiş olanlar: deneme sayısı (kaç kendini-onarma turu), uçtan uca üretim gecikmesi, ses-görüntü senkronu, algısal/öğrenme etkisi.
      Manim başarısız olursa 1280x720 başlık kartı fallback'ına düşülür (manimator.py:904-915); yedi artefaktın tamamı 854x480 olduğu için
      diskteki üretimler gerçek Manim render'ıdır, sessiz fallback değildir. Altyazı ipuçları LLM'in beyan ettiği duration_s'nden türetilir (varsayılan 10,0 sn), TTS sesinden ölçülmez; .vtt yazılmaz.
      Figure Walkthrough kolunun diskte tek artefaktı yoktur. manim ve edge-tts requirements.txt'te kilitli değildir.

hardware_and_environment:
  host: "Geliştirme ve Test İş İstasyonu (kişisel tüketici dizüstüsü; donanım imzaları experiments/check_env.py ile doğrulanır)"
  cpu: "Intel Core i7-14650HX (16 çekirdek / 24 iş parçacığı; WMI taban 2,2 GHz, 5,2 GHz üretici turbo değeri)"
  gpu: "NVIDIA GeForce RTX 5060 Laptop GPU (8.151 MiB GDDR6, Sürücü 617.42, CUDA 13.0)"
  ram: "34.057.011.200 B (~32 GB) DDR5"
  os: "Windows 11 64-bit (derleme 26300)"
  python_version: "3.12.9"
  pytorch_version: "2.12.0+cu130"
  transformers_version: "5.16.1"
  reportlab_version: "4.5.1"
  other_pinned: "fastapi 0.141.1, sacrebleu 2.6.0, manim 0.21.0, edge-tts 7.2.8, yt-dlp 2026.8.19, pypdf 6.14.2, pypdfium2 5.13.0 (not: manim/edge-tts requirements.txt'te kilitli değil)"
  reverify_command: "cd experiments && python check_env.py"

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
    vad_chunker: "experiments/vad_chunker.py (ENERJİ tabanlı parçalayıcı; Silero DEĞİL. 30 ms çerçeve, eşik = max(1e-4, 25. yüzdeğil gürültü tabanı × 2,2), ≤200 ms boşluk kapatma, >0,4 s durakta sınır, 12–28 s pencereler. Yalnızca run_asr.py --use-vad ile; arayüz çağırmaz)"
    interventions_summarizer: "experiments/summarize_interventions.py (temel → VAD → VAD+biasing → LLM düzeltme merdivenini artefaktlardan yeniden kurar → outputs/interventions_summary.json)"
    interventions_table_maker: "experiments/make_interventions_table.py (JSON'dan paper/tables/asr_interventions.tex ve asr_interventions_perclip.tex üretir; sayılar elle yazılmaz)"
    post_corrector: "experiments/asr_post_correction.py (LLM tabanlı transkript düzeltme; altın terim kapsamı alias-duyarlı eşleşmeyle raporlanır)"
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
    graduation_report_build: "YETKİLİ ZİNCİR: notları düzenle (Bitirme_Projesi_Raporu.md) -> `cd notes && python md2tex.py && latexmk -g -pdf Bitirme_Projesi_Raporu.tex`. UYARI: `notes/generate_final_report_pdf.py` (ReportLab) AYNI dosya yoluna yazar ve iptal edilmiş iki-katmanlı başlık ile eski sayıları taşır; resmi PDF'i sessizce çakıştırmasın diye canonical yaza karşı kendini korumaya aldı (`--allow-overwrite-report` verilmezse `Bitirme_Projesi_Raporu.legacy-reportlab.pdf` üretir). Bu script bakılmıyor, karşılaştırılmıyor ve jüriye teslim edilmiyor."
    memorandum_pdf: "memorandum.pdf & memorandum.png (Yönetici özeti ve kilit göstergeler)"
    revision_pdf: "notes/What_Added_What_Improved.pdf (Revizyon ve iyileştirmeler raporu)"
    what_changed: "notes/WHAT_CHANGED.md (Pass 1, Pass 2, Pass 3 değişiklik günlüğü)"
    pass2_status: "notes/PASS2_STATUS.md (Deneysel takip ve donanım durum tablosu)"
    dataset_card: "experiments/data/DATASET_CARD.md (Veri kümesi kartı ve yasal izinler)"
    hardware_doc: "experiments/HARDWARE.md (Çözümleme bayrakları ve makine profili)"
  presentation:
    deck_html: "presentation/index.html (İnteraktif web sunumu, konuşmacı modu ve süre sayacı)"
    slides_marp: "presentation/slides.md (Marp uyumlu slayt metni ve konuşmacı notları)"
    qa_guide: "presentation/DEFENSE_QA_GUIDE.md (18 zor jüri sorusu ve 10 guardrail savunma rehberi)"
    launcher: "presentation/serve_presentation.py & run_presentation.bat"

deployment_recommendations:
  accuracy_oriented:
    asr: "Whisper Large-v3 (TR WER 0.021 / EN WER 0.010)"
    llm: "Qwen3.5-9B veya GPT-4.1"
    best_use_case: "Yüksek doğruluk gerektiren resmi ders arşivleme, çeviri ve terminoloji sözlüğü çıkarma"
  latency_speed_oriented:
    asr: "Whisper Large-v3 Turbo (TR RTF 0,044 / EN RTF 0,025; aynı ana makinede ölçülmüş havuzlanmış duvar saati hızı TR 3,4× (110,56 sn → 32,53 sn), EN 7,7× (170,72 sn → 22,10 sn))"
    llm: "Qwen3.5-4B veya GPT-4o-mini"
    best_use_case: "Eşzamanlı canlı ders altyazılama, anlık terim kartı gösterimi ve etkileşimli öğrenci stüdyosu"

future_work:
  - "Sistemin Moodle veya Canvas gibi açık kaynaklı LMS platformlarına eklenti olarak entegrasyonu"
  - "Öğrenci geçmiş sınav ve video izleme kayıtlarından beslenen derin Öğrenci Bilgi Takibi (Knowledge Tracing)"
  - "Veri kümesinin 7 klip ötesine (30+ saat) genişletilerek tam çift-kör bağımsız etiketlemenin tamamlanması"
  - "Geliştirilen yerel PoC stüdyosunun Docker konteyneri, Redis iş kuyruğu ve PostgreSQL ile üretime hazır hale getirilmesi"
