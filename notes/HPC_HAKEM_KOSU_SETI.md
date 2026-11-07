# HPC / EVREN hakem (judge) kolu koşum seti

Bu sayfa, **makine kolu** içindir. Burada üretilen R/D etiketleri insan etiketinin
yerine geçmez; rapor ve tabloda daima **human–LLM agreement** (insan–LLM uyumu)
olarak raporlanır ve `gold_how=machine_only` olarak kalır. **Inter-annotator
agreement (IAA) denmez** — IAA için iki gerçek insan kolu gerekir.

## 0. Neden HPC/EVREN?

Yerel RTX 5060 (8 GB) üzerinde ölçülen durum (2026-10-30, 40 maddelik sabit alt küme,
`experiments/logs/judge_capability.log`, özet: `experiments/outputs/judge_capability_audit.json`):

| Aday judge | Menşe | Rubrik stili | 40 maddelik ayrıştırılma | Yorum |
|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-1.7B-Instruct | HuggingFace | keyvalue + `R=` prefill | **7/40 (%17,5)** | yakanlanan 7 kaydın **7'si de format yankısı** (`NEDEN=<en çok 12 kelime>`); güvenilir verdict **0** |
| unsloth/Llama-3.2-1B-Instruct | Meta | keyvalue + `R=` prefill | **0/40 (%0,0)** | hiç verdict üretmedi |
| Qwen/Qwen3.5-4B | Qwen (**extractörün kendisi**) | json (protocol-v1) | **39/40 (%97,5)** | çalışıyor ama kendi ürettiği adayların **%97,44'üne R=1, %92,31'ine D=1** veriyor → aile-işi self-grading kontrol kolü |

1,7B ve 1B kollarda model uzun rubrik metnini **olduğu gibi geri yazıyor**
(ör. continuation `R=<0 veya 1>\nD=<0 veya 1>`); verdict üretmediği için `parse()` `None`
dönüyor. Bu, "küçük yerel modelle ikinci kol kurulabilir" tasarım varsayımının
ölçülmüş redsidir — dolayısıyla çapraz-menşe kol için A40 veya EVREN gerekir.

## 1. BOUN HPC — A40 48 GB (önerilen)

Önkoşul: `experiments/annotation/labels_panel_input.csv` (454 madde = 235 makine
adayı + 219 altın-transkriptten bağlamı kurtarılmış pilot madde) repo ile birlikte
HPC düğümünde bulunmalı; çalışma dizimi `experiments/` olsun.

```bash
# tek kol = tek çıktı dosyası; askıya alınırsa aynı komutu yeniden koş (resume'li)
python experiments/llm_judge_protocol.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt-style json \
  --sheet experiments/annotation/labels_panel_input.csv \
  --out experiments/annotation/labels_panel_qwen25_7b_a40.csv

# menşe çeşitliliği için ikinci aday (Qwen ailesi dışı; HF token gerekebilir)
python experiments/llm_judge_protocol.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --prompt-style json \
  --sheet experiments/annotation/labels_panel_input.csv \
  --out experiments/annotation/labels_panel_mistral7b_a40.csv
```

Notlar:
* `experiments/run_llm.py` CUDA varsa **nf4** yükler; A40'ta 7B nf4 ≈ 4-5 GB, bf16 da sığar.
  bf16 istenirse `quantization_config` satırını koşuda atlayacak şekilde yerel bir dal aç
  (bu dosya submission kodu; HPC kopyasında değiştirmek daha güvenli).
* `llm_judge_protocol.py` **aynı rubriği** (PROTOCOL.md R/D) kullanır; A40'taki
  `run_hpc_llm_judge.py` koşusu **farklı rubrik** (B1 düz-dil) kullandı, iki sayı
  yan yana konur ama birbiriyle kıyaslanmaz.
* Süre tahmini: 454 madde × ~1-3 sn/madde (7B, 200 token bütçe) ≈ 15-25 dk.

## 2. EVREN (TÜBİTAK/SSB, H200) — anahtar bekleniyor

2026-10-30 ölçümü: `evren-llmapi.ssyz.org.tr/v1` iki modelde de **HTTP 401
"Geçersiz API anahtarı"** döndürdü (`experiments/logs/evren_key_probe.log`,
`EVREN_API_KEY` tanımlı ama süresi dolmuş; aynı gün iki kez ölçüldü). Anahtar yenilenince:

```bash
# Qwen ailesi DIŞI, bulut — çapraz-menşe kol için birinci tercih
python experiments/llm_judge_protocol.py --model evren/deepseek-v4.1-flash \
  --sheet experiments/annotation/labels_panel_input.csv \
  --out experiments/annotation/labels_panel_evren_deepseek.csv

# Google menşesi, daha büyük model
python experiments/llm_judge_protocol.py --model evren/gemma-4-31b \
  --sheet experiments/annotation/labels_panel_input.csv \
  --out experiments/annotation/labels_panel_evren_gemma.csv
```

Anahtar/sanal makine kontrolü (tek satır, etiket üretmez):

```bash
python - <<'PY'
import sys; sys.path.insert(0, "experiments")
from run_llm import generate_hf
print(generate_hf("evren/deepseek-v4.1-flash", "Yalnızca OK yaz.", max_new_tokens=8))
PY
```

## 3. Uyumlama + triage (yerelde de koşar, GPU istemez)

```bash
python experiments/panel_agreement.py \
  --arms experiments/annotation/labels_panel_qwen25_7b_a40.csv \
         experiments/annotation/labels_panel_mistral7b_a40.csv \
         experiments/annotation/labels_panel_evren_deepseek.csv \
         experiments/annotation/labels_7b_judge.csv \
  --human-candidates experiments/annotation/labels_a1.csv \
  --queue 80
```

Çıktılar:
* `experiments/outputs/panel_agreement.json` — model-model uyumu (**IAA değil**),
  insan–LLM uyumu (`labels_a1`/`labels_a2` gerçek etiketlerine karşı; aday kümesinde
  ayrıca `--human-candidates` koluna karşı) ve aynı satırlarda yeniden hesaplanan
  insan-insan çizgisi.
* `experiments/annotation/labels_todo_human_queue.csv` — uyuşmazlıklar önce
  gelecek şekilde tabakalı kuyruk; **elle etiketlenmiş maddeler kuyruktan düşer**.
  2026-10-30 ölçümü: 235 adayın 180'i elle etiketlenmiş durumdayken kuyruk **55
  maddeye** inmiştir (tamamı EN, 8 klip).

Formu kuyruğa yönelt:

```bash
python experiments/annotation/make_annotation_form.py --annotator a1 \
  --items experiments/annotation/labels_todo_human_queue.csv
python experiments/annotation/make_annotation_form.py --annotator a2 \
  --items experiments/annotation/labels_todo_human_queue.csv
```

## 4. Raporlama kuralları (değişmez)

1. Panel/hakem sayısı **hiçbir tabloda** ikinci insan kolü gibi gösterilmez.
2. κ değerleri için başlık: `model-model agreement` veya `human–LLM agreement`.
3. Panelin accept kararı `gold_how=machine_only` statüsünü değiştirmez; altın
   statü yalnız insan R/D etiketiyle (`labels_a1`/`labels_a2`) kazanılır.
4. Extractör ailesinden bir judge kullanılıyorsa (Qwen3.5-4B gibi) **self-preference
   riski** tablo dibinde ayrıca not edilir.
5. İnsan etiketi yazan kol (a1'in 180 elle etiketi) **tek değerlendiricilidir**; ikinci
   insan kolü olmadığı için bu satırlar hiçbir yerde IAA/κ-insan-insan olarak
   adlandırılamaz. 180 aday satırı 30.10.2026'da `merge_labels.py` ile
   `labels_a1.csv`'e katıştırılmıştır (233 pilot + 180 aday = 413 satır; 233 pilot
   satırı bayt bayt korundu, `iaa.py` yeniden koşulduğunda çift-etiketli küme hâlâ
   $n=115$, $\kappa_R=0{,}782$). Uyum hesaplarında `--human-candidates` artık
   `labels_a1.csv`'yi gösterir.
6. Form ilerlemesi yalnız tarayıcının `localStorage` anahtarında (`eduterm-anno-<a1|a2>`)
   yaşar; "CSV indir"e basılmadan diskte kopyası olmaz. Otomatik doğrulama betikleri bu
   anahtara dokunmaz. 2026-10-30'da bir doğrulama beteri anahtarı silmiş, 180 etiket
   tarayıcının LevelDB yedek günlüğünden (`Partitions/qoder-browser/Local Storage/leveldb/*.log`)
   geri çıkarılıp diske alınmıştır; kayıp yoktur, ancak kopya artık iki dosyada tutulur.
