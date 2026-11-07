"""Generate a self-contained keyboard annotation form (no server, no deps).

Usage
-----
python experiments/annotation/make_annotation_form.py --annotator a2
python experiments/annotation/make_annotation_form.py --annotator a1 --video TR-19
python experiments/annotation/make_annotation_form.py --annotator a1 --items 15clip

Reads the item sheet given by --items (default: the 32-clip machine_only candidates),
drops items that already have labels in
annotation/labels_<annotator>.csv, embeds the rest as JSON inside the HTML, and
writes annotation_form_<annotator>_<sheet>.html next to it. The annotator opens
the file, labels with the keyboard, and downloads a CSV that merges back into
labels_<annotator>.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "experiments" / "annotation"
ITEMS_ALIASES = {
    "32clip": ANN / "labels_todo_32clip_machine.csv",
    "15clip": ANN / "labels_todo_15clip.csv",
    "100plus-sample": ANN / "labels_todo_sample_100plus.csv",
}

TEMPLATE = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EduTerm etiketleme — {{ANNOTATOR}}</title>
<style>
  :root {
    --bg: #f4f6f9; --card: #ffffff; --ink: #0f172a; --muted: #64748b;
    --line: #d8dee9; --accent: #0e7490; --ok: #15803d; --no: #b91c1c;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 16px/1.55 "Segoe UI", system-ui, sans-serif; }
  header { position: sticky; top: 0; z-index: 5; background: #0f172a; color: #e2e8f0;
           padding: 10px 20px; display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
  header b { color: #67e8f9; font-weight: 600; }
  .bar { flex: 1 1 240px; height: 8px; background: #1e293b; border-radius: 99px; overflow: hidden; min-width: 160px; }
  .bar > i { display: block; height: 100%; width: 0; background: #22d3ee; transition: width .15s; }
  main { max-width: 980px; margin: 26px auto 120px; padding: 0 20px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 26px 28px; }
  .meta { color: var(--muted); font-size: 14px; letter-spacing: .02em; }
  .term { font-size: 32px; font-weight: 650; margin: 10px 0 18px; line-height: 1.2; }
  .label { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 4px; }
  .expl { font-size: 19px; margin-bottom: 20px; }
  .ctx { border-left: 0 solid; background: #f1f5f9; border-radius: 8px; padding: 14px 16px; font-size: 15px; color: #334155; }
  .ctx mark { background: #fef3c7; padding: 0 2px; }
  .mode { margin-top: 26px; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
  .pill { padding: 6px 14px; border-radius: 999px; font-weight: 600; font-size: 14px; }
  .pill.r { background: #e0f2fe; color: #075985; }
  .pill.d { background: #ede9fe; color: #5b21b6; }
  .pill.idle { background: #e2e8f0; color: #475569; }
  .keys { color: var(--muted); font-size: 14px; }
  kbd { background: #0f172a; color: #e2e8f0; border-radius: 5px; padding: 2px 7px; font: 600 13px/1.4 ui-monospace, monospace; }
  .options { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
  button.opt { border: 1px solid var(--line); background: #fff; border-radius: 10px; padding: 14px;
               font-size: 15px; cursor: pointer; text-align: left; color: var(--ink); }
  button.opt:hover { border-color: var(--accent); }
  button.opt b { display: block; font-size: 17px; margin-bottom: 2px; }
  button.opt.yes b { color: var(--ok); }
  button.opt.no b { color: var(--no); }
  textarea { width: 100%; margin-top: 14px; border: 1px solid var(--line); border-radius: 8px;
             padding: 10px; font: 15px/1.5 "Segoe UI", sans-serif; resize: vertical; min-height: 54px; }
  .row { display: flex; justify-content: space-between; gap: 16px; margin-top: 18px; flex-wrap: wrap; }
  .row button { border: 1px solid var(--line); background: #fff; border-radius: 8px; padding: 9px 16px;
                font-size: 14px; cursor: pointer; }
  .row button.primary { background: #0f172a; color: #fff; border-color: #0f172a; }
  .legend { color: var(--muted); font-size: 13px; margin-top: 20px; line-height: 1.7; }
  .done { text-align: center; padding: 40px 0; }
  .done h2 { margin: 0 0 8px; }
  .toast { position: fixed; left: 50%; bottom: 26px; transform: translateX(-50%);
           background: #0f172a; color: #e2e8f0; padding: 10px 18px; border-radius: 10px;
           font-size: 14px; opacity: 0; transition: opacity .2s; pointer-events: none; }
  .toast.on { opacity: 1; }
</style>
</head>
<body>
<header>
  <span><b>EduTerm</b> · etiketleyici: {{ANNOTATOR}}</span>
  <div class="bar"><i id="fill"></i></div>
  <span id="count" class="keys">0/0</span>
  <button class="rowbtn" id="export" style="border:0;background:#22d3ee;color:#083344;border-radius:8px;padding:7px 14px;font-weight:600;cursor:pointer">CSV indir</button>
</header>
<main>
  <div class="card" id="card"></div>
  <div class="legend">
    <b>R1</b> terim transkript bağlamıyla tutarlı &nbsp;·&nbsp; <b>R0</b> değil (yanlış dil, derste yok, alan terimi değil)<br>
    <b>D1</b> açıklama TDK/OED tanımına göre doğru &nbsp;·&nbsp; <b>D0</b> değil (hatalı, döngüsel, jargon)<br>
    Emin değilsen 0 işaretle ve not bırak. <kbd>1</kbd>/<kbd>0</kbd> seç, <kbd>Backspace</kbd> geri, <kbd>s</kbd> atla, <kbd>n</kbd> not.
  </div>
</main>
<div class="toast" id="toast"></div>
<script>
const ITEMS = {{ITEMS}};
const ANN = "{{ANNOTATOR}}";
const KEY = "eduterm-anno-" + ANN;
const state = JSON.parse(localStorage.getItem(KEY) || '{"labels":{},"index":0}');
const labels = state.labels || {};
let index = Math.min(state.index || 0, ITEMS.length);

const card = document.getElementById('card');
const fill = document.getElementById('fill');
const countEl = document.getElementById('count');
const toastEl = document.getElementById('toast');

let mode = 'R';
let draft = null;

function save() { localStorage.setItem(KEY, JSON.stringify({labels, index})); }

function toast(msg) {
  toastEl.textContent = msg; toastEl.classList.add('on');
  clearTimeout(toast.t); toast.t = setTimeout(() => toastEl.classList.remove('on'), 1200);
}

function esc(s) { return String(s || '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function markTerm(ctx, term) {
  const safe = esc(ctx);
  if (!term) return safe;
  const t = esc(term).replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
  return safe.replace(new RegExp('(' + t + ')', 'ig'), '<mark>$1</mark>');
}

function labeledCount() { return ITEMS.filter(it => labels[it.item_id] && labels[it.item_id].R !== undefined && labels[it.item_id].D !== undefined).length; }

function render() {
  const done = labeledCount();
  fill.style.width = (ITEMS.length ? (100 * done / ITEMS.length) : 0) + '%';
  countEl.textContent = done + '/' + ITEMS.length;

  if (index >= ITEMS.length) {
    card.innerHTML = '<div class="done"><h2>Bitti.</h2><p class="meta">' + done + ' madde etiketlendi. Sağ üstteki <b>CSV indir</b> ile dosyayı dışa aktar.</p></div>';
    return;
  }
  const it = ITEMS[index];
  const cur = labels[it.item_id];
  if (!cur || (cur.R === undefined) || (cur.D === undefined)) {
    mode = (!cur || cur.R === undefined) ? 'R' : 'D';
  } else {
    mode = 'R';
  }
  const shown = cur ? Object.assign({}, cur) : {};
  if (draft && draft.item_id === it.item_id) { shown.R = draft.R; shown.D = draft.D; shown.comment = draft.comment; }

  const modeHtml = mode === 'R'
    ? '<span class="pill r">şimdi: R — bağlam uyumu</span><span class="keys"><kbd>1</kbd> tutarlı · <kbd>0</kbd> değil</span>'
    : '<span class="pill d">şimdi: D — tanım doğruluğu</span><span class="keys"><kbd>1</kbd> doğru · <kbd>0</kbd> değil</span>';

  card.innerHTML =
    '<div class="meta">' + esc(it.video_id) + ' · ' + esc(it.lang) + ' · ' + (index + 1) + '. madde' +
      (cur && cur.R !== undefined ? ' · R=' + cur.R : '') +
      (cur && cur.D !== undefined ? ' · D=' + cur.D : '') + '</div>' +
    '<div class="term">' + esc(it.term) + '</div>' +
    '<div class="label">Açıklama</div>' +
    '<div class="expl">' + esc(it.explanation) + '</div>' +
    (it.context ? '<div class="label">Transkript bağlamı</div><div class="ctx">' + markTerm(it.context, it.term) + '</div>' : '') +
    '<div class="mode">' + modeHtml + '</div>' +
    '<div class="options">' +
      '<button class="opt yes" data-v="1"><b>1 — evet</b>' + (mode === 'R' ? 'Terim bağlama uygun' : 'Tanım doğru') + '</button>' +
      '<button class="opt no" data-v="0"><b>0 — hayır</b>' + (mode === 'R' ? 'Bağlama uygun değil' : 'Tanım hatalı') + '</button>' +
    '</div>' +
    '<textarea id="comment" placeholder="Not (isteğe bağlı, n ile odaklan)">' + esc(shown.comment || '') + '</textarea>' +
    '<div class="row">' +
      '<button id="prev">← önceki (Backspace)</button>' +
      '<button id="skip">atla (s) →</button>' +
      '<button id="exp2">CSV indir</button>' +
    '</div>';

  card.querySelectorAll('button.opt').forEach(b => b.addEventListener('click', () => choose(b.dataset.v, b)));
  document.getElementById('prev').addEventListener('click', prev);
  document.getElementById('skip').addEventListener('click', skip);
  document.getElementById('exp2').addEventListener('click', exportCsv);
}

function choose(v, btn) {
  if (index >= ITEMS.length) return;
  const it = ITEMS[index];
  const cur = labels[it.item_id] || {};
  const val = Number(v);
  const comment = (document.getElementById('comment') || {}).value || cur.comment || '';
  if (mode === 'R') {
    labels[it.item_id] = {R: val, D: cur.D, comment};
    draft = {item_id: it.item_id, R: val, D: cur.D, comment};
    mode = 'D';
    save();
    render();
    toast('R=' + val + ' · şimdi D');
  } else {
    labels[it.item_id] = {R: cur.R, D: val, comment};
    draft = null;
    index += 1;
    save();
    render();
    toast('kaydedildi');
  }
}

function prev() {
  if (index > 0) { index -= 1; draft = null; save(); render(); }
}

function skip() { if (index < ITEMS.length) { index += 1; draft = null; save(); render(); } }

function exportCsv() {
  const head = ['item_id','video_id','model','lang','term','explanation','R','D','comment'];
  const rows = [head.join(',')];
  for (const it of ITEMS) {
    const l = labels[it.item_id];
    if (!l || l.R === undefined || l.D === undefined) continue;
    const cell = x => '"' + String(x === undefined || x === null ? '' : x).replace(/"/g, '""') + '"';
    rows.push([it.item_id, it.video_id, it.model, it.lang, it.term, it.explanation, l.R, l.D, l.comment || ''].map(cell).join(','));
  }
  const blob = new Blob(['\\ufeff' + rows.join('\\r\\n')], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'labels_' + ANN + '_new.csv';
  document.body.appendChild(a); a.click(); a.remove();
  toast(rows.length - 1 + ' satır indirildi');
}

document.getElementById('export').addEventListener('click', exportCsv);

document.addEventListener('keydown', e => {
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'textarea') {
    if (e.key === 'Escape') { e.target.blur(); }
    return;
  }
  if (e.key === '1' || e.key === '0') { e.preventDefault(); choose(e.key, null); }
  else if (e.key === 'Backspace') { e.preventDefault(); prev(); }
  else if (e.key === 's' || e.key === 'S') { e.preventDefault(); skip(); }
  else if (e.key === 'n' || e.key === 'N') { const t = document.getElementById('comment'); if (t) { e.preventDefault(); t.focus(); } }
});

render();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True, choices=["a1", "a2"])
    ap.add_argument(
        "--items",
        default="32clip",
        help="item sheet: '32clip' (machine_only candidates), '15clip', '100plus-sample' (stratified human sample) or a CSV path",
    )
    ap.add_argument("--video", default=None, help="restrict to one video_id")
    ap.add_argument("--all", action="store_true", help="ignore already-labeled rows")
    args = ap.parse_args()

    items_csv = ITEMS_ALIASES.get(args.items, Path(args.items))
    tag = re.sub(r"\W+", "-", items_csv.stem.replace("labels_todo_", ""))
    BUILDERS = {
        "32clip": "build_item_sheet_32_machine.py",
        "sample-100plus": "build_sample_100plus.py",
    }
    builder = BUILDERS.get(tag, "build_item_sheet_15clip.py")
    if not items_csv.exists():
        raise SystemExit(f"{items_csv} yok; önce {builder} çalıştır.")

    old = ANN / f"labels_{args.annotator}.csv"
    done = set()
    if old.exists() and not args.all:
        with old.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("R", "")).strip() and str(r.get("D", "")).strip():
                    done.add(r["item_id"])

    items = []
    reader_rows = []
    with items_csv.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            reader_rows.append(r)
            if r["item_id"] in done:
                continue
            if args.video and r["video_id"] != args.video:
                continue
            items.append(
                {
                    "item_id": r["item_id"],
                    "video_id": r["video_id"],
                    "model": r["model"],
                    "lang": r["lang"],
                    "term": r["term"],
                    "explanation": r["explanation"],
                    "context": r.get("context", ""),
                }
            )

    if not items:
        if args.video:
            clips = sorted({i["video_id"] for i in reader_rows})
            raise SystemExit(f"{items_csv.name}: video_id={args.video} eşleşmedi; klipler {clips}")
        raise SystemExit(f"{items_csv.name}: etiketlenecek madde kalmadı (hepsi {old.name} içinde R/D dolu)")

    html = (
        TEMPLATE.replace("{{ANNOTATOR}}", args.annotator)
        .replace("{{ITEMS}}", json.dumps(items, ensure_ascii=False))
    )
    out = ANN / f"annotation_form_{args.annotator}_{tag}.html"
    out.write_text(html, encoding="utf-8")
    print(f"{out} — {len(items)} madde ({len({i['video_id'] for i in items})} klip, kaynak: {items_csv.name})")


if __name__ == "__main__":
    main()
