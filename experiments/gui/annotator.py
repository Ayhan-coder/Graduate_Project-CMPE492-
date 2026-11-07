#!/usr/bin/env python3
"""EduTerm AI - Gold Standard Annotation Studio (Altın Standart Etiketleme Stüdyosu).

A zero-dependency local web app for fast, guided human annotation of technical
terms, CEFR B1 definitions, and Bloom taxonomy levels across lecture clips.
Saves data directly to disk in standard JSON/JSONL format readable by agents and models.
"""

from __future__ import annotations

import functools
import http.server
import json
import logging
import math
import os
import re
import socket
import socketserver
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "experiments" / "data"
MANIFEST_FILE = DATA_DIR / "manifest_30plus.jsonl"
INVENTORY_FILE = DATA_DIR / "data_inventory.json"
GOLD_TERMS_FILE = DATA_DIR / "gold_terms.json"
ANNOTATIONS_DIR = DATA_DIR / "gold_annotations"

ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Annotator")


def calculate_atesman_score(text: str) -> float:
    """Calculate Ateşman Turkish Readability Index (50-69 = Medium / B1 level)."""
    if not text or not text.strip():
        return 60.0
    vowels = "aeıioöuü"
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not words:
        return 60.0
    syllables = sum(sum(1 for c in w if c in vowels) for w in words)
    spw = syllables / max(1, len(words))
    wps = len(words) / max(1, len(sentences))
    score = 198.825 - (40.175 * spw) - (2.610 * wps)
    return max(0.0, min(100.0, round(score, 1)))


def load_all_clips() -> list[dict]:
    """Load clips from manifest_30plus.jsonl and data_inventory.json."""
    clips = []
    seen_ids = set()

    # 1. From manifest_30plus.jsonl
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cid = item.get("id")
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        # Check transcript
                        text = item.get("reference", "")
                        ref_file = item.get("reference_file")
                        if not text and ref_file:
                            candidate_path = ROOT_DIR / "experiments" / ref_file
                            if candidate_path.exists():
                                text = candidate_path.read_text(encoding="utf-8", errors="replace")
                        clips.append({
                            "id": cid,
                            "alt_id": item.get("id"),
                            "lang": item.get("lang", "tr"),
                            "title": item.get("title", ""),
                            "domain": item.get("domain", ""),
                            "channel": item.get("channel", ""),
                            "duration_s": item.get("duration_s", 0),
                            "transcript": text
                        })

    # 2. From data_inventory.json
    if INVENTORY_FILE.exists():
        with open(INVENTORY_FILE, "r", encoding="utf-8") as f:
            inv = json.load(f)
            records = inv.get("records", [])
            for r in records:
                cid = r.get("id")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    # Resolve transcript text if exists
                    t_path = r.get("transcript", {}).get("file_path")
                    text = ""
                    if t_path:
                        full_tp = ROOT_DIR / t_path
                        if full_tp.exists():
                            text = full_tp.read_text(encoding="utf-8", errors="replace")
                    clips.append({
                        "id": cid,
                        "alt_id": r.get("alt_id", cid),
                        "lang": r.get("lang", "tr"),
                        "title": r.get("title", ""),
                        "domain": r.get("domain", ""),
                        "channel": r.get("channel", ""),
                        "duration_s": r.get("duration_s", 0),
                        "transcript": text
                    })

    # Load existing annotations state
    gold_terms_dict = {}
    if GOLD_TERMS_FILE.exists():
        try:
            with open(GOLD_TERMS_FILE, "r", encoding="utf-8") as f:
                gold_terms_dict = json.load(f)
        except Exception:
            pass

    for c in clips:
        cid = c["id"]
        alt_id = c["alt_id"]
        annot_file = ANNOTATIONS_DIR / f"{cid}.json"
        c["is_annotated"] = False
        c["term_count"] = 0
        c["bloom_level"] = "Anlama"

        if annot_file.exists():
            try:
                data = json.loads(annot_file.read_text(encoding="utf-8"))
                c["is_annotated"] = True
                c["term_count"] = len(data.get("terms", []))
                c["bloom_level"] = data.get("bloom_level", "Anlama")
            except Exception:
                pass
        elif cid in gold_terms_dict or alt_id in gold_terms_dict:
            entry = gold_terms_dict.get(cid) or gold_terms_dict.get(alt_id)
            c["is_annotated"] = True
            c["term_count"] = len(entry.get("terms", []))
            c["bloom_level"] = entry.get("bloom_level", "Anlama")

    return sorted(clips, key=lambda x: x["id"])


def get_clip_annotation(cid: str) -> dict:
    """Retrieve annotation for a clip if it exists."""
    annot_file = ANNOTATIONS_DIR / f"{cid}.json"
    if annot_file.exists():
        try:
            return json.loads(annot_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Check gold_terms.json
    if GOLD_TERMS_FILE.exists():
        try:
            with open(GOLD_TERMS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                if cid in d:
                    return d[cid]
                # Try normalized id TR-1 vs TR-01
                norm_id = cid.replace("-0", "-")
                if norm_id in d:
                    return d[norm_id]
        except Exception:
            pass

    return {
        "id": cid,
        "bloom_level": "Anlama",
        "terms": []
    }


def save_clip_annotation(cid: str, data: dict):
    """Save annotation to gold_annotations/{cid}.json and sync with gold_terms.json."""
    data["id"] = cid
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Calculate atesman score for all terms (multi-level B1, B2, C1)
    for t in data.get("terms", []):
        defn_b1 = t.get("definition_b1", "")
        defn_b2 = t.get("definition_b2", "")
        defn_c1 = t.get("definition_c1", "")

        t["atesman_b1"] = calculate_atesman_score(defn_b1) if defn_b1 else None
        t["atesman_b2"] = calculate_atesman_score(defn_b2) if defn_b2 else None
        t["atesman_c1"] = calculate_atesman_score(defn_c1) if defn_c1 else None

        # Primary level score backward compatibility
        prim = t.get("primary_level", "B1")
        if prim == "B2":
            t["atesman_score"] = t["atesman_b2"] or 42.0
        elif prim == "C1":
            t["atesman_score"] = t["atesman_c1"] or 20.0
        else:
            t["atesman_score"] = t["atesman_b1"] or 60.0

    # 1. Save to gold_annotations/{cid}.json
    out_file = ANNOTATIONS_DIR / f"{cid}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 2. Sync to gold_terms.json
    gold_terms_dict = {}
    if GOLD_TERMS_FILE.exists():
        try:
            with open(GOLD_TERMS_FILE, "r", encoding="utf-8") as f:
                gold_terms_dict = json.load(f)
        except Exception:
            pass

    gold_terms_dict[cid] = {
        "domain": data.get("domain", ""),
        "target_level": "CEFR Multi-Level (B1/B2/C1)",
        "lecture_title": data.get("title", ""),
        "bloom_level": data.get("bloom_level", "Anlama"),
        "terms": [
            {
                "canonical": t.get("canonical", ""),
                "aliases": t.get("aliases", []),
                "primary_level": t.get("primary_level", "B1"),
                "definition_b1": t.get("definition_b1", ""),
                "definition_b2": t.get("definition_b2", ""),
                "definition_c1": t.get("definition_c1", ""),
                "atesman_b1": t.get("atesman_b1"),
                "atesman_b2": t.get("atesman_b2"),
                "atesman_c1": t.get("atesman_c1"),
                "atesman_score": t.get("atesman_score", 60.0)
            }
            for t in data.get("terms", [])
        ]
    }

    with open(GOLD_TERMS_FILE, "w", encoding="utf-8") as f:
        json.dump(gold_terms_dict, f, ensure_ascii=False, indent=2)

    logger.info("Saved annotation for %s with %d terms to disk (Multi-level CEFR).", cid, len(data.get("terms", [])))


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EduTerm AI — Altın Standart Etiketleme Stüdyosu</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #070b14;
      --bg-panel: rgba(15, 23, 42, 0.75);
      --bg-card: rgba(30, 41, 59, 0.5);
      --border: rgba(255, 255, 255, 0.08);
      --border-focus: #38bdf8;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-green: #34d399;
      --accent-purple: #a855f7;
      --accent-amber: #fbbf24;
      --accent-red: #f87171;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      background: radial-gradient(circle at 10% 10%, #0f172a 0%, #020617 100%);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    header {
      height: 60px;
      padding: 0 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
      background: rgba(10, 15, 26, 0.8);
      backdrop-filter: blur(12px);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 800;
      font-size: 18px;
      color: #fff;
    }
    .brand-badge {
      font-size: 11px;
      padding: 3px 8px;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid rgba(56, 189, 248, 0.4);
      color: var(--accent);
      border-radius: 6px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .top-stats {
      display: flex;
      align-items: center;
      gap: 18px;
      font-size: 13.5px;
      color: var(--text-muted);
    }
    .top-stats strong { color: var(--accent); font-family: 'JetBrains Mono', monospace; }
    .btn {
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.05);
      color: var(--text);
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover { background: rgba(255, 255, 255, 0.1); border-color: rgba(255, 255, 255, 0.2); }
    .btn-primary { background: var(--accent); color: #020617; border-color: var(--accent); font-weight: 700; }
    .btn-primary:hover { background: #7dd3fc; border-color: #7dd3fc; }
    .btn-guide { background: rgba(168, 85, 247, 0.15); color: #c084fc; border-color: rgba(168, 85, 247, 0.4); }
    .btn-guide:hover { background: rgba(168, 85, 247, 0.25); }

    .main-layout {
      flex: 1;
      display: grid;
      grid-template-columns: 320px 1fr 480px;
      overflow: hidden;
    }

    /* Sol Sütun: Klip Listesi */
    .sidebar {
      border-right: 1px solid var(--border);
      background: rgba(10, 16, 30, 0.6);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .sidebar-search {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
    }
    .search-input {
      width: 100%;
      padding: 9px 12px;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: #fff;
      font-size: 13px;
      outline: none;
    }
    .search-input:focus { border-color: var(--accent); }
    .clip-list {
      flex: 1;
      overflow-y: auto;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .clip-item {
      padding: 12px 14px;
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.4);
      border: 1px solid transparent;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .clip-item:hover { background: rgba(30, 41, 59, 0.5); border-color: rgba(255, 255, 255, 0.05); }
    .clip-item.active { background: rgba(56, 189, 248, 0.1); border-color: rgba(56, 189, 248, 0.4); }
    .clip-item-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .clip-id { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: var(--accent); }
    .clip-status { font-size: 11px; padding: 2px 6px; border-radius: 4px; font-weight: 600; }
    .status-done { background: rgba(52, 211, 153, 0.15); color: var(--accent-green); border: 1px solid rgba(52, 211, 153, 0.3); }
    .status-pending { background: rgba(251, 191, 36, 0.15); color: var(--accent-amber); border: 1px solid rgba(251, 191, 36, 0.3); }
    .clip-title { font-size: 12.5px; font-weight: 600; color: #e2e8f0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .clip-domain { font-size: 11.5px; color: var(--text-muted); }

    /* Orta Sütun: Transkript & Metin Okuma */
    .center-panel {
      display: flex;
      flex-direction: column;
      overflow: hidden;
      border-right: 1px solid var(--border);
      background: rgba(12, 19, 35, 0.4);
    }
    .panel-header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(15, 23, 42, 0.4);
    }
    .panel-header h2 { font-size: 16px; font-weight: 700; color: #fff; }
    .selection-hint {
      font-size: 12px;
      color: var(--accent);
      background: rgba(56, 189, 248, 0.1);
      padding: 4px 10px;
      border-radius: 6px;
      display: none;
      align-items: center;
      gap: 6px;
      cursor: pointer;
    }
    .transcript-viewer {
      flex: 1;
      padding: 24px;
      overflow-y: auto;
      font-size: 15px;
      line-height: 1.8;
      color: #cbd5e1;
      user-select: text;
    }
    .transcript-viewer::selection { background: rgba(56, 189, 248, 0.35); color: #fff; }

    /* Sağ Sütun: Etiketleme Formu */
    .right-panel {
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background: rgba(10, 16, 30, 0.8);
    }
    .form-container {
      flex: 1;
      overflow-y: auto;
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .section-title {
      font-size: 12.5px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    /* Bloom Selector */
    .bloom-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .bloom-btn {
      padding: 10px 8px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.6);
      color: var(--text);
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s;
    }
    .bloom-btn:hover { border-color: rgba(255, 255, 255, 0.2); }
    .bloom-btn.active {
      background: rgba(168, 85, 247, 0.2);
      border-color: var(--accent-purple);
      color: #d8b4fe;
      font-weight: 700;
      box-shadow: 0 0 12px rgba(168, 85, 247, 0.2);
    }
    /* Term Form */
    .term-box {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .form-group label {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
    }
    .input-text, .input-textarea {
      width: 100%;
      background: rgba(10, 15, 26, 0.9);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px 12px;
      color: #fff;
      font-size: 13.5px;
      font-family: inherit;
      outline: none;
    }
    .input-text:focus, .input-textarea:focus { border-color: var(--accent); }
    .input-textarea { resize: vertical; min-height: 70px; line-height: 1.5; }
    
    .atesman-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12px;
      padding: 6px 10px;
      background: rgba(10, 15, 26, 0.6);
      border-radius: 6px;
      margin-top: 2px;
    }
    .atesman-badge {
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      padding: 2px 6px;
      border-radius: 4px;
    }
    .badge-b1 { background: rgba(52, 211, 153, 0.15); color: var(--accent-green); border: 1px solid rgba(52, 211, 153, 0.3); }
    .badge-b2 { background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-c1 { background: rgba(168, 85, 247, 0.15); color: var(--accent-purple); border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-warn { background: rgba(251, 191, 36, 0.15); color: var(--accent-amber); border: 1px solid rgba(251, 191, 36, 0.3); }

    /* CEFR Level Selector */
    .cefr-tabs {
      display: flex;
      gap: 6px;
      margin-bottom: 6px;
    }
    .cefr-tab-btn {
      flex: 1;
      padding: 6px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: rgba(10, 15, 26, 0.7);
      color: var(--text-muted);
      font-size: 11.5px;
      font-weight: 700;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s;
    }
    .cefr-tab-btn:hover { color: var(--text); border-color: rgba(255, 255, 255, 0.2); }
    .cefr-tab-btn.active[data-level="B1"] {
      background: rgba(52, 211, 153, 0.15);
      color: var(--accent-green);
      border-color: rgba(52, 211, 153, 0.4);
    }
    .cefr-tab-btn.active[data-level="B2"] {
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border-color: rgba(56, 189, 248, 0.4);
    }
    .cefr-tab-btn.active[data-level="C1"] {
      background: rgba(168, 85, 247, 0.15);
      color: var(--accent-purple);
      border-color: rgba(168, 85, 247, 0.4);
    }

    /* Term List */
    .term-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 240px;
      overflow-y: auto;
    }
    .term-card {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 13px;
    }
    .term-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .term-canonical { font-weight: 700; color: var(--accent); }
    .term-del-btn {
      color: var(--accent-red);
      cursor: pointer;
      font-size: 11px;
      background: transparent;
      border: none;
      padding: 2px 4px;
    }
    .term-def { font-size: 12px; color: #94a3b8; line-height: 1.4; }

    /* Alt Bar: Kaydet */
    .footer-action {
      padding: 16px 24px;
      border-top: 1px solid var(--border);
      background: rgba(10, 15, 26, 0.9);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* Modal: Rehber */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 100;
    }
    .modal-overlay.open { display: flex; }
    .modal-content {
      width: min(800px, 92vw);
      max-height: 85vh;
      background: #0f172a;
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 16px;
      padding: 32px;
      overflow-y: auto;
      box-shadow: 0 25px 60px rgba(0, 0, 0, 0.9);
    }
    .modal-content h2 { color: var(--accent); margin-bottom: 16px; font-size: 22px; }
    .modal-content h3 { color: #d8b4fe; margin: 18px 0 8px; font-size: 16px; }
    .modal-content p, .modal-content li { font-size: 14px; line-height: 1.6; color: #cbd5e1; }
    .modal-content ul { padding-left: 20px; margin-bottom: 12px; }
    .modal-close {
      margin-top: 24px;
      float: right;
    }
  </style>
</head>
<body>

  <!-- Üst Bar -->
  <header>
    <div class="brand">
      <span>EduTerm AI</span>
      <span class="brand-badge">Altın Standart Etiketleme Stüdyosu</span>
    </div>
    <div class="top-stats">
      <span>Toplam Klip: <strong id="statTotal">0</strong></span>
      <span>Etiketlenen: <strong id="statDone" style="color: var(--accent-green);">0</strong></span>
      <span>Toplam Terim: <strong id="statTerms" style="color: #c084fc;">0</strong></span>
    </div>
    <div style="display: flex; gap: 10px;">
      <button class="btn btn-guide" onclick="openGuide()">📖 Etiketleme Kılavuzu (Nasıl Yapılır?)</button>
    </div>
  </header>

  <!-- Ana Görünüm -->
  <div class="main-layout">

    <!-- Sol Sütun: Klip Listesi -->
    <div class="sidebar">
      <div class="sidebar-search">
        <input type="text" class="search-input" id="clipSearch" placeholder="🔍 Klip ID veya başlık ara..." oninput="filterClips()">
      </div>
      <div class="clip-list" id="clipList">
        <!-- JS ile dolacak -->
      </div>
    </div>

    <!-- Orta Sütun: Ders Transkripti -->
    <div class="center-panel">
      <div class="panel-header">
        <div>
          <h2 id="currentClipTitle">Bir klip seçin...</h2>
          <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;" id="currentClipSub">Disiplin • Süre</div>
        </div>
        <div class="selection-hint" id="selectionHint" onclick="addSelectedTextAsTerm()">
          <span>➕ Seçilen Metni Terim Yap: <strong id="selectedWord"></strong></span>
        </div>
      </div>
      <div class="transcript-viewer" id="transcriptViewer" onmouseup="checkTextSelection()">
        Lütfen sol listeden etiketlemek istediğiniz bir ders klibini seçin.
      </div>
    </div>

    <!-- Sağ Sütun: Etiketleme Paneli -->
    <div class="right-panel">
      <div class="form-container">

        <!-- 1. Bloom Taksonomisi -->
        <div>
          <div class="section-title">1. Dersin Bilişsel Seviyesi (Bloom Taksonomisi)</div>
          <div class="bloom-grid">
            <button class="bloom-btn" data-level="Hatırlama" onclick="setBloom('Hatırlama')">Hatırlama</button>
            <button class="bloom-btn" data-level="Anlama" onclick="setBloom('Anlama')">Anlama</button>
            <button class="bloom-btn" data-level="Uygulama" onclick="setBloom('Uygulama')">Uygulama</button>
            <button class="bloom-btn" data-level="Analiz" onclick="setBloom('Analiz')">Analiz</button>
            <button class="bloom-btn" data-level="Değerlendirme" onclick="setBloom('Değerlendirme')">Değerlendirme</button>
            <button class="bloom-btn" data-level="Yaratma" onclick="setBloom('Yaratma')">Yaratma</button>
          </div>
        </div>

        <!-- 2. Terim Ekleme Formu -->
        <div>
          <div class="section-title">2. Yeni Teknik Terim Ekle</div>
          <div class="term-box">
            <div class="form-group">
              <label>Teknik Terim Adı (Canonical Name)</label>
              <input type="text" class="input-text" id="termCanonical" placeholder="Örn: Özdeğerler, DNA Replikasyonu...">
            </div>
            <div class="form-group">
              <label>Eş Anlamlılar / Kısaltmalar (Virgülle ayırın)</label>
              <input type="text" class="input-text" id="termAliases" placeholder="Örn: eigenvalues, özvektör, char equation">
            </div>
            <div class="form-group">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <label>Hedef Dil Düzeyi (CEFR)</label>
                <span style="font-size: 11px; color: var(--text-muted);" id="cefrTargetHint">B1: 50–69 (Sade)</span>
              </div>
              <div class="cefr-tabs">
                <button type="button" class="cefr-tab-btn active" data-level="B1" onclick="selectCefrLevel('B1')">🌱 B1 (Sade / Lisans 1)</button>
                <button type="button" class="cefr-tab-btn" data-level="B2" onclick="selectCefrLevel('B2')">⚙️ B2 (Teknik / Lisans 3-4)</button>
                <button type="button" class="cefr-tab-btn" data-level="C1" onclick="selectCefrLevel('C1')">🔬 C1 (Uzman / Y.Lisans)</button>
              </div>
              <textarea class="input-textarea" id="termDef" placeholder="Teknik jargona boğmadan, sade açıklama..." oninput="updateAtesmanPreview()"></textarea>
              <div class="atesman-bar">
                <span>Ateşman Okunabilirlik: <strong id="atesmanScore">60.0</strong></span>
                <span class="atesman-badge badge-b1" id="atesmanBadge">B1 (İdeal)</span>
              </div>
            </div>
            <button class="btn btn-primary" onclick="addTermToList()" style="justify-content: center;">
              ➕ Terimi Listeye Ekle
            </button>
          </div>
        </div>

        <!-- 3. Eklenen Terimler -->
        <div>
          <div class="section-title" style="display: flex; justify-content: space-between;">
            <span>3. Eklenen Altın Terimler (<strong id="termCounter">0</strong>)</span>
          </div>
          <div class="term-list" id="termList">
            <!-- JS ile dolacak -->
          </div>
        </div>

      </div>

      <!-- Alt Bar: Diske Kaydet -->
      <div class="footer-action">
        <span style="font-size: 12px; color: var(--text-muted);" id="saveStatus">Değişiklikler kaydedilmedi</span>
        <button class="btn btn-primary" onclick="saveCurrentAnnotation()">
          💾 Altın Etiketleri Diske Kaydet (JSON)
        </button>
      </div>
    </div>

  </div>

  <!-- Rehber Modalı -->
  <div class="modal-overlay" id="guideModal">
    <div class="modal-content">
      <h2>📖 EduTerm Altın Standart Etiketleme Kılavuzu</h2>
      <p>Bu kılavuz, etiketlemeyi uluslararası standartlara (IAA & Cohen's Kappa) tam uyumlu yapmanızı sağlar.</p>

      <h3>1. Hangi kelime "Teknik Terim"dir?</h3>
      <ul>
        <li><strong>Evet:</strong> Alana özgü kavramlar (örn: <em>"Kirchhoff Akım Kanunu", "Backpropagation", "Eylemsizlik Momenti", "Açık Rıza"</em>).</li>
        <li><strong>Hayır:</strong> Günlük konuşma kelimeleri (örn: <em>"Önemli bir yöntem", "Sistem çalışması", "Verimli sonuç"</em> terim değildir).</li>
        <li><strong>Kural:</strong> Bir derste genellikle <strong>5 ila 12 adet</strong> kilit teknik terim yer alır. Hepsini değil, dersin öğrenilmesini sağlayan çekirdek terimleri seçin.</li>
      </ul>

      <h3>2. B1 Açıklaması Nasıl Yazılmalıdır?</h3>
      <ul>
        <li>Tanımı okuyan bir 1. sınıf lisans öğrencisi sözlüğe bakmadan anlayabilmelidir.</li>
        <li>Cümleler kısa (10-18 kelime), fiiller net olmalıdır.</li>
        <li>Kutunun altındaki <strong>Ateşman Puanı yeşil (50–69)</strong> renkte kalmalıdır. Çok uzun cümleler kurarsanız puan düşer.</li>
      </ul>

      <h3>3. Bloom Taksonomisi Nasıl Seçilir?</h3>
      <ul>
        <li><strong>Hatırlama:</strong> Sadece tanım ve formül ezberletiliyorsa.</li>
        <li><strong>Anlama (En yaygın):</strong> Kavramın mantığı, neden-sonuç ilişkisi ve grafiği açıklanıyorsa.</li>
        <li><strong>Uygulama:</strong> Sayısal örnek, devre hesabı veya kod yazılıyorsa.</li>
        <li><strong>Analiz:</strong> İki yöntem kıyaslanıyor (örn: <em>Overfitting nasıl önlenir? BST vs AVL farkı</em>).</li>
      </ul>

      <button class="btn btn-primary modal-close" onclick="closeGuide()">Anladım, Kapat</button>
    </div>
  </div>

  <script>
    let allClips = [];
    let currentClip = null;
    let currentTerms = [];
    let currentBloom = 'Anlama';
    let currentCefr = 'B1';

    // CEFR Seçimi
    function selectCefrLevel(level) {
      currentCefr = level;
      document.querySelectorAll('.cefr-tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-level') === level);
      });
      const hint = document.getElementById('cefrTargetHint');
      const defEl = document.getElementById('termDef');
      if (level === 'B1') {
        hint.innerText = 'B1 Hedefi: 50.0 – 69.9 (Kısa & Sade)';
        defEl.placeholder = 'Teknik jargona boğmadan, kısa ve sade B1 açıklaması (Örn: "NumPy, sayısal verileri hızlıca işleyen Python aracıdır.")...';
      } else if (level === 'B2') {
        hint.innerText = 'B2 Hedefi: 35.0 – 49.9 (Teknik & İşlevsel)';
        defEl.placeholder = 'Kavramın mekanizmasını ve teknik işleyişini açıklayan B2 tanımı (Örn: "Vektörel matris işlemleriyle bellek verimliliği sağlayan kütüphanedir.")...';
      } else {
        hint.innerText = 'C1 Hedefi: 0.0 – 34.9 (Derin & Akademik)';
        defEl.placeholder = 'Teorik arka plan, soyut mimari ve akademik C1 jargonu içeren tanım (Örn: "Bitişik bellek blokları üzerinde strided indisleme ve broadcasting prensiplerini optimize eden mimaridir.")...';
      }
      updateAtesmanPreview();
    }

    // 1. Sayfa Açılışında Klipleri Yükle
    async function init() {
      const res = await fetch('/api/clips');
      allClips = await res.json();
      renderClipList(allClips);
      updateTopStats();
      if (allClips.length > 0) {
        selectClip(allClips[0].id);
      }
    }

    function updateTopStats() {
      document.getElementById('statTotal').innerText = allClips.length;
      const done = allClips.filter(c => c.is_annotated).length;
      document.getElementById('statDone').innerText = done;
      const terms = allClips.reduce((acc, c) => acc + (c.term_count || 0), 0);
      document.getElementById('statTerms').innerText = terms;
    }

    function renderClipList(clips) {
      const container = document.getElementById('clipList');
      container.innerHTML = clips.map(c => `
        <div class="clip-item ${currentClip && currentClip.id === c.id ? 'active' : ''}" onclick="selectClip('${c.id}')">
          <div class="clip-item-top">
            <span class="clip-id">${c.id}</span>
            <span class="clip-status ${c.is_annotated ? 'status-done' : 'status-pending'}">
              ${c.is_annotated ? '✅ ' + c.term_count + ' Terim' : '⏳ Bekliyor'}
            </span>
          </div>
          <div class="clip-title">${c.title || 'Başlıksız Klip'}</div>
          <div class="clip-domain">${c.domain || 'Genel'} • ${(c.duration_s / 60).toFixed(0)} dk</div>
        </div>
      `).join('');
    }

    function filterClips() {
      const q = document.getElementById('clipSearch').value.toLowerCase();
      const filtered = allClips.filter(c => 
        c.id.toLowerCase().includes(q) || 
        (c.title && c.title.toLowerCase().includes(q)) ||
        (c.domain && c.domain.toLowerCase().includes(q))
      );
      renderClipList(filtered);
    }

    // 2. Klip Seçimi
    async function selectClip(cid) {
      currentClip = allClips.find(c => c.id === cid);
      if (!currentClip) return;

      renderClipList(allClips);

      document.getElementById('currentClipTitle').innerText = `[${currentClip.id}] ${currentClip.title}`;
      document.getElementById('currentClipSub').innerText = `${currentClip.domain} • ${currentClip.channel} • ${(currentClip.duration_s / 60).toFixed(0)} dk`;
      document.getElementById('transcriptViewer').innerText = currentClip.transcript || 'Bu klip için henüz transkript metni bulunamadı.';

      // Get existing annotation
      const res = await fetch(`/api/annotation?id=${cid}`);
      const data = await res.json();
      currentTerms = data.terms || [];
      currentBloom = data.bloom_level || 'Anlama';

      setBloom(currentBloom);
      renderTermList();
      document.getElementById('saveStatus').innerText = 'Değişiklikler kaydedilmedi';
      document.getElementById('saveStatus').style.color = 'var(--text-muted)';
    }

    // 3. Bloom Seçici
    function setBloom(level) {
      currentBloom = level;
      document.querySelectorAll('.bloom-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-level') === level);
      });
    }

    // 4. Metin Seçimi (Transkriptten Fareyle Terim Seçme)
    function checkTextSelection() {
      const sel = window.getSelection().toString().trim();
      const hint = document.getElementById('selectionHint');
      const word = document.getElementById('selectedWord');
      if (sel && sel.length >= 2 && sel.length <= 50) {
        word.innerText = sel;
        hint.style.display = 'inline-flex';
      } else {
        hint.style.display = 'none';
      }
    }

    function addSelectedTextAsTerm() {
      const sel = document.getElementById('selectedWord').innerText;
      if (sel) {
        document.getElementById('termCanonical').value = sel;
        document.getElementById('termDef').focus();
        document.getElementById('selectionHint').style.display = 'none';
      }
    }

    // 5. Ateşman Skor Önizlemesi (B1 / B2 / C1 Çok Düzeyli Doğrulama)
    function updateAtesmanPreview() {
      const text = document.getElementById('termDef').value;
      const scoreEl = document.getElementById('atesmanScore');
      const badgeEl = document.getElementById('atesmanBadge');

      if (!text.trim()) {
        scoreEl.innerText = currentCefr === 'B1' ? '60.0' : (currentCefr === 'B2' ? '42.0' : '25.0');
        badgeEl.className = currentCefr === 'B1' ? 'atesman-badge badge-b1' : (currentCefr === 'B2' ? 'atesman-badge badge-b2' : 'atesman-badge badge-c1');
        badgeEl.innerText = `${currentCefr} (Bekleniyor)`;
        return;
      }
      const vowels = 'aeıioöuü';
      const words = text.toLowerCase().match(/\\b\\w+\\b/g) || [];
      const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
      if (words.length === 0) return;
      
      let syllables = 0;
      words.forEach(w => {
        for (let c of w) if (vowels.includes(c)) syllables++;
      });
      const spw = syllables / words.length;
      const wps = words.length / Math.max(1, sentences.length);
      const score = Math.max(0, Math.min(100, Math.round((198.825 - (40.175 * spw) - (2.610 * wps)) * 10) / 10));
      scoreEl.innerText = score.toFixed(1);

      if (currentCefr === 'B1') {
        if (score >= 50 && score <= 70) {
          badgeEl.className = 'atesman-badge badge-b1';
          badgeEl.innerText = '✅ B1 (İdeal Sade: 50–70)';
        } else if (score < 50) {
          badgeEl.className = 'atesman-badge badge-warn';
          badgeEl.innerText = '⚠️ B1 İçin Ağır (Cümleyi Kısaltın)';
        } else {
          badgeEl.className = 'atesman-badge badge-warn';
          badgeEl.innerText = '⚠️ Çok Basit (A1/A2)';
        }
      } else if (currentCefr === 'B2') {
        if (score >= 35 && score < 50) {
          badgeEl.className = 'atesman-badge badge-b2';
          badgeEl.innerText = '✅ B2 (İdeal Teknik: 35–49)';
        } else if (score >= 50) {
          badgeEl.className = 'atesman-badge badge-warn';
          badgeEl.innerText = '⚠️ B2 İçin Fazla Basit (Detay Ekleyin)';
        } else {
          badgeEl.className = 'atesman-badge badge-warn';
          badgeEl.innerText = '⚠️ Çok Ağır (C1 Düzeyinde)';
        }
      } else { // C1
        if (score < 35) {
          badgeEl.className = 'atesman-badge badge-c1';
          badgeEl.innerText = '✅ C1 (İdeal Akademik: <35)';
        } else {
          badgeEl.className = 'atesman-badge badge-warn';
          badgeEl.innerText = '⚠️ C1 İçin Basit (Daha Teknik Olmalı)';
        }
      }
    }

    // 6. Terim Ekleme & Silme
    function addTermToList() {
      const canon = document.getElementById('termCanonical').value.trim();
      const aliasesRaw = document.getElementById('termAliases').value.trim();
      const def = document.getElementById('termDef').value.trim();

      if (!canon) {
        alert('Lütfen bir terim adı girin!');
        return;
      }

      const aliases = aliasesRaw ? aliasesRaw.split(',').map(s => s.trim()).filter(Boolean) : [];
      if (!aliases.includes(canon.toLowerCase())) {
        aliases.unshift(canon.toLowerCase());
      }

      const currentScore = parseFloat(document.getElementById('atesmanScore').innerText) || 60.0;
      
      const newTerm = {
        canonical: canon,
        aliases: aliases,
        primary_level: currentCefr,
        definition_b1: currentCefr === 'B1' ? def : '',
        definition_b2: currentCefr === 'B2' ? def : '',
        definition_c1: currentCefr === 'C1' ? def : '',
        atesman_score: currentScore
      };
      // Set level specific
      if (currentCefr === 'B1') newTerm.atesman_b1 = currentScore;
      if (currentCefr === 'B2') newTerm.atesman_b2 = currentScore;
      if (currentCefr === 'C1') newTerm.atesman_c1 = currentScore;

      currentTerms.push(newTerm);

      // Formu temizle
      document.getElementById('termCanonical').value = '';
      document.getElementById('termAliases').value = '';
      document.getElementById('termDef').value = '';
      selectCefrLevel('B1');

      renderTermList();
    }

    function removeTerm(index) {
      currentTerms.splice(index, 1);
      renderTermList();
    }

    function renderTermList() {
      const container = document.getElementById('termList');
      document.getElementById('termCounter').innerText = currentTerms.length;

      if (currentTerms.length === 0) {
        container.innerHTML = '<div style="font-size: 12.5px; color: var(--text-muted); text-align: center; padding: 20px;">Henüz bu klip için terim eklenmedi.</div>';
        return;
      }

      container.innerHTML = currentTerms.map((t, i) => {
        const lvl = t.primary_level || 'B1';
        const lvlClass = lvl === 'B1' ? 'badge-b1' : (lvl === 'B2' ? 'badge-b2' : 'badge-c1');
        const def = t.definition_b1 || t.definition_b2 || t.definition_c1 || '<em>Açıklama girilmedi</em>';
        const score = t.atesman_score != null ? t.atesman_score : 60.0;

        return `
          <div class="term-card">
            <div class="term-card-header">
              <div style="display: flex; align-items: center; gap: 8px;">
                <span class="term-canonical">${t.canonical}</span>
                <span class="atesman-badge ${lvlClass}" style="font-size: 10px;">${lvl} (${score})</span>
              </div>
              <button class="term-del-btn" onclick="removeTerm(${i})">Sil ✕</button>
            </div>
            <div style="font-size: 11px; color: #64748b;">${(t.aliases || []).join(', ')}</div>
            <div class="term-def">${def}</div>
          </div>
        `;
      }).join('');
    }

    // 7. Diske Kaydet
    async function saveCurrentAnnotation() {
      if (!currentClip) return;

      const payload = {
        id: currentClip.id,
        title: currentClip.title,
        domain: currentClip.domain,
        bloom_level: currentBloom,
        terms: currentTerms
      };

      const res = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const result = await res.json();
      if (result.success) {
        document.getElementById('saveStatus').innerText = `✅ Diske kaydedildi (${new Date().toLocaleTimeString()})`;
        document.getElementById('saveStatus').style.color = 'var(--accent-green)';
        currentClip.is_annotated = currentTerms.length > 0;
        currentClip.term_count = currentTerms.length;
        renderClipList(allClips);
        updateTopStats();
      } else {
        alert('Kaydetme hatası: ' + result.error);
      }
    }

    // Modal
    function openGuide() { document.getElementById('guideModal').classList.add('open'); }
    function closeGuide() { document.getElementById('guideModal').classList.remove('open'); }

    window.onload = init;
  </script>
</body>
</html>"""


class AnnotationHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Quiet logging

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif path == "/api/clips":
            clips = load_all_clips()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(clips, ensure_ascii=False).encode("utf-8"))
            return

        elif path == "/api/annotation":
            qs = urllib.parse.parse_qs(parsed.query)
            cid = qs.get("id", [""])[0]
            annot = get_clip_annotation(cid)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(annot, ensure_ascii=False).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/save":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                data = json.loads(body.decode("utf-8"))
                cid = data.get("id")
                if not cid:
                    raise ValueError("Missing clip ID")
                save_clip_annotation(cid, data)
                resp = {"success": True, "id": cid, "count": len(data.get("terms", []))}
            except Exception as e:
                resp = {"success": False, "error": str(e)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def find_free_port(start_port: int = 8090, max_attempts: int = 50) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    port = find_free_port(8090)
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("127.0.0.1", port), AnnotationHandler) as httpd:
        url = f"http://127.0.0.1:{port}"
        print("=" * 72)
        print("  EduTerm AI — Altın Standart Etiketleme Stüdyosu Başlatıldı!")
        print(f"  Yerel Adres: {url}")
        print("=" * 72)
        print("  Özellikler:")
        print("    ✨ Sol Panel   : 32 Klip Listesi ve canlı tamamlama durumu")
        print("    📄 Orta Panel  : Ders Transkripti (Fareyle metin seçip tek tıkla terim yap)")
        print("    🧠 Sağ Panel   : Bloom Taksonomisi + Canlı Ateşman B1 Puanı Denetimi")
        print("    💾 Çıktı       : gold_annotations/{clip_id}.json ve gold_terms.json")
        print("=" * 72)
        print("  Durdurmak için pencereyi kapatabilir veya Ctrl+C yapabilirsiniz.")

        threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nEtiketleme Stüdyosu kapatıldı.")


if __name__ == "__main__":
    main()

