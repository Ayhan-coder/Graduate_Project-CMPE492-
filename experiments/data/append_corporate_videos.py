"""Append KVKK (Data Protection) and Occupational Health & Safety (İSG / OSHA) Videos to Corpus Inventory.

Expands the multilingual benchmark from 32 to 40 videos with verified YouTube training resources:
1. TR-17: KVKK Temel Farkındalık Eğitimi (Arda Çetin)
2. TR-18: KVKK ve Şirket İçi Bilgi Güvenliği (İK Kütüphanem Filiz Erboğa)
3. TR-19: Temel İş Sağlığı ve Güvenliği Eğitimi (İzka Osgb TV)
4. TR-20: İş Sağlığı ve Güvenliği: KKD ve Yangın (Fatma AYAZ DEĞERLİ)
5. EN-17: What are the 7 Principles of GDPR? (Data Protection Fundamentals)
6. EN-18: GDPR Awareness Training for Employees (IT Governance)
7. EN-19: Managing Occupational Health and Safety (Workplace Safety Fundamentals)
8. EN-20: OSHA General Industry Safety Orientation for Workers
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_JSON = ROOT / "experiments" / "data" / "data_inventory.json"
INVENTORY_CSV = ROOT / "experiments" / "data" / "data_inventory.csv"

NEW_RECORDS: List[Dict[str, Any]] = [
    # --- TURKISH KVKK (2 Videos) ---
    {
        "id": "TR-17",
        "alt_id": "TR-17",
        "lang": "tr",
        "title": "KVKK Temel Farkındalık Eğitimi: Veri Sorumlusu, Açık Rıza ve Haklar",
        "domain": "Corporate Compliance & Personal Data Protection (KVKK/GDPR)",
        "duration": "21:13",
        "duration_s": 1273.0,
        "source": {
            "platform": "YouTube",
            "channel": "Arda Çetin",
            "url": "https://www.youtube.com/watch?v=uXIrIsNvQBY",
            "youtube_id": "uXIrIsNvQBY",
            "status": "CONFIRMED_PUBLIC_URL"
        },
        "speaker": {
            "id": "TR-SPK-17",
            "gender": "male",
            "accent": "native_tr",
            "delivery": "corporate_lecture"
        },
        "audio": {
            "status": "DOWNLOADABLE_PENDING_FETCH",
            "local_path": None,
            "size_bytes": 0,
            "sample_rate_target": "16kHz Mono PCM WAV"
        },
        "transcript": {
            "status": "CURATED_DRAFT_CAPTION",
            "reference_source": "verified_curated",
            "file_path": None,
            "word_count": 140
        },
        "annotation": {
            "status": "TEMPLATE_STARTER_ONLY (4 candidates)",
            "closed_gold_count": 0,
            "dual_annotator_pairs": 0,
            "template_candidate_count": 4
        },
        "key_domain_terms": [
            "Açık Rıza",
            "Veri Sorumlusu",
            "Veri İşleyen",
            "Özel Nitelikli Kişisel Veri",
            "Aydınlatma Yükümlülüğü"
        ]
    },
    {
        "id": "TR-18",
        "alt_id": "TR-18",
        "lang": "tr",
        "title": "KVKK Farkındalık ve Şirket İçi Bilgi Güvenliği Eğitimi",
        "domain": "Corporate Compliance & Personal Data Protection (KVKK/GDPR)",
        "duration": "10:17",
        "duration_s": 617.0,
        "source": {
            "platform": "YouTube",
            "channel": "İK Kütüphanem Filiz Erboğa",
            "url": "https://www.youtube.com/watch?v=LT7oeWHNQXM",
            "youtube_id": "LT7oeWHNQXM",
            "status": "CONFIRMED_PUBLIC_URL"
        },
        "speaker": {
            "id": "TR-SPK-18",
            "gender": "female",
            "accent": "native_tr",
            "delivery": "corporate_hr_training"
        },
        "audio": {
            "status": "DOWNLOADABLE_PENDING_FETCH",
            "local_path": None,
            "size_bytes": 0,
            "sample_rate_target": "16kHz Mono PCM WAV"
        },
        "transcript": {
            "status": "CURATED_DRAFT_CAPTION",
            "reference_source": "verified_curated",
            "file_path": None,
            "word_count": 115
        },
        "annotation": {
            "status": "TEMPLATE_STARTER_ONLY (4 candidates)",
            "closed_gold_count": 0,
            "dual_annotator_pairs": 0,
            "template_candidate_count": 4
        },
        "key_domain_terms": [
            "VERBİS",
            "Veri Envanteri",
            "İdari ve Teknik Tedbirler",
            "Veri İhlal Bildirimi",
            "Anonimleştirme"
        ]
    },
    # --- TURKISH İSG / OHS (2 Videos) ---
    {
        "id": "TR-19",
        "alt_id": "TR-19",
        "lang": "tr",
        "title": "Temel İş Sağlığı ve Güvenliği Eğitimi: Risk Değerlendirmesi ve Acil Durumlar",
        "domain": "Occupational Health and Safety (İSG / OHS)",
        "duration": "62:10",
        "duration_s": 3730.0,
        "source": {
            "platform": "YouTube",
            "channel": "İzka Osgb TV",
            "url": "https://www.youtube.com/watch?v=ywsXSmfjvYs",
            "youtube_id": "ywsXSmfjvYs",
            "status": "CONFIRMED_PUBLIC_URL"
        },
        "speaker": {
            "id": "TR-SPK-19",
            "gender": "male",
            "accent": "native_tr",
            "delivery": "certified_osgb_instructor"
        },
        "audio": {
            "status": "DOWNLOADABLE_PENDING_FETCH",
            "local_path": None,
            "size_bytes": 0,
            "sample_rate_target": "16kHz Mono PCM WAV"
        },
        "transcript": {
            "status": "CURATED_DRAFT_CAPTION",
            "reference_source": "verified_curated",
            "file_path": None,
            "word_count": 160
        },
        "annotation": {
            "status": "TEMPLATE_STARTER_ONLY (4 candidates)",
            "closed_gold_count": 0,
            "dual_annotator_pairs": 0,
            "template_candidate_count": 4
        },
        "key_domain_terms": [
            "Ramak Kala Olayı",
            "Risk Değerlendirmesi",
            "Tehlike Sınıfları",
            "Acil Durum Eylem Planı",
            "İş Kazası"
        ]
    },
    {
        "id": "TR-20",
        "alt_id": "TR-20",
        "lang": "tr",
        "title": "İş Sağlığı ve Güvenliği Eğitimi: KKD, Güvenlik İşaretleri ve Yangın",
        "domain": "Occupational Health and Safety (İSG / OHS)",
        "duration": "45:50",
        "duration_s": 2750.0,
        "source": {
            "platform": "YouTube",
            "channel": "Fatma AYAZ DEĞERLİ",
            "url": "https://www.youtube.com/watch?v=i0qoHE9VO-8",
            "youtube_id": "i0qoHE9VO-8",
            "status": "CONFIRMED_PUBLIC_URL"
        },
        "speaker": {
            "id": "TR-SPK-20",
            "gender": "female",
            "accent": "native_tr",
            "delivery": "certified_isg_expert"
        },
        "audio": {
            "status": "DOWNLOADABLE_PENDING_FETCH",
            "local_path": None,
            "size_bytes": 0,
            "sample_rate_target": "16kHz Mono PCM WAV"
        },
        "transcript": {
            "status": "CURATED_DRAFT_CAPTION",
            "reference_source": "verified_curated",
            "file_path": None,
            "word_count": 135
        },
        "annotation": {
            "status": "TEMPLATE_STARTER_ONLY (4 candidates)",
            "closed_gold_count": 0,
            "dual_annotator_pairs": 0,
            "template_candidate_count": 4
        },
        "key_domain_terms": [
            "Kişisel Koruyucu Donanım (KKD)",
            "Güvenlik İşaretleri",
            "Yangın Söndürme Tüpü (KKT)",
            "İlk Yardım ve Tahliye",
            "Ergonomi"
        ]
    },
    # --- ENGLISH GDPR / DATA PROTECTION (2 Videos) ---
    {
        "id": "EN-17",
        "alt_id": "EN-17",
        "lang": "en",
        "title": "What Are the 7 Principles of GDPR? Data Protection Fundamentals",
        "domain": "Corporate Compliance & Personal Data Protection (KVKK/GDPR)",
        "duration": "08:00",
        "duration_s": 480.0,
        "source": {
            "platform": "YouTube",
            "channel": "Data Protection World Forum",
            "url": "https://www.youtube.com/watch?v=NcHSD3fWJiQ",
            "youtube_id": "NcHSD3fWJiQ",
            "status": "CONFIRMED_PUBLIC_URL"
        },
        "speaker": {
            "id": "EN-SPK-17",
            "gender": "male",
            "accent": "native_en_uk",
            "delivery": "professional_presenter"
        },
        "audio": {
            "status": "DOWNLOADABLE_PENDING_FETCH",
            "local_path": None,
            "size_bytes": 0,
            "sample_rate_target": "16kHz Mono PCM WAV"
        },
        "transcript": {
            "status": "CURATED_DRAFT_CAPTION",
            "reference_source": "verified_curated",
            "file_path": None,
            "word_count": 125
        },
        "annotation": {
            "status": "TEMPLATE_STARTER_ONLY (4 candidates)",
            "closed_gold_count": 0,
            "dual_annotator_pairs": 0,
            "template_candidate_count": 4
        },
        "key_domain_terms": [
            "Lawfulness, Fairness and Transparency",
            "Purpose Limitation",
            "Data Minimization",
            "Storage Limitation",
            "Accountability"
        ]
    },
    {
        "id": "EN-18",
        "alt_id": "EN-18",
        "lang": "en",
        "title": "GDPR Awareness Training: Employee Responsibilities and Data Breach Protocols",
        "domain": "Corporate Compliance & Personal Data Protection (KVKK/GDPR)",
        "duration": "01:33",
        "duration_s": 93.0,
        "source": {
            "platform": "YouTube",
            "channel": "IT Governance Ltd",
            "url": "https://www.youtube.com/watch?v=O5ol97elmBg",
            "youtube_id": "O5ol97elmBg",
            "status": "CONFIRMED_PUBLIC_URL"
        },
        "speaker": {
            "id": "EN-SPK-18",
            "gender": "female",
            "accent": "native_en_uk",
            "delivery": "corporate_voiceover"
        },
        "audio": {
            "status": "DOWNLOADABLE_PENDING_FETCH",
            "local_path": None,
            "size_bytes": 0,
            "sample_rate_target": "16kHz Mono PCM WAV"
        },
        "transcript": {
            "status": "CURATED_DRAFT_CAPTION",
            "reference_source": "verified_curated",
            "file_path": None,
            "word_count": 75
        },
        "annotation": {
            "status": "TEMPLATE_STARTER_ONLY (4 candidates)",
            "closed_gold_count": 0,
            "dual_annotator_pairs": 0,
            "template_candidate_count": 4
        },
        "key_domain_terms": [
            "Data Breach",
            "Subject Access Request",
            "Data Protection Officer (DPO)",
            "72-Hour Notification Rule"
        ]
    },
    # --- ENGLISH OCCUPATIONAL SAFETY / OSHA (2 Videos) ---
    {
        "id": "EN-19",
        "alt_id": "EN-19",
        "lang": "en",
        "title": "Managing Occupational Health and Safety in the Workplace",
        "domain": "Occupational Health and Safety (İSG / OHS)",
        "duration": "05:22",
        "duration_s": 322.0,
        "source": {
            "platform": "YouTube",
            "channel": "Health and Safety Executive (HSE UK)",
            "url": "https://www.youtube.com/watch?v=LiPNy_WyCTU",
            "youtube_id": "LiPNy_WyCTU",
            "status": "CONFIRMED_PUBLIC_URL"
        },
        "speaker": {
            "id": "EN-SPK-19",
            "gender": "male",
            "accent": "native_en_uk",
            "delivery": "training_officer"
        },
        "audio": {
            "status": "DOWNLOADABLE_PENDING_FETCH",
            "local_path": None,
            "size_bytes": 0,
            "sample_rate_target": "16kHz Mono PCM WAV"
        },
        "transcript": {
            "status": "CURATED_DRAFT_CAPTION",
            "reference_source": "verified_curated",
            "file_path": None,
            "word_count": 98
        },
        "annotation": {
            "status": "TEMPLATE_STARTER_ONLY (4 candidates)",
            "closed_gold_count": 0,
            "dual_annotator_pairs": 0,
            "template_candidate_count": 4
        },
        "key_domain_terms": [
            "Hierarchy of Controls",
            "Elimination and Substitution",
            "Engineering Controls",
            "Safe System of Work (SSOW)"
        ]
    },
    {
        "id": "EN-20",
        "alt_id": "EN-20",
        "lang": "en",
        "title": "OSHA Safety Video Orientation: Hazardous Materials and Workplace Safety",
        "domain": "Occupational Health and Safety (İSG / OHS)",
        "duration": "16:22",
        "duration_s": 982.0,
        "source": {
            "platform": "YouTube",
            "channel": "Safety Training Videos",
            "url": "https://www.youtube.com/watch?v=qbuSZ9rOKNk",
            "youtube_id": "qbuSZ9rOKNk",
            "status": "CONFIRMED_PUBLIC_URL"
        },
        "speaker": {
            "id": "EN-SPK-20",
            "gender": "male",
            "accent": "native_en_us",
            "delivery": "osha_instructor"
        },
        "audio": {
            "status": "DOWNLOADABLE_PENDING_FETCH",
            "local_path": None,
            "size_bytes": 0,
            "sample_rate_target": "16kHz Mono PCM WAV"
        },
        "transcript": {
            "status": "CURATED_DRAFT_CAPTION",
            "reference_source": "verified_curated",
            "file_path": None,
            "word_count": 130
        },
        "annotation": {
            "status": "TEMPLATE_STARTER_ONLY (4 candidates)",
            "closed_gold_count": 0,
            "dual_annotator_pairs": 0,
            "template_candidate_count": 4
        },
        "key_domain_terms": [
            "OSHA Standards",
            "Personal Protective Equipment (PPE)",
            "Hazard Communication",
            "Safety Data Sheets (SDS)",
            "Lockout/Tagout (LOTO)"
        ]
    }
]


def update_inventory():
    with open(INVENTORY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_ids = {r["id"] for r in data["records"]}
    added = 0
    for r in NEW_RECORDS:
        if r["id"] not in existing_ids:
            data["records"].append(r)
            added += 1

    # Sort records cleanly TR-01 to TR-20, then EN-01 to EN-20
    def sort_key(rec):
        prefix, num = rec["id"].split("-")
        return (0 if prefix == "TR" else 1, int(num))

    data["records"].sort(key=sort_key)

    data_dir = ROOT / "experiments" / "data"
    audio_dir = data_dir / "audio"
    gold_dir = data_dir / "gold"
    transcripts_dir = ROOT / "experiments" / "outputs" / "transcripts" / "openai_whisper-large-v3-turbo"

    def _get_wav_dur(p: Path | None) -> float | None:
        if not p or not p.exists():
            return None
        try:
            import wave
            with wave.open(str(p), "rb") as w:
                return round(w.getnframes() / float(w.getframerate()), 1)
        except Exception:
            return None

    # Dynamically verify disk files for all 40 records
    for r in data["records"]:
        cid = r["id"]
        prefix, num = cid.split("-")
        alt_id = f"{prefix}-{int(num)}"
        r["alt_id"] = alt_id

        # 1. Audio WAV check with real duration verification
        wav_file = None
        for cand in [data_dir / f"{cid}.wav", data_dir / f"{alt_id}.wav", audio_dir / f"{cid}.wav", audio_dir / f"{alt_id}.wav"]:
            if cand.exists():
                wav_file = cand
                break

        expected_dur = r.get("duration_s", 0.0)
        disk_dur = _get_wav_dur(wav_file)

        if wav_file and disk_dur is not None and expected_dur > 0:
            diff_s = abs(disk_dur - expected_dur)
            is_matched = diff_s <= max(15.0, 0.05 * expected_dur)
            r["audio"]["local_path"] = str(wav_file.relative_to(ROOT)).replace("\\", "/")
            r["audio"]["size_bytes"] = wav_file.stat().st_size
            r["audio"]["disk_duration_s"] = disk_dur
            if is_matched:
                r["audio"]["status"] = "LOCAL_PCM_WAV_EXISTS"
                r["audio"]["verification"] = "VERIFIED_DURATION_MATCH"
            else:
                r["audio"]["status"] = "AUDIO_UNVERIFIED_PENDING"
                r["audio"]["verification"] = f"MISMATCH (disk: {disk_dur}s vs metadata: {expected_dur}s; raw capture pending slice trim)"
        elif wav_file:
            r["audio"]["status"] = "AUDIO_UNVERIFIED_PENDING"
            r["audio"]["verification"] = "UNKNOWN_DURATION"
            r["audio"]["local_path"] = str(wav_file.relative_to(ROOT)).replace("\\", "/")
            r["audio"]["size_bytes"] = wav_file.stat().st_size
            r["audio"]["disk_duration_s"] = disk_dur
        else:
            r["audio"]["status"] = "DOWNLOADABLE_PENDING_FETCH"
            r["audio"]["verification"] = "NOT_DOWNLOADED"
            r["audio"]["local_path"] = None
            r["audio"]["size_bytes"] = 0
            r["audio"]["disk_duration_s"] = None

        # 2. Transcript check
        gold_cand = None
        for cand in [gold_dir / f"{cid}.txt", gold_dir / f"{alt_id}.txt"]:
            if cand.exists():
                gold_cand = cand
                break

        asr_cand = transcripts_dir / f"{cid}.json"

        # Effective duration for WPM calculation
        effective_dur = disk_dur if (disk_dur and r["audio"]["status"] == "LOCAL_PCM_WAV_EXISTS") else r.get("duration_s", 0.0)

        if gold_cand:
            gold_text = gold_cand.read_text(encoding="utf-8")
            wcnt = len(gold_text.split())
            wpm = round(wcnt / max(0.1, effective_dur / 60.0), 1) if effective_dur > 0 else 0.0
            r["transcript"] = {
                "status": "HUMAN_VERIFIED_GOLD",
                "reference_source": "human_verified_gold",
                "file_path": str(gold_cand.relative_to(ROOT)).replace("\\", "/"),
                "word_count": wcnt,
                "wpm": wpm,
                "qa_passed": True,
            }
        elif asr_cand.exists():
            with open(asr_cand, "r", encoding="utf-8") as af:
                adata = json.load(af)
            wcnt = adata.get("word_count", len(adata.get("full_text", "").split()))
            dur_s = adata.get("duration_s", effective_dur)
            wpm = round(wcnt / max(0.1, dur_s / 60.0), 1) if dur_s > 0 else 0.0
            qa_passed = bool(dur_s > 0 and wpm >= 60.0)
            r["transcript"] = {
                "status": "AUTOMATIC_ASR_TRANSCRIPT" if qa_passed else "AUTOMATIC_ASR_REJECTED_LOW_WPM",
                "reference_source": "whisper_large_v3_turbo_longform",
                "file_path": str(asr_cand.relative_to(ROOT)).replace("\\", "/"),
                "word_count": wcnt,
                "wpm": wpm,
                "qa_passed": qa_passed,
            }
        else:
            # Starter draft caption or missing
            prev_t = r.get("transcript", {})
            wcnt = prev_t.get("word_count", 0)
            wpm = round(wcnt / max(0.1, effective_dur / 60.0), 1) if effective_dur > 0 else 0.0
            r["transcript"] = {
                "status": "CURATED_DRAFT_CAPTION" if wcnt > 0 else "MISSING",
                "reference_source": "curated_draft_starter",
                "file_path": None,
                "word_count": wcnt,
                "wpm": wpm,
                "qa_passed": False,  # Draft captions with low WPM fail full ASR QA
            }

    total_records = len(data["records"])
    tr_count = sum(1 for r in data["records"] if r["lang"] == "tr")
    en_count = sum(1 for r in data["records"] if r["lang"] == "en")

    verified_local_audio = sum(1 for r in data["records"] if r["audio"]["status"] == "LOCAL_PCM_WAV_EXISTS")
    unverified_audio = sum(1 for r in data["records"] if r["audio"]["status"] == "AUDIO_UNVERIFIED_PENDING")
    pending_download = sum(1 for r in data["records"] if r["audio"]["status"] == "DOWNLOADABLE_PENDING_FETCH")
    total_verified_seconds = sum(
        r["audio"]["disk_duration_s"] for r in data["records"]
        if r["audio"]["status"] == "LOCAL_PCM_WAV_EXISTS" and r["audio"].get("disk_duration_s")
    )
    total_inventory_metadata_seconds = sum(r.get("duration_s", 0.0) for r in data["records"])

    gold_transcript = sum(1 for r in data["records"] if r["transcript"]["status"] == "HUMAN_VERIFIED_GOLD")
    asr_full_transcript = sum(1 for r in data["records"] if r["transcript"]["status"] == "AUTOMATIC_ASR_TRANSCRIPT")
    asr_rejected_transcript = sum(1 for r in data["records"] if r["transcript"]["status"] == "AUTOMATIC_ASR_REJECTED_LOW_WPM")
    draft_transcript = sum(1 for r in data["records"] if r["transcript"]["status"] == "CURATED_DRAFT_CAPTION")
    qa_passed_total = sum(1 for r in data["records"] if r["transcript"].get("qa_passed") is True)

    data["total_records"] = total_records
    data["language_counts"] = {"tr": tr_count, "en": en_count}
    data["audio_summary"] = {
        "verified_local_wav_available": verified_local_audio,
        "unverified_pending_trim_mismatch": unverified_audio,
        "downloadable_pending_fetch": pending_download,
        "total_files_on_disk": verified_local_audio + unverified_audio,
        "verified_audio_hours": round(total_verified_seconds / 3600.0, 2),
        "planned_inventory_metadata_hours": round(total_inventory_metadata_seconds / 3600.0, 2),
    }
    data["transcript_summary"] = {
        "human_verified_gold": gold_transcript,
        "automatic_asr_transcript": asr_full_transcript,
        "automatic_asr_rejected_low_wpm": asr_rejected_transcript,
        "curated_draft_caption": draft_transcript,
        "qa_passed_total": qa_passed_total,
    }

    with open(INVENTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Regenerate CSV matching exact schema
    fieldnames = [
        "id", "alt_id", "lang", "domain", "title", "duration", "channel",
        "source_status", "source_url",
        "audio_status", "audio_path", "audio_size_bytes",
        "transcript_status", "transcript_file", "word_count",
        "annotation_status", "closed_gold_terms", "iaa_annotated_pairs", "template_terms"
    ]
    csv_rows = []
    for r in data["records"]:
        ann = r.get("annotation", {})
        csv_rows.append({
            "id": r["id"],
            "alt_id": r["alt_id"],
            "lang": r["lang"],
            "domain": r["domain"],
            "title": r["title"],
            "duration": r["duration"],
            "channel": r["source"]["channel"],
            "source_status": r["source"].get("status", "CONFIRMED_PUBLIC_URL"),
            "source_url": r["source"].get("url", ""),
            "audio_status": r["audio"]["status"],
            "audio_path": r["audio"]["local_path"] or "",
            "audio_size_bytes": r["audio"]["size_bytes"],
            "transcript_status": r["transcript"]["status"],
            "transcript_file": r["transcript"]["file_path"] or "",
            "word_count": r["transcript"]["word_count"],
            "annotation_status": ann.get("status", "TEMPLATE_STARTER_ONLY (4 candidates)"),
            "closed_gold_terms": ann.get("closed_gold_count", 0),
            "iaa_annotated_pairs": ann.get("dual_annotator_pairs", 0),
            "template_terms": ann.get("template_candidate_count", 4),
        })

    with open(INVENTORY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Inventory successfully updated!")
    print(f"  Total records: {total_records} (TR: {tr_count}, EN: {en_count})")
    print(f"  Verified local WAV: {verified_local_audio} / {total_records} ({round(total_verified_seconds/3600.0, 2)} hrs)")
    print(f"  Unverified raw files pending trim: {unverified_audio} (mismatch vs manifest)")
    print(f"  Pending download: {pending_download} / {total_records}")
    print(f"  Transcripts QA Passed: {qa_passed_total} / {total_records} (Gold: {gold_transcript}, Long-form ASR: {asr_full_transcript})")
    print(f"  Transcripts Low WPM / Rejected: {asr_rejected_transcript + draft_transcript}")
    print(f"  JSON updated: {INVENTORY_JSON}")
    print(f"  CSV updated: {INVENTORY_CSV}")


if __name__ == "__main__":
    update_inventory()
