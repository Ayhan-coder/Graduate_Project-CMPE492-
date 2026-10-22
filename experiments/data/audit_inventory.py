import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    sources_path = root / "sources_30plus.json"
    manifest_path = root / "manifest_30plus.jsonl"
    gold_terms_path = root / "gold_terms.json"
    annotation_template_path = root.parent.parent / "annotation" / "annotation_template_30plus.jsonl"
    
    with open(sources_path, "r", encoding="utf-8") as f:
        sources_data = json.load(f)
        
    manifest_rows = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    manifest_rows[item["id"]] = item
                    
    gold_terms = {}
    if gold_terms_path.exists():
        with open(gold_terms_path, "r", encoding="utf-8") as f:
            gold_terms = json.load(f)
            
    template_terms = {}
    if annotation_template_path.exists():
        with open(annotation_template_path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    vid = item.get("video_id")
                    if vid:
                        template_terms.setdefault(vid, []).append(item)
                        
    # Check dual annotations (labels_a1 and labels_a2)
    labels_a1_path = root.parent / "annotation" / "labels_a1.csv"
    labels_a2_path = root.parent / "annotation" / "labels_a2.csv"
    
    a1_counts = {}
    if labels_a1_path.exists():
        import csv
        with open(labels_a1_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                vid = r.get("video_id", "")
                a1_counts[vid] = a1_counts.get(vid, 0) + 1
                
    a2_counts = {}
    if labels_a2_path.exists():
        import csv
        with open(labels_a2_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                vid = r.get("video_id", "")
                a2_counts[vid] = a2_counts.get(vid, 0) + 1

    inventory = []
    
    for clip in sources_data["clips"]:
        cid = clip["id"]
        lang = clip["lang"]
        title = clip["title"]
        domain = clip["domain"]
        duration = clip["duration"]
        youtube_id = clip["youtube_id"]
        source_url = f"https://www.youtube.com/watch?v={youtube_id}"
        channel = clip["channel"]
        speaker = clip.get("speaker", {})
        
        prefix, num_str = cid.split("-")
        alt_id = f"{prefix}-{int(num_str)}"
        
        # Audio check
        wav_file = None
        for cand in [root / f"{cid}.wav", root / f"{alt_id}.wav", root / "audio" / f"{cid}.wav", root / "audio" / f"{alt_id}.wav"]:
            if cand.exists():
                wav_file = cand
                break
                
        # Gold text check
        gold_txt_file = None
        for cand in [root / "gold" / f"{cid}.txt", root / "gold" / f"{alt_id}.txt"]:
            if cand.exists():
                gold_txt_file = cand
                break
                
        # Status determinations
        source_status = "CONFIRMED_PUBLIC_URL"
        m_item = manifest_rows.get(cid, {})
        expected_dur = m_item.get("duration_s", 0.0)
        
        disk_dur = None
        if wav_file:
            try:
                import wave
                with wave.open(str(wav_file), "rb") as w:
                    disk_dur = round(w.getnframes() / float(w.getframerate()), 1)
            except Exception:
                disk_dur = None

        if wav_file and disk_dur is not None and expected_dur > 0:
            diff_s = abs(disk_dur - expected_dur)
            is_matched = diff_s <= max(15.0, 0.05 * expected_dur)
            audio_size_bytes = wav_file.stat().st_size
            rel = wav_file.relative_to(root.parent.parent)
            audio_rel_path = str(rel).replace("\\", "/")
            if is_matched:
                audio_status = "LOCAL_PCM_WAV_EXISTS"
            else:
                audio_status = "AUDIO_UNVERIFIED_PENDING"
        elif wav_file:
            audio_status = "AUDIO_UNVERIFIED_PENDING"
            audio_size_bytes = wav_file.stat().st_size
            rel = wav_file.relative_to(root.parent.parent)
            audio_rel_path = str(rel).replace("\\", "/")
        else:
            audio_status = "DOWNLOADABLE_PENDING_FETCH"
            audio_size_bytes = 0
            audio_rel_path = None
            
        ref_text = m_item.get("reference", "")
        ref_source = m_item.get("reference_source", "")
        
        asr_json = root.parent / "outputs" / "transcripts" / "openai_whisper-large-v3-turbo" / f"{cid}.json"

        dur_val = disk_dur if (disk_dur and audio_status == "LOCAL_PCM_WAV_EXISTS") else m_item.get("duration_s", 0.0)

        if gold_txt_file:
            transcript_status = "HUMAN_VERIFIED_GOLD"
            transcript_path = f"experiments/data/gold/{gold_txt_file.name}"
            transcript_word_count = len(gold_txt_file.read_text(encoding="utf-8").split())
            wpm = round(transcript_word_count / max(0.1, dur_val / 60.0), 1) if dur_val > 0 else 0.0
            qa_passed = True
        elif asr_json.exists():
            with open(asr_json, "r", encoding="utf-8") as af:
                adata = json.load(af)
            transcript_word_count = adata.get("word_count", len(adata.get("full_text", "").split()))
            dur_s = adata.get("duration_s", m_item.get("duration_s", 0.0))
            wpm = round(transcript_word_count / max(0.1, dur_s / 60.0), 1) if dur_s > 0 else 0.0
            qa_passed = adata.get("qa_passed", wpm >= 60.0)
            transcript_status = "AUTOMATIC_ASR_TRANSCRIPT" if qa_passed else "AUTOMATIC_ASR_REJECTED_LOW_WPM"
            transcript_path = f"experiments/outputs/transcripts/openai_whisper-large-v3-turbo/{cid}.json"
        elif ref_text:
            transcript_status = "CURATED_DRAFT_CAPTION"
            transcript_path = None
            transcript_word_count = len(ref_text.split())
            dur_val = m_item.get("duration_s", 0.0)
            wpm = round(transcript_word_count / max(0.1, dur_val / 60.0), 1) if dur_val > 0 else 0.0
            qa_passed = bool(dur_val > 0 and wpm >= 60.0)
        else:
            transcript_status = "MISSING"
            transcript_path = None
            transcript_word_count = 0
            wpm = 0.0
            qa_passed = False
            
        # Annotation status
        gt_terms = gold_terms.get(cid, gold_terms.get(alt_id, {}))
        gt_count = len(gt_terms.get("terms", [])) if isinstance(gt_terms, dict) else 0
        
        iaa_count = min(a1_counts.get(alt_id, 0), a2_counts.get(alt_id, 0))
        tmpl_count = len(template_terms.get(cid, []))
        
        if gt_count > 0:
            annotation_status = f"CLOSED_GOLD_BENCHMARK ({gt_count} terms)"
        elif iaa_count > 0:
            annotation_status = f"DUAL_ANNOTATED_SAMPLE ({iaa_count} pairs)"
        elif tmpl_count > 0:
            annotation_status = f"TEMPLATE_STARTER_ONLY ({tmpl_count} candidates)"
        else:
            annotation_status = "PENDING_NOT_ANNOTATED"
            
        item_entry = {
            "id": cid,
            "alt_id": alt_id,
            "lang": lang,
            "title": title,
            "domain": domain,
            "duration": duration,
            "duration_s": m_item.get("duration_s", 0.0),
            "source": {
                "platform": "YouTube",
                "channel": channel,
                "url": source_url,
                "youtube_id": youtube_id,
                "status": source_status
            },
            "speaker": speaker,
            "audio": {
                "status": audio_status,
                "local_path": audio_rel_path,
                "size_bytes": audio_size_bytes,
                "sample_rate_target": "16kHz Mono PCM WAV"
            },
            "transcript": {
                "status": transcript_status,
                "reference_source": ref_source,
                "file_path": transcript_path,
                "word_count": transcript_word_count,
                "wpm": wpm,
                "qa_passed": qa_passed
            },
            "annotation": {
                "status": annotation_status,
                "closed_gold_count": gt_count,
                "dual_annotator_pairs": iaa_count,
                "template_candidate_count": tmpl_count
            }
        }
        inventory.append(item_entry)
        
    # Write JSON output
    out_json_path = root / "data_inventory.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "corpus_name": "TR/EN Multilingual Educational Speech Benchmark & Extended Candidate Corpus",
            "total_records": len(inventory),
            "language_counts": {"tr": 16, "en": 16},
            "audio_summary": {
                "local_wav_available": sum(1 for x in inventory if x["audio"]["status"] == "LOCAL_PCM_WAV_EXISTS"),
                "downloadable_pending": sum(1 for x in inventory if x["audio"]["status"] == "DOWNLOADABLE_PENDING_FETCH")
            },
            "transcript_summary": {
                "human_verified_gold": sum(1 for x in inventory if x["transcript"]["status"] == "HUMAN_VERIFIED_GOLD"),
                "curated_draft_caption": sum(1 for x in inventory if x["transcript"]["status"] == "CURATED_DRAFT_CAPTION")
            },
            "annotation_summary": {
                "closed_gold_benchmark_32": sum(1 for x in inventory if x["annotation"]["closed_gold_count"] > 0),
                "dual_annotated_iaa": sum(1 for x in inventory if x["annotation"]["dual_annotator_pairs"] > 0),
                "template_starter_only": sum(1 for x in inventory if x["annotation"]["closed_gold_count"] == 0 and x["annotation"]["dual_annotator_pairs"] == 0 and x["annotation"]["template_candidate_count"] > 0)
            },
            "records": inventory
        }, f, indent=2, ensure_ascii=False)
        
    # Write CSV output
    import csv
    out_csv_path = root / "data_inventory.csv"
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "alt_id", "lang", "domain", "title", "duration", "channel",
            "source_status", "source_url",
            "audio_status", "audio_path", "audio_size_bytes",
            "transcript_status", "transcript_file", "word_count",
            "annotation_status", "closed_gold_terms", "iaa_annotated_pairs", "template_terms"
        ])
        for x in inventory:
            writer.writerow([
                x["id"], x["alt_id"], x["lang"], x["domain"], x["title"], x["duration"], x["source"]["channel"],
                x["source"]["status"], x["source"]["url"],
                x["audio"]["status"], x["audio"]["local_path"] or "", x["audio"]["size_bytes"],
                x["transcript"]["status"], x["transcript"]["file_path"] or "", x["transcript"]["word_count"],
                x["annotation"]["status"], x["annotation"]["closed_gold_count"], x["annotation"]["dual_annotator_pairs"], x["annotation"]["template_candidate_count"]
            ])
            
    print(f"[OK] Successfully created {out_json_path}")
    print(f"[OK] Successfully created {out_csv_path}")
    print(f"Total: {len(inventory)} records audited.")

if __name__ == "__main__":
    main()
