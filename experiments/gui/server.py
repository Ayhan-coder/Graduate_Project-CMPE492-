"""FastAPI Backend Server for Speech-to-Terminology Web GUI.

Provides:
- REST API for YouTube metadata extraction.
- Asynchronous task processing with Server-Sent Events (SSE) progress streaming.
- Local audio/video file uploads.
- Synchronized transcript, term cards, and annotation CSV/JSON export.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gui.pipeline import (
    convert_local_audio,
    download_youtube_audio,
    extract_terms,
    get_youtube_info,
    transcribe_audio,
)

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMP_DIR = Path(__file__).resolve().parent / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="EduTerm AI Web GUI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jobs state store: job_id -> {"status": ..., "events": [], "result": ..., "audio_file": ...}
JOBS: dict[str, dict] = {}


class YouTubeRequest(BaseModel):
    url: str
    start_s: float = 0.0
    duration_s: Optional[float] = None
    asr_model: str = "openai/whisper-large-v3-turbo"
    lang: str = "auto"
    llm_model: str = "Qwen/Qwen3.5-4B"


class URLInfoRequest(BaseModel):
    url: str


@app.get("/api/system-info")
def system_info():
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "CPU"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda else 0
    return {
        "cuda": cuda,
        "gpu": gpu,
        "vram_gb": vram_gb,
        "torch_version": torch.__version__,
    }


@app.get("/api/models")
def get_models():
    return {
        "asr_models": [
            {
                "id": "openai/whisper-large-v3-turbo",
                "name": "Whisper Large-v3 Turbo (Recommended)",
                "desc": "Ultra-fast (~0.025 RTF on RTX 5060), high accuracy on TR & EN",
                "default": True,
            },
            {
                "id": "openai/whisper-large-v3",
                "name": "Whisper Large-v3",
                "desc": "Highest benchmark accuracy, slightly slower (~0.20 RTF)",
                "default": False,
            },
            {
                "id": "openai/whisper-medium",
                "name": "Whisper Medium",
                "desc": "Balanced baseline model",
                "default": False,
            },
            {
                "id": "selimc/whisper-large-v3-turbo-turkish",
                "name": "Whisper Turbo (Turkish Fine-Tuned)",
                "desc": "Specialized checkpoint for Turkish lecture speech",
                "default": False,
            },
        ],
        "llm_models": [
            {
                "id": "Qwen/Qwen3.5-4B",
                "name": "Qwen3.5-4B (4-bit Quantized CUDA)",
                "desc": "Local open model, zero API costs, fits in ~3.1 GB VRAM",
                "default": True,
            },
        ],
        "languages": [
            {"code": "auto", "name": "Auto-Detect Language"},
            {"code": "tr", "name": "Turkish (tr)"},
            {"code": "en", "name": "English (en)"},
        ],
    }


@app.post("/api/youtube-info")
def fetch_yt_info(req: URLInfoRequest):
    try:
        info = get_youtube_info(req.url)
        return info
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _push_event(job_id: str, event_type: str, data: dict):
    if job_id in JOBS:
        JOBS[job_id]["events"].append({"event": event_type, "data": data})


def _run_pipeline_job(
    job_id: str,
    audio_wav: Path,
    asr_model: str,
    lang: str,
    llm_model: str,
    video_meta: Optional[dict] = None,
):
    try:
        JOBS[job_id]["status"] = "processing"

        # Step 1: Transcribe ASR
        _push_event(job_id, "step", {"step": "asr", "message": f"Running ASR ({asr_model})..."})

        def asr_cb(msg: str):
            _push_event(job_id, "log", {"message": msg})

        asr_res = transcribe_audio(audio_wav, model_id=asr_model, lang=lang, progress_cb=asr_cb)
        _push_event(
            job_id,
            "asr_done",
            {
                "text": asr_res["text"],
                "duration_s": asr_res["duration_s"],
                "latency_s": asr_res["latency_s"],
                "rtf": asr_res["rtf"],
            },
        )

        # Step 2: Terminology Extraction with LLM
        _push_event(job_id, "step", {"step": "llm", "message": f"Extracting domain terminology ({llm_model})..."})

        def llm_cb(msg: str):
            _push_event(job_id, "log", {"message": msg})

        detected_lang = lang if lang in ("tr", "en") else ("tr" if any(c in asr_res["text"] for c in "çğıöşüÇĞİÖŞÜ") else "en")
        llm_res = extract_terms(asr_res["text"], lang=detected_lang, model_id=llm_model, progress_cb=llm_cb)

        _push_event(
            job_id,
            "llm_done",
            {
                "n_terms": llm_res["n_terms"],
                "latency_s": llm_res["latency_s"],
            },
        )

        # Complete result payload
        final_payload = {
            "job_id": job_id,
            "video_meta": video_meta,
            "lang": detected_lang,
            "asr": asr_res,
            "llm": llm_res,
            "has_audio_file": True,
        }

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = final_payload
        _push_event(job_id, "done", final_payload)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        JOBS[job_id]["status"] = "error"
        _push_event(job_id, "error", {"message": str(exc)})


@app.post("/api/process-youtube")
def process_youtube(req: YouTubeRequest):
    job_id = str(uuid.uuid4())
    audio_wav = TEMP_DIR / f"{job_id}.wav"

    try:
        meta = get_youtube_info(req.url)
    except Exception:
        meta = {"id": "", "title": req.url, "url": req.url, "duration": 0}

    JOBS[job_id] = {
        "status": "queued",
        "events": [],
        "result": None,
        "audio_file": audio_wav,
        "meta": meta,
    }

    def worker():
        try:
            _push_event(job_id, "step", {"step": "download", "message": "Downloading YouTube audio..."})

            def dl_cb(msg: str):
                _push_event(job_id, "log", {"message": msg})

            dur = download_youtube_audio(
                req.url,
                audio_wav,
                start_s=req.start_s,
                duration_s=req.duration_s,
                progress_cb=dl_cb,
            )
            _push_event(job_id, "download_done", {"duration_s": dur})
            _run_pipeline_job(job_id, audio_wav, req.asr_model, req.lang, req.llm_model, meta)
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            _push_event(job_id, "error", {"message": str(e)})

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "meta": meta}


@app.post("/api/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    start_s: float = Form(0.0),
    duration_s: Optional[float] = Form(None),
    asr_model: str = Form("openai/whisper-large-v3-turbo"),
    lang: str = Form("auto"),
    llm_model: str = Form("Qwen/Qwen3.5-4B"),
):
    job_id = str(uuid.uuid4())
    temp_upload = TEMP_DIR / f"upload_{job_id}_{file.filename}"
    audio_wav = TEMP_DIR / f"{job_id}.wav"

    with temp_upload.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    meta = {
        "id": "",
        "title": file.filename,
        "url": "",
        "channel": "Local Upload",
        "duration": 0,
        "thumbnail": "",
    }

    JOBS[job_id] = {
        "status": "queued",
        "events": [],
        "result": None,
        "audio_file": audio_wav,
        "meta": meta,
    }

    def worker():
        try:
            _push_event(job_id, "step", {"step": "convert", "message": "Converting uploaded file..."})

            def conv_cb(msg: str):
                _push_event(job_id, "log", {"message": msg})

            dur = convert_local_audio(
                temp_upload,
                audio_wav,
                start_s=start_s,
                duration_s=duration_s,
                progress_cb=conv_cb,
            )
            try:
                temp_upload.unlink()
            except Exception:
                pass

            _push_event(job_id, "convert_done", {"duration_s": dur})
            _run_pipeline_job(job_id, audio_wav, asr_model, lang, llm_model, meta)
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            _push_event(job_id, "error", {"message": str(e)})

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "meta": meta}


@app.get("/api/stream/{job_id}")
async def stream_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_index = 0
        while True:
            job = JOBS.get(job_id)
            if not job:
                break
            events = job["events"]
            while last_index < len(events):
                ev = events[last_index]
                last_index += 1
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"
            if job["status"] in ("completed", "error"):
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/audio/{job_id}")
def get_audio(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("audio_file") or not Path(job["audio_file"]).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(job["audio_file"], media_type="audio/wav")


@app.get("/api/export-csv/{job_id}")
def export_csv(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Result not found")
    res = job["result"]
    terms = res["llm"]["terms"]
    meta = res.get("video_meta") or {}
    video_id = meta.get("id") or "custom"
    asr_tag = res["asr"]["model"]
    llm_tag = res["llm"]["model"]
    tag = f"{asr_tag}|{llm_tag}"

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["item_id", "video_id", "model", "lang", "term", "explanation", "R", "D", "comment"],
    )
    writer.writeheader()
    for i, t in enumerate(terms, start=1):
        writer.writerow({
            "item_id": f"{video_id}-{tag}-{i:03d}",
            "video_id": video_id,
            "model": tag,
            "lang": res.get("lang", "tr"),
            "term": t.get("term", ""),
            "explanation": t.get("explanation", ""),
            "R": "",
            "D": "",
            "comment": "",
        })
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=terms_{job_id[:8]}.csv"},
    )


# Mount static assets
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return "<h1>EduTerm AI GUI Loading...</h1>"
    return index_file.read_text(encoding="utf-8")
