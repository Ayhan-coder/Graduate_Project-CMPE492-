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
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("eduterm.gui")

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gui.exam import (
    ATTENTION_MILESTONES,
    ATTENTION_RESEARCH_INFO,
    analysis_after_exam,
    generate_practice_exam,
    load_exam,
    make_popquiz_question,
    public_questions,
    save_exam,
    score_exam,
)
from gui.learner import (
    annotate_terms,
    build_adaptive_analysis,
    build_learner_prompt_block,
    empty_profile,
    load_profile,
    public_profile,
    record_exam,
    save_profile,
    update_mastery,
)
from gui.pipeline import (
    JobCancelledError,
    YouTubeDownloadError,
    parse_youtube_dl_error,
    clear_gpu_memory,
    convert_local_audio,
    detect_lang,
    download_youtube_audio,
    extract_terms,
    extract_youtube_id,
    fetch_instant_transcript,
    build_slide_specification,
    generate_cornell_study_sheet,
    build_concept_knowledge_graph,
    find_all_term_occurrences,
    get_or_load_whisper,
    get_youtube_info,
    preload_models,
    transcribe_audio,
)
from gui.manimator import (
    produce_student_explainer_video,
    VIDEOS_DIR,
    DEFAULT_EVREN_MODEL,
    DEFAULT_VIDEO_QUALITY,
    VIDEO_QUALITY_TIERS,
    TEMP_DIR as MANIMATOR_TEMP_DIR,
)
from gui.socratic import generate_socratic_response

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMP_DIR = Path(__file__).resolve().parent / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_UPLOAD_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".mp4", ".webm", ".flac",
    ".mkv", ".mov", ".avi", ".wmv", ".m4v", ".ogg", ".opus", ".aac", ".ts"
}
ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4", ".webm", ".mkv", ".mov", ".avi", ".wmv", ".m4v", ".ts"
}
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".wmv": "video/x-ms-wmv",
    ".m4v": "video/mp4",
    ".ts": "video/mp2t",
}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB limit

app = FastAPI(title="EduTerm AI Web GUI", version="1.0.0")


@app.on_event("startup")
def prewarm_default_models():
    def _warmup_worker():
        try:
            time.sleep(1.5)  # Allow server to bind and start listening
            if is_job_active():
                return
            with PIPELINE_LOCK:
                if is_job_active():
                    return
                logger.info("Pre-warming default models in background so runs are instant...")
                get_or_load_whisper("openai/whisper-large-v3-turbo")
                from run_llm import get_or_load_llm
                get_or_load_llm("Qwen/Qwen3.5-4B")
                logger.info("Default models (Whisper + Qwen) are resident in memory.")
        except Exception as exc:
            logger.debug("Background pre-warm notice: %s", exc)

    threading.Thread(target=_warmup_worker, daemon=True).start()

# Jobs state store: job_id -> {"status": ..., "events": [], "result": ..., "audio_file": ..., "cancel_event": ...}
JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()
PIPELINE_LOCK = threading.Lock()


def is_job_active() -> bool:
    """Return True if any job is currently uploading, queued, or processing."""
    return any(j.get("status") in ("queued", "uploading", "processing") for j in JOBS.values())


def get_active_job() -> Optional[dict]:
    """Return summary of the currently active job, if one exists."""
    for jid, j in JOBS.items():
        if j.get("status") in ("queued", "uploading", "processing"):
            return {
                "job_id": jid,
                "status": j.get("status"),
                "meta": j.get("meta"),
            }
    return None


JOB_TTL_S = 1800.0  # 30-minute TTL for finished jobs and audio files


def cleanup_temp_dir(max_age_s: float = 0.0) -> int:
    """Remove temporary audio and video files from TEMP_DIR.
    If max_age_s > 0, only removes files modified more than max_age_s seconds ago.
    If max_age_s <= 0, removes all orphan temporary files not currently in use by an active job.
    """
    removed = 0
    now = time.time()
    if not TEMP_DIR.exists():
        return 0
    with JOB_LOCK:
        active_files = set()
        for j in JOBS.values():
            if j.get("status") in ("queued", "uploading", "processing"):
                if j.get("audio_file"):
                    active_files.add(Path(j["audio_file"]).resolve())
                if j.get("video_file"):
                    active_files.add(Path(j["video_file"]).resolve())
    for f in TEMP_DIR.iterdir():
        if f.is_file():
            try:
                if f.resolve() in active_files:
                    continue
                if max_age_s <= 0 or (now - f.stat().st_mtime) >= max_age_s:
                    f.unlink(missing_ok=True)
                    removed += 1
            except Exception as e:
                logger.warning("Failed to delete temp file %s: %s", f, e)
    return removed


def cleanup_expired_jobs(ttl_s: float = JOB_TTL_S) -> int:
    """Drop completed/error/cancelled jobs older than ttl_s and delete their audio/video files."""
    now = time.time()
    expired_ids = []
    with JOB_LOCK:
        for jid, job in list(JOBS.items()):
            if job.get("status") in ("queued", "uploading", "processing"):
                continue
            created = job.get("created_at", 0.0)
            accessed = job.get("last_accessed", created)
            if (now - accessed) >= ttl_s:
                expired_ids.append(jid)

        for jid in expired_ids:
            job = JOBS.pop(jid, None)
            if job:
                if job.get("audio_file"):
                    try:
                        Path(job["audio_file"]).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning("Failed to remove audio file for expired job %s: %s", jid, e)
                if job.get("video_file"):
                    try:
                        Path(job["video_file"]).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning("Failed to remove video file for expired job %s: %s", jid, e)

    cleanup_temp_dir(max_age_s=ttl_s)
    return len(expired_ids)


def _temp_cleaner_daemon():
    while True:
        try:
            time.sleep(300)  # Every 5 minutes
            cleanup_expired_jobs(ttl_s=JOB_TTL_S)
        except Exception as e:
            logger.warning("Background temp cleaner encountered error: %s", e)


# Startup sweep of TEMP_DIR: purge any leftover temporary files from prior runs
cleanup_temp_dir(max_age_s=0.0)

# Start background periodic cleaner thread (daemon)
_cleaner_thread = threading.Thread(target=_temp_cleaner_daemon, daemon=True, name="EduTermTempCleaner")
_cleaner_thread.start()


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    expected_token = os.environ.get("EDUTERM_AUTH_TOKEN")
    if expected_token and request.url.path.startswith("/api/"):
        auth_header = request.headers.get("Authorization") or request.headers.get("X-Auth-Token")
        token_param = request.query_params.get("token")
        provided = None
        if auth_header:
            if auth_header.lower().startswith("bearer "):
                provided = auth_header[7:].strip()
            else:
                provided = auth_header.strip()
        elif token_param:
            provided = token_param.strip()

        if not provided or not secrets.compare_digest(provided, expected_token):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized: invalid or missing authentication token."},
            )
    return await call_next(request)


DEFAULT_MAX_SCOPE_S = 600.0  # 10 minutes default safety cap for audio processing


class YouTubeRequest(BaseModel):
    url: str
    start_s: float = 0.0
    duration_s: Optional[float] = None
    asr_model: str = "openai/whisper-large-v3-turbo"
    lang: str = "auto"
    llm_model: str = "Qwen/Qwen3.5-4B"
    confirm_long: bool = False


class URLInfoRequest(BaseModel):
    url: str


class ExamGenerateRequest(BaseModel):
    job_id: str
    use_llm: bool = True


class ExamAnswer(BaseModel):
    question_id: str
    selected_index: Optional[int] = None
    self_grade: Optional[str] = None


class ExamSubmitRequest(BaseModel):
    exam_id: str
    answers: list[ExamAnswer]


class PopQuizRequest(BaseModel):
    job_id: str
    current_time_s: float = 0.0


class PopQuizAnswerRequest(BaseModel):
    term: str
    correct: bool
    video_id: Optional[str] = ""
    job_id: Optional[str] = ""


class CornellNotesRequest(BaseModel):
    job_id: str
    custom_title: Optional[str] = None


class HelperSlideRequest(BaseModel):
    job_id: str
    current_time_s: float = 0.0
    llm_analogy: bool = False
    generate_deep_slide: bool = False


class ManimatorGenerateRequest(BaseModel):
    job_id: Optional[str] = ""
    current_time_s: float = 0.0
    topic: Optional[str] = ""
    context: Optional[str] = ""
    lang: str = "tr"
    video_style: str = "manim_animation"  # "manim_animation" or "figure_walkthrough"
    duration_s: int = 50
    voice: Optional[str] = None
    llm_backend: str = "evren"  # "evren" | "local" | "heuristic"; degrades down this chain
    evren_model: str = DEFAULT_EVREN_MODEL
    video_quality: str = DEFAULT_VIDEO_QUALITY  # a key of gui.manimator.VIDEO_QUALITY_TIERS
    model_id: str = "Qwen/Qwen3.5-4B"  # local weights used when llm_backend == "local"


MANIMATOR_HISTORY: list[dict] = []


@app.get("/api/health")
def health():
    return {"ok": True, "service": "eduterm"}


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
                "id": "evren/qwen3.8-flash-next",
                "name": "⚡ EVREN Qwen 3.8 Flash (TÜBİTAK/SSB Bulut • H200 • Ultra Hızlı)",
                "desc": "Yerli H200 GPU altyapısı, semantik katman istemci VRAM'inde ağırlık barındırmaz (ASR katmanı yine yerelde çalışır). Ücretli kredi harcar; kullanımı EVREN_API_KEY ve EVREN_ACCEPT_TERMS=1 opt-in'ine bağlıdır",
                "default": False,
            },
            {
                "id": "evren/deepseek-v4.1-flash",
                "name": "🧠 EVREN DeepSeek V4.1 Flash (TÜBİTAK/SSB Bulut • Derin Muhakeme)",
                "desc": "Gelişmiş akıl yürütme, derin pedagojik bağlam ve CEFR B1 açıklama gücü",
                "default": False,
            },
            {
                "id": "evren/glm-5.3",
                "name": "🔬 EVREN GLM-5.3 (TÜBİTAK/SSB Bulut • Kompleks Analiz)",
                "desc": "Yoğun terminoloji ve teknik dersler için genişletilmiş akıl yürütme",
                "default": False,
            },
            {
                "id": "Qwen/Qwen3.5-4B",
                "name": "Qwen3.5-4B (Yerel CUDA • RTX 5060 • 4-bit)",
                "desc": "Lightweight open 4B model, yerel varsayılan (ücretsiz/çevrimdışı). Bu makinede ölçüldü (outputs/local_vram_probe.json): 3.08 GB ağırlık kalıcı, tam klip çağrısında 4.72 GB tepe ayırıcı / 5.26 GB cihaz kullanımı, 8.12 GB RTX 5060 üzerinde",
                "default": True,
            },
            {
                "id": "Qwen/Qwen3-8B",
                "name": "Qwen3-8B (8 Billion Parameters, High Accuracy)",
                "desc": "Powerful 8B open model (~5.2 GB VRAM in 4-bit - projection from the measured 4B figure, not instrumented)",
                "default": False,
            },
            {
                "id": "Qwen/Qwen3.5-9B",
                "name": "Qwen3.5-9B (9 Billion Parameters, Maximum Capacity)",
                "desc": "Deepest open reasoning capacity benchmark (~6.2 GB VRAM in 4-bit - projection, not instrumented; on this 8 GB card it ran offloaded to host memory at 513 s/clip mean)",
                "default": False,
            },
            {
                "id": "google/gemma-4-E4B-it",
                "name": "Gemma 4 (8 Billion Parameters, Google Open Alternative)",
                "desc": "Google Gemma 4 (~8B total parameters across multimodal towers)",
                "default": False,
            },
            {
                "id": "meta-llama/Llama-3.1-8B-Instruct",
                "name": "Llama 3.1 / 3.3 (8 Billion Parameters • Meta Open Weights • 4-bit CUDA)",
                "desc": "Meta Llama 3.x 8B instruction model for terminology extraction (~5.4 GB VRAM in 4-bit)",
                "default": False,
            },
        ],
        "languages": [
            {"code": "auto", "name": "Auto-Detect Language"},
            {"code": "tr", "name": "Turkish (tr)"},
            {"code": "en", "name": "English (en)"},
        ],
    }


def get_allowed_model_ids() -> tuple[set[str], set[str]]:
    """Return (allowed_asr_ids, allowed_llm_ids) from /api/models and models.json."""
    models_api = get_models()
    asr_ids = {m["id"] for m in models_api.get("asr_models", []) if "id" in m}
    llm_ids = {m["id"] for m in models_api.get("llm_models", []) if "id" in m}

    models_file = ROOT / "models.json"
    if models_file.exists():
        try:
            reg = json.loads(models_file.read_text(encoding="utf-8"))
            for section in ("asr_selected", "asr_pilot"):
                for m in reg.get(section, []):
                    if "id" in m:
                        asr_ids.add(m["id"])
            for m in reg.get("llm_selected", []):
                if "id" in m:
                    llm_ids.add(m["id"])
        except Exception:
            pass

    return asr_ids, llm_ids


@app.post("/api/youtube-info")
def fetch_yt_info(req: URLInfoRequest):
    cleaned = req.url.strip() if req.url else ""
    if not cleaned:
        raise HTTPException(status_code=400, detail="Please enter a valid YouTube URL.")
    vid = extract_youtube_id(cleaned)
    if not vid:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Only valid 11-character YouTube video IDs or URLs are allowed.",
        )
    try:
        info = get_youtube_info(cleaned)
        return info
    except Exception as exc:
        logger.error("Failed to fetch YouTube metadata for '%s': %s", cleaned, exc, exc_info=True)
        parsed_err = parse_youtube_dl_error(str(exc), url=cleaned)
        if parsed_err.error_type != "generic_yt_error":
            detail_msg = parsed_err.user_friendly_message
        else:
            detail_msg = "Failed to retrieve YouTube video metadata. Please verify the URL."
        raise HTTPException(
            status_code=400,
            detail=detail_msg,
        )


@app.get("/api/transcript-preview")
@app.post("/api/transcript-preview")
def get_transcript_preview(url: str = "", req: Optional[URLInfoRequest] = None):
    """Instant transcript and terminology pre-fetch before starting full pipeline."""
    target_url = url or (req.url if req else "")
    cleaned = target_url.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Please provide a valid YouTube URL or clip identifier.")
    try:
        payload = fetch_instant_transcript(cleaned)
        job_id = payload.get("job_id")
        if job_id:
            JOBS[job_id] = {
                "status": "completed",
                "result": payload,
                "events": [],
                "created_at": time.time(),
                "last_accessed": time.time(),
            }
        return payload
    except Exception as exc:
        logger.error("Failed to generate transcript preview for '%s': %s", cleaned, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load transcript preview.")


def _push_event(job_id: str, event_type: str, data: dict):
    if job_id in JOBS:
        JOBS[job_id]["events"].append({"event": event_type, "data": data})


def _set_job_status(job_id: str, status: str, result: Optional[dict] = None):
    if job_id in JOBS:
        JOBS[job_id]["status"] = status
        JOBS[job_id]["last_accessed"] = time.time()
        if result is not None:
            JOBS[job_id]["result"] = result


def align_terms_with_segments(terms: list[dict], segments: list[dict]) -> list[dict]:
    """Map each extracted terminology item to its timestamps and all ASR segment occurrences."""
    for item in terms:
        term_text = (item.get("term") or "").strip()
        if not term_text:
            item["start_s"] = None
            item["end_s"] = None
            item["timestamp"] = None
            item["occurrences"] = []
            item["occurrences_count"] = 0
            continue

        occurrences = find_all_term_occurrences(term_text, segments)
        item["occurrences"] = occurrences
        item["occurrences_count"] = len(occurrences)

        if occurrences:
            item["start_s"] = occurrences[0]["start_s"]
            item["end_s"] = occurrences[0]["end_s"]
            item["timestamp"] = occurrences[0]["timestamp"]
        else:
            item["start_s"] = None
            item["end_s"] = None
            item["timestamp"] = None

    return terms


def _run_pipeline_job(
    job_id: str,
    audio_wav: Path,
    asr_model: str,
    lang: str,
    llm_model: str,
    video_meta: Optional[dict] = None,
    cancel_event: Optional[threading.Event] = None,
):
    def is_cancelled() -> bool:
        if cancel_event and cancel_event.is_set():
            return True
        job = JOBS.get(job_id)
        return bool(job and job.get("status") == "cancelled")

    try:
        with PIPELINE_LOCK:
            if is_cancelled():
                raise JobCancelledError("Job cancelled before pipeline execution.")

            _set_job_status(job_id, "processing")

            # Step 1: Transcribe ASR
            _push_event(job_id, "step", {"step": "asr", "message": f"Running ASR ({asr_model})..."})

            def asr_cb(msg: str):
                if is_cancelled():
                    raise JobCancelledError("ASR cancelled.")
                _push_event(job_id, "log", {"message": msg})

            asr_res = transcribe_audio(
                audio_wav,
                model_id=asr_model,
                lang=lang,
                progress_cb=asr_cb,
                is_cancelled=is_cancelled,
            )

            if is_cancelled():
                raise JobCancelledError("Job cancelled after ASR.")

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
            transcript_text = (asr_res.get("text") or "").strip()
            if not transcript_text:
                _push_event(job_id, "step", {"step": "llm", "message": "Transcript is empty. Skipping LLM terminology extraction."})
                _push_event(job_id, "log", {"message": "No spoken text detected in audio; skipping LLM extraction."})
                detected_lang = lang if lang in ("tr", "en") else "en"
                llm_res = {
                    "terms": [],
                    "n_terms": 0,
                    "latency_s": 0.0,
                    "raw_output": "",
                    "model_id": llm_model,
                }
                _push_event(
                    job_id,
                    "llm_done",
                    {
                        "n_terms": 0,
                        "latency_s": 0.0,
                    },
                )
            else:
                _push_event(job_id, "step", {"step": "llm", "message": f"Extracting domain terminology ({llm_model})..."})

                def llm_cb(msg: str):
                    if is_cancelled():
                        raise JobCancelledError("LLM extraction cancelled.")
                    _push_event(job_id, "log", {"message": msg})

                detected_lang = lang if lang in ("tr", "en") else detect_lang(asr_res["text"])
                profile = load_profile()
                learner_context = build_learner_prompt_block(profile)
                if learner_context:
                    _push_event(
                        job_id,
                        "log",
                        {"message": "Applying saved practice-exam history as learner weight on terminology analysis."},
                    )
                llm_res = extract_terms(
                    asr_res["text"],
                    lang=detected_lang,
                    model_id=llm_model,
                    progress_cb=llm_cb,
                    is_cancelled=is_cancelled,
                    learner_context=learner_context or None,
                )

                if is_cancelled():
                    raise JobCancelledError("Job cancelled after LLM extraction.")

                # Align extracted terms to timestamped transcript segments
                align_terms_with_segments(llm_res.get("terms", []), asr_res.get("segments", []))

                _push_event(
                    job_id,
                    "llm_done",
                    {
                        "n_terms": llm_res["n_terms"],
                        "latency_s": llm_res["latency_s"],
                    },
                )

            profile = load_profile()
            annotate_terms(llm_res.get("terms", []), profile)
            adaptive = build_adaptive_analysis(profile, llm_res.get("terms", []))

            # Complete result payload
            has_vid = bool(JOBS.get(job_id, {}).get("video_file"))
            is_vid = (video_meta.get("is_video", False) if isinstance(video_meta, dict) else False) or has_vid
            final_payload = {
                "job_id": job_id,
                "video_meta": video_meta,
                "lang": detected_lang,
                "asr": asr_res,
                "llm": llm_res,
                "adaptive": adaptive,
                "has_audio_file": True,
                "has_video_file": has_vid,
                "is_video": is_vid,
            }

            _set_job_status(job_id, "completed", result=final_payload)
            _push_event(job_id, "done", final_payload)

    except JobCancelledError:
        logger.info("Pipeline job %s cancelled by user.", job_id)
        _set_job_status(job_id, "cancelled")
        _push_event(job_id, "cancelled", {"message": "Job was cancelled by the user."})
        clear_gpu_memory()
    except Exception as exc:
        if is_cancelled():
            logger.info("Pipeline job %s cancelled during processing.", job_id)
            _set_job_status(job_id, "cancelled")
            _push_event(job_id, "cancelled", {"message": "Job was cancelled by the user."})
            clear_gpu_memory()
        else:
            logger.error("Pipeline job %s failed: %s", job_id, exc, exc_info=True)
            _set_job_status(job_id, "error")
            _push_event(job_id, "job_error", {"message": "Processing failed due to an internal server error."})
    finally:
        if is_cancelled():
            clear_gpu_memory()


@app.post("/api/process-youtube")
def process_youtube(req: YouTubeRequest):
    cleaned_url = req.url.strip() if req.url else ""
    vid = extract_youtube_id(cleaned_url)
    if not vid:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Only valid 11-character YouTube video IDs or URLs are allowed.",
        )

    allowed_asr, allowed_llm = get_allowed_model_ids()
    if req.asr_model not in allowed_asr:
        raise HTTPException(
            status_code=400,
            detail=f"Disallowed ASR model '{req.asr_model}'. Allowed models: {', '.join(sorted(allowed_asr))}",
        )
    if req.llm_model not in allowed_llm:
        raise HTTPException(
            status_code=400,
            detail=f"Disallowed LLM model '{req.llm_model}'. Allowed models: {', '.join(sorted(allowed_llm))}",
        )

    job_id = str(uuid.uuid4())
    audio_wav = TEMP_DIR / f"{job_id}.wav"
    canonical_url = f"https://www.youtube.com/watch?v={vid}"

    try:
        meta = get_youtube_info(canonical_url)
    except Exception:
        meta = {
            "id": vid,
            "title": f"YouTube Video ({vid})",
            "url": canonical_url,
            "channel": "YouTube",
            "duration": 0,
            "thumbnail": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
        }

    cancel_event = threading.Event()
    with JOB_LOCK:
        if is_job_active():
            raise HTTPException(
                status_code=429,
                detail="Another speech-to-terminology job is currently running. Only one active job is allowed at a time.",
            )
        now = time.time()
        JOBS[job_id] = {
            "status": "queued",
            "events": [],
            "result": None,
            "audio_file": audio_wav,
            "meta": meta,
            "cancel_event": cancel_event,
            "created_at": now,
            "last_accessed": now,
        }

    effective_duration_s = req.duration_s
    if not req.confirm_long:
        if effective_duration_s is None or effective_duration_s > DEFAULT_MAX_SCOPE_S:
            effective_duration_s = DEFAULT_MAX_SCOPE_S

    def worker():
        try:
            if cancel_event.is_set():
                JOBS[job_id]["status"] = "cancelled"
                _push_event(job_id, "cancelled", {"message": "Job was cancelled by the user."})
                clear_gpu_memory()
                return

            _push_event(job_id, "step", {"step": "download", "message": "Downloading YouTube audio..."})

            def dl_cb(msg: str):
                if cancel_event.is_set():
                    raise JobCancelledError("Download cancelled.")
                _push_event(job_id, "log", {"message": msg})

            dur = download_youtube_audio(
                canonical_url,
                audio_wav,
                start_s=req.start_s,
                duration_s=effective_duration_s,
                progress_cb=dl_cb,
                is_cancelled=cancel_event.is_set,
            )
            _push_event(job_id, "download_done", {"duration_s": dur})
            _run_pipeline_job(job_id, audio_wav, req.asr_model, req.lang, req.llm_model, meta, cancel_event=cancel_event)
        except JobCancelledError:
            logger.info("YouTube job %s cancelled.", job_id)
            _set_job_status(job_id, "cancelled")
            _push_event(job_id, "cancelled", {"message": "Job was cancelled by the user."})
            clear_gpu_memory()
        except YouTubeDownloadError as yt_err:
            logger.warning("YouTube download error for job %s: %s", job_id, yt_err)
            _set_job_status(job_id, "error")
            _push_event(
                job_id,
                "job_error",
                {
                    "message": yt_err.user_friendly_message,
                    "error_type": yt_err.error_type,
                    "suggest_upload": yt_err.suggest_upload,
                    "detail": str(yt_err),
                },
            )
            clear_gpu_memory()
        except Exception as e:
            if cancel_event.is_set():
                _set_job_status(job_id, "cancelled")
                _push_event(job_id, "cancelled", {"message": "Job was cancelled by the user."})
                clear_gpu_memory()
            else:
                logger.error("Job %s download worker failed: %s", job_id, e, exc_info=True)
                _set_job_status(job_id, "error")
                parsed_err = parse_youtube_dl_error(str(e))
                _push_event(
                    job_id,
                    "job_error",
                    {
                        "message": parsed_err.user_friendly_message,
                        "error_type": parsed_err.error_type,
                        "suggest_upload": parsed_err.suggest_upload,
                        "detail": str(e),
                    },
                )

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
    confirm_long: bool = Form(False),
):
    allowed_asr, allowed_llm = get_allowed_model_ids()
    if asr_model not in allowed_asr:
        raise HTTPException(
            status_code=400,
            detail=f"Disallowed ASR model '{asr_model}'. Allowed models: {', '.join(sorted(allowed_asr))}",
        )
    if llm_model not in allowed_llm:
        raise HTTPException(
            status_code=400,
            detail=f"Disallowed LLM model '{llm_model}'. Allowed models: {', '.join(sorted(allowed_llm))}",
        )

    job_id = str(uuid.uuid4())
    cancel_event = threading.Event()

    with JOB_LOCK:
        if is_job_active():
            raise HTTPException(
                status_code=429,
                detail="Another speech-to-terminology job is currently running. Only one active job is allowed at a time.",
            )
        # Atomically reserve slot to prevent race condition during upload streaming
        now = time.time()
        JOBS[job_id] = {
            "status": "uploading",
            "events": [],
            "result": None,
            "audio_file": None,
            "meta": None,
            "cancel_event": cancel_event,
            "created_at": now,
            "last_accessed": now,
        }

    raw_filename = file.filename or ""
    # 1. Path traversal protection: isolate strict basename and remove path separators
    basename = Path(raw_filename.replace("\\", "/")).name.strip()
    if not basename:
        with JOB_LOCK:
            JOBS.pop(job_id, None)
        raise HTTPException(status_code=400, detail="A valid filename must be provided.")

    # 2. File extension validation whitelist
    suffix = Path(basename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        with JOB_LOCK:
            JOBS.pop(job_id, None)
        allowed_str = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed formats: {allowed_str}",
        )

    temp_upload = TEMP_DIR / f"upload_{job_id}_{basename}"

    # Verify temp_upload stays strictly inside TEMP_DIR
    if not temp_upload.resolve().is_relative_to(TEMP_DIR.resolve()):
        with JOB_LOCK:
            JOBS.pop(job_id, None)
        raise HTTPException(status_code=400, detail="Invalid upload filename path.")

    audio_wav = TEMP_DIR / f"{job_id}.wav"

    # 3. Stream-copy with a strict 500 MB limit
    chunk_size = 1024 * 1024  # 1 MB chunk
    total_bytes = 0
    try:
        with temp_upload.open("wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    buffer.close()
                    if temp_upload.exists():
                        temp_upload.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds the maximum allowed limit of 500 MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        with JOB_LOCK:
            JOBS.pop(job_id, None)
        raise
    except Exception as exc:
        with JOB_LOCK:
            JOBS.pop(job_id, None)
        logger.error("Failed to save uploaded file %s for job %s: %s", temp_upload, job_id, exc, exc_info=True)
        if temp_upload.exists():
            temp_upload.unlink()
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    if total_bytes == 0:
        with JOB_LOCK:
            JOBS.pop(job_id, None)
        if temp_upload.exists():
            temp_upload.unlink()
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    is_video = suffix in ALLOWED_VIDEO_EXTENSIONS
    meta = {
        "id": "",
        "title": basename,
        "url": "",
        "channel": "Local Upload",
        "duration": 0,
        "thumbnail": "",
        "is_video": is_video,
        "media_type": suffix,
    }

    with JOB_LOCK:
        JOBS[job_id]["status"] = "queued"
        JOBS[job_id]["audio_file"] = audio_wav
        if is_video:
            JOBS[job_id]["video_file"] = temp_upload
        JOBS[job_id]["meta"] = meta

    effective_duration_s = duration_s
    if not confirm_long:
        if effective_duration_s is None or effective_duration_s > DEFAULT_MAX_SCOPE_S:
            effective_duration_s = DEFAULT_MAX_SCOPE_S

    def worker():
        try:
            if cancel_event.is_set():
                JOBS[job_id]["status"] = "cancelled"
                _push_event(job_id, "cancelled", {"message": "Job was cancelled by the user."})
                clear_gpu_memory()
                return

            _push_event(job_id, "step", {"step": "convert", "message": "Converting uploaded file..."})

            def conv_cb(msg: str):
                if cancel_event.is_set():
                    raise JobCancelledError("Conversion cancelled.")
                _push_event(job_id, "log", {"message": msg})

            dur = convert_local_audio(
                temp_upload,
                audio_wav,
                start_s=start_s,
                duration_s=effective_duration_s,
                progress_cb=conv_cb,
                is_cancelled=cancel_event.is_set,
            )
            if not is_video:
                try:
                    temp_upload.unlink()
                except Exception:
                    pass

            _push_event(job_id, "convert_done", {"duration_s": dur})
            _run_pipeline_job(job_id, audio_wav, asr_model, lang, llm_model, meta, cancel_event=cancel_event)
        except JobCancelledError:
            logger.info("Upload job %s cancelled.", job_id)
            _set_job_status(job_id, "cancelled")
            _push_event(job_id, "cancelled", {"message": "Job was cancelled by the user."})
            clear_gpu_memory()
        except Exception as e:
            if cancel_event.is_set():
                _set_job_status(job_id, "cancelled")
                _push_event(job_id, "cancelled", {"message": "Job was cancelled by the user."})
                clear_gpu_memory()
            else:
                logger.error("Job %s conversion worker failed: %s", job_id, e, exc_info=True)
                _set_job_status(job_id, "error")
                _push_event(job_id, "job_error", {"message": "Failed to convert uploaded media."})

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "meta": meta}


@app.get("/api/job/{job_id}")
def get_job_status(job_id: str):
    """Query current status and result for a given job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["last_accessed"] = time.time()
    error_msg = None
    if job.get("status") == "error":
        for ev in reversed(job.get("events", [])):
            if ev.get("event") == "job_error":
                error_msg = ev.get("data", {}).get("message")
                break
    return {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "result": job.get("result"),
        "meta": job.get("meta"),
        "error": error_msg,
    }


@app.get("/api/stream/{job_id}")
async def stream_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_index = 0
        heartbeat_ticks = 0
        while True:
            job = JOBS.get(job_id)
            if not job:
                break
            events = job["events"]
            while last_index < len(events):
                ev = events[last_index]
                last_index += 1
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"
            if job["status"] in ("completed", "error", "cancelled"):
                break
            heartbeat_ticks += 1
            if heartbeat_ticks >= 5:  # Heartbeat every ~2 seconds to prevent idle socket drop
                yield ": keepalive\n\n"
                heartbeat_ticks = 0
            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
def health_check():
    """Lightweight health check endpoint."""
    return {"status": "ok", "service": "eduterm", "version": "1.0.0"}


@app.get("/api/active-job")
def get_active_job_endpoint():
    """Return currently active job status if one is running."""
    active = get_active_job()
    return {"active": active is not None, "job": active}


@app.post("/api/cancel-job")
@app.post("/api/cancel/{job_id}")
def cancel_job(job_id: Optional[str] = None):
    """Cancel an in-flight job, set cancellation flag, and release GPU memory."""
    with JOB_LOCK:
        target_job_id = job_id
        if not target_job_id:
            active = get_active_job()
            if active:
                target_job_id = active["job_id"]

        if not target_job_id or target_job_id not in JOBS:
            raise HTTPException(status_code=404, detail="No active job found to cancel.")

        job = JOBS[target_job_id]
        if job.get("status") not in ("queued", "uploading", "processing"):
            return {
                "job_id": target_job_id,
                "status": job.get("status"),
                "message": f"Job is not active (status: {job.get('status')}).",
            }

        job["status"] = "cancelled"
        cancel_event = job.get("cancel_event")
        if cancel_event:
            cancel_event.set()

        _push_event(target_job_id, "cancelled", {"message": "Job was cancelled by the user."})

    try:
        clear_gpu_memory()
    except Exception as exc:
        logger.warning("Error clearing GPU memory during cancel: %s", exc)

    return {
        "job_id": target_job_id,
        "status": "cancelled",
        "message": "Job cancellation initiated and GPU memory released.",
    }


@app.get("/api/audio/{job_id}")
def get_audio(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("audio_file") or not Path(job["audio_file"]).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    job["last_accessed"] = time.time()
    return FileResponse(job["audio_file"], media_type="audio/wav")


@app.get("/api/video/{job_id}")
def get_video(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["last_accessed"] = time.time()
    v_file = job.get("video_file")
    if v_file and Path(v_file).exists():
        vpath = Path(v_file)
        suffix = vpath.suffix.lower()
        media_type = VIDEO_MIME_TYPES.get(suffix, "video/mp4")
        return FileResponse(vpath, media_type=media_type)
    # Fallback if only audio exists
    a_file = job.get("audio_file")
    if a_file and Path(a_file).exists():
        return FileResponse(a_file, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Video or media file not found for this job")


@app.get("/api/export-csv/{job_id}")
def export_csv(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Result not found")
    job["last_accessed"] = time.time()
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
        fieldnames=["item_id", "video_id", "model", "lang", "term", "timestamp", "explanation", "R", "D", "comment"],
    )
    writer.writeheader()
    for i, t in enumerate(terms, start=1):
        writer.writerow({
            "item_id": f"{video_id}-{tag}-{i:03d}",
            "video_id": video_id,
            "model": tag,
            "lang": res.get("lang", "tr"),
            "term": t.get("term", ""),
            "timestamp": t.get("timestamp") or "",
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


@app.get("/api/export-json/{job_id}")
def export_json(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Result not found")
    job["last_accessed"] = time.time()
    payload = json.dumps(job["result"], ensure_ascii=False, indent=2)
    return StreamingResponse(
        iter([payload]),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=eduterm_{job_id[:8]}.json"},
    )



@app.post("/api/exam/generate")
def create_exam(req: ExamGenerateRequest):
    job = JOBS.get(req.job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Job or extracted terms not found")
    res = job["result"]
    terms = res.get("llm", {}).get("terms", [])
    if not terms:
        raise HTTPException(status_code=400, detail="No extracted terms available for this lecture.")
    transcript = res.get("asr", {}).get("text", "")
    lang = res.get("lang", "en")
    llm_model = res.get("llm", {}).get("model", "Qwen/Qwen3.5-4B")
    profile = load_profile()
    exam = None
    if req.use_llm:
        import concurrent.futures
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    generate_practice_exam,
                    terms=terms,
                    transcript=transcript,
                    lang=lang,
                    model_id=llm_model,
                    profile=profile,
                    use_llm=True,
                )
                exam = fut.result(timeout=15.0)
        except Exception as exc:
            logger.warning("LLM exam generation timed out or failed (%s); using instant flashcards.", exc)
            exam = None

    if not exam:
        exam = generate_practice_exam(
            terms=terms,
            transcript=transcript,
            lang=lang,
            profile=profile,
            use_llm=False,
        )
    meta = res.get("video_meta") or {}
    exam["job_id"] = req.job_id
    exam["video_id"] = meta.get("id", "")
    exam["video_title"] = meta.get("title", "")
    save_exam(exam)
    return {
        "exam_id": exam["exam_id"],
        "source": exam["source"],
        "lang": exam["lang"],
        "questions": public_questions(exam["questions"]),
        "total": len(exam["questions"]),
        "video_title": exam["video_title"],
    }


@app.get("/api/exam/attention-research")
def get_attention_research():
    """Return scientific literature metadata, findings, and drop-off timestamps."""
    return ATTENTION_RESEARCH_INFO


@app.get("/api/exam/{exam_id}")
def get_exam(exam_id: str):
    exam = load_exam(exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    if exam.get("submitted"):
        return exam
    return {
        "exam_id": exam["exam_id"],
        "source": exam["source"],
        "lang": exam["lang"],
        "questions": public_questions(exam["questions"]),
        "total": len(exam["questions"]),
        "video_title": exam.get("video_title", ""),
        "submitted": False,
    }


@app.post("/api/exam/submit")
def submit_exam(req: ExamSubmitRequest):
    exam = load_exam(req.exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    if exam.get("submitted") and exam.get("result"):
        return exam["result"]
    answers_dicts = [a.model_dump() if hasattr(a, "model_dump") else a.dict() for a in req.answers]
    scored = score_exam(exam["questions"], answers_dicts)

    profile = load_profile()
    exam_record = {
        "exam_id": exam["exam_id"],
        "job_id": exam.get("job_id"),
        "video_id": exam.get("video_id", ""),
        "video_title": exam.get("video_title", ""),
        "score": scored["score"],
        "correct": scored["correct"],
        "total": scored["total"],
        "weak_terms": scored["weak_terms"],
        "answers": scored["answers"],
    }
    record_exam(profile, exam_record)
    save_profile(profile)

    adaptive_info = build_adaptive_analysis(profile)
    summary_text = analysis_after_exam(scored, adaptive_info)

    result_payload = {
        "exam_id": exam["exam_id"],
        "score": scored["score"],
        "correct": scored["correct"],
        "total": scored["total"],
        "weak_terms": scored["weak_terms"],
        "strong_terms": scored["strong_terms"],
        "answers": scored["answers"],
        "analysis_summary": summary_text,
        "adaptive": adaptive_info,
        "profile_stats": profile.get("stats"),
    }

    exam["submitted"] = True
    exam["result"] = result_payload
    save_exam(exam)
    return result_payload


@app.post("/api/exam/popquiz")
def get_popquiz_question(req: PopQuizRequest):
    """Return a single instant pop-quiz question based on the lecture's terms at current playback time."""
    job = JOBS.get(req.job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Job or extracted terms not found")
    res = job["result"]
    terms = res.get("llm", {}).get("terms", [])
    if not terms:
        raise HTTPException(status_code=400, detail="No extracted terms available for this lecture.")
    lang = res.get("lang", "en")
    profile = load_profile()
    pq = make_popquiz_question(
        terms=terms,
        current_time_s=req.current_time_s,
        lang=lang,
        profile=profile,
    )
    if not pq:
        raise HTTPException(status_code=404, detail="Unable to generate pop-quiz question.")
    meta = res.get("video_meta") or {}
    pq["video_id"] = meta.get("id", "")
    pq["video_title"] = meta.get("title", "")
    return pq


@app.post("/api/exam/quick-answer")
def record_quick_answer(req: PopQuizAnswerRequest):
    """Record a single pop-quiz answer into the student learner profile in real-time."""
    if not req.term:
        raise HTTPException(status_code=400, detail="Term is required")
    profile = load_profile()
    rec = update_mastery(profile, req.term, req.correct, video_id=req.video_id or "")
    save_profile(profile)
    return {
        "status": "ok",
        "term": rec.get("term"),
        "mastery": rec.get("mastery"),
        "focus": rec.get("focus"),
        "seen": rec.get("seen"),
        "stats": profile.get("stats"),
    }



@app.get("/api/learner/profile")
def get_learner_profile():
    return public_profile()


@app.post("/api/learner/reset")
def reset_learner_profile():
    prof = empty_profile()
    save_profile(prof)
    return {"ok": True, "profile": public_profile(prof)}


@app.post("/api/helper-slide")
def get_helper_slide(req: HelperSlideRequest):
    """Generate an instant summary helper slide based on the lecture section just listened to."""
    job = JOBS.get(req.job_id)
    if not job or not job.get("result"):
        cand_id = f"preview_{req.job_id}" if not req.job_id.startswith("preview_") else req.job_id
        if cand_id in JOBS and JOBS[cand_id].get("result"):
            job = JOBS[cand_id]
        else:
            try:
                payload = fetch_instant_transcript(req.job_id)
                JOBS[payload["job_id"]] = {"result": payload, "status": "completed"}
                job = JOBS[payload["job_id"]]
            except Exception:
                pass
    if not job or not job.get("result"):
        for j in reversed(list(JOBS.values())):
            if j.get("result"):
                job = j
                break
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Job or lecture result not found")
    res = job["result"]
    asr = res.get("asr", {})
    segments = asr.get("segments", [])
    terms = res.get("llm", {}).get("terms", [])
    lang = res.get("lang", "tr")

    t = max(0.0, float(req.current_time_s))
    window_start = max(0.0, t - 50.0)
    window_end = t + 5.0

    recent_segments = [
        s for s in segments
        if s.get("start") is not None and s.get("end") is not None
        and float(s["end"]) >= window_start and float(s["start"]) <= window_end
    ]
    recent_text = " ".join(s.get("text", "").strip() for s in recent_segments) if recent_segments else asr.get("text", "")[:350]

    recent_lower = recent_text.lower()
    active_terms = []
    for term_obj in terms:
        t_name = term_obj.get("term", "").strip()
        if not t_name:
            continue
        in_window = False
        if term_obj.get("start") is not None:
            ts = float(term_obj["start"])
            if window_start <= ts <= window_end:
                in_window = True
        if in_window or (t_name.lower() in recent_lower):
            active_terms.append(term_obj)

    if not active_terms and terms:
        active_terms = terms[:2]

    topic_title = active_terms[0].get("term") if active_terms else "Ders Bölümü"

    core_summary = ""
    if active_terms:
        core_summary = " • ".join(f"{t['term']}: {t.get('explanation', '')}" for t in active_terms[:2])
    else:
        core_summary = recent_text[:200]

    # Build structured presentation slide specification (with Mermaid diagram & cards)
    llm_model_name = res.get("llm", {}).get("model", "Qwen/Qwen3.5-4B")
    use_deep_llm = bool(req.generate_deep_slide or req.llm_analogy)
    slide_spec = build_slide_specification(
        topic=topic_title,
        recent_text=recent_text,
        active_terms=active_terms,
        lang=lang,
        model_id=llm_model_name,
        use_llm=use_deep_llm,
    )

    analogy_text = slide_spec.get("analogy")

    start_m, start_s = divmod(int(window_start), 60)
    cur_m, cur_s = divmod(int(t), 60)
    time_range = f"{start_m:02d}:{start_s:02d} - {cur_m:02d}:{cur_s:02d}"

    return {
        "topic": topic_title,
        "time_range": time_range,
        "current_time_s": t,
        "rewind_time_s": max(0.0, t - 10.0),
        "core_summary": slide_spec.get("core_takeaway") or core_summary,
        "terms": active_terms[:4],
        "recent_transcript": recent_text,
        "analogy": analogy_text,
        "slide_spec": slide_spec,
    }


@app.post("/api/cornell-notes")
def get_cornell_study_sheet(req: CornellNotesRequest):
    """Generate or retrieve a full Cornell Note-Taking Study Sheet for the active lecture."""
    job = JOBS.get(req.job_id)
    if not job or not job.get("result"):
        cand_id = f"preview_{req.job_id}" if not req.job_id.startswith("preview_") else req.job_id
        if cand_id in JOBS and JOBS[cand_id].get("result"):
            job = JOBS[cand_id]
        else:
            try:
                payload = fetch_instant_transcript(req.job_id)
                JOBS[payload["job_id"]] = {"result": payload, "status": "completed"}
                job = JOBS[payload["job_id"]]
            except Exception:
                pass
    if not job or not job.get("result"):
        for j in reversed(list(JOBS.values())):
            if j.get("result"):
                job = j
                break
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Job or lecture result not found")

    res = job["result"]
    lang = res.get("lang", "tr")
    sheet = generate_cornell_study_sheet(res, custom_title=req.custom_title, lang=lang)
    return sheet


@app.get("/api/export-cornell-markdown/{job_id}")
def export_cornell_markdown(job_id: str):
    """Download the Cornell Study Sheet as a clean Markdown (.md) document."""
    job = JOBS.get(job_id)
    if not job or not job.get("result"):
        cand_id = f"preview_{job_id}" if not job_id.startswith("preview_") else job_id
        if cand_id in JOBS and JOBS[cand_id].get("result"):
            job = JOBS[cand_id]
        else:
            try:
                payload = fetch_instant_transcript(job_id)
                JOBS[payload["job_id"]] = {"result": payload, "status": "completed"}
                job = JOBS[payload["job_id"]]
            except Exception:
                pass
    if not job or not job.get("result"):
        for j in reversed(list(JOBS.values())):
            if j.get("result"):
                job = j
                break
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Lecture not found")

    res = job["result"]
    lang = res.get("lang", "tr")
    sheet = generate_cornell_study_sheet(res, lang=lang)
    md_content = sheet.get("markdown", "")
    filename = f"cornell_study_sheet_{job_id}.md"

    from fastapi.responses import Response
    return Response(
        content=md_content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class ConceptGraphRequest(BaseModel):
    job_id: Optional[str] = None
    custom_title: Optional[str] = None
    terms: Optional[list[dict]] = None
    segments: Optional[list[dict]] = None
    lang: Optional[str] = "tr"


@app.post("/api/concept-graph")
def get_concept_knowledge_graph(req: ConceptGraphRequest):
    """Generate an Ausubel-grounded Concept Knowledge Graph with video timestamps and semantic relations."""
    res = None
    if req.job_id:
        job = JOBS.get(req.job_id)
        if not job or not job.get("result"):
            cand_id = f"preview_{req.job_id}" if not req.job_id.startswith("preview_") else req.job_id
            if cand_id in JOBS and JOBS[cand_id].get("result"):
                job = JOBS[cand_id]
            else:
                try:
                    payload = fetch_instant_transcript(req.job_id)
                    JOBS[payload["job_id"]] = {"result": payload, "status": "completed"}
                    job = JOBS[payload["job_id"]]
                except Exception:
                    pass
        if job and job.get("result"):
            res = job["result"]

    if not res:
        # Check if terms were directly posted
        if req.terms:
            res = {
                "llm": {"terms": req.terms},
                "asr": {"segments": req.segments or []},
                "video_meta": {"title": req.custom_title or "Ders Kavram Ağı"},
                "lang": req.lang or "tr",
            }
        else:
            # Fallback to latest available job with result
            for j in reversed(list(JOBS.values())):
                if j.get("result"):
                    res = j["result"]
                    break

    if not res:
        # Predefined benchmark fallback
        res = {
            "llm": {"terms": []},
            "asr": {"segments": []},
            "video_meta": {"title": req.custom_title or "Proje Yönetimi & EVM"},
            "lang": req.lang or "tr",
        }

    lang = req.lang or res.get("lang", "tr")
    graph = build_concept_knowledge_graph(res, lang=lang)
    if req.custom_title:
        graph["title"] = req.custom_title

    return {"status": "ok", "graph": graph}


class SocraticChatRequest(BaseModel):
    job_id: Optional[str] = None
    message: str
    history: Optional[list[dict]] = None
    current_time_s: Optional[float] = None
    terms: Optional[list[dict]] = None
    segments: Optional[list[dict]] = None
    lang: Optional[str] = "tr"
    use_llm: Optional[bool] = False


@app.post("/api/socratic-chat")
def socratic_chat_endpoint(req: SocraticChatRequest):
    """Handle student questions using Socratic scaffolding based on Vanzo et al. (ACL 2025)."""
    res = None
    if req.job_id:
        job = JOBS.get(req.job_id)
        if not job or not job.get("result"):
            cand_id = f"preview_{req.job_id}" if not req.job_id.startswith("preview_") else req.job_id
            if cand_id in JOBS and JOBS[cand_id].get("result"):
                job = JOBS[cand_id]
            else:
                try:
                    payload = fetch_instant_transcript(req.job_id)
                    JOBS[payload["job_id"]] = {"result": payload, "status": "completed"}
                    job = JOBS[payload["job_id"]]
                except Exception:
                    pass
        if job and job.get("result"):
            res = job["result"]

    terms = req.terms or (res.get("llm", {}).get("terms") if res else [])
    segments = req.segments or (res.get("asr", {}).get("segments") if res else [])
    lang = req.lang or (res.get("lang") if res else "tr")

    result = generate_socratic_response(
        user_message=req.message,
        history=req.history or [],
        current_time_s=req.current_time_s,
        terms=terms,
        segments=segments,
        lang=lang,
        use_llm=bool(req.use_llm),
    )
    return result


@app.post("/api/manimator/generate")
def generate_manimator_video(req: ManimatorGenerateRequest):
    """Generate custom narrated animated video tailored to the exact concept the student struggles with."""
    recent_text = req.context or ""
    active_terms = []
    lang = req.lang or "tr"
    topic_title = (req.topic or "").strip()

    # If job_id provided, extract context around current_time_s
    if req.job_id:
        job = JOBS.get(req.job_id)
        if not job or not job.get("result"):
            cand_id = f"preview_{req.job_id}" if not req.job_id.startswith("preview_") else req.job_id
            if cand_id in JOBS and JOBS[cand_id].get("result"):
                job = JOBS[cand_id]
            else:
                try:
                    payload = fetch_instant_transcript(req.job_id)
                    JOBS[payload["job_id"]] = {"result": payload, "status": "completed"}
                    job = JOBS[payload["job_id"]]
                except Exception:
                    pass

        if job and job.get("result"):
            res = job["result"]
            asr = res.get("asr", {})
            segments = asr.get("segments", [])
            terms = res.get("llm", {}).get("terms", [])
            lang = res.get("lang", lang)

            t = max(0.0, float(req.current_time_s))
            window_start = max(0.0, t - 20.0)
            window_end = t + 8.0

            if not recent_text:
                recent_segments = [
                    s for s in segments
                    if s.get("start") is not None and s.get("end") is not None
                    and float(s["end"]) >= window_start and float(s["start"]) <= window_end
                ]
                if recent_segments:
                    recent_text = " ".join(s.get("text", "").strip() for s in recent_segments)
                elif segments:
                    timed_segments = [s for s in segments if s.get("start") is not None and s.get("end") is not None]
                    if timed_segments:
                        nearest = min(
                            timed_segments,
                            key=lambda s: abs((float(s["start"]) + float(s["end"])) / 2 - t),
                        )
                        recent_text = str(nearest.get("text", "")).strip()
                elif asr.get("text"):
                    recent_text = asr.get("text", "")[:350]

            if not topic_title:
                # 1. First priority: find term closest to timestamp t
                timed_candidates = []
                for term_obj in terms:
                    t_start = term_obj.get("start")
                    if t_start is not None:
                        dist = abs(float(t_start) - t)
                        if dist <= 30.0:
                            timed_candidates.append((dist, term_obj.get("term", "").strip()))
                if timed_candidates:
                    timed_candidates.sort(key=lambda x: x[0])
                    topic_title = timed_candidates[0][1]

                # 2. Second priority: term explicitly mentioned in recent_text
                if not topic_title and recent_text:
                    recent_lower = recent_text.lower()
                    for term_obj in terms:
                        t_name = term_obj.get("term", "").strip()
                        if t_name and t_name.lower() in recent_lower:
                            topic_title = t_name
                            break

                if not topic_title and terms and not req.job_id:
                    topic_title = terms[0].get("term", "Ders Bölümü")

    if not topic_title:
        topic_title = "Ders Kavramı"

    if req.video_quality not in VIDEO_QUALITY_TIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown video_quality {req.video_quality!r}; expected one of {sorted(VIDEO_QUALITY_TIERS)}",
        )

    try:
        explainer_result = produce_student_explainer_video(
            topic=topic_title,
            context=recent_text,
            lang=lang,
            video_style=req.video_style or "manim_animation",
            duration_s=req.duration_s or 50,
            voice=req.voice,
            backend=req.llm_backend or "evren",
            evren_model=req.evren_model or DEFAULT_EVREN_MODEL,
            local_model_id=req.model_id or "Qwen/Qwen3.5-4B",
            video_quality=req.video_quality,
        )
    except Exception as exc:
        logger.error("Manimator generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Manimator video generation failed: {exc}")

    # Record remediation event in learner profile
    try:
        focus_terms = [
            t_item.get("term", "").strip()
            for t_item in explainer_result.get("terms", [])[:2]
        ]
        focus_terms = [t for t in focus_terms if t]
        if focus_terms:
            profile = load_profile()
            for t_str in focus_terms:
                update_mastery(
                    profile,
                    t_str,
                    False,
                    video_id=explainer_result.get("video_id", ""),
                )
            save_profile(profile)
    except Exception:
        logger.warning("Failed to record manimator remediation in learner profile", exc_info=True)

    MANIMATOR_HISTORY.insert(0, {
        "video_id": explainer_result["video_id"],
        "title": explainer_result["title"],
        "video_url": explainer_result["video_url"],
        "duration_s": explainer_result["duration_s"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_backend": explainer_result.get("script_backend"),
        "video_quality": explainer_result.get("video_quality"),
        "terms": [t.get("term") for t in explainer_result.get("terms", [])],
    })

    return explainer_result


@app.post("/api/manimator/pdf-to-video")
async def generate_pdf_video(
    file: UploadFile = File(...),
    topic: Optional[str] = Form(""),
    video_style: str = Form("manim_animation"),
    duration_s: int = Form(50),
    voice: Optional[str] = Form(None),
    lang: str = Form("tr"),
    llm_backend: str = Form("evren"),
    evren_model: str = Form(DEFAULT_EVREN_MODEL),
    video_quality: str = Form(DEFAULT_VIDEO_QUALITY),
):
    """Upload PDF and turn it directly into a narrated Manim or Figure Walkthrough video."""
    safe_name = Path((file.filename or "").replace("\\", "/")).name.strip() or "Ders_Belgesi.pdf"
    if video_quality not in VIDEO_QUALITY_TIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown video_quality {video_quality!r}; expected one of {sorted(VIDEO_QUALITY_TIERS)}",
        )
    temp_pdf = Path(tempfile.gettempdir()) / f"upload_{uuid.uuid4().hex[:8]}_{safe_name}"
    try:
        contents = await file.read()
        temp_pdf.write_bytes(contents)

        doc_topic = topic.strip() or Path(safe_name).stem.replace("_", " ")

        explainer_result = produce_student_explainer_video(
            topic=doc_topic,
            context="",
            lang=lang,
            video_style=video_style,
            duration_s=duration_s,
            voice=voice,
            pdf_path=temp_pdf,
            backend=llm_backend or "evren",
            evren_model=evren_model or DEFAULT_EVREN_MODEL,
            video_quality=video_quality,
        )

        MANIMATOR_HISTORY.insert(0, {
            "video_id": explainer_result["video_id"],
            "title": explainer_result["title"],
            "video_url": explainer_result["video_url"],
            "duration_s": explainer_result["duration_s"],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": safe_name,
        })

        return explainer_result
    except Exception as exc:
        logger.error("PDF to Video failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF video processing failed: {exc}")
    finally:
        temp_pdf.unlink(missing_ok=True)


@app.get("/api/manimator/history")
def get_manimator_history():
    """Retrieve history of generated explainer videos."""
    return {"videos": MANIMATOR_HISTORY[:25]}


@app.get("/api/manimator/video/{filename}")
def stream_manimator_video(filename: str):
    """Stream generated MP4 video with range support."""
    safe_name = os.path.basename(filename)
    path = VIDEOS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4", filename=safe_name)


@app.get("/api/manimator/audio/{filename}")
def stream_manimator_audio(filename: str):
    """Stream generated narration audio (MP3)."""
    safe_name = os.path.basename(filename)
    path = MANIMATOR_TEMP_DIR / safe_name
    if not path.resolve().is_relative_to(MANIMATOR_TEMP_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid audio filename")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/mpeg", filename=safe_name)


# Mount static assets
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return "<h1>EduTerm AI GUI Loading...</h1>"
    resp = HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/poster", response_class=HTMLResponse)
def serve_poster():
    poster_file = STATIC_DIR / "poster.html"
    if not poster_file.exists():
        return "<h1>A3 Poster Sunumu Yükleniyor...</h1>"
    resp = HTMLResponse(content=poster_file.read_text(encoding="utf-8"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.get("/api/download/poster-pdf")
def download_poster_pdf():
    pdf_path = ROOT.parent / "notes" / "Bitirme_Projesi_A3_Posteri.pdf"
    if not pdf_path.exists():
        try:
            import importlib.util
            script_path = ROOT.parent / "notes" / "generate_a3_poster_pdf.py"
            spec = importlib.util.spec_from_file_location("generate_a3_poster_pdf", str(script_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.build_poster_pdf()
        except Exception as e:
            logger.error("A3 poster PDF build failed: %s", e)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="A3 Poster PDF bulunamadı.")
    return FileResponse(pdf_path, media_type="application/pdf", filename="Bitirme_Projesi_A3_Posteri.pdf")
