"""EVREN LLM Client & Provider Integration.

Integrates the National AI / LLM Platform (EVREN - T3 / TÜBİTAK BİLGEM / SSB)
using the OpenAI-compatible REST API at https://evren-llmapi.ssyz.org.tr/v1.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_EVREN_BASE_URL = "https://evren-llmapi.ssyz.org.tr/v1"
# EVREN qwen3.8-flash-next is a reasoning model and bills thinking tokens against
# max_tokens, so a long transcript can consume the whole budget and answer with "".
DEFAULT_EVREN_REASONING_EFFORT = "none"

# Automatically find and load .env files if present
def _load_env_files() -> None:
    current = Path(__file__).resolve().parent
    candidates = [
        current / ".env",
        current.parent / ".env",
    ]
    for env_file in candidates:
        if env_file.exists():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass

_load_env_files()


def get_evren_api_key() -> str:
    """Return the EVREN API key from the environment or a .env file.

    The key is never hardcoded: it grants access to a national inference platform
    that is billed per token and is tracked in version control.
    """
    key = (
        os.environ.get("EVREN_API_KEY")
        or os.environ.get("EVREN_TOKEN")
        or ""
    ).strip()
    if not key:
        raise RuntimeError(
            "EVREN API key is not configured. Set EVREN_API_KEY in the environment "
            "or in experiments/.env (that file is git-ignored)."
        )
    return key


def get_evren_base_url() -> str:
    """Return the EVREN base URL."""
    return (
        os.environ.get("EVREN_BASE_URL")
        or DEFAULT_EVREN_BASE_URL
    ).strip().rstrip("/")


# Catalog of EVREN cloud models supported on the platform
EVREN_MODELS = [
    {
        "id": "evren/qwen3.8-flash-next",
        "evren_id": "qwen3.8-flash-next",
        "name": "EVREN Qwen 3.8 Flash (TÜBİTAK/SSB Bulut • H200 GPU • Ultra Hızlı)",
        "desc": "Yerli sunucu altyapısı (H200 kümesi), düşük gecikme ve yüksek Türkçe/İngilizce terim doğruluğu",
        "default": True,
    },
    {
        "id": "evren/deepseek-v4.1-flash",
        "evren_id": "deepseek-v4.1-flash",
        "name": "EVREN DeepSeek V4.1 Flash (TÜBİTAK/SSB Bulut • Derin Muhakeme)",
        "desc": "Gelişmiş akıl yürütme, derin semantik bağlam ve CEFR B1 pedagojik açıklama kalitesi",
        "default": False,
    },
    {
        "id": "evren/glm-5.3",
        "evren_id": "glm-5.3",
        "name": "EVREN GLM-5.3 (TÜBİTAK/SSB Bulut • Kompleks Analiz)",
        "desc": "Zor ve yoğun terminoloji içeren ders anlatımları için genişletilmiş akıl yürütme modeli",
        "default": False,
    },
    {
        "id": "evren/gemma-4-31b",
        "evren_id": "gemma-4-31b",
        "name": "EVREN Gemma 4 31B (TÜBİTAK/SSB Bulut • Büyük Model)",
        "desc": "31 Milyar parametreli yüksek kapasiteli açık ağırlıklı bulut modeli",
        "default": False,
    },
    {
        "id": "evren/auto",
        "evren_id": "auto",
        "name": "EVREN Auto (TÜBİTAK/SSB Akıllı Yönlendirme)",
        "desc": "İsteme göre en uygun EVREN modelini otomatik seçen dinamik yönlendirici",
        "default": False,
    },
]

_EVREN_KNOWN_IDS = {
    m["id"].lower() for m in EVREN_MODELS
} | {
    m["evren_id"].lower() for m in EVREN_MODELS
} | {
    "dots-ocr", "deepseek-v4-flash", "qwen3-embedding-8b", "qwen3-reranker-8b",
    "deepseek-ocr-2", "qwen3-asr-1.7b", "gemma-4-31b", "qwen3-vl-30b",
    "qwen3-guard-4b", "auto", "mimo-v2.6-pro", "qwen3.8-flash-next",
    "clef", "laya", "deepseek-v4.1-flash", "glm-5.3"
}


def is_evren_model(model_id: str) -> bool:
    """Return True if model_id targets an EVREN cloud model."""
    if not model_id:
        return False
    mid = model_id.strip().lower()
    return mid.startswith("evren/") or mid.startswith("evren:") or mid in _EVREN_KNOWN_IDS


def normalize_evren_model_id(model_id: str) -> str:
    """Strip 'evren/' prefix to get the raw model identifier expected by the API."""
    if not model_id:
        return "qwen3.8-flash-next"
    mid = model_id.strip()
    if mid.lower().startswith("evren/"):
        return mid[6:]
    if mid.lower().startswith("evren:"):
        return mid[6:]
    return mid


def accept_evren_terms(api_key: Optional[str] = None) -> bool:
    """Accept EVREN platform terms of service (v1) via the REST endpoint."""
    key = api_key or get_evren_api_key()
    base_url = get_evren_base_url()
    url = f"{base_url}/terms/accept"
    payload = json.dumps({"version": 1}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "EduTerm-AI/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return resp.status in (200, 201)
    except Exception as exc:
        logger.warning("Could not auto-accept EVREN terms: %s", exc)
        return False


_OPENAI_CLIENT = None
_LAST_CLIENT_KEY = None


def get_evren_client(api_key: Optional[str] = None):
    """Return a configured OpenAI client pointed at the EVREN API."""
    global _OPENAI_CLIENT, _LAST_CLIENT_KEY
    from openai import OpenAI

    key = api_key or get_evren_api_key()
    base_url = get_evren_base_url()

    if _OPENAI_CLIENT is not None and _LAST_CLIENT_KEY == key:
        return _OPENAI_CLIENT

    _OPENAI_CLIENT = OpenAI(api_key=key, base_url=base_url)
    _LAST_CLIENT_KEY = key
    return _OPENAI_CLIENT


def strip_think_tags(text: str) -> str:
    """Strip chain-of-thought <think>...</think> tags if present."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def generate_evren(
    model_id: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Generate completion via EVREN cloud LLM endpoint.

    Raises RuntimeError when the platform produces no answer text, so an empty
    response can never be recorded as a finished experiment.
    """
    target_model = normalize_evren_model_id(model_id)
    client = get_evren_client()
    effort = (
        reasoning_effort
        or os.environ.get("EVREN_REASONING_EFFORT")
        or DEFAULT_EVREN_REASONING_EFFORT
    ).strip()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    def _create(effort_value: str):
        return client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=effort_value,
        )

    def _attempt(effort_value: str):
        try:
            return _create(effort_value)
        except Exception as exc:
            if _accept_terms_if_opted_in(exc):
                return _create(effort_value)
            raise

    # The thinking tokens of a reasoning model are billed against max_tokens, so the
    # whole budget can vanish into the (discarded) reasoning field. Retry once with
    # reasoning off before declaring the call empty.
    efforts = ["none"] if effort == "none" else [effort, "none"]
    failure = None
    for attempt_effort in efforts:
        resp = _attempt(attempt_effort)
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if content:
            if choice.finish_reason == "length":
                logger.warning(
                    "EVREN model '%s' ran out of budget at max_tokens=%d; the answer is truncated "
                    "and may not parse as complete JSON.",
                    target_model,
                    max_tokens,
                )
            return strip_think_tags(content)
        reasoning = ((getattr(choice.message, "model_extra", None) or {}).get("reasoning") or "")
        failure = (attempt_effort, choice.finish_reason, len(reasoning))

    effort_used, finish_reason, reasoning_chars = failure
    raise RuntimeError(
        f"EVREN model '{target_model}' returned no answer text "
        f"(reasoning_effort={effort_used}, finish_reason={finish_reason}, "
        f"reasoning_chars={reasoning_chars}, max_tokens={max_tokens}). "
        "Raise max_tokens or shorten the prompt."
    )


def _accept_terms_if_opted_in(exc: Exception) -> bool:
    """Accept the platform terms only on explicit operator opt-in; else keep the error.

    Clicking through a terms-of-service agreement is a legal act by a person, so the
    client may not do it silently on every 403.
    """
    err_msg = str(exc)
    if "terms_not_accepted" not in err_msg and "403" not in err_msg:
        return False
    if os.environ.get("EVREN_ACCEPT_TERMS") != "1":
        logger.error(
            "EVREN refused the request because the platform terms are not accepted (%s). "
            "Read and accept them yourself, or set EVREN_ACCEPT_TERMS=1 to let this client "
            "post the acceptance on your behalf.",
            err_msg[:200],
        )
        return False
    logger.info("EVREN terms not accepted; accepting under explicit EVREN_ACCEPT_TERMS=1 opt-in.")
    return accept_evren_terms()
