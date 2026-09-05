"""Load gold transcripts from data/gold/{id}.txt and resolve a manifest row."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
GOLD_DIR = ROOT / "data" / "gold"


def gold_path(clip_id: str) -> Path:
    return GOLD_DIR / f"{clip_id}.txt"


def load_gold_text(clip_id: str) -> str:
    path = gold_path(clip_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def resolve_reference(row: dict, *, root: Path | None = None) -> str:
    """Prefer reference_file / data/gold/{id}.txt, then inline reference."""
    base = root or ROOT
    rel = row.get("reference_file")
    if rel:
        path = Path(rel)
        if not path.is_absolute():
            path = (base / path).resolve()
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    cid = row.get("id")
    if cid:
        disk = load_gold_text(str(cid))
        if disk:
            return disk
    return (row.get("reference") or "").strip()
