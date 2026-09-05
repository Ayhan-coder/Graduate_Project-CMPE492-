"""Smoke test for GUI server endpoints using TestClient."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from gui.server import app

client = TestClient(app)


def test_homepage():
    res = client.get("/")
    assert res.status_code == 200
    assert "EduTerm" in res.text
    print("test_homepage: OK")


def test_system_info():
    res = client.get("/api/system-info")
    assert res.status_code == 200
    data = res.json()
    assert "cuda" in data
    assert "gpu" in data
    print(f"test_system_info: OK (GPU: {data['gpu']})")


def test_models():
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data["asr_models"]) >= 4
    assert len(data["llm_models"]) >= 1
    print("test_models: OK")


def test_youtube_info():
    res = client.post("/api/youtube-info", json={"url": "https://www.youtube.com/watch?v=rz5JCvMlCMs"})
    assert res.status_code == 200
    data = res.json()
    assert "id" in data
    assert "title" in data
    assert "thumbnail" in data
    print(f"test_youtube_info: OK (Title: {data['title'][:40]}...)")


if __name__ == "__main__":
    test_homepage()
    test_system_info()
    test_models()
    test_youtube_info()
    print("\nAll GUI API tests PASSED successfully!")
