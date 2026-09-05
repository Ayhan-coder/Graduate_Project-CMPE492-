import os
import sys
import time
import hashlib
from pathlib import Path
import requests

URL = "https://huggingface.co/Qwen/Qwen3.5-4B/resolve/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/model.safetensors-00001-of-00002.safetensors"
EXPECTED_SHA = "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"
EXPECTED_SIZE = 5329398688

ROOT = Path(__file__).resolve().parent
BLOBS_DIR = ROOT / "hf_cache" / "models--Qwen--Qwen3.5-4B" / "blobs"
SNAPSHOT_DIR = ROOT / "hf_cache" / "models--Qwen--Qwen3.5-4B" / "snapshots" / "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TARGET_BLOB = BLOBS_DIR / EXPECTED_SHA
TARGET_SNAPSHOT = SNAPSHOT_DIR / "model.safetensors-00001-of-00002.safetensors"

INCOMPLETE_FILE = BLOBS_DIR / f"{EXPECTED_SHA}.4dbde662.incomplete"

def main():
    if TARGET_SNAPSHOT.exists() and TARGET_SNAPSHOT.stat().st_size == EXPECTED_SIZE:
        print("Target snapshot already exists and is complete!")
        return

    if TARGET_BLOB.exists() and TARGET_BLOB.stat().st_size == EXPECTED_SIZE:
        print("Target blob complete, creating hardlink...")
        if TARGET_SNAPSHOT.exists():
            TARGET_SNAPSHOT.unlink()
        os.link(TARGET_BLOB, TARGET_SNAPSHOT)
        print("Done!")
        return

    curr_size = INCOMPLETE_FILE.stat().st_size if INCOMPLETE_FILE.exists() else 0
    print(f"Current size: {curr_size} / {EXPECTED_SIZE} ({curr_size / EXPECTED_SIZE * 100:.1f}%)")

    chunk_size = 1024 * 1024 * 4 # 4MB
    retries = 20

    while curr_size < EXPECTED_SIZE and retries > 0:
        headers = {"Range": f"bytes={curr_size}-{EXPECTED_SIZE - 1}"}
        try:
            print(f"Requesting bytes {curr_size} to {EXPECTED_SIZE - 1} ...")
            with requests.get(URL, headers=headers, stream=True, timeout=30) as r:
                if r.status_code not in (200, 206):
                    print(f"HTTP error {r.status_code}: {r.text[:200]}")
                    time.sleep(2)
                    retries -= 1
                    continue
                with open(INCOMPLETE_FILE, "ab") as f:
                    t_last = time.time()
                    bytes_last = curr_size
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            curr_size += len(chunk)
                            now = time.time()
                            if now - t_last >= 2.0:
                                speed = (curr_size - bytes_last) / (now - t_last) / (1024 * 1024)
                                print(f"Progress: {curr_size / (1024*1024):.1f} MB / {EXPECTED_SIZE / (1024*1024):.1f} MB ({curr_size / EXPECTED_SIZE * 100:.1f}%) @ {speed:.2f} MB/s", flush=True)
                                t_last = now
                                bytes_last = curr_size
        except Exception as exc:
            print(f"Connection error: {exc}. Retrying...")
            time.sleep(2)
            retries -= 1
            if INCOMPLETE_FILE.exists():
                curr_size = INCOMPLETE_FILE.stat().st_size

    if curr_size != EXPECTED_SIZE:
        print(f"Failed to complete download. Size is {curr_size}")
        sys.exit(1)

    print("Download complete! Verifying SHA256...")
    h = hashlib.sha256()
    with open(INCOMPLETE_FILE, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    digest = h.hexdigest()
    print(f"Computed SHA256: {digest}")
    if digest != EXPECTED_SHA:
        print(f"SHA256 MISMATCH! Expected {EXPECTED_SHA}")
        sys.exit(1)

    print("SHA256 matches! Moving into place...")
    if TARGET_BLOB.exists():
        TARGET_BLOB.unlink()
    INCOMPLETE_FILE.rename(TARGET_BLOB)

    if TARGET_SNAPSHOT.exists():
        TARGET_SNAPSHOT.unlink()
    os.link(TARGET_BLOB, TARGET_SNAPSHOT)
    print("Successfully linked model.safetensors-00001-of-00002.safetensors!")

if __name__ == "__main__":
    main()
