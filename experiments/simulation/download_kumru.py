import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

target_dir = Path("/users/ali.gunder/models/Kumru-2B")
target_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading vngrs-ai/Kumru-2B to {target_dir}...")
snapshot_download(
    repo_id="vngrs-ai/Kumru-2B",
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,
    resume_download=True
)
print("Kumru-2B downloaded successfully!")

