import os
from huggingface_hub import snapshot_download

model_id = "Trendyol/Trendyol-LLM-8B-T1"
target_dir = "/users/ali.gunder/models/Trendyol-LLM-8B-T1"

print(f"Downloading {model_id} to {target_dir}...")
os.makedirs(target_dir, exist_ok=True)

snapshot_download(
    repo_id=model_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
)
print("Download completed successfully!")

