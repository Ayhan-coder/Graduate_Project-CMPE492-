import json
from pathlib import Path

config_path = Path("/users/ali.gunder/models/Trendyol-LLM-8B-T1/config.json")
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("Old model_type:", cfg.get("model_type"), "architectures:", cfg.get("architectures"))
    cfg["model_type"] = "qwen2"
    cfg["architectures"] = ["Qwen2ForCausalLM"]
    cfg["sliding_window"] = 131072
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("Updated config to qwen2 / Qwen2ForCausalLM successfully!")
else:
    print("Config path not found!")

