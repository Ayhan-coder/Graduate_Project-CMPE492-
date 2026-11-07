import urllib.request
import json

candidates = [
    "Trendyol/Trendyol-LLM-7b-chat-v1.0",
    "Trendyol/Trendyol-LLM-7b-chat-dpo-v1.0",
    "vngrs-ai/Kumru-2B",
    "vngrs-ai/Kumru-2.5B",
    "yentinglin/Llama-3-8B-Turkish-Instruct",
    "erythropoietin/Llama-3-8B-Turkish-Instruct"
]

print("Checking Hugging Face Hub for verified Turkish models...")
for model_id in candidates:
    url = f"https://huggingface.co/api/models/{model_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.load(resp)
            print(f"[FOUND] {model_id} | Downloads: {data.get('downloads', 0)} | Pipeline: {data.get('pipeline_tag')}")
    except Exception as e:
        print(f"[NOT FOUND / ERROR] {model_id}: {e}")

