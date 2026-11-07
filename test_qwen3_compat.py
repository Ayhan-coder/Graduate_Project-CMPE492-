import sys
try:
    from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
    print("AutoConfig and Qwen2 imported successfully.")
except Exception as e:
    print("Import error:", e)

