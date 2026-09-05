import sys
import traceback
import torch

try:
    print("Testing CUDA availability...")
    print("CUDA available:", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name(0))
    print("Total memory GB:", torch.cuda.get_device_properties(0).total_memory / (1024**3))

    from transformers import AutoModelForCausalLM
    snapshot = 'hf_cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    
    print("Attempting to load model onto CPU first...")
    m = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16)
    print("Loaded on CPU successfully! Total params:", sum(p.numel() for p in m.parameters()))

    print("Attempting to move model to CUDA...")
    m = m.to("cuda")
    print("Moved to CUDA successfully! Allocated:", torch.cuda.memory_allocated() / (1024**3))
except Exception as e:
    print("CAUGHT EXCEPTION:")
    traceback.print_exc()
    sys.exit(1)
