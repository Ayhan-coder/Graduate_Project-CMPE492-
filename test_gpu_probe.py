import os
import sys

print("CUDA_VISIBLE_DEVICES from Slurm:", os.environ.get("CUDA_VISIBLE_DEVICES"))

try:
    import torch
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print("Device Count:", count)
        for i in range(count):
            try:
                name = torch.cuda.get_device_name(i)
                print(f"Device {i}: {name}")
                # Try simple tensor alloc
                t = torch.tensor([1.0, 2.0], device=f"cuda:{i}")
                print(f"  -> Allocation on cuda:{i} SUCCEEDED: {t}")
            except Exception as e:
                print(f"  -> Allocation on cuda:{i} FAILED: {e}")
except Exception as e:
    print("Torch import/check error:", e)

