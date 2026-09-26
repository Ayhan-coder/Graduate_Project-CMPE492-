"""Shard Qwen/Qwen3.5-9B safetensors into <=3.5GB shards to prevent Windows 4GB mmap access violation."""

import json
import struct
from pathlib import Path
import torch
from safetensors.torch import save_file
from fetch_models import resolve_hub_cache, hub_slug


def reshard_qwen():
    snap = resolve_hub_cache("Qwen/Qwen3.5-9B") / hub_slug("Qwen/Qwen3.5-9B") / "snapshots" / "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    index_file = snap / "model.safetensors.index.json"
    with open(index_file, "r", encoding="utf-8") as f:
        idx_data = json.load(f)

    weight_map = idx_data["weight_map"]
    old_shards = sorted(set(weight_map.values()))
    print(f"Current shards: {old_shards}")

    # Check if already resharded
    if all((snap / s).stat().st_size <= 3_800_000_000 for s in old_shards if (snap / s).exists()):
        print("All shards already <= 3.8GB! Nothing to do.")
        return

    # Collect all tensor info across old shards
    all_tensors = []  # list of (name, old_shard_name, shape, dtype, data_start + offset, size)
    for shard_name in old_shards:
        shard_path = snap / shard_name
        with open(shard_path, "rb") as f:
            h_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(h_len).decode("utf-8"))
        data_start = 8 + h_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            s, e = info["data_offsets"]
            t_size = e - s
            all_tensors.append({
                "name": name,
                "shard_path": shard_path,
                "offset": data_start + s,
                "size": t_size,
                "shape": info["shape"],
                "dtype": info["dtype"],
            })

    print(f"Total tensors across all shards: {len(all_tensors)}")

    # Group into new shards <= 3.2 GB
    MAX_SHARD_BYTES = 3_200_000_000
    new_groups = []
    curr_group = []
    curr_size = 0

    for t in all_tensors:
        if curr_group and (curr_size + t["size"] > MAX_SHARD_BYTES):
            new_groups.append(curr_group)
            curr_group = [t]
            curr_size = t["size"]
        else:
            curr_group.append(t)
            curr_size += t["size"]
    if curr_group:
        new_groups.append(curr_group)

    num_new_shards = len(new_groups)
    print(f"Splitting into {num_new_shards} new shards (max ~3.2 GB each)...")

    new_weight_map = {}
    total_bytes = 0

    for shard_idx, group in enumerate(new_groups, 1):
        shard_filename = f"model-{shard_idx:05d}-of-{num_new_shards:05d}.safetensors"
        shard_out_path = snap / shard_filename
        print(f"Writing shard {shard_idx}/{num_new_shards}: {shard_filename} ({len(group)} tensors)...")

        tensors_dict = {}
        # Open source files as needed
        open_files = {}
        try:
            for t in group:
                sp = t["shard_path"]
                if sp not in open_files:
                    open_files[sp] = open(sp, "rb")
                f = open_files[sp]
                f.seek(t["offset"])
                raw = f.read(t["size"])
                # parse dtype
                dt = torch.bfloat16 if t["dtype"] == "BF16" else torch.float32
                tensor = torch.frombuffer(bytearray(raw), dtype=dt).view(t["shape"]).clone()
                tensors_dict[t["name"]] = tensor
                new_weight_map[t["name"]] = shard_filename
                total_bytes += t["size"]
        finally:
            for f in open_files.values():
                f.close()

        save_file(tensors_dict, str(shard_out_path), metadata={"format": "pt"})
        print(f"  Saved {shard_filename} ({shard_out_path.stat().st_size / 1e9:.2f} GB)")

    # Save new index
    new_index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": new_weight_map,
    }
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)
    print("Updated model.safetensors.index.json")

    # Remove old oversized shards
    for old_s in old_shards:
        old_p = snap / old_s
        if old_p.exists() and old_s not in [f"model-{i:05d}-of-{num_new_shards:05d}.safetensors" for i in range(1, num_new_shards + 1)]:
            old_p.unlink()
            print(f"Removed old shard {old_s}")

    print("\nResharding Qwen/Qwen3.5-9B complete!")


if __name__ == "__main__":
    reshard_qwen()
