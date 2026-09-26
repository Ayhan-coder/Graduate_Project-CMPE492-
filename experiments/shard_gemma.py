"""Shard google/gemma-4-E4B-it into <=3.8GB safetensors shards.

Windows PyTorch/safetensors has an internal 32-bit offset/mmap limitation
that crashes with an access violation when reading single safetensors files >4GB.
Sharding into smaller chunks fixes this completely.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    hub_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--google--gemma-4-E4B-it"
    )
    snap_dir = hub_dir / "snapshots" / "ee0ef6023621cff504d758262d4e04895a5af4a2"
    source_file = snap_dir / "model.safetensors"

    if not source_file.exists():
        print(f"Source file not found at {source_file}")
        return

    print(f"Reading header from {source_file}...")
    with open(source_file, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))

    tensor_keys = [k for k in header.keys() if k != "__metadata__"]
    print(f"Total tensors: {len(tensor_keys)}")

    data_start = 8 + header_size
    max_shard_size = 3_800_000_000  # ~3.8 GB

    # Group into shards
    shards: list[list[str]] = []
    current_shard: list[str] = []
    current_shard_size = 0

    for k in tensor_keys:
        info = header[k]
        s, e = info["data_offsets"]
        t_size = e - s
        if current_shard and (current_shard_size + t_size > max_shard_size):
            shards.append(current_shard)
            current_shard = [k]
            current_shard_size = t_size
        else:
            current_shard.append(k)
            current_shard_size += t_size

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    print(f"Splitting into {num_shards} shards (max ~3.8 GB each)...")

    weight_map = {}
    total_bytes = 0

    with open(source_file, "rb") as f:
        for idx, shard_keys in enumerate(shards, 1):
            shard_name = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
            shard_path = snap_dir / shard_name
            print(f"Writing shard {idx}/{num_shards}: {shard_name} ({len(shard_keys)} tensors)...")

            tensors = {}
            for k in shard_keys:
                info = header[k]
                s, e = info["data_offsets"]
                t_size = e - s
                total_bytes += t_size

                f.seek(data_start + s)
                raw = f.read(t_size)
                # Create a writable copy
                t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(info["shape"]).clone()
                tensors[k] = t
                weight_map[k] = shard_name

            save_file(tensors, str(shard_path), metadata={"format": "pt"})
            print(f"  Saved {shard_name} ({shard_path.stat().st_size / 1e9:.2f} GB)")

    # Write model.safetensors.index.json
    index_data = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    index_path = snap_dir / "model.safetensors.index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
    print(f"Created index file: {index_path.name}")

    # Remove the un-sharded single file
    if source_file.is_symlink() or source_file.is_file():
        source_file.unlink()
        print(f"Removed monolithic {source_file.name}")

    print("\nSharding complete! Checking generated files...")
    for item in sorted(snap_dir.glob("model*")):
        print(f"  {item.name} -> {item.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
