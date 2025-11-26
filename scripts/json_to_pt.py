import argparse
import json
from pathlib import Path

import numpy as np
import torch


def json_to_volume(sample: dict) -> np.ndarray:
    """
    Convert one JSON sample from sample_caves.py into a 3D volume.

    JSON structure:
      {
        "sample_id": 0,
        "shape": [1, 1, D, H, W],
        "num_voxels": ...,
        "num_cave_voxels": ...,
        "voxels": [ {"x": int, "y": int, "z": int}, ... ]
      }

    We build a volume with shape (D, H, W) where 1 = cave interior, 0 = outside.
    """
    shape = sample.get("shape", [1, 1, 32, 32, 32])
    if len(shape) != 5:
        raise ValueError(f"Expected shape [B,C,D,H,W], got {shape}")
    _, _, D, H, W = shape

    vol = np.zeros((D, H, W), dtype=np.uint8)

    for v in sample.get("voxels", []):
        x = int(v["x"])
        y = int(v["y"])
        z = int(v["z"])
        if 0 <= z < D and 0 <= y < H and 0 <= x < W:
            vol[z, y, x] = 1  # mark cave voxel

    return vol


def build_slice_seq_from_volume(vol: np.ndarray, max_slices: int = 16) -> np.ndarray:
    """
    Build a simple slice sequence from the 3D volume for viz_cave.py.

    vol: (D, H, W), values in {0,1}
    Returns: (N, H, W), N <= max_slices
    """
    D, H, W = vol.shape

    # Take up to max_slices slices evenly spaced along depth
    if D <= max_slices:
        indices = list(range(D))
    else:
        step = max(1, D // max_slices)
        indices = list(range(0, D, step))
        indices = indices[:max_slices]

    slices = [vol[d] for d in indices]
    slice_seq = np.stack(slices, axis=0)  # (N, H, W)

    return slice_seq.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Convert cave_sample JSON to a .pt file with full_volume + slice_seq.")
    ap.add_argument("json_path", type=str, help="Path to cave_sample_*.json (output of scripts.sample_caves).")
    ap.add_argument("--out-pt", type=str, required=True, help="Output .pt file path to create.")
    args = ap.parse_args()

    json_path = Path(args.json_path)
    out_path = Path(args.out_pt)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # sample_caves.py writes a list of samples; pick the first by default
    if isinstance(data, list):
        sample = data[0]
    else:
        sample = data

    vol = json_to_volume(sample)  # (D, H, W), 1 = cave interior
    slice_seq = build_slice_seq_from_volume(vol, max_slices=16)  # (N, H, W)

    # Convert to torch tensors
    full_volume_t = torch.from_numpy(vol)        # (D, H, W)
    slice_seq_t = torch.from_numpy(slice_seq)    # (N, H, W)

    payload = {
        "full_volume": full_volume_t,
        "slice_seq": slice_seq_t,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"Saved converted cave to: {out_path}")
    print(f"full_volume shape: {tuple(full_volume_t.shape)}, "
          f"slice_seq shape: {tuple(slice_seq_t.shape)}")


if __name__ == "__main__":
    main()
