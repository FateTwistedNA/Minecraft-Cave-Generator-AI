# scripts/eval_generated.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def load_volume(path: Path) -> np.ndarray:
    """
    Load a cave volume from a .pt file.

    Supports:
      - dict with key 'full_volume'
      - bare tensor
    Returns a numpy array of shape (D, H, W) with values in {0,1},
    where 0 = air, 1 = rock.
    """
    data = torch.load(path, map_location="cpu")

    if isinstance(data, dict):
        if "full_volume" in data:
            vol = data["full_volume"]
        else:
            # fall back: take first tensor-looking thing
            for v in data.values():
                if torch.is_tensor(v):
                    vol = v
                    break
            else:
                raise ValueError(f"No tensor found in dict from {path}")
    elif torch.is_tensor(data):
        vol = data
    else:
        raise ValueError(f"Unsupported data type in {path}: {type(data)}")

    vol = vol.squeeze()  # remove batch/channel if present
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {tuple(vol.shape)} from {path}")

    vol_np = vol.detach().cpu().numpy()
    return vol_np


def compute_air_stats(vol_np: np.ndarray) -> Tuple[float, int, int]:
    """
    Given (D,H,W) volume with 0=air, 1=rock, compute:
      air_ratio, num_air, num_rock
    """
    total = vol_np.size
    num_air = int((vol_np == 0).sum())
    num_rock = int((vol_np == 1).sum())
    air_ratio = num_air / total
    return air_ratio, num_air, num_rock


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute simple stats (shape, air_ratio) for real and generated caves."
    )
    ap.add_argument(
        "--real-pt",
        nargs="*",
        type=str,
        default=[],
        help="One or more real cave .pt files from data/caves_pt.",
    )
    ap.add_argument(
        "--gen-pt",
        nargs="*",
        type=str,
        default=[],
        help="One or more generated cave .pt files (e.g., gen_cave_final.pt).",
    )
    args = ap.parse_args()

    if not args.real_pt and not args.gen_pt:
        ap.error("Provide at least one --real-pt or --gen-pt file.")

    def report(label: str, p: Path) -> None:
        vol_np = load_volume(p)
        D, H, W = vol_np.shape
        air_ratio, num_air, num_rock = compute_air_stats(vol_np)
        print(f"[{label}] {p}")
        print(f"  shape        : (D,H,W) = ({D}, {H}, {W})")
        print(f"  total voxels : {vol_np.size}")
        print(f"  air voxels   : {num_air}")
        print(f"  rock voxels  : {num_rock}")
        print(f"  air_ratio    : {air_ratio:.4f}")
        print()

    for p_str in args.real_pt:
        report("REAL", Path(p_str))

    for p_str in args.gen_pt:
        report("GEN ", Path(p_str))


if __name__ == "__main__":
    main()
