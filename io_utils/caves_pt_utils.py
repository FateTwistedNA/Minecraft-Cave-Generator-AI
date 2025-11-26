# io_utils/caves_pt_utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Any, List

import torch
import pandas as pd
import numpy as np

# Known extreme outlier to skip if desired
OUTLIER_FILES = {}


@dataclass
class CaveStats:
    path: Path
    shape_raw: tuple    # original tensor shape
    shape_3d: tuple     # after squeezing channel
    air_ratio: float
    is_outlier: bool


def iter_cave_paths(root: Path, skip_outliers: bool = True) -> Iterable[Path]:
    root = Path(root)
    for p in sorted(root.glob("*.pt")):
        if skip_outliers and p.name in OUTLIER_FILES:
            continue
        yield p


def load_raw_cave(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    obj = torch.load(Path(path), map_location=map_location)
    if "full_volume" not in obj:
        raise ValueError(f"{path} missing 'full_volume'")
    if "slice_seq" not in obj:
        obj["slice_seq"] = []
    return obj


def get_3d_volume_from_raw(obj: Dict[str, Any]) -> torch.Tensor:
    """
    Take the raw cave object and return a 3D tensor (D,H,W) of uint8.

    Expected raw shape: (1, D, H, W).
    """
    vol = obj["full_volume"]
    if vol.ndim == 4 and vol.shape[0] == 1:
        vol3d = vol[0]  # (D,H,W)
    elif vol.ndim == 3:
        vol3d = vol
    else:
        raise ValueError(f"Unexpected full_volume shape {tuple(vol.shape)}")
    return vol3d.to(dtype=torch.uint8)


def compute_cave_stats(path: Path) -> CaveStats:
    obj = load_raw_cave(path)
    vol_raw = obj["full_volume"]
    vol3d = get_3d_volume_from_raw(obj)  # (D,H,W)
    v_np = vol3d.cpu().numpy()
    air_ratio = float((v_np == 0).sum()) / v_np.size
    return CaveStats(
        path=Path(path),
        shape_raw=tuple(vol_raw.shape),
        shape_3d=tuple(vol3d.shape),
        air_ratio=air_ratio,
        is_outlier=path.name in OUTLIER_FILES,
    )


def build_manifest(caves_root: Path, out_csv: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in iter_cave_paths(caves_root, skip_outliers=False):
        try:
            s = compute_cave_stats(p)
            rows.append(
                {
                    "file": p.name,
                    "shape_raw": list(s.shape_raw),
                    "shape_3d": list(s.shape_3d),
                    "air_ratio": s.air_ratio,
                    "is_outlier": s.is_outlier,
                }
            )
        except Exception as e:
            print(f"[WARN] Failed {p}: {e}")
            rows.append(
                {
                    "file": p.name,
                    "shape_raw": None,
                    "shape_3d": None,
                    "air_ratio": None,
                    "is_outlier": True,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote manifest to {out_csv} ({len(df)} rows)")
    return df
