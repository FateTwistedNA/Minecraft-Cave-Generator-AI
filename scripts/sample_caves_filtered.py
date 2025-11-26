import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.train_vae import VAE3D


# ---------- helpers ----------

def volume_to_bin(probs: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    """
    probs: numpy array, typically (1,1,D,H,W) or (D,H,W) with values in [0,1].
    Returns uint8 array with shape (D,H,W), values in {0,1}:
      0 = air, 1 = rock.
    """
    arr = np.asarray(probs)
    # squeeze batch / channel dims
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume after squeeze, got {arr.shape}")
    vol_bin = (arr > threshold).astype(np.uint8)
    return vol_bin  # (D,H,W)


def volume_to_coords(vol_bin: np.ndarray, air_value: int = 0):
    """
    vol_bin: (D,H,W) uint8 or bool; 0/1 volume.
    air_value: which voxel value represents 'cave air' (0 if 1=rock).
    Returns list of {"x": x, "y": y, "z": z} local coordinates.
    """
    vol_bin = np.asarray(vol_bin)
    if vol_bin.ndim != 3:
        raise ValueError(f"volume_to_coords expected (D,H,W), got {vol_bin.shape}")

    zs, ys, xs = np.where(vol_bin == air_value)
    coords = [{"x": int(x), "y": int(y), "z": int(z)} for z, y, x in zip(zs, ys, xs)]
    return coords


def compute_air_ratio_extent_count(vol_bin: np.ndarray, air_value: int = 0):
    """
    Returns (air_ratio, (dz, dy, dx), n_air).
      air_ratio = (#air voxels) / total voxels
      extents   = bounding-box size of the air region
      n_air     = number of air voxels
    """
    vol_bin = np.asarray(vol_bin)
    mask = (vol_bin == air_value)
    n_air = int(mask.sum())
    total = int(mask.size)
    air_ratio = float(n_air) / float(total) if total > 0 else 0.0

    if n_air == 0:
        return air_ratio, (0, 0, 0), n_air

    zs, ys, xs = np.where(mask)
    dz = int(zs.max() - zs.min() + 1)
    dy = int(ys.max() - ys.min() + 1)
    dx = int(xs.max() - xs.min() + 1)
    return air_ratio, (dz, dy, dx), n_air


def is_small_cave(air_ratio: float, extents, n_air: int) -> bool:
    """
    Heuristic for 'small-ish' caves for presentation.

    Updated:
      - Ignore extents, since the VAE tends to sprinkle air everywhere,
        so the bounding box is almost always (32,32,32).
      - Focus instead on how many air voxels there are (sparser = "smaller").
    """
    # total volume is 32^3 = 32768
    # we call it "small-ish" if it's not almost solid but also not filling
    # most of the box with air.
    if n_air == 0:
        return False

    # Require a moderate air ratio: not super sparse, not almost empty,
    # but definitely not a huge swiss-cheese cave either.
    if not (0.25 <= air_ratio <= 0.55):
        return False

    # also cap absolute air count (kind of a safety net)
    if n_air > 19000:
        return False

    return True



# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Sample filtered 'small' caves from a trained 3D VAE.")
    ap.add_argument("--weights", type=str, required=True, help="Path to vae3d_z*.pt")
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument(
        "--num-samples",
        type=int,
        default=3,
        dest="num_samples",
        help="Number of *accepted* caves to save."
    )
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--max-tries", type=int, default=200,
                    help="Maximum random draws before giving up.")
    ap.add_argument("--json-out", type=str, default="small_caves.json",
                    help="Where to write JSON coordinates + stats.")
    ap.add_argument("--pt-prefix", type=str, default="gen_small_cave",
                    help="Prefix for generated .pt volumes.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, file=sys.stderr)

    # Load model
    model = VAE3D(z_dim=args.z_dim).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    accepted = []
    pt_outputs = []
    tries = 0

    with torch.no_grad():
        while len(accepted) < args.num_samples and tries < args.max_tries:
            tries += 1
            z = torch.randn(1, args.z_dim, device=device)
            logits = model.decode(z)             # (1,1,D,H,W)
            probs = torch.sigmoid(logits).cpu().numpy()

            vol_bin = volume_to_bin(probs, threshold=args.threshold)  # (D,H,W)
            air_ratio, extents, n_air = compute_air_ratio_extent_count(vol_bin, air_value=0)

            # Debug line so you can see what’s being tried (optional)
            print(
                f"[TRY {tries}] air_ratio={air_ratio:.3f}, extents={extents}, n_air={n_air}",
                file=sys.stderr
            )

            if not is_small_cave(air_ratio, extents, n_air):
                continue

            coords = volume_to_coords(vol_bin, air_value=0)
            idx = len(accepted)

            sample_rec = {
                "sample_id": idx,
                "shape": [1, 1] + list(vol_bin.shape),  # mimic VAE output shape
                "num_voxels": int(vol_bin.size),
                "num_cave_voxels": len(coords),
                "air_ratio": air_ratio,
                "extents": {"dz": extents[0], "dy": extents[1], "dx": extents[2]},
                "voxels": coords,
            }
            accepted.append(sample_rec)

            pt_path = f"{args.pt_prefix}_{idx}.pt"
            torch.save(
                {
                    "full_volume": torch.from_numpy(vol_bin.astype(np.uint8)),
                    "slice_seq": None,
                },
                pt_path,
            )
            pt_outputs.append(pt_path)

            print(
                f"[ACCEPTED {idx}] air_ratio={air_ratio:.3f}, extents={extents}, "
                f"n_air={n_air}, saved to {pt_path}",
                file=sys.stderr,
            )

    if not accepted:
        print("No caves matched the filter; try relaxing thresholds.", file=sys.stderr)
        return

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(accepted, f, indent=2)

    print(f"\nSaved {len(accepted)} filtered caves to {args.json_out}", file=sys.stderr)
    print("PT files:", *pt_outputs, sep="\n  ", file=sys.stderr)


if __name__ == "__main__":
    main()
