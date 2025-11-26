import numpy as np
import torch
import json
import argparse
import sys
from pathlib import Path

from scripts.train_vae import VAE3D 


def volume_to_coords(vol_bin: np.ndarray, air_value: int = 0):
    """
    vol_bin: numpy array with shape (B,C,D,H,W) or (C,D,H,W) or (D,H,W),
             values in {0,1}.
    air_value: which value is treated as 'cave air' (typically 0 if 1=rock).
    Returns list of dicts: {"x": x, "y": y, "z": z} in local coordinates.
    """
    vol_bin = np.asarray(vol_bin)

    # Squeeze batch/channel dimensions if present
    if vol_bin.ndim == 5:      # (B,C,D,H,W)
        vol_bin = vol_bin[0, 0]
    elif vol_bin.ndim == 4:    # (C,D,H,W) or (B,D,H,W)
        vol_bin = vol_bin[0]
    elif vol_bin.ndim != 3:    # must be (D,H,W) now
        raise ValueError(f"Unexpected volume shape {vol_bin.shape}")

    D, H, W = vol_bin.shape  # (z,y,x)

    # indices where volume == air_value
    zs, ys, xs = np.where(vol_bin == air_value)

    coords = []
    for z, y, x in zip(zs, ys, xs):
        coords.append({"x": int(x), "y": int(y), "z": int(z)})
    return coords



def main():
    ap = argparse.ArgumentParser(description="Sample caves from trained 3D VAE and output voxel coords.")
    ap.add_argument("--weights", type=str, required=True, help="Path to vae3d_z*.pt")
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="", help="Write JSON to this file instead of stdout.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE3D(z_dim=args.z_dim).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    samples_out = []
    with torch.no_grad():
        for i in range(args.num_samples):
            z = torch.randn(1, args.z_dim, device=device)
            
            logits = model.decode(z)                     # (1,1,32,32,32)
            probs = torch.sigmoid(logits)
            
            probs_np = probs.cpu().numpy()
            print(f"[Sample {i}] probs stats:",
                  "min=", probs_np.min(),
                  "max=", probs_np.max(),
                  "mean=", probs_np.mean(),
                  "shape=", probs_np.shape)

            vol = (probs > args.threshold).float()       # (1,1,32,32,32)
            vol_np = vol.cpu().numpy()                   # still (1,1,32,32,32)
            print("Sample", i, "volume stats:",
                  "min=", vol_np.min(),
                  "max=", vol_np.max(),
                  "mean=", vol_np.mean(),
                  "shape=", vol_np.shape)
            
            #IMPORTANT: for this dataset, 1 == air/cave (similar to the viz_cave.py)
            coords = volume_to_coords(vol_np, air_value=1
                                      )
            samples_out.append(
                {
                    "sample_id": i,
                    "shape": list(vol_np.shape),
                    "num_voxels": int(vol_np.size),
                    "num_cave_voxels": len(coords),
                    "voxels": coords,
                }
            )

    # print(json.dumps(samples_out, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(samples_out, f, indent=2)
    else:
        print("No --out specified, skipping JSON dump.")


if __name__ == "__main__":
    main()
