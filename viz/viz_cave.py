# How to call:
# python viz_cave.py ./data4/cave_0.pt
# last part is just the path to the .pt file you want to see. Every .pt file is a cave.

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def load_sample(pt_path):
    """
    Load a cave sample from a .pt file.

    Supports:
      1) Dict with keys like ("slice_seq", "full_volume", etc.)  <-- original dataset/json_to_pt
      2) Plain tensor saved directly (e.g., gen_small_cave_0.pt) <-- generated tensors
    """
    data = torch.load(pt_path, map_location="cpu")

    slice_seq = None
    full_vol = None

    # Case 1: dict-like .pt (original caves or json_to_pt output)
    if isinstance(data, dict):
        # Accept multiple possible key names
        slice_keys = ("slice_seq", "slice_sequence", "slice_seq_resized",
                      "outline_sequence", "outline_seq")
        vol_keys = ("full_volume", "full_vol", "volume", "voxels", "vox_resized")

        for k in slice_keys:
            if k in data:
                slice_seq = data[k]
                break

        for k in vol_keys:
            if k in data:
                full_vol = data[k]
                break

        # Fallback: if no explicit keys, try any tensor-ish values
        if full_vol is None:
            tensors = [v for v in data.values()
                       if torch.is_tensor(v) or isinstance(v, np.ndarray)]
            if len(tensors) >= 1:
                full_vol = tensors[0]

    # Case 2: plain tensor / numpy array saved directly
    elif torch.is_tensor(data) or isinstance(data, np.ndarray):
        full_vol = data

    else:
        raise ValueError("Unsupported .pt format (not a dict or tensor)")

    # --- Normalize full_vol to numpy, shape (D, H, W) ---

    if torch.is_tensor(full_vol):
        full_vol = full_vol.cpu().numpy()

    # Possible shapes: (1,1,D,H,W), (1,D,H,W), (D,H,W)
    if full_vol.ndim == 5 and full_vol.shape[0] == 1 and full_vol.shape[1] == 1:
        full_vol = full_vol[0, 0]         # -> (D, H, W)
    elif full_vol.ndim == 4:
        # (1, D, H, W) or (D, 1, H, W)
        if full_vol.shape[0] == 1:
            full_vol = full_vol[0]
        elif full_vol.shape[1] == 1:
            full_vol = full_vol[:, 0]
    elif full_vol.ndim == 3:
        pass  # already (D, H, W)
    else:
        raise ValueError(f"Unsupported full_vol shape: {full_vol.shape}")

    D, H, W = full_vol.shape

    # --- If we don't have slice_seq, synthesize it from full_vol ---
    if slice_seq is None:
        # Take up to 16 slices evenly spaced along depth
        step = max(1, D // 16)
        slice_seq = full_vol[::step, :, :]   # shape (N_slices, H, W)

    # Convert slice_seq to numpy if tensor
    if torch.is_tensor(slice_seq):
        slice_seq = slice_seq.cpu().numpy()

    # Clean up slice_seq shape: (N_slices, H, W)
    if slice_seq.ndim == 4 and slice_seq.shape[0] == 1:
        slice_seq = slice_seq[0]
    elif slice_seq.ndim == 2:
        # single slice 2D -> add slice dimension
        slice_seq = slice_seq[None, :, :]

    # Binarize both arrays to 0/1 for visualization
    slice_seq = (slice_seq > 0).astype(np.uint8)
    full_vol  = (full_vol  > 0).astype(np.uint8)

    return slice_seq, full_vol


# -------- 2D sequence visualizer ----------
def show_slice_sequence(slice_seq, fps=6, loop=False):
    """
    slice_seq: (N, H, W) numpy uint8
    """
    N, H, W = slice_seq.shape
    plt.ion()
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(slice_seq[0], cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title("Slice 0 / {}".format(N))
    for i in range(N):
        im.set_data(slice_seq[i])
        ax.set_title(f"Slice {i+1}/{N}")
        plt.pause(1.0 / fps)
    if loop:
        show_slice_sequence(slice_seq, fps=fps, loop=loop)
    else:
        plt.ioff()
        plt.show()

# -------- 3D Open3D visualizer ----------
def show_voxel_open3d(full_vol, downsample=1.0, color=(1.0,1.0,1.0)):
    """
    full_vol: (D, H, W) where values==1 indicate air (voxels to visualize)
    We convert to a point cloud of air voxels, and display as VoxelGrid.
    downsample: voxel size in world units for downsampling the point cloud (float)
    """
    D, H, W = full_vol.shape
    # Create coordinates for air voxels
    zs, ys, xs = np.where(full_vol == 1)  # careful ordering
    # Note: original arrays often have ordering (D, H, W) -> (z,y,x)
    # We'll map to (x,y,z) for Open3D
    if len(xs) == 0:
        print("No air voxels found in this cave.")
        return

    pts = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Downsample the pointcloud for performance if requested
    if downsample > 0 and downsample != 1.0:
        pcd = pcd.voxel_down_sample(voxel_size=downsample)

    # Convert to a VoxelGrid for nicer rendering
    try:
        voxel_size = max(1.0, downsample)
        vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        o3d.visualization.draw_geometries([vg])
    except Exception:
        # fallback: draw point cloud
        o3d.visualization.draw_geometries([pcd])

# -------- CLI ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_cave_from_pt.py /path/to/cave_X.pt [--fps 6] [--downsample 1.0] [--no3d]")
        return
    ptp = Path(sys.argv[1])
    fps = 6
    downsample = 1.0
    do3d = True
    # parse simple args
    for a in sys.argv[2:]:
        if a.startswith("--fps="):
            fps = float(a.split("=",1)[1])
        if a.startswith("--downsample="):
            downsample = float(a.split("=",1)[1])
        if a == "--no3d":
            do3d = False

    slice_seq, full_vol = load_sample(ptp)
    print("Loaded slice_seq shape:", slice_seq.shape)
    print("Loaded full_vol shape:", full_vol.shape)

    show_slice_sequence(slice_seq, fps=fps)

    if do3d:
        show_voxel_open3d(full_vol, downsample=downsample)

if __name__ == "__main__":
    main()
