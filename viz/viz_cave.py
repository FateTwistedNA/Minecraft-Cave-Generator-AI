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
    data = torch.load(pt_path, map_location="cpu")
    # Accept either keys names from our exporter or a direct dict
    # Try common keys:
    if isinstance(data, dict):
        # Accept multiple possible key names
        slice_keys = ("slice_seq", "slice_sequence", "slice_seq_resized", "outline_sequence", "outline_seq")
        vol_keys = ("full_volume", "full_vol", "volume", "voxels", "vox_resized")
        slice_seq = None
        full_vol = None
        for k in slice_keys:
            if k in data:
                slice_seq = data[k]
                break
        for k in vol_keys:
            if k in data:
                full_vol = data[k]
                break
        # fallback: if only two tensors saved arbitrarily
        if slice_seq is None or full_vol is None:
            # try to find tensors in values
            tensors = [v for v in data.values() if torch.is_tensor(v) or isinstance(v, np.ndarray)]
            if len(tensors) >= 2:
                slice_seq = tensors[0]
                full_vol = tensors[1]
    else:
        raise ValueError("Unsupported .pt format")

    if slice_seq is None or full_vol is None:
        raise ValueError("Could not find required keys (slice_seq, full_volume) in the .pt file.")

    # Convert to numpy
    if torch.is_tensor(slice_seq):
        slice_seq = slice_seq.cpu().numpy()
    if torch.is_tensor(full_vol):
        full_vol = full_vol.cpu().numpy()

    # Normalize shapes:
    # slice_seq expected shape: (N, H, W)
    # full_vol expected shape: (1, D, H, W) or (D,H,W)
    if full_vol.ndim == 4 and full_vol.shape[0] == 1:
        full_vol = full_vol[0]  # (D, H, W)
    elif full_vol.ndim == 4 and full_vol.shape[1] == 1:
        # sometimes shape (1,D,H,W)
        full_vol = np.squeeze(full_vol, axis=0)
    elif full_vol.ndim == 3:
        pass
    else:
        # try to reshape if possible
        raise ValueError(f"Unsupported full_vol shape: {full_vol.shape}")

    # ensure uint8 / 0/1
    slice_seq = (slice_seq > 0).astype(np.uint8)
    full_vol = (full_vol > 0).astype(np.uint8)

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
