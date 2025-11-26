# COSC 4368 – Minecraft Cave Generator (VAE + Tensor Dataset)
## **Group members : Nam Cao, ...**
This project is a course project for **COSC 4368 – Fundamentals of AI**.  
The goal is to **learn 3D cave structures from Minecraft** and then **generate new, plausible caves** that can be spawned in-game via a custom Minecraft mod.

We:

- Start from a large dataset of **Minecraft cave volumes** (`.pt` files with 3D tensors).
- Train a **3D Variational Autoencoder (VAE)** on those cave volumes.
- Sample new caves from the trained VAE.
- Export the generated caves as:
  - **`.pt` volumes** (for visualization).
  - **JSON lists of air-voxel coordinates** (for use by a Minecraft mod that places caves in the world).

> **Important:** This repo focuses on the **ML side** (training + sampling + visualization).  
> A separate Minecraft mod (by Adem) reads the JSON files and converts local voxel coordinates into Minecraft world coordinates, then surrounds them with stone to form actual caves.

---

## Repository Structure

```text
Minecraft-Data-Parser/
├─ data/
│  ├─ caves_pt/               # Real Minecraft cave volumes (.pt), NOT committed (≈ 6 GB)
│  ├─ caves_manifest.csv      # Summary of the cave dataset (shapes, air_ratio, outliers)
├─ io_utils/
│  ├─ __init__.py
│  └─ caves_pt_utils.py       # Helpers for loading raw .pt caves and building manifests
├─ models/
│  ├─ __init__.py
│  └─ vae3d.py                # 3D VAE model (VAE3D: encoder/decoder for 32×32×32 volumes)
├─ scripts/
│  ├─ scan_caves.py           # Build caves_manifest.csv from data/caves_pt/*.pt
│  ├─ train_vae.py            # Train the 3D VAE on Minecraft cave volumes
│  ├─ sample_caves.py         # Sample random caves from the trained VAE → JSON (coords) + stats
│  ├─ json_to_pt.py           # Convert JSON coords back into a 3D tensor .pt volume
│  ├─ eval_generated.py       # Compare real vs generated caves (air ratios, counts)
│  └─ sample_caves_filtered.py# (Optional) sample caves that satisfy simple filters
├─ viz/
│  ├─ viz_cave.py             # Visualize 3D cave volumes (slices + simple 3D view)
├─ .gitignore
└─ README.md
```
## Data Format
### Real cave files (data/caves_pt/*.pt)

Each cave file **cave_N.pt** is a PyTorch dict with at least:

- **full_volume**: **torch.Tensor** with shape **(D, H, W) = (129, 128, 128)**

    - 1 = rock / solid

    - 0 = air / empty

- **slice_seq**: **torch.Tensor** with shape **(T, H, W)**
    - A sequence of 2D slices along the depth/height axis.

These files are big (~6 GB total) and are not tracked in git. They live in:
```
data/caves_pt/
  cave_0.pt
  cave_1.pt
  cave_2.pt
  ...
```
### Generated cave files

When we sample caves, we work at a smaller resolution:

- Generated volumes are typically (32, 32, 32), stored as a tensor with shape `(1, 1, 32, 32, 32)` in `.pt`.

Values follow the same convention:

- `0` = air (cave)

- `1` = rock (solid)

We convert these to JSON as lists of air voxels:
```json
[
  {
    "sample_id": 0,
    "shape": [1, 1, 32, 32, 32],
    "num_voxels": 32768,
    "num_cave_voxels": 11297,
    "voxels": [
      {"x": 0, "y": 0, "z": 0},
      {"x": 1, "y": 0, "z": 0},
      ...
    ]
  }
]
```

- **`num_voxels`** = total voxels = **`32 * 32 * 32`**.

- **`num_cave_voxels`** = how many of those voxels are air (**`0`**), i.e., the cave interior size.

- **`(x, y, z)`** are local coordinates inside that 32³ box.
Adem’s Minecraft mod uses these local coordinates to place caves in the game.

## Environment Setup
```markdown
Tested with:

- Python 3.12

- Windows + PowerShell

- Virtual environment (venv) recommended.
```
### 1. Create and activate **`venv`**

From the repo root:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

Minimal set (what the project actually uses):
```powershell
python -m pip install --upgrade pip

python -m pip install `
  torch `
  numpy `
  matplotlib `
  open3d
```

If `open3d` or other packages fail on Windows due to path length,
enable **Windows Long Path Support** and retry (pip will show a helpful URL).

## 1. Scan the dataset (build manifest)

Once you have your `data/caves_pt/ folder` populated with real caves (`cave_0.pt`, etc.), run:
```powershell
python -m scripts.scan_caves --caves-root .\data\caves_pt --out .\data\caves_manifest.csv
```

This produces a CSV like:
```csv
file,shape_raw,shape_3d,air_ratio,is_outlier
cave_0.pt,"[1, 129, 128, 128]","[129, 128, 128]",0.5504,False
cave_103.pt,"[1, 129, 128, 128]","[129, 128, 128]",0.6731,False
...
```

- **`air_ratio`** = (# air voxels) / (# total voxels)

- **`is_outlier`** flags caves like **cave_4.pt** if they are extreme or corrupted.

This manifest is useful for:

- Sanity-checking the dataset.

- Picking interesting real caves to visualize or compare against.

## 2. Train the 3D VAE

The VAE learns a distribution over downsampled Minecraft caves (32×32×32).

Example training run (what we used):
```powershell
python -m scripts.train_vae `
  --caves-root .\data\caves_pt `
  --epochs 25 `
  --batch-size 2 `
  --z-dim 64 `
  --max-samples 1000
```

Output (example):
```text
Dataset size: 1000 caves
Epoch 1: loss=14564.16, recon=13223.42, kld=1340.75
...
Epoch 25: loss=3295.94, recon=3089.10, kld=206.84
Saved model weights to vae3d_z64.pt
```

- `--max-samples` lets you cap how many caves you train on (useful for speed).

- At the end, the model weights are saved to `vae3d_z64.pt` in the repo root.

**Note**: All sampling uses `vae3d_z64.pt`.
The generated `.pt` cave volumes (like `gen_cave_final.pt`) are outputs, not the weights.

## 3. Sample caves from the trained VAE
### Basic random sampling

Generate a few caves and dump them to JSON:
```powershell
python -m scripts.sample_caves `
  --weights .\vae3d_z64.pt `
  --z-dim 64 `
  --num-samples 3 `
  --threshold 0.03 `
  --out caves_samples_3.json
```

Example log:
```text
[Sample 0] probs stats: min= 4.1e-08 max= 0.9999 mean= 0.1370 shape= (1, 1, 32, 32, 32)
Sample 0 volume stats: min= 0.0 max= 1.0 mean= 0.3448 shape= (1, 1, 32, 32, 32)
...
```

- **`--threshold`** is applied to the sigmoid probabilities to binarize the volume.

    - Lower thresholds → more air (bigger caves).

    - Higher thresholds → more rock (smaller caves).

- The JSON (`caves_samples_3.json`) contains local voxel coordinates for each sample.

### Single cave for Minecraft integration

Generate a single “final” cave:
```powershell
python -m scripts.sample_caves `
  --weights .\vae3d_z64.pt `
  --z-dim 64 `
  --num-samples 1 `
  --threshold 0.03 `
  --out cave_sample_final.json
```

Then convert this JSON to a `.pt` volume (for visualization):
```powershell
python -m scripts.json_to_pt cave_sample_final.json --out-pt gen_cave_final.pt
```

Now you have:

- `cave_sample_final.json` → what Adem’s mod will ingest.

- `gen_cave_final.pt` → same cave stored as a (32,32,32) tensor for visualization.

## 4. Visualize caves (real + generated)
### Visualize a real cave
```powershell
python .\viz\viz_cave.py .\data\caves_pt\cave_0.pt
```

This opens a simple matplotlib viewer that shows:

- Slice sequence (2D slices through the cave).

- Keyboard navigation (arrow keys) to move through slices.

Example output in console:
```text
Loaded slice_seq shape: (8, 128, 128)
Loaded full_vol shape: (129, 128, 128)
```
### Visualize a generated cave
```powershell
python .\viz\viz_cave.py .\gen_cave_final.pt
```

or any other generated cave:

```powershell
python .\viz\viz_cave.py .\gen_cave_2.pt
python .\viz\viz_cave.py .\gen_small_cave_0.pt
```

For pure tensor `.pt` files (like `gen_small_cave_0.pt`), `viz_cave.py`:

- Treats the `(1,1,32,32,32)` tensor as a `(32,32,32)` volume.

- Synthesizes a small sequence of 2D slices along depth to visualize.

## 5. Compare real vs generated caves (metrics)

Use `eval_generated.py` to compare air ratios and voxel counts.

Example:
```powershell
python -m scripts.eval_generated --real-pt `
  .\data\caves_pt\cave_0.pt `
  .\data\caves_pt\cave_103.pt `
  .\data\caves_pt\cave_1003.pt `
  .\data\caves_pt\cave_1008.pt `
  .\data\caves_pt\cave_1019.pt `
  .\data\caves_pt\cave_1025.pt `
  --gen-pt .\gen_cave_final.pt
```

Output example:
``` text
[REAL] data\caves_pt\cave_0.pt
  shape        : (D,H,W) = (129, 128, 128)
  total voxels : 2113536
  air voxels   : 1163376
  rock voxels  : 950160
  air_ratio    : 0.5504

...

[GEN ] gen_cave_final.pt
  shape        : (D,H,W) = (32, 32, 32)
  total voxels : 32768
  air voxels   : 19720
  rock voxels  : 13048
  air_ratio    : 0.6018
```

This is useful for:

- Showing that the generated caves’ **`air_ratio`** is in a plausible range compared to real caves.

- Supporting quantitative claims in the presentation/report.

## 6. Sampling “small caves” with filters

`sample_caves_filtered.py` is a helper to sample caves that satisfy basic criteria, e.g.:

- Air ratio between 0.3 and 0.6.

- Approximate size thresholds, etc.

Example usage (tune arguments as needed):
```powershell
python -m scripts.sample_caves_filtered `
  --weights .\vae3d_z64.pt `
  --z-dim 64 `
  --num-samples 3 `
  --threshold 0.03 `
  --max-tries 200 `
  --json-out small_caves.json `
  --pt-prefix gen_small_cave
```

This will:

- Try up to `max_tries` random samples.

- Accept those that pass the filters.

- Save:

    - JSON with voxel coordinates (`small_caves.json`)

    - `.pt` volumes (`gen_small_cave_0.pt`, `gen_small_cave_1.pt`, …)
      which can be visualized with viz_cave.py.

If no caves match, relax the filters in the script (e.g., wider `air_ratio` range).

## What we have achieved

- Dataset handling for thousands of Minecraft caves (**`caves_manifest.csv`**).

- **3D VAE** that learns a distribution over downsampled cave volumes.

- Random cave generation with:

    - Controllable threshold to adjust cave “fullness”.

    - Export to JSON voxel coordinates for integration with a Minecraft mod.

- Visualization tools to inspect real and generated caves (2D slices).

- Basic metrics (`air_ratio`) to compare real vs generated caves.

These pieces are enough to:

- Show real cave examples (slices / 3D view) in the presentation.

- Show generated caves side-by-side and discuss similarities/differences.

- Provide JSON cave definitions to **Adem’s Minecraft mod** so caves can be placed in an actual Minecraft world.
