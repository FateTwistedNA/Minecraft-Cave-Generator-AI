"""
mc_world_to_dataset.py (patched for legacy OldBlock support)

Converts a Minecraft Java Edition world save into model-friendly artifacts:
- world_meta.json (from level.dat)
- per-chunk heightmaps (.npy)
- per-chunk block_counts.json
- tile_entities.jsonl (optional)
- index.csv

Usage:
  pip install "anvil-parser>=0.4.0" "nbtlib>=1.12.1" numpy
  python mc_world_to_dataset.py "/path/to/.minecraft/saves/YourWorld" --out ./dataset_out
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable

import numpy as np

try:
    import anvil  # anvil-parser
except Exception as e:
    print("ERROR: Could not import 'anvil' (anvil-parser). Install it with:", file=sys.stderr)
    print("  pip install anvil-parser", file=sys.stderr)
    raise

try:
    import nbtlib
except Exception as e:
    print("ERROR: Could not import 'nbtlib'. Install it with:", file=sys.stderr)
    print("  pip install nbtlib==1.12.1", file=sys.stderr)
    raise


# Accept several spellings for air across versions
AIR_ALIASES = {
    "minecraft:air", "minecraft:cave_air", "minecraft:void_air",
    "air", "cave_air", "void_air", "legacy:air"
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Minecraft world to model-friendly data.")
    p.add_argument("world", type=str, help="Path to the world directory (folder containing level.dat).")
    p.add_argument("--out", type=str, default="./mc_dataset_out", help="Output directory.")
    p.add_argument("--dims", type=str, nargs="*", default=["overworld", "nether", "end"],
                   help="Dimensions to export: any subset of {overworld, nether, end}.")
    p.add_argument("--ymin", type=int, default=0, help="Minimum Y to scan (default 0).")
    p.add_argument("--ymax", type=int, default=255, help="Maximum Y to scan inclusive (default 255; set 319 for 1.18+).")
    p.add_argument("--max-regions", type=int, default=None, help="Limit number of region files per dimension (for testing).")
    p.add_argument("--skip-entities", action="store_true", help="Skip tile-entity extraction to speed up.")
    return p.parse_args()


def dim_paths(world_dir: Path, dims: Iterable[str]) -> Dict[str, Path]:
    mapping = {}
    for d in dims:
        if d == "overworld":
            mapping[d] = world_dir / "region"
        elif d == "nether":
            mapping[d] = world_dir / "DIM-1" / "region"
        elif d == "end":
            mapping[d] = world_dir / "DIM1" / "region"
        else:
            print(f"Warning: unknown dimension '{d}' (skipping).", file=sys.stderr)
    return mapping


REGION_RE = re.compile(r"r\.(-?\d+)\.(-?\d+)\.mca$")


def parse_region_coords(region_filename: str) -> Optional[Tuple[int, int]]:
    m = REGION_RE.search(region_filename)
    if not m:
        return None
    rx, rz = int(m.group(1)), int(m.group(2))
    return rx, rz


def load_level_dat(world_dir: Path) -> Dict:
    level_path = world_dir / "level.dat"
    if not level_path.exists():
        return {}
    nbt = nbtlib.load(str(level_path))
    data = nbt.get("Data") or {}

    def to_py(x):
        if isinstance(x, nbtlib.tag.Compound):
            return {k: to_py(v) for k, v in x.items()}
        if isinstance(x, nbtlib.tag.List):
            return [to_py(v) for v in x]
        if isinstance(x, (nbtlib.tag.Int, nbtlib.tag.Long, nbtlib.tag.Short, nbtlib.tag.Byte)):
            return int(x)
        if isinstance(x, (nbtlib.tag.Double, nbtlib.tag.Float)):
            return float(x)
        if isinstance(x, (nbtlib.tag.String,)):
            return str(x)
        return x

    return to_py(data)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def block_to_name(b) -> str:
    """Return a best-effort namespaced id for a block, robust to legacy OldBlock."""
    if b is None:
        return "minecraft:air"

    # Legacy path: OldBlock often has .id (0 == air) and maybe .data/metadata
    try:
        legacy_id = getattr(b, "id", None)
        if legacy_id is not None:
            try:
                if int(legacy_id) == 0:
                    return "minecraft:air"
            except Exception:
                pass
    except Exception:
        pass

    # Common newer styles
    for attr in ("name", "namespaced_name"):
        try:
            v = getattr(b, attr)
            v = v() if callable(v) else v
            if isinstance(v, str) and v:
                return v if ":" in v else f"minecraft:{v}"
        except Exception:
            pass

    # Some objects expose .namespace and .id (string id part)
    try:
        ns = getattr(b, "namespace", None)
        nid = getattr(b, "id", None)
        if isinstance(ns, str) and isinstance(nid, str):
            return f"{ns}:{nid}"
    except Exception:
        pass

    # Fallback to string form
    try:
        s = str(b)
        if s:
            return s if ":" in s else f"minecraft:{s}"
    except Exception:
        pass

    return "minecraft:air"


def is_air(name: str) -> bool:
    base = name.lower()
    return base in AIR_ALIASES or base.endswith(":air") or base in {"cave_air", "void_air"}


def export_world(world_dir: Path, out_dir: Path, dims: Iterable[str], ymin: int, ymax: int,
                 max_regions: Optional[int], skip_entities: bool) -> None:
    ensure_dir(out_dir)
    meta = load_level_dat(world_dir)
    with open(out_dir / "world_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    chunks_dir = out_dir / "chunks"
    ensure_dir(chunks_dir)
    tiles_path = out_dir / "tile_entities.jsonl"
    index_path = out_dir / "index.csv"

    index_fields = [
        "dimension", "region_rx", "region_rz",
        "chunk_x_in_region", "chunk_z_in_region",
        "world_chunk_x", "world_chunk_z",
        "has_chunk", "blocks_total", "unique_blocks",
        "heightmap_min", "heightmap_max", "heightmap_mean"
    ]
    index_fp = open(index_path, "w", newline="", encoding="utf-8")
    index_writer = csv.DictWriter(index_fp, fieldnames=index_fields)
    index_writer.writeheader()

    tiles_fp = None
    if not skip_entities:
        tiles_fp = open(tiles_path, "w", encoding="utf-8")

    dim_to_path = dim_paths(world_dir, dims)
    for dim, rpath in dim_to_path.items():
        if not rpath.exists():
            print(f"[{dim}] No region dir at: {rpath} (skipping).", file=sys.stderr)
            continue

        region_files = sorted(p for p in rpath.glob("r.*.*.mca"))
        if max_regions is not None:
            region_files = region_files[:max_regions]

        print(f"[{dim}] Found {len(region_files)} region files.", file=sys.stderr)

        for ri, rf in enumerate(region_files):
            coords = parse_region_coords(rf.name)
            if not coords:
                print(f"  Skipping file with unexpected name: {rf.name}", file=sys.stderr)
                continue
            rx, rz = coords
            try:
                region = anvil.Region.from_file(str(rf))
            except Exception as e:
                print(f"  ERROR opening region {rf}: {e}", file=sys.stderr)
                continue

            dim_region_dir = chunks_dir / dim / f"r.{rx}.{rz}"
            ensure_dir(dim_region_dir)

            for cx in range(32):
                for cz in range(32):
                    # Quick header check: 0 offset means no chunk
                    try:
                        off, length = region.chunk_location(cx, cz)
                    except Exception:
                        off, length = 0, 0
                    if off == 0 and length == 0:
                        row = {
                            "dimension": dim,
                            "region_rx": rx, "region_rz": rz,
                            "chunk_x_in_region": cx, "chunk_z_in_region": cz,
                            "world_chunk_x": rx * 32 + cx, "world_chunk_z": rz * 32 + cz,
                            "has_chunk": 0, "blocks_total": 0, "unique_blocks": 0,
                            "heightmap_min": "", "heightmap_max": "", "heightmap_mean": "",
                        }
                        index_writer.writerow(row)
                        continue

                    try:
                        chunk = region.get_chunk(cx, cz)
                    except Exception as e:
                        print(f"    WARN: cannot read chunk ({cx},{cz}) in r.{rx}.{rz}: {e}", file=sys.stderr)
                        continue

                    block_counts: Dict[str, int] = {}
                    heightmap = np.full((16, 16), fill_value=ymin, dtype=np.int32)

                    # Heightmap: scan downward until first non-air at (x,z)
                    for x in range(16):
                        for z in range(16):
                            top_y = ymin
                            for y in range(ymax, ymin - 1, -1):
                                try:
                                    b = chunk.get_block(x, y, z)
                                except Exception:
                                    b = None
                                name = block_to_name(b)
                                if not is_air(name):
                                    top_y = y
                                    break
                            heightmap[x, z] = top_y

                    # Block counts (fast path via stream_chunk when available)
                    blocks_total = 0
                    try:
                        for b in chunk.stream_chunk():
                            name = block_to_name(b)
                            block_counts[name] = block_counts.get(name, 0) + 1
                            blocks_total += 1
                    except Exception:
                        for x in range(16):
                            for z in range(16):
                                for y in range(ymin, ymax + 1):
                                    try:
                                        b = chunk.get_block(x, y, z)
                                    except Exception:
                                        b = None
                                    name = block_to_name(b)
                                    block_counts[name] = block_counts.get(name, 0) + 1
                                    blocks_total += 1

                    # Tile entities
                    if tiles_fp is not None:
                        try:
                            for te in getattr(chunk, "tile_entities", []) or []:
                                def conv(val):
                                    try:
                                        return val.value
                                    except Exception:
                                        try:
                                            return str(val)
                                        except Exception:
                                            return None
                                rec = {k: conv(v) for k, v in te.items()}
                                rec["dimension"] = dim
                                rec["region_rx"] = rx
                                rec["region_rz"] = rz
                                rec["chunk_x_in_region"] = cx
                                rec["chunk_z_in_region"] = cz
                                json.dump(rec, tiles_fp)
                                tiles_fp.write("\\n")
                        except Exception as e:
                            print(f"    WARN: tile entity read failed at chunk ({cx},{cz}) r.{rx}.{rz}: {e}", file=sys.stderr)

                    # Save artifacts for this chunk
                    base = f"c.{cx}.{cz}"
                    np.save(dim_region_dir / f"{base}.heightmap.npy", heightmap)
                    with open(dim_region_dir / f"{base}.block_counts.json", "w", encoding="utf-8") as f:
                        json.dump(block_counts, f, indent=2, ensure_ascii=False)

                    hm_vals = heightmap.flatten()
                    row = {
                        "dimension": dim,
                        "region_rx": rx, "region_rz": rz,
                        "chunk_x_in_region": cx, "chunk_z_in_region": cz,
                        "world_chunk_x": rx * 32 + cx, "world_chunk_z": rz * 32 + cz,
                        "has_chunk": 1,
                        "blocks_total": int(blocks_total),
                        "unique_blocks": int(len(block_counts)),
                        "heightmap_min": int(hm_vals.min()) if hm_vals.size else "",
                        "heightmap_max": int(hm_vals.max()) if hm_vals.size else "",
                        "heightmap_mean": float(hm_vals.mean()) if hm_vals.size else "",
                    }
                    index_writer.writerow(row)

            print(f"  [{dim}] Processed region r.{rx}.{rz} ({ri+1}/{len(region_files)}).")

    index_fp.close()
    if tiles_fp is not None:
        tiles_fp.close()

    print(f"Done. Outputs in: {out_dir}")


def main():
    args = parse_args()
    world_dir = Path(args.world).expanduser().resolve()
    if not (world_dir / "level.dat").exists():
        print(f"ERROR: '{world_dir}' does not look like a world folder (missing level.dat).", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out).expanduser().resolve()
    export_world(world_dir, out_dir, args.dims, args.ymin, args.ymax, args.max_regions, args.skip_entities)


if __name__ == "__main__":
    main()
