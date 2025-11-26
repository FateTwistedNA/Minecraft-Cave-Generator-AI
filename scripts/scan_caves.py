# scripts/scan_caves.py
from __future__ import annotations

import argparse
from pathlib import Path

from io_utils.caves_pt_utils import build_manifest, OUTLIER_FILES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caves-root", required=True)
    ap.add_argument("--out", default="caves_manifest.csv")
    args = ap.parse_args()

    caves_root = Path(args.caves_root)
    out_csv = Path(args.out)
    print(f"Scanning caves in {caves_root} (known outliers: {OUTLIER_FILES})")
    build_manifest(caves_root, out_csv)


if __name__ == "__main__":
    main()
