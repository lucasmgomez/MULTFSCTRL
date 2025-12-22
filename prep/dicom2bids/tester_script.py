#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from collections import defaultdict

import pydicom

HCPT_PATTERN = re.compile(r"hcptrt[._-](\d+)\b", re.IGNORECASE)

def read_series_desc(path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return str(getattr(ds, "SeriesDescription", "") or "").strip()
    except Exception:
        return ""

def extract_block_from_name(path: Path):
    m = HCPT_PATTERN.search(path.name)
    return m.group(1) if m else None

def main():
    ap = argparse.ArgumentParser(description="List block number -> SeriesDescription from DICOM filenames")
    ap.add_argument("dicom_dir", nargs="?", default=".", help="Directory to search (default: current dir)")
    args = ap.parse_args()

    base = Path(args.dicom_dir)
    if not base.exists():
        print(f"Missing directory: {base}")
        return

    by_block = defaultdict(set)
    no_block = []

    for p in base.rglob("*.dcm"):
        blk = extract_block_from_name(p)
        sd = read_series_desc(p)
        if blk:
            by_block[blk].add(sd or "<empty SeriesDescription>")
        else:
            no_block.append((p, sd))

    for blk in sorted(by_block, key=lambda x: int(x)):
        descs = sorted(by_block[blk])
        print(f"Block {blk}:")
        for d in descs:
            print(f"  - {d}")

    if no_block:
        print("\nFiles with no 'hcptrt.<num>' block in filename:")
        for p, sd in no_block:
            print(f"  {p} -> {sd or '<empty SeriesDescription>'}")

if __name__ == "__main__":
    main()