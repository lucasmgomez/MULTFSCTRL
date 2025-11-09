#!/usr/bin/env python3
"""
Copy .tsv files to a new folder, appending last-modified TIME (HH-MM-SS) to filenames.

Examples:
  src/data.tsv         -> dest/data_14-35-22.tsv
  src/reports/a.tsv    -> dest/reports/a_09-12-05.tsv        (default: preserves folder tree)
  --flatten example    -> dest/a_09-12-05.tsv                (puts all files directly in dest)

Usage:
  python copy_tsv_with_mtime.py /path/to/src /path/to/dest --dry-run
  python copy_tsv_with_mtime.py /path/to/src /path/to/dest -r
  python copy_tsv_with_mtime.py /path/to/src /path/to/dest --flatten
  python copy_tsv_with_mtime.py /path/to/src /path/to/dest --utc
  python copy_tsv_with_mtime.py /path/to/src /path/to/dest --overwrite
  python copy_tsv_with_mtime.py /path/to/src /path/to/dest --append-always
"""

import argparse
from pathlib import Path
from datetime import datetime, timezone
import shutil
import re

# Detects a trailing _HH-MM-SS (or -HH-MM-SS) to avoid double-appending
TIME_RE = re.compile(r"(?:^|[_-])(\d{2})-(\d{2})-(\d{2})(?:$|\.)")

def already_has_time(stem: str) -> bool:
    return bool(TIME_RE.search(stem))

def time_suffix(p: Path, use_utc: bool) -> str:
    ts = p.stat().st_mtime
    dt = datetime.fromtimestamp(ts, tz=timezone.utc if use_utc else None)
    return dt.strftime("%H-%M-%S")

def target_name(p: Path, append_time: bool, sep: str, use_utc: bool) -> str:
    if append_time:
        return f"{p.stem}{sep}{time_suffix(p, use_utc)}{p.suffix}"
    else:
        return p.name  # keep original name (e.g., if it already had a time)

def next_available_path(dst: Path) -> Path:
    """If dst exists, return dst with _v2, _v3, ... before the suffix."""
    if not dst.exists():
        return dst
    base = dst.with_suffix("")
    suffix = dst.suffix
    i = 2
    while True:
        candidate = dst.with_name(f"{base.name}_v{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1

def copy_one(src: Path, dst_dir: Path, *,
             use_utc: bool, sep: str,
             overwrite: bool, append_always: bool) -> None:
    if src.suffix.lower() != ".tsv":
        return

    # Decide whether to append a time to the filename
    append_time = append_always or not already_has_time(src.stem)
    name = target_name(src, append_time, sep, use_utc)

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / name
    if not overwrite:
        dst_path = next_available_path(dst_path)

    shutil.copy2(src, dst_path)  # preserves timestamps & metadata
    print(f"COPIED: {src} -> {dst_path}")

def main():
    ap = argparse.ArgumentParser(description="Copy .tsv files with HH-MM-SS mtime appended.")
    ap.add_argument("src", type=Path, help="Source folder to scan")
    ap.add_argument("dest", type=Path, help="Destination folder to copy into")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--flatten", action="store_true",
                    help="Do not preserve folder structure; copy all into dest root")
    ap.add_argument("--utc", action="store_true", help="Use UTC time (default: local time)")
    ap.add_argument("--sep", default="_", help="Separator before time (default: _ )")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing files at destination (default: create _v2, _v3, ...)")
    ap.add_argument("--append-always", action="store_true",
                    help="Append time even if filename already ends with HH-MM-SS")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done, without copying")
    args = ap.parse_args()

    if not args.src.exists() or not args.src.is_dir():
        ap.error(f"{args.src} is not a valid folder")

    # Gather files
    paths = (args.src.rglob("*.tsv") if args.recursive
             else (p for p in args.src.iterdir() if p.is_file() and p.suffix.lower() == ".tsv"))

    any_found = False
    for src in sorted(paths):
        any_found = True
        # Decide destination directory (preserve tree by default)
        if args.flatten:
            dst_dir = args.dest
        else:
            rel = src.parent.relative_to(args.src)
            dst_dir = args.dest / rel

        if args.dry_run:
            append_time = args.append_always or not already_has_time(src.stem)
            name = target_name(src, append_time, args.sep, args.utc)
            intended = (dst_dir / name)
            if not args.overwrite:
                intended = next_available_path(intended)
            print(f"DRY-RUN: {src} -> {intended}")
        else:
            copy_one(src, dst_dir,
                     use_utc=args.utc,
                     sep=args.sep,
                     overwrite=args.overwrite,
                     append_always=args.append_always)

    if not any_found:
        print("No .tsv files found.")

if __name__ == "__main__":
    main()