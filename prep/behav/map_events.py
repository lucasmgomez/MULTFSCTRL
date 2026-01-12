#!/usr/bin/env python3
import argparse
import re
import shutil
from pathlib import Path

ALNUM_RE = re.compile(r'[^A-Za-z0-9]+')
BLOCK_RE = re.compile(r"(?i)(?:^|[_-])block[_-]?(\d+)(?=$|[^A-Za-z0-9])")

def _clean_label(x: str) -> str:
    """Drop non-alphanumeric chars but keep case (e.g., 'ctg' -> 'ctg')."""
    return ALNUM_RE.sub('', x or '')

def normalize_block(block_str: str) -> str:
    """Convert a 'block_X' chunk into run-XX (0-based → 1-based)."""
    m = BLOCK_RE.search(block_str)
    if not m:
        return "run-01"
    n = int(m.group(1))
    return f"run-{n+1:02d}"

def extract_sub_ses_from_path(path: Path):
    """Look for sub-XX and ses-YY in the directory structure."""
    sub = None
    ses = None
    for part in path.parts:
        if sub is None and re.match(r"sub-\w+", part, re.IGNORECASE):
            sub = re.match(r"(sub-\w+)", part, re.IGNORECASE).group(1)
        if ses is None and re.match(r"ses-\w+", part, re.IGNORECASE):
            ses = re.match(r"(ses-\w+)", part, re.IGNORECASE).group(1)
    return sub, ses

def parse_behav_filename(path: Path):
    """
    From e.g.
      sub-01/ses-01/sub-01_ses-1_20251106-120104_task-1back_ctg_block_0_events_12-33-56.tsv

    Extract:
      sub-01, ses-01, task-1back, acq-ctg, run-01, ext='.tsv'
    """
    name = path.name

    # --- subject & session: prefer folder structure, fallback to filename ---
    sub_from_path, ses_from_path = extract_sub_ses_from_path(path)

    # subject
    if sub_from_path:
        sub = sub_from_path
    else:
        m = re.search(r"(sub-\w+)", name, re.IGNORECASE)
        sub = m.group(1) if m else "sub-XX"

    # session, zero-pad if numeric
    if ses_from_path:
        m_s = re.match(r"ses-?(\d+)", ses_from_path, re.IGNORECASE)
        if m_s:
            ses_num = m_s.group(1).zfill(2)
            ses = f"ses-{ses_num}"
        else:
            ses = ses_from_path
    else:
        m = re.search(r"ses-?(\d+)", name, re.IGNORECASE)
        ses_num = m.group(1).zfill(2) if m else "01"
        ses = f"ses-{ses_num}"

    # --- task ---
    m = re.search(r"task-([A-Za-z0-9]+)", name)
    task = m.group(1) if m else "task"

    # --- acquisition label (between task-XXX_ and _block) ---
    m = re.search(r"task-[A-Za-z0-9]+_(.+?)_block", name)
    acq_raw = m.group(1) if m else ""
    acq = _clean_label(acq_raw)
    acq_part = f"_acq-{acq}" if acq else ""

    # --- block → run ---
    m = BLOCK_RE.search(name)
    block_token = m.group(0) if m else "block_0"
    run = normalize_block(block_token)

    # --- extension (support .tsv or .tsv.gz etc.) ---
    ext = "".join(path.suffixes)  # '.tsv', '.tsv.gz', etc.
    if not ext:
        ext = ".tsv"

    return sub, ses, task, acq_part, run, ext

def build_bids_target(bids_root: Path, src_file: Path):
    sub, ses, task, acq_part, run, ext = parse_behav_filename(src_file)

    # Put directly into <bids_root>/sub-XX/ses-YY/
    ses_dir = bids_root / sub / ses
    ses_dir.mkdir(parents=True, exist_ok=True)

    # Filename: sub-XX_ses-YY_task-<task>[_acq-<acq>]_run-XX_events.ext
    bids_name = f"{sub}_{ses}_task-{task}{acq_part}_{run}_events{ext}"
    return ses_dir / bids_name

def main():
    ap = argparse.ArgumentParser(
        description="Copy behavior TSV files into BIDS sub-XX/ses-YY folders with standardized names."
    )
    ap.add_argument("--src_root", help="Root directory containing behavior files (with sub-XX/ses-YY structure).")
    ap.add_argument("--bids_root", help="Root of BIDS dataset to copy into.")
    ap.add_argument("--pattern", default="*.tsv",
                    help="Glob pattern for behavior files (default: *.tsv).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show planned copies but do not actually copy.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite destination files if they already exist.")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    bids_root = Path(args.bids_root)

    if not src_root.exists():
        raise SystemExit(f"Source root does not exist: {src_root}")

    files = list(src_root.rglob(args.pattern))

    if not files:
        print(f"No files matching pattern '{args.pattern}' found under {src_root}")
        return

    for f in files:
        dest = build_bids_target(bids_root, f)

        if dest.exists() and not args.overwrite:
            print(f"[SKIP] Dest exists (use --overwrite to replace): {dest}")
            continue

        if args.dry_run:
            print(f"[DRY] {f}  →  {dest}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            print(f"COPIED: {f}  →  {dest}")

if __name__ == "__main__":
    main()

"""
python map_events.py \
    --src_root /mnt/tempdata/lucas/fmri/recordings/TR/behav \
    --bids_root /mnt/tempdata/lucas/fmri/recordings/TR/neural/final_all_ses/glasser_resampled \
    --pattern "*.tsv" \
    --overwrite
"""