#!/usr/bin/env python3
import argparse, re, sys
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd
import pydicom

# -------------------- Config --------------------
FUNC_KEYWORDS = ("func", "task", "wm")
LETTER_RE = re.compile(r"(?:[_\-]wm[_\-]?|task[_\-]wm[_\-]?)([ABC])\b", re.IGNORECASE)

BLOCK_RE = re.compile(r"(?i)(^|[_-])block[_-]?(\d+)(?=$|[^A-Za-z0-9])")
ALNUM_RE = re.compile(r'[^A-Za-z0-9]+')

def _clean_label(x: str) -> str:
    """Drop non-alphanumeric chars but keep case (e.g., 'loc_ABAB'→'locABAB')."""
    return ALNUM_RE.sub('', x or '')

def normalize_block_token(s: str) -> str:
    """
    Replace ...[_-]block_0 → ...run-01, ...[_-]block-1 → ...run-02, etc.
    Keeps the leading separator (start/underscore/hyphen).
    """
    def _sub(m):
        sep = m.group(1)  # '' at start, or '_' / '-'
        n = int(m.group(2))
        return f"{sep}run-{n+1:02d}"
    return BLOCK_RE.sub(_sub, str(s))

def block_to_entities(block_raw: str):
    """From 'interdms_loc_ABAB_block_0' → ('interdms_loc_ABAB','01')."""
    blk = normalize_block_token(block_raw)                # interdms_loc_ABAB_run-01
    m = re.match(r"(.+)_run-(\d+)$", blk)
    if not m:
        return blk, "01"
    return m.group(1), m.group(2)  # (variant, run)

def safe(ds, key, default=""):
    try:
        v = getattr(ds, key, None)
        return "" if v is None else str(v)
    except Exception:
        return default

def read_header(f):
    return pydicom.dcmread(f, stop_before_pixels=True, force=True)

def read_full(f):
    return pydicom.dcmread(f, force=True)

def looks_functional(prot, sdesc):
    hay = f"{prot} | {sdesc}".lower()
    return any(k in hay for k in FUNC_KEYWORDS)

def extract_letter(prot):
    if not prot: return ""
    m = LETTER_RE.search(prot)
    return (m.group(1).upper() if m else "")

def is_sbref(ds, prot, sdesc):
    hay = f"{prot} | {sdesc}".lower()
    if "sbref" in hay or "single-band" in hay:
        return True
    try:
        it = ds.get("ImageType", None)
        if it and any("sbref" in str(v).lower() for v in it):
            return True
    except Exception:
        pass
    try:
        ntp = getattr(ds, "NumberOfTemporalPositions", None)
        if ntp and int(ntp) == 1 and "bold" not in hay:
            return True
    except Exception:
        pass
    return False

def to_int(x, default=10**9):
    try: return int(str(x))
    except: return default

def build_letter_queues(design_df):
    by_letter = defaultdict(list)
    for _, r in design_df.iterrows():
        L = str(r["scan_type"]).strip().upper()
        block = str(r["block_file_name"]).strip()
        if L in ("A","B","C"):
            by_letter[L].append(block)
    return {L: deque(v) for L, v in by_letter.items()}

def plan_next_block(letter_queues, L):
    q = letter_queues.get(L)
    if not q or len(q) == 0:
        return None
    return q.popleft()

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(
        description="Map DICOM SeriesDescription/ProtocolName to design-based names in exact run order, then validate."
    )
    ap.add_argument("-dicom_dir", help="Directory with original DICOMs (searched recursively).")
    ap.add_argument("-design_tsv", help="study_design.tsv with columns: session, block_file_name, scan_type")
    ap.add_argument("-out_root", help="Root directory under which a 'mapped' folder will be created.")
    ap.add_argument("--session", required=True, type=str, help="Session number to select (e.g., 1)")
    ap.add_argument("--copy_nonfunc", action="store_true",
                help="Copy non-functional DICOMs unchanged into mapped/.")
    ap.add_argument("--dry_run", action="store_true", help="Preview without writing DICOMs.")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if validation fails.")
    ap.add_argument("--patient_study_prefix", default="multfs_pilot",
                    help="Prefix for PatientName, e.g. 'multfs_pilot'")
    ap.add_argument("--patient_sub", required=True,
                    help="Subject label (e.g., '01' or 'sub_01'—numbers will be zero-padded to 2 by default)")
    ap.add_argument("--patient_ses", required=True,
                    help="Session label (e.g., '01'—numbers will be zero-padded to 2 by default)")
    args = ap.parse_args()

    def _zp2(x: str) -> str:
        xs = str(x)
        return xs.zfill(2) if xs.isdigit() else xs

    target_patient_name = f"{args.patient_study_prefix}_sub_{_zp2(args.patient_sub)}_ses_{_zp2(args.patient_ses)}"

    dicom_dir = Path(args.dicom_dir)
    if not dicom_dir.exists():
        print(f"Missing DICOM dir: {dicom_dir}", file=sys.stderr); sys.exit(1)

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Read & filter design by session --------
    design = pd.read_csv(args.design_tsv, sep="\t")
    must = {"session","block_file_name","scan_type"}
    if not must.issubset(design.columns):
        print(f"Design TSV must have columns: {', '.join(must)}", file=sys.stderr); sys.exit(1)

    design["_session_str"] = design["session"].astype(str).str.strip()
    target_session = str(args.session).strip()

    design_sess = design[design["_session_str"] == target_session].copy()
    if design_sess.empty:
        avail = sorted(design["_session_str"].unique().tolist())
        print(f"No rows found for session={target_session}. Available sessions: {avail}", file=sys.stderr)
        sys.exit(1)

    # Planned lists for validation
    planned_blocks_raw = design_sess["block_file_name"].astype(str).str.strip().tolist()

    def make_expected(block_raw):
        variant, runnum = block_to_entities(block_raw)
        parts = variant.split("_", 1)
        if len(parts) == 2:
            task = parts[0]
            acq  = _clean_label(parts[1])
        else:
            task = parts[0]
            acq  = ""
        acq_part = f"_acq-{acq}" if acq else ""
        return f"func_task-{task}{acq_part}_run-{runnum}"

    planned_seriesdesc = [make_expected(b) for b in planned_blocks_raw]
    planned_protocol   = [make_expected(b) for b in planned_blocks_raw]

    # Queues: nth A/B/C from design
    letter_queues = build_letter_queues(design_sess)

    # -------- Scan DICOMs and group by series --------
    all_paths = set()
    written_paths = set()
    series = {}
    for f in dicom_dir.rglob("*.dcm"):
        try:
            ds = read_header(f)
        except Exception:
            continue

        all_paths.add(f)

        prot = safe(ds, "ProtocolName")
        sdesc = safe(ds, "SeriesDescription")
        if not prot and not sdesc:
            continue
        if not looks_functional(prot, sdesc):
            continue

        suid = safe(ds, "SeriesInstanceUID") or f"SNUM-{safe(ds,'SeriesNumber')}"
        rec = series.setdefault(suid, {
            "SeriesInstanceUID": suid,
            "SeriesNumber": safe(ds, "SeriesNumber"),
            "AcquisitionDateTime": safe(ds, "AcquisitionDateTime") or safe(ds, "AcquisitionTime"),
            "ProtocolName": prot,
            "SeriesDescription": sdesc,
            "Letter": extract_letter(prot or sdesc),
            "Files": [],
            "Kind": None,  # "sbref" | "bold" | "mixed"
        })
        rec["Files"].append(f)

        try:
            sb = is_sbref(ds, prot, sdesc)
        except Exception:
            sb = False
        if rec["Kind"] is None:
            rec["Kind"] = "sbref" if sb else "bold"
        elif rec["Kind"] != ("sbref" if sb else "bold"):
            rec["Kind"] = "mixed"

    if not series:
        print("No functional series found to map. Adjust FUNC_KEYWORDS.", file=sys.stderr)
        sys.exit(1)

    # Sort series in acquisition order
    ser_list = list(series.values())
    ser_list.sort(key=lambda r: (to_int(r["SeriesNumber"]), r["AcquisitionDateTime"]))

    # Build BOLD stream and SBRef list
    bold_series = [s for s in ser_list if s["Kind"] in ("bold", "mixed")]
    sbref_series = [s for s in ser_list if s["Kind"] in ("sbref", "mixed")]

    # Index SBRefs by ProtocolName to find nearest SeriesNumber match
    sbrefs_by_prot = defaultdict(list)
    for s in sbref_series:
        sbrefs_by_prot[s["ProtocolName"]].append(s)
    for prot in sbrefs_by_prot:
        sbrefs_by_prot[prot].sort(key=lambda r: to_int(r["SeriesNumber"]))

    # Pair each BOLD with nearest SBRef (|ΔSeriesNumber| <= 2)
    pairs = []
    used_sbref_uids = set()
    for b in bold_series:
        sn = to_int(b["SeriesNumber"])
        cands = sbrefs_by_prot.get(b["ProtocolName"], [])
        if not cands:
            cands = [s for s in sbref_series if s["SeriesDescription"] == b["SeriesDescription"]]
        best = None
        best_dist = 9999
        for s in cands:
            if s["SeriesInstanceUID"] in used_sbref_uids:
                continue
            d = abs(to_int(s["SeriesNumber"]) - sn)
            if d < best_dist and d <= 2:
                best, best_dist = s, d
        if best:
            used_sbref_uids.add(best["SeriesInstanceUID"])
        pairs.append((b, best))  # (BOLD, SBRef or None)

    # -------- Map runs in order using nth A/B/C queues --------
    mapping_records = []  # per-run record (by acquisition order)
    for idx, (b, s) in enumerate(pairs, start=1):
        L = extract_letter(b["ProtocolName"] or b["SeriesDescription"]).upper()
        if L not in ("A","B","C"):
            print(f"[WARN] Run {idx}: couldn’t extract A/B/C from ProtocolName='{b['ProtocolName']}'. Skipping mapping.", file=sys.stderr)
            continue

        block_raw = plan_next_block(letter_queues, L)
        if block_raw is None:
            print(f"[WARN] Run {idx} (letter {L}): no remaining blocks in design. Leaving original names.", file=sys.stderr)
            continue

        # Run tag and (task, acq) from design
        variant, runnum = block_to_entities(block_raw)
        parts = variant.split("_", 1)
        if len(parts) == 2:
            task = parts[0]
            acq  = _clean_label(parts[1])
        else:
            task = parts[0]
            acq  = ""

        acq_part = f"_acq-{acq}" if acq else ""
        new_seriesdesc = f"func_task-{task}{acq_part}_run-{runnum}"
        new_protocol   = new_seriesdesc

        # Apply to BOLD files
        for f in b["Files"]:
            ds_full = read_full(f)
            ds_full.SeriesDescription = new_seriesdesc
            ds_full.ProtocolName     = new_protocol
            ds_full.PatientName      = target_patient_name
            ds_full.PatientID        = target_patient_name   # optional but recommended
            rel = f.relative_to(dicom_dir)
            out_path = out_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not args.dry_run:
                pydicom.dcmwrite(out_path, ds_full, write_like_original=False)
            written_paths.add(out_path)

        # Apply to SBRef files (if present)
        if s is not None:
            for f in s["Files"]:
                ds_full = read_full(f)
                ds_full.SeriesDescription = new_seriesdesc
                ds_full.ProtocolName     = new_protocol
                ds_full.PatientName      = target_patient_name
                ds_full.PatientID        = target_patient_name
                rel = f.relative_to(dicom_dir)
                out_path = out_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if not args.dry_run:
                    pydicom.dcmwrite(out_path, ds_full, write_like_original=False)

        mapping_records.append({
            "run_index": idx,
            "letter": L,
            "block_file_name": block_raw,
            "seriesdesc": new_seriesdesc,
            "protocol": new_protocol,
            "patient": target_patient_name,
            "bold_seriesuid": b["SeriesInstanceUID"],
            "sbref_seriesuid": (s["SeriesInstanceUID"] if s else ""),
            "seriesnumber": b["SeriesNumber"],
            "acq_datetime": b["AcquisitionDateTime"],
        })

    # -------- VALIDATION: planned vs observed (by acquisition order) --------
    print("\n=== VALIDATION (ordered by acquisition time) ===")
    observed_seriesdesc = [r["seriesdesc"] for r in mapping_records]
    observed_protocol   = [r["protocol"]   for r in mapping_records]

    print("\nPlanned blocks (raw from TSV):")
    print(planned_blocks_raw)

    print("\nExpected mapped names (SeriesDescription/ProtocolName):")
    print(planned_seriesdesc)  # == planned_protocol

    print("\nObserved mapped SeriesDescription list:")
    print(observed_seriesdesc)

    print("\nObserved mapped ProtocolName list:")
    print(observed_protocol)

    ok_seriesdesc = (planned_seriesdesc == observed_seriesdesc)
    ok_protocol   = (planned_protocol   == observed_protocol)

    print(f"\nSeriesDescription match: {ok_seriesdesc}")
    if not ok_seriesdesc:
        m = min(len(planned_seriesdesc), len(observed_seriesdesc))
        for i in range(m):
            if planned_seriesdesc[i] != observed_seriesdesc[i]:
                print(f"  MISMATCH at position {i+1}: plan='{planned_seriesdesc[i]}' vs obs='{observed_seriesdesc[i]}'")
        if len(planned_seriesdesc) != len(observed_seriesdesc):
            print(f"  LENGTH mismatch: plan={len(planned_seriesdesc)} obs={len(observed_seriesdesc)}")

    print(f"ProtocolName match: {ok_protocol}")
    if not ok_protocol:
        m = min(len(planned_protocol), len(observed_protocol))
        for i in range(m):
            if planned_protocol[i] != observed_protocol[i]:
                print(f"  MISMATCH at position {i+1}: plan='{planned_protocol[i]}' vs obs='{observed_protocol[i]}'")
        if len(planned_protocol) != len(observed_protocol):
            print(f"  LENGTH mismatch: plan={len(planned_protocol)} obs={len(observed_protocol)}")

    if args.strict and not (ok_seriesdesc and ok_protocol):
        print("\n[STRICT] Validation failed.", file=sys.stderr)
        sys.exit(2)

    # -------- Write a log CSV in mapped/ --------
    if mapping_records:
        df_map = pd.DataFrame(mapping_records).sort_values(["run_index"])
        log_path = out_dir / f"mapping_log.session-{target_session}.csv"
        df_map.to_csv(log_path, index=False)
        print(f"\nMapping log written: {log_path}")

    if not mapping_records:
        print("\nNote: no runs were mapped (see warnings above).")

    # -------- Copy (and rewrite) non-functional DICOMs if requested --------
    if args.copy_nonfunc and not args.dry_run:
        rewritten = 0
        skipped   = 0
        for f in all_paths:
            rel  = f.relative_to(dicom_dir)
            dest = out_dir / rel

            if dest in written_paths:
                continue

            try:
                ds_hdr = read_header(f)
                prot   = safe(ds_hdr, "ProtocolName")
                sdesc  = safe(ds_hdr, "SeriesDescription")
                is_func = looks_functional(prot, sdesc)
            except Exception:
                skipped += 1
                print(f"[WARN] Could not read DICOM header for non-functional copy: {f}", file=sys.stderr)
                continue

            if is_func:
                skipped += 1
                continue

            try:
                ds_full = read_full(f)
                ds_full.PatientName = target_patient_name
                ds_full.PatientID   = target_patient_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                pydicom.dcmwrite(dest, ds_full, write_like_original=False)
                written_paths.add(dest)
                rewritten += 1
            except Exception:
                skipped += 1
                print(f"[WARN] Could not rewrite non-functional DICOM: {f}", file=sys.stderr)
                continue

        print(f"\nRewrote {rewritten} non-functional DICOMs with new PatientName/ID into: {out_dir} "
              f"(skipped {skipped})")

if __name__ == "__main__":
    main()


"""
python map_dcms.py \
    -dicom_dir /mnt/tempdata/lucas/fmri/recordings/TR/neural/ses-1/dicom/raw \
    -design_tsv /mnt/tempdata/lucas/fmri/recordings/TR/study_design/study_designs/sub-01_design.tsv \
    -out_root /mnt/tempdata/lucas/fmri/recordings/TR/neural/ses-1/dicom/mapped/sub-01/ses-01 \
    --copy_nonfunc \
    --session 1 \
    --strict \
    --patient_study_prefix multfs_pilot \
    --patient_sub 01 \
    --patient_ses 01
"""