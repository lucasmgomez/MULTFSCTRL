#!/usr/bin/env python3
"""
Split-half reliability (4 sessions) for LSS betas within a Glasser ROI, aligned by tc.

CHANGES vs your current script:
- With Fisher-z transform.
- Spearman–Brown is applied PER SPLIT (to each split r), then we also average SB values.
- Outputs both:
    r_mean_raw: mean of raw split correlations
    r_sb_mean: mean of per-split SB-corrected correlations
  plus per-split r and per-split SB in split_rs/split_rs_sb.

Keeps your QC counting + verbose warnings.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib


# ----------------------------
# ROI mask (Glasser dlabel)
# ----------------------------
def make_roi_mask(dlabel_path: str, roi_names):
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0]  # {id: (name, rgba)}

    roi_set = {r.strip() for r in roi_names if r.strip()}
    if not roi_set:
        raise ValueError("roi_names is empty after parsing.")

    keys = [k for k, (name, _) in label_dict.items() if name in roi_set]
    found = [label_dict[k][0] for k in keys]
    missing = sorted(list(roi_set - set(found)))
    if missing:
        all_names = sorted([name for _, (name, _) in label_dict.items()])
        suggestions = [n for n in all_names if any(m.lower() in n.lower() for m in missing)][:40]
        raise ValueError(
            f"Missing ROI labels: {missing}\n"
            f"Found: {found}\n"
            f"Suggestions: {suggestions}"
        )

    mask = np.isin(data, keys)
    if mask.sum() == 0:
        raise ValueError("ROI mask has 0 vertices (space mismatch or wrong labels).")

    print("Matched ROI labels:", found)
    print("Total ROI vertices:", int(mask.sum()))
    return mask, "+".join(found)


# ----------------------------
# Naming helpers
# ----------------------------
TASKDIR_RE = re.compile(r"^task-(?P<task>[^_]+)_acq-(?P<acq>.+)_run-(?P<run>\d+)$")
BETA_RE = re.compile(
    r"""
    (?:
        # LSS format
        lss-(EncTarget|DelayTarget)_(Enc\d{4}|Del\d{4})_events_beta-target
      |
        # LSA format
        lsa_[^_]+(?:_[^_]+)*_(Enc|Del)_(Enc\d{4}|Del\d{4})_beta
    )
    \.dscalar\.nii$
    """,
    re.VERBOSE
)


def parse_taskdir_name(taskdir_basename: str):
    """
    Example folder: task-interdms_acq-obj_ABAB_run-01
    Returns: task='interdms', acq='obj_ABAB', run='01'
    """
    m = TASKDIR_RE.match(taskdir_basename)
    if not m:
        return None
    return m.group("task"), m.group("acq"), m.group("run")


def blockfile_name(task: str, acq: str, run: str):
    """block index = run-01 -> 0, run-02 -> 1"""
    block_idx = int(run) - 1
    return f"{task}_{acq}_block_{block_idx}.csv"


def behav_baseevents_name(task: str, acq: str, run: str):
    """taskname includes variant: 1back_ctg, interdms_obj_ABAB, ctxdm_col"""
    return f"task-{task}_{acq}_run-{run}_base-events.tsv"


# ----------------------------
# Mapping Trial -> tc
# ----------------------------
def load_trial_tc(block_csv_path: str):
    df = pd.read_csv(block_csv_path)
    if "tc" not in df.columns:
        raise ValueError(f"'tc' column not found in block file: {block_csv_path}")

    tc_map = {}
    for i, val in enumerate(df["tc"].tolist(), start=1):
        tc_map[f"Trial{i:02d}"] = str(val)
    return tc_map


def load_eventid_to_trialpos(base_events_tsv: str):
    df = pd.read_csv(base_events_tsv, sep="\t")
    needed = {"event_id", "phase", "trial", "pos"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {base_events_tsv}: {missing}")

    out = {}
    for _, row in df.iterrows():
        eid = str(row["event_id"])
        phase = str(row["phase"])
        trial = str(row["trial"])
        pos = int(row["pos"])
        out[eid] = (phase, trial, pos)
    return out


# ----------------------------
# Load ROI-mean betas per event, keyed by (tc, pos)
# ----------------------------
def collect_session_series(
    task_dir: str,
    roi_mask: np.ndarray,
    base_events_tsv: str,
    block_csv: str,
    verbose: bool = True,
):
    """
    Returns:
      phase_to_series: dict phase -> { (tc, pos) -> roi_beta_mean }
      stats: dict with counts of skipped/missing items
    """
    eventid_map = load_eventid_to_trialpos(base_events_tsv)
    tc_map = load_trial_tc(block_csv)

    phase_to_series = {"Encoding": {}, "Delay": {}}

    stats = {
        "total_beta_files": 0,
        "matched_beta_pattern": 0,
        "used": 0,
        "missing_eventid": 0,
        "missing_trial_tc": 0,
        "bad_phase": 0,
        "overwritten_keys": 0,
    }

    beta_files = sorted(glob.glob(os.path.join(task_dir, "*beta*.dscalar.nii")))
    if not beta_files:
        raise FileNotFoundError(f"No beta dscalars found in: {task_dir}")

    for f in beta_files:
        stats["total_beta_files"] += 1
        bname = os.path.basename(f)
        m = BETA_RE.search(bname)
        if not m:
            continue

        stats["matched_beta_pattern"] += 1
        eid = m.group(2) or m.group(4)  # last capture is always Enc#### or Del####

        if eid not in eventid_map:
            stats["missing_eventid"] += 1
            if verbose:
                print(f"[WARN] Missing eventid mapping for {eid} ({bname})")
            continue

        phase, trial, pos = eventid_map[eid]
        if phase not in phase_to_series:
            stats["bad_phase"] += 1
            if verbose:
                print(f"[WARN] Unexpected phase='{phase}' for {eid} in {base_events_tsv}")
            continue

        if trial not in tc_map:
            stats["missing_trial_tc"] += 1
            if verbose:
                print(f"[WARN] Missing tc mapping for trial={trial} ({block_csv})")
            continue

        tc = tc_map[trial]
        key = (tc, pos)

        beta = nib.load(f).get_fdata().squeeze().astype(np.float32)
        val = float(beta[roi_mask].mean())

        if key in phase_to_series[phase]:
            stats["overwritten_keys"] += 1
            if verbose:
                print(f"[WARN] Duplicate key {key} in phase {phase}; overwriting ({bname})")

        phase_to_series[phase][key] = val
        stats["used"] += 1

    if verbose:
        print(
            f"[INFO] {os.path.basename(task_dir)}: used={stats['used']} "
            f"/ total_files={stats['total_beta_files']} "
            f"(matched_pattern={stats['matched_beta_pattern']}, "
            f"missing_eventid={stats['missing_eventid']}, "
            f"missing_trial_tc={stats['missing_trial_tc']}, "
            f"bad_phase={stats['bad_phase']}, "
            f"overwritten={stats['overwritten_keys']})"
        )

    return phase_to_series, stats


# ----------------------------
# Reliability helpers
# ----------------------------
def pearsonr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return np.nan
    return float((a * b).sum() / denom)

def split_half_4sessions(session_names):
    splits = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    return [([session_names[i] for i in a], [session_names[i] for i in b]) for a, b in splits]


def average_across_sessions(series_by_session: dict, sessions: list, verbose=True):
    """
    series_by_session: {ses -> {key -> value}}
    Returns:
      averaged_series: dict key -> mean value
      stats: dict with common keys + dropped counts
    """
    key_sets = {ses: set(series_by_session[ses].keys()) for ses in sessions}

    common_keys = set.intersection(*key_sets.values()) if key_sets else set()

    stats = {
        "n_sessions": len(sessions),
        "n_common_keys": len(common_keys),
        "keys_per_session": {ses: len(keys) for ses, keys in key_sets.items()},
        "dropped_per_session": {ses: len(key_sets[ses] - common_keys) for ses in sessions},
    }

    if verbose:
        print("[INFO] Averaging across sessions:", sessions)
        for ses in sessions:
            print(
                f"  {ses}: keys={stats['keys_per_session'][ses]} "
                f"(dropped={stats['dropped_per_session'][ses]})"
            )
        print(f"  → Common keys used: {stats['n_common_keys']}")

    averaged = {k: float(np.mean([series_by_session[ses][k] for ses in sessions])) for k in common_keys}
    return averaged, stats

def spearman_brown_correction(r):
    """SB correction for split-half reliability. Returns nan if r is nan."""
    if not np.isfinite(r):
        return np.nan
    denom = (1.0 + r)
    if denom == 0:
        return np.nan
    return float((2.0 * r) / denom)


def fisher_z(r):
    """Fisher r-to-z transform (stable clamp to avoid infs)."""
    r = np.clip(r, -0.999999, 0.999999)
    return float(np.arctanh(r))


def inv_fisher_z(z):
    """Inverse Fisher transform."""
    return float(np.tanh(z))


def mean_fisher(rs):
    """
    Average correlations in Fisher-z space, ignoring NaNs.
    Returns nan if no finite values.
    """
    good = [r for r in rs if np.isfinite(r)]
    if not good:
        return np.nan
    zbar = float(np.mean([fisher_z(r) for r in good]))
    return inv_fisher_z(zbar)


def compute_split_half_with_fisher(
    series_by_session,
    ses_list,
    apply_spearman_brown=True,
    verbose=True
):
    """
    Fisher-z version of your split-half routine:

    - Compute split r in raw space per split.
    - Apply Spearman–Brown PER split (raw -> r_sb_split).
    - Average across splits using Fisher-z transform:
        r_mean = inv_fisher(mean(fisher(r_split)))
        r_sb_mean = inv_fisher(mean(fisher(r_sb_split)))  (if SB on)
    - noise_ceiling = sqrt(max(r_sb_mean,0)) (or sqrt(max(r_mean,0)) if SB off)

    Returns dict with:
      r_splits, r_sb_splits,
      r_mean, r_sb_mean,
      noise_ceiling,
      n_events_min, split_ns, split_debug
    """
    splits = split_half_4sessions(ses_list)

    r_splits = []
    r_sb_splits = []
    split_ns = []
    split_debug = []

    for halfA, halfB in splits:
        A, A_stats = average_across_sessions(series_by_session, halfA, verbose=verbose)
        B, B_stats = average_across_sessions(series_by_session, halfB, verbose=verbose)

        common = sorted(set(A.keys()) & set(B.keys()))
        n_common = len(common)

        split_debug.append({
            "halfA": ",".join(halfA),
            "halfB": ",".join(halfB),
            "A_common": A_stats["n_common_keys"],
            "B_common": B_stats["n_common_keys"],
            "AB_common": n_common,
        })

        if n_common < 3:
            if verbose:
                print(f"[WARN] Split {halfA} vs {halfB}: only {n_common} common events — skipping")
            r_splits.append(np.nan)
            r_sb_splits.append(np.nan)
            split_ns.append(n_common)
            continue

        a_vals = [A[k] for k in common]
        b_vals = [B[k] for k in common]
        r = pearsonr(a_vals, b_vals)

        r_splits.append(r)
        split_ns.append(n_common)

        if apply_spearman_brown:
            r_sb_splits.append(spearman_brown_correction(r))
        else:
            r_sb_splits.append(np.nan)

    # Fisher-z averaging across splits
    r_mean = mean_fisher(r_splits)
    r_sb_mean = mean_fisher(r_sb_splits) if apply_spearman_brown else np.nan

    # Final noise ceiling AFTER averaging (and sqrt), clip negatives to 0
    reliability_for_ceiling = r_sb_mean if np.isfinite(r_sb_mean) else r_mean
    if np.isfinite(reliability_for_ceiling):
        reliability_for_ceiling = max(float(reliability_for_ceiling), 0.0)
        noise_ceiling = float(np.sqrt(reliability_for_ceiling))
    else:
        noise_ceiling = np.nan

    return {
        "r_splits": r_splits,
        "r_sb_splits": r_sb_splits,
        "r_mean": r_mean,
        "r_sb_mean": r_sb_mean,
        "noise_ceiling": noise_ceiling,
        "n_events_min": int(np.min(split_ns) if split_ns else 0),
        "split_ns": split_ns,
        "split_debug": split_debug,
    }
# ----------------------------
# Main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        "Split-half reliability (4 sessions) for LSS betas within a Glasser ROI, aligned by tc (w/ Fisher-z)."
    )

    ap.add_argument("--base_dir", required=True,
                    help="Neural LSS output base, e.g. .../glm_runs/lss/64kDense")
    ap.add_argument("--behav_base", required=True,
                    help="Behavior base, e.g. /mnt/.../TR/behav")
    ap.add_argument("--blockfiles_dir", required=True,
                    help="Root containing blockfiles/session01, session02, ... (design CSVs)")
    ap.add_argument("--sub", required=True, help="e.g. sub-01")
    ap.add_argument("--sessions", required=True,
                    help="Comma-separated sessions in order, e.g. ses-01,ses-02,ses-03,ses-04")
    ap.add_argument("--events_type", required=True, help="type of events")

    ap.add_argument("--dlabel", required=True,
                    help="Glasser dlabel in SAME space as dscalars (Dense64k).")
    ap.add_argument("--roi_names", required=True,
                    help='Comma-separated exact ROI label names, e.g. "L_V1_ROI,R_V1_ROI" or "L_46_ROI".')

    ap.add_argument("--out_dir", required=True,
                    help="Where to write outputs.")
    ap.add_argument("--spearman_brown", action="store_true",
                    help="Apply Spearman–Brown correction PER split, and also report mean SB across splits.")
    ap.add_argument("--only_tasks_regex", default=None,
                    help="Optional regex to filter which task dirs to include (matches task dir name).")
    ap.add_argument("--verbose", action="store_true",
                    help="Verbose logging about dropped/missing events.")

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ses_list = [s.strip() for s in args.sessions.split(",") if s.strip()]
    if len(ses_list) != 4:
        raise ValueError("This script expects exactly 4 sessions for split-half across 4 repeats.")

    roi_mask, roi_label = make_roi_mask(args.dlabel, args.roi_names.split(","))

    # Discover task/run folders from session 1
    ses1_dir = os.path.join(args.base_dir, args.sub, ses_list[0])
    if not os.path.isdir(ses1_dir):
        raise FileNotFoundError(f"Missing session dir: {ses1_dir}")

    task_dirs = sorted([d for d in os.listdir(ses1_dir) if d.startswith("task-")])
    if args.only_tasks_regex:
        pat = re.compile(args.only_tasks_regex)
        task_dirs = [d for d in task_dirs if pat.search(d)]
    if not task_dirs:
        raise ValueError("No task-* directories found (or all filtered out).")

    rows = []
    qc_rows = []

    for taskdir in task_dirs:
        parsed = parse_taskdir_name(taskdir)
        if not parsed:
            print("[WARN] Skipping unparsable task dir:", taskdir)
            continue
        task, acq, run = parsed

        series_by_session_enc = {}
        series_by_session_del = {}
        ok = True

        for ses in ses_list:
            neural_task_dir = os.path.join(args.base_dir, args.sub, ses, taskdir)
            if not os.path.isdir(neural_task_dir):
                print("[WARN] Missing neural dir:", neural_task_dir)
                ok = False
                break

            behav_events_dir = os.path.join(args.behav_base, args.sub, ses, f"events_{args.events_type}")
            base_events = os.path.join(behav_events_dir, behav_baseevents_name(task, acq, run))
            if not os.path.isfile(base_events):
                print("[WARN] Missing base-events:", base_events)
                ok = False
                break

            sess_num = int(ses.split("-")[1])
            block_ses_dir = os.path.join(args.blockfiles_dir, f"session{sess_num:02d}")
            block_csv = os.path.join(block_ses_dir, blockfile_name(task, acq, run))
            if not os.path.isfile(block_csv):
                print("[WARN] Missing block CSV:", block_csv)
                ok = False
                break

            phase_series, load_stats = collect_session_series(
                task_dir=neural_task_dir,
                roi_mask=roi_mask,
                base_events_tsv=base_events,
                block_csv=block_csv,
                verbose=args.verbose,
            )

            series_by_session_enc[ses] = phase_series["Encoding"]
            series_by_session_del[ses] = phase_series["Delay"]

            qc_rows.append({
                "taskdir": taskdir,
                "task": task,
                "acq": acq,
                "run": run,
                "roi": roi_label,
                "session": ses,
                **load_stats,
                "n_keys_encoding": len(phase_series["Encoding"]),
                "n_keys_delay": len(phase_series["Delay"]),
            })

        if not ok:
            continue

        enc_stats = compute_split_half_with_fisher(
            series_by_session_enc, ses_list,
            apply_spearman_brown=args.spearman_brown,
            verbose=args.verbose
        )
        del_stats = compute_split_half_with_fisher(
            series_by_session_del, ses_list,
            apply_spearman_brown=args.spearman_brown,
            verbose=args.verbose
        )

        rows.append({
            "taskdir": taskdir,
            "task": task,
            "acq": acq,
            "run": run,
            "roi": roi_label,
            "phase": "Encoding",

            "r_mean": enc_stats["r_mean"],
            "r_sb_mean": enc_stats["r_sb_mean"],
            "noise_ceiling": enc_stats["noise_ceiling"],  # sqrt(max(mean reliability,0))

            "n_events_min": enc_stats["n_events_min"],
            "split_r": ",".join([f"{x:.4f}" if np.isfinite(x) else "nan" for x in enc_stats["r_splits"]]),
            "split_r_sb": ",".join([f"{x:.4f}" if np.isfinite(x) else "nan" for x in enc_stats["r_sb_splits"]]),
            "split_ns": ",".join([str(n) for n in enc_stats["split_ns"]]),
        })

        rows.append({
            "taskdir": taskdir,
            "task": task,
            "acq": acq,
            "run": run,
            "roi": roi_label,
            "phase": "Delay",

            "r_mean": del_stats["r_mean"],
            "r_sb_mean": del_stats["r_sb_mean"],
            "noise_ceiling": del_stats["noise_ceiling"],

            "n_events_min": del_stats["n_events_min"],
            "split_r": ",".join([f"{x:.4f}" if np.isfinite(x) else "nan" for x in del_stats["r_splits"]]),
            "split_r_sb": ",".join([f"{x:.4f}" if np.isfinite(x) else "nan" for x in del_stats["r_sb_splits"]]),
            "split_ns": ",".join([str(n) for n in del_stats["split_ns"]]),
        })

        print(
            f"[OK] {taskdir} ROI={roi_label} "
            f"Enc r_mean={enc_stats['r_mean']:.3f} r_sb_mean={enc_stats['r_sb_mean']:.3f} ceil={enc_stats['noise_ceiling']:.3f} | "
            f"Del r_mean={del_stats['r_mean']:.3f} r_sb_mean={del_stats['r_sb_mean']:.3f} ceil={del_stats['noise_ceiling']:.3f}"
        )

    out_tsv = os.path.join(args.out_dir, "split_half_roi_reliability_fisherz.tsv")

    df_out = pd.DataFrame(rows)

    # Append if file exists; otherwise create it (with header)
    file_exists = os.path.isfile(out_tsv)
    df_out.to_csv(
        out_tsv,
        sep="\t",
        index=False,
        mode="a" if file_exists else "w",
        header=not file_exists,
    )

    print(("Appended to:" if file_exists else "Wrote:"), out_tsv)

if __name__ == "__main__":
    main()

"""
Example:
python split_half_fisherz.py \
  --base_dir /mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lss/64kDense \
  --behav_base /mnt/tempdata/lucas/fmri/recordings/TR/behav \
  --blockfiles_dir /home/lucas/projects/task_stimuli/data/multfs/trevor/blockfiles \
  --sub sub-01 \
  --sessions ses-01,ses-02,ses-03,ses-04 \
  --dlabel /home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii \
  --roi_names "L_10r_ROI, R_10r_ROI, L_10d_ROI, R_10d_ROI" \
  --out_dir /mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lss/64kDense/sub-01/_reliability_roi \
  --spearman_brown \
  --verbose
"""