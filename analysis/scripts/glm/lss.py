#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import run_glm


def parse_args():
    p = argparse.ArgumentParser("Run LSS GLMs on fMRIPrep CIFTI dtseries (fsLR 91k).")

    p.add_argument("--dtseries", required=True,
                   help="Path to *_bold.dtseries.nii for ONE run/block.")
    p.add_argument("--lss_events_dir", required=True,
                   help="Directory with LSS events TSVs for this run (task-*_run-*_LSS/).")
    p.add_argument("--confounds_tsv", required=True,
                   help="Path to fMRIPrep *_desc-confounds_timeseries.tsv for the same run.")
    p.add_argument("--tr", type=float, default=1.49, help="TR in seconds (default 1.49).")

    p.add_argument("--output_dir", required=True, help="Where to write outputs.")
    p.add_argument("--glob_pattern", default="*events.tsv", help="Pattern for LSS TSVs (default *events.tsv).")

    p.add_argument("--confounds_cols",
                   default="trans_x,trans_y,trans_z,rot_x,rot_y,rot_z",
                   help="Comma-separated confound columns to include.")
    p.add_argument("--add_fd", action="store_true",
                   help="Also include framewise_displacement if present.")

    p.add_argument("--hrf_model", default="glover", choices=["glover", "spm"],
                   help="HRF model for design matrix.")
    p.add_argument("--high_pass", type=float, default=0.01,
                   help="High-pass cutoff in Hz (cosine drift). Default 0.01 (~100s).")

    p.add_argument("--noise_model", default="ar1", choices=["ar1", "ols"],
                   help="Noise model for run_glm (default ar1).")

    p.add_argument("--save_dscalar", action="store_true",
                   help="Save a CIFTI dscalar beta map for each target event.")
    p.add_argument("--max_models", type=int, default=None,
                   help="Limit number of LSS models for debugging.")
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def load_confounds(confounds_tsv, confounds_cols, add_fd):
    conf = pd.read_csv(confounds_tsv, sep="\t")
    cols = [c.strip() for c in confounds_cols.split(",") if c.strip()]
    if add_fd and "framewise_displacement" in conf.columns:
        cols.append("framewise_displacement")

    missing = [c for c in cols if c not in conf.columns]
    if missing:
        raise ValueError(f"Missing confound columns in {confounds_tsv}: {missing}")

    Xc = conf[cols].copy().fillna(0.0)
    return Xc, cols


def find_target_regressor(design_cols):
    # Your LSS TSVs use trial_type like "{task}_EncTarget" / "{task}_DelayTarget"
    # nilearn design columns will include those names.
    pats = [re.compile(r"EncTarget", re.I), re.compile(r"DelayTarget", re.I)]
    for pat in pats:
        hits = [c for c in design_cols if pat.search(c)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # pick shortest match if multiple
            return sorted(hits, key=len)[0]
    raise RuntimeError(f"Could not find EncTarget/DelayTarget in design matrix columns:\n{design_cols}")


def save_dscalar_from_dtseries(dt_img, beta_vec, out_path, scalar_name="beta"):
    """
    Save a single beta vector (n_grayordinates,) as a CIFTI dscalar file.
    Uses the brain-model axis from the dtseries header so it stays in fsLR 91k space.
    """
    if beta_vec.ndim != 1:
        raise ValueError("beta_vec must be 1D (n_grayordinates,)")

    # dtseries axes: (time, brainmodels)
    time_axis, bm_axis = dt_img.header.get_axis(0), dt_img.header.get_axis(1)
    scalar_axis = nib.cifti2.ScalarAxis([scalar_name])

    # data for dscalar: (n_scalars, n_grayordinates)
    data2d = beta_vec[np.newaxis, :].astype(np.float32)

    hdr = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
    out_img = nib.cifti2.Cifti2Image(data2d, hdr, dt_img.nifti_header)
    nib.save(out_img, out_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dt = nib.load(args.dtseries)
    Y = dt.get_fdata(dtype=np.float32)  # shape: (n_time, n_grayordinates)
    n_scans = Y.shape[0]

    # Confounds
    confounds, conf_cols = load_confounds(args.confounds_tsv, args.confounds_cols, args.add_fd)
    if len(confounds) != n_scans:
        raise ValueError(f"Confounds rows ({len(confounds)}) != dtseries timepoints ({n_scans})")

    frame_times = np.arange(n_scans, dtype=np.float64) * args.tr

    lss_files = sorted(glob.glob(os.path.join(args.lss_events_dir, args.glob_pattern)))
    if not lss_files:
        raise FileNotFoundError(f"No LSS events TSVs found in {args.lss_events_dir} with pattern {args.glob_pattern}")
    if args.max_models is not None:
        lss_files = lss_files[:args.max_models]

    beta_rows = []  # optional: store metadata + (optional) write a table later

    for i, ev_path in enumerate(lss_files, start=1):
        ev_base = os.path.basename(ev_path).replace(".tsv", "")
        out_beta = os.path.join(args.output_dir, f"{ev_base}_beta-target.dscalar.nii")

        if args.save_dscalar and os.path.exists(out_beta) and (not args.overwrite):
            print(f"[{i}/{len(lss_files)}] Skip existing: {os.path.basename(out_beta)}")
            continue

        events = pd.read_csv(ev_path, sep="\t")[["onset", "duration", "trial_type"]]

        # Build design matrix (HRF + cosine drifts + confounds)
        X = make_first_level_design_matrix(
            frame_times=frame_times,
            events=events,
            hrf_model=args.hrf_model,
            drift_model="cosine",
            high_pass=args.high_pass,
            add_regs=confounds.to_numpy(),
            add_reg_names=conf_cols,
        )

        target_col = find_target_regressor(list(X.columns))
        target_idx = list(X.columns).index(target_col)

        # Fit GLM for all grayordinates
        labels, results = run_glm(Y, X.values, noise_model=args.noise_model)

        # For run_glm, results is a dict keyed by label (e.g., 0) -> RegressionResults
        # We'll just take the first label.
        first_label = list(results.keys())[0]
        res = results[first_label]

        # theta shape: (n_regressors, n_grayordinates)
        beta_vec = res.theta[target_idx, :]

        if args.save_dscalar:
            save_dscalar_from_dtseries(dt, beta_vec, out_beta, scalar_name=target_col)

        beta_rows.append({
            "lss_events_file": os.path.basename(ev_path),
            "target_regressor": target_col,
            "saved_beta_map": os.path.basename(out_beta) if args.save_dscalar else "",
        })

        print(f"[{i}/{len(lss_files)}] Done {os.path.basename(ev_path)} target={target_col}")

    # Save a small manifest
    manifest = pd.DataFrame(beta_rows)
    manifest.to_csv(os.path.join(args.output_dir, "lss_manifest.tsv"), sep="\t", index=False)
    print("All done. Wrote lss_manifest.tsv")


if __name__ == "__main__":
    main()