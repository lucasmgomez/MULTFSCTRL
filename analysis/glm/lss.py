#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.glm.first_level import make_first_level_design_matrix, run_glm


def parse_args():
    p = argparse.ArgumentParser(
        "Run LSS GLMs on CIFTI dtseries (vertex-wise), e.g., space-Glasser64k_bold.dtseries.nii."
    )

    p.add_argument("--dtseries", required=True,
                   help="Path to *_space-Glasser64k_*_bold.dtseries.nii for ONE run/block (vertex-wise dtseries).")
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

    # Optional normalization (often helpful for vertex-wise, but leave off by default)
    p.add_argument("--zscore_time", action="store_true",
                   help="Z-score each vertex time series (across time) before fitting. Default off.")

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
    pats = [re.compile(r"EncTarget", re.I), re.compile(r"DelayTarget", re.I)]
    for pat in pats:
        hits = [c for c in design_cols if pat.search(c)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return sorted(hits, key=len)[0]
    raise RuntimeError(f"Could not find EncTarget/DelayTarget in design matrix columns:\n{design_cols}")


def save_dscalar_from_dtseries(dt_img, beta_vec, out_path, scalar_name="beta"):
    """
    Save a single beta vector (n_nodes,) as a CIFTI dscalar file.
    Preserves the brain-model axis from the input dtseries (so output stays in Glasser64k vertex space).
    """
    if beta_vec.ndim != 1:
        raise ValueError("beta_vec must be 1D (n_nodes,)")

    # dtseries axes: (time, brainmodels)
    bm_axis = dt_img.header.get_axis(1)
    scalar_axis = nib.cifti2.ScalarAxis([scalar_name])

    data2d = beta_vec[np.newaxis, :].astype(np.float32)  # (1, n_nodes)
    hdr = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
    out_img = nib.cifti2.Cifti2Image(data2d, hdr, dt_img.nifti_header)
    nib.save(out_img, out_path)


def load_dtseries_as_time_by_nodes(dt_path):
    """
    Load a CIFTI dtseries and return (dt_img, Y) where Y is (n_time, n_nodes).
    Some CIFTIs can load transposed depending on how they were written; this guards against that.
    """
    dt = nib.load(dt_path)
    Y = dt.get_fdata(dtype=np.float32)

    if Y.ndim != 2:
        raise ValueError(f"Expected 2D dtseries data. Got shape {Y.shape}")

    # In CIFTI dtseries, axis 0 is time by convention.
    # But if someone saved a transposed array, we can detect using header axis lengths.
    time_axis = dt.header.get_axis(0)
    bm_axis = dt.header.get_axis(1)

    n_time_hdr = len(time_axis)
    n_nodes_hdr = len(bm_axis)

    if Y.shape == (n_time_hdr, n_nodes_hdr):
        return dt, Y
    elif Y.shape == (n_nodes_hdr, n_time_hdr):
        # transpose to (time, nodes)
        return dt, Y.T
    else:
        raise ValueError(
            f"dtseries data shape {Y.shape} does not match header axes "
            f"(time={n_time_hdr}, nodes={n_nodes_hdr})."
        )


def zscore_timewise(Y):
    """Z-score each node across time (mean 0, std 1)."""
    mu = Y.mean(axis=0, keepdims=True)
    sd = Y.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (Y - mu) / sd


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dt, Y = load_dtseries_as_time_by_nodes(args.dtseries)
    n_scans, n_nodes = Y.shape
    print(f"Loaded dtseries: n_scans={n_scans}, n_nodes={n_nodes}")

    if args.zscore_time:
        Y = zscore_timewise(Y)
        print("Applied timewise z-scoring per vertex.")

    confounds, conf_cols = load_confounds(args.confounds_tsv, args.confounds_cols, args.add_fd)
    if len(confounds) != n_scans:
        raise ValueError(
            f"Confounds rows ({len(confounds)}) != dtseries timepoints ({n_scans}). "
            "Check you're using the matching confounds file for this run."
        )

    frame_times = np.arange(n_scans, dtype=np.float64) * args.tr

    lss_files = sorted(glob.glob(os.path.join(args.lss_events_dir, args.glob_pattern)))
    if not lss_files:
        raise FileNotFoundError(
            f"No LSS events TSVs found in {args.lss_events_dir} with pattern {args.glob_pattern}"
        )

    if args.max_models is not None:
        lss_files = lss_files[:args.max_models]

    beta_rows = []

    for i, ev_path in enumerate(lss_files, start=1):
        ev_base = os.path.basename(ev_path).replace(".tsv", "")
        out_beta = os.path.join(args.output_dir, f"{ev_base}_beta-target.dscalar.nii")

        if args.save_dscalar and os.path.exists(out_beta) and (not args.overwrite):
            print(f"[{i}/{len(lss_files)}] Skip existing: {os.path.basename(out_beta)}")
            continue

        events = pd.read_csv(ev_path, sep="\t")[["onset", "duration", "trial_type"]]

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

        labels, results = run_glm(Y, X.values, noise_model=args.noise_model)

        beta_vec = np.zeros(Y.shape[1], dtype=np.float32)
        for lab, res in results.items():
            idx = np.where(labels == lab)[0]
            beta_vec[idx] = res.theta[target_idx, :]

        # sanity check
        expected = len(dt.header.get_axis(1))
        if beta_vec.shape[0] != expected:
            raise ValueError(f"beta_vec has {beta_vec.shape[0]} nodes, expected {expected}")

        if args.save_dscalar:
            save_dscalar_from_dtseries(dt, beta_vec, out_beta, scalar_name=target_col)

        beta_rows.append({
            "lss_events_file": os.path.basename(ev_path),
            "target_regressor": target_col,
            "beta_dscalar": os.path.basename(out_beta) if args.save_dscalar else "",
        })

        print(f"[{i}/{len(lss_files)}] Done {os.path.basename(ev_path)} target={target_col}")

    manifest = pd.DataFrame(beta_rows)
    manifest_path = os.path.join(args.output_dir, "lss_manifest.tsv")
    manifest.to_csv(manifest_path, sep="\t", index=False)
    print(f"All done. Wrote {manifest_path}")


if __name__ == "__main__":
    main()

"""
python ./lss.py  \
    --dtseries /mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/64kDense/sub-01/ses-01/sub-01_ses-01_task-ctxdm_acq-col_run-01_space-Glasser64k_bold.dtseries.nii \
    --lss_events_dir /mnt/tempdata/lucas/fmri/recordings/TR/behav/sub-01/ses-1/events/task-ctxdm_col_run-01_LSS \
    --confounds_tsv /mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/confounds/sub-01/ses-01/sub-01_ses-01_task-ctxdm_acq-col_run-01_desc-confounds_timeseries.tsv \
    --tr 1.49 \
    --output_dir /mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lss/64kDense/sub-01/ses-01/task-ctxdm_acq-col_run-01 \
    --add_fd \
    --save_dscalar \
    --zscore_time \
    --overwrite
"""