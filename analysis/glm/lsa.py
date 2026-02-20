#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.glm.first_level import make_first_level_design_matrix, run_glm


def parse_args():
    p = argparse.ArgumentParser(
        "Run ONE LSA GLM on a CIFTI dtseries (vertex-wise). "
        "Events file must have one unique trial_type per event."
    )

    p.add_argument("--dtseries", required=True,
                   help="Path to *_space-Glasser64k_*_bold.dtseries.nii for ONE run/block.")
    p.add_argument("--lsa_events_tsv", required=True,
                   help="Path to the LSA events TSV for this run (task-*_run-*_LSA/task-*_run-*_lsa-events.tsv).")
    p.add_argument("--confounds_tsv", required=True,
                   help="Path to fMRIPrep *_desc-confounds_timeseries.tsv for the same run.")
    p.add_argument("--tr", type=float, default=1.49, help="TR in seconds (default 1.49).")

    p.add_argument("--output_dir", required=True, help="Where to write outputs.")
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

    # Output controls
    p.add_argument("--save_dscalar_per_event", action="store_true",
                   help="Save one beta dscalar per event regressor (can be lots of files).")
    p.add_argument("--save_multidscalar", action="store_true",
                   help="Save a single multi-scalar dscalar with one scalar per event regressor.")
    p.add_argument("--max_events", type=int, default=None,
                   help="Limit number of event regressors saved (debugging). Applies to BOTH outputs.")
    p.add_argument("--overwrite", action="store_true")

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


def load_dtseries_as_time_by_nodes(dt_path):
    dt = nib.load(dt_path)
    Y = dt.get_fdata(dtype=np.float32)
    if Y.ndim != 2:
        raise ValueError(f"Expected 2D dtseries data. Got shape {Y.shape}")

    time_axis = dt.header.get_axis(0)
    bm_axis = dt.header.get_axis(1)
    n_time_hdr = len(time_axis)
    n_nodes_hdr = len(bm_axis)

    if Y.shape == (n_time_hdr, n_nodes_hdr):
        return dt, Y
    if Y.shape == (n_nodes_hdr, n_time_hdr):
        return dt, Y.T
    raise ValueError(
        f"dtseries data shape {Y.shape} does not match header axes "
        f"(time={n_time_hdr}, nodes={n_nodes_hdr})."
    )


def zscore_timewise(Y):
    mu = Y.mean(axis=0, keepdims=True)
    sd = Y.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (Y - mu) / sd


def sanitize_name(s: str, maxlen: int = 200) -> str:
    """
    Make a filesystem-safe-ish name.
    """
    s2 = re.sub(r"[^A-Za-z0-9._+-]+", "_", s)
    s2 = re.sub(r"_+", "_", s2).strip("_")
    if len(s2) > maxlen:
        s2 = s2[:maxlen]
    return s2


def save_dscalar_from_dtseries(dt_img, beta_vec, out_path, scalar_name="beta"):
    if beta_vec.ndim != 1:
        raise ValueError("beta_vec must be 1D (n_nodes,)")

    bm_axis = dt_img.header.get_axis(1)
    scalar_axis = nib.cifti2.ScalarAxis([scalar_name])

    data2d = beta_vec[np.newaxis, :].astype(np.float32)  # (1, n_nodes)
    hdr = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
    out_img = nib.cifti2.Cifti2Image(data2d, hdr, dt_img.nifti_header)
    nib.save(out_img, out_path)


def save_multidscalar_from_dtseries(dt_img, beta_2d, scalar_names, out_path):
    """
    beta_2d: (n_scalars, n_nodes)
    scalar_names: list length n_scalars
    """
    beta_2d = np.asarray(beta_2d, dtype=np.float32)
    if beta_2d.ndim != 2:
        raise ValueError("beta_2d must be 2D (n_scalars, n_nodes)")
    if len(scalar_names) != beta_2d.shape[0]:
        raise ValueError("scalar_names length must match beta_2d first dim")

    bm_axis = dt_img.header.get_axis(1)
    scalar_axis = nib.cifti2.ScalarAxis(list(scalar_names))

    hdr = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
    out_img = nib.cifti2.Cifti2Image(beta_2d, hdr, dt_img.nifti_header)
    nib.save(out_img, out_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not (args.save_dscalar_per_event or args.save_multidscalar):
        print("[WARN] Neither --save_dscalar_per_event nor --save_multidscalar was set.")
        print("       I will still fit the GLM and write the manifest, but no beta maps will be saved.")

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

    if not os.path.isfile(args.lsa_events_tsv):
        raise FileNotFoundError(f"Missing LSA events TSV: {args.lsa_events_tsv}")

    events = pd.read_csv(args.lsa_events_tsv, sep="\t")
    needed = {"onset", "duration", "trial_type"}
    missing = needed - set(events.columns)
    if missing:
        raise ValueError(f"Events TSV missing columns {missing}: {args.lsa_events_tsv}")

    events = events[["onset", "duration", "trial_type"]].copy()
    if events.empty:
        raise ValueError(f"No rows in LSA events TSV: {args.lsa_events_tsv}")

    # Unique event regressors (this is the entire point of LSA)
    event_types = list(pd.unique(events["trial_type"].astype(str)))
    event_types = [e for e in event_types if e.strip()]

    if args.max_events is not None:
        event_types = event_types[:args.max_events]
        # also filter events so design matrix doesn't include a million we won't save
        events = events[events["trial_type"].isin(event_types)].copy()

    print(f"LSA events: {len(events)} rows, {len(event_types)} unique trial_type regressors")

    frame_times = np.arange(n_scans, dtype=np.float64) * args.tr

    X = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model=args.hrf_model,
        drift_model="cosine",
        high_pass=args.high_pass,
        add_regs=confounds.to_numpy(),
        add_reg_names=conf_cols,
    )

    design_cols = list(X.columns)
    print(f"Design matrix: shape={X.shape} (cols={len(design_cols)})")

    # Map trial_type -> column index in design matrix.
    # With make_first_level_design_matrix, each trial_type becomes a column with that exact name.
    missing_cols = [tt for tt in event_types if tt not in X.columns]
    if missing_cols:
        preview = "\n".join(design_cols[:80])
        raise RuntimeError(
            "Some trial_type values were not found as columns in the design matrix.\n"
            f"Missing ({len(missing_cols)}): {missing_cols[:20]}{'...' if len(missing_cols)>20 else ''}\n"
            "First design columns:\n" + preview
        )

    # Fit one GLM for all regressors
    print(f"Fitting GLM (noise_model={args.noise_model}) ...")
    labels, results = run_glm(Y, X.values, noise_model=args.noise_model)
    print("GLM fit complete.")

    # Helper to extract a theta row into a full n_nodes vector
    def extract_beta_for_row(row_idx: int) -> np.ndarray:
        beta_vec = np.zeros(n_nodes, dtype=np.float32)
        for lab, res in results.items():
            idx = np.where(labels == lab)[0]
            beta_vec[idx] = res.theta[row_idx, :].astype(np.float32)
        return beta_vec

    manifest_rows = []

    # Optionally assemble multi-scalar output
    multi_betas = []
    multi_names = []

    for i, tt in enumerate(event_types, start=1):
        col_idx = design_cols.index(tt)

        out_name = sanitize_name(tt)
        out_beta = os.path.join(args.output_dir, f"lsa_{out_name}_beta.dscalar.nii")

        if args.save_dscalar_per_event and os.path.exists(out_beta) and (not args.overwrite):
            print(f"[{i}/{len(event_types)}] Skip existing: {os.path.basename(out_beta)}")
            beta_vec = None
        else:
            beta_vec = extract_beta_for_row(col_idx)

            # sanity check against header
            expected = len(dt.header.get_axis(1))
            if beta_vec.shape[0] != expected:
                raise ValueError(f"beta_vec has {beta_vec.shape[0]} nodes, expected {expected}")

            if args.save_dscalar_per_event:
                save_dscalar_from_dtseries(dt, beta_vec, out_beta, scalar_name=tt)

        if args.save_multidscalar:
            if beta_vec is None:
                # if we skipped due to overwrite, we still need betas for multi output
                beta_vec = extract_beta_for_row(col_idx)
            multi_betas.append(beta_vec[np.newaxis, :])  # keep 2D
            multi_names.append(tt)

        manifest_rows.append({
            "trial_type": tt,
            "design_col": tt,
            "design_col_index": int(col_idx),
            "beta_dscalar": os.path.basename(out_beta) if args.save_dscalar_per_event else "",
        })

        if i % 25 == 0 or i == len(event_types):
            print(f"[{i}/{len(event_types)}] Processed trial_type={tt}")

    # Write multi-scalar file
    if args.save_multidscalar and multi_betas:
        beta_2d = np.vstack(multi_betas).astype(np.float32)  # (n_events, n_nodes)
        out_multi = os.path.join(args.output_dir, "lsa_betas_all_events.dscalar.nii")
        if os.path.exists(out_multi) and (not args.overwrite):
            print(f"Skip existing multi-dscalar: {out_multi}")
        else:
            save_multidscalar_from_dtseries(dt, beta_2d, multi_names, out_multi)
            print(f"Wrote multi-dscalar: {out_multi} (scalars={beta_2d.shape[0]})")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.output_dir, "lsa_manifest.tsv")
    manifest.to_csv(manifest_path, sep="\t", index=False)
    print(f"All done. Wrote {manifest_path}")


if __name__ == "__main__":
    main()

"""
Example:
python ./lsa.py \
  --dtseries /mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/64kDense/sub-01/ses-01/sub-01_ses-01_task-ctxdm_acq-col_run-01_space-Glasser64k_bold.dtseries.nii \
  --lsa_events_tsv /mnt/tempdata/lucas/fmri/recordings/TR/behav/sub-01/ses-01/events/task-ctxdm_col_run-01_LSA/task-ctxdm_col_run-01_lsa-events.tsv \
  --confounds_tsv /mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/confounds/sub-01/ses-01/sub-01_ses-01_task-ctxdm_acq-col_run-01_desc-confounds_timeseries.tsv \
  --tr 1.49 \
  --output_dir /mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lsa/64kDense/sub-01/ses-01/task-ctxdm_acq-col_run-01 \
  --add_fd \
  --noise_model ar1 \
  --zscore_time \
  --save_dscalar_per_event \
  --overwrite
"""