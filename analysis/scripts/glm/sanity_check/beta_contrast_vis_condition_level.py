# python beta_contrast_vis_condition_level.py \
#     --folder /project/def-pbellec/xuan/fmri_dataset_project/data/condition_level_betas \
#     --stats_output /project/def-pbellec/xuan/fmri_dataset_project/data/condition_level_betas/stats \
#     --p_thresh 0.05 \
#     --output_csv /project/def-pbellec/xuan/fmri_dataset_project/data/condition_level_betas/stats/summary.csv

# Description: Batch process GLM HDF5 output files to compute t, p, z values and optionally filter significant parcels with FDR correction

#!/usr/bin/env python3
# Batch stats with FDR + diagnostics (enc regressor nonzeros, cXtXc, dof)

#!/usr/bin/env python3
"""
Batch stats with FDR + diagnostics for GLM outputs that contain TWO task regressors:
'encoding' and 'delay'.

Default contrast: encoding - delay.
Other options via --contrast:
  - enc                (tests encoding vs 0)
  - delay              (tests delay vs 0)
  - enc-minus-delay    (encoding - delay)
  - delay-minus-enc    (delay - encoding)

It reads *_betas.h5 files produced by the per-run GLM script which stores:
  - betas              (P x 2) in order [encoding, delay]
  - XtX_inv            (K x K) for the full design matrix order
  - sigma2             (P,)
  - attrs task_col_start, task_col_end, task_regressor_names, dof
  - design_matrix + design_col_names (optional but used for diagnostics)

Outputs per run under --stats_output mirroring input tree:
  - *_stats_<contrast>.h5 with t, p, z, masks, and diagnostics
And optionally aggregates a CSV summary via --output_csv.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection


# -----------------------
# Helpers
# -----------------------

def _safe_decode(x):
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    return x


def _rename_to_stats_path(folder: Path, h5_path: Path, stats_output_dir: Path, contrast_key: str) -> Path:
    rel_path = h5_path.relative_to(folder)
    name = rel_path.name
    if name.endswith("_betas.h5"):
        out_name = name[:-10] + f"_stats_{contrast_key}.h5"   # strip "_betas.h5"
    else:
        out_name = rel_path.stem + f"_stats_{contrast_key}.h5"
    return stats_output_dir / rel_path.parent / out_name


def _build_contrast_vector(K: int, enc_idx: int, dly_idx: int, contrast_key: str) -> np.ndarray:
    c = np.zeros((K,), dtype=np.float32)
    if contrast_key == "enc":
        c[enc_idx] = 1.0
    elif contrast_key == "delay":
        c[dly_idx] = 1.0
    elif contrast_key == "enc-minus-delay":
        c[enc_idx] = 1.0
        c[dly_idx] = -1.0
    elif contrast_key == "delay-minus-enc":
        c[enc_idx] = -1.0
        c[dly_idx] = 1.0
    else:
        raise ValueError(f"Unknown contrast_key: {contrast_key}")
    return c


# -----------------------
# Core compute
# -----------------------

def compute_t_p_z(h5_path: Path, p_thresh=0.05, contrast_key="enc-minus-delay", verbose=False):
    with h5py.File(h5_path, "r") as f:
        betas = f["betas"][()]                   # (P x 2) expected: [encoding, delay]
        XtX_inv = f["XtX_inv"][()]               # (K x K)
        sigma2 = f["sigma2"][()]                 # (P,)
        dof = int(f.attrs["dof"])               # scalar
        task_col_start = int(f.attrs["task_col_start"])  # index in X
        task_col_end = int(f.attrs["task_col_end"])      # exclusive end
        # Names (optional but helpful)
        task_names = None
        if "task_regressor_names" in f.attrs:
            tn = f.attrs["task_regressor_names"]
            if isinstance(tn, np.ndarray):
                task_names = [_safe_decode(x) for x in tn]
            else:
                task_names = [_safe_decode(tn)]
        X = f["design_matrix"][()] if "design_matrix" in f else None
        col_names = None
        if "design_col_names" in f:
            col_names = [ _safe_decode(x) for x in f["design_col_names"][()] ]

    # Determine indices of encoding and delay within the FULL design
    # Preferred: use design_col_names to locate by name; fallback: assume order [encoding, delay] at task slice
    if col_names is not None and task_names is not None and len(task_names) >= 2:
        enc_idx = None
        dly_idx = None
        for j in range(task_col_start, task_col_end):
            name = col_names[j].lower()
            if name == "encoding":
                enc_idx = j
            elif name == "delay":
                dly_idx = j
        # fallback if names not found exactly
        if enc_idx is None:
            enc_idx = task_col_start
        if dly_idx is None:
            dly_idx = task_col_start + 1
    else:
        enc_idx = task_col_start
        dly_idx = task_col_start + 1

    K = XtX_inv.shape[0]
    c = _build_contrast_vector(K, enc_idx, dly_idx, contrast_key)

    # Diagnostics: nonzero counts for task columns
    enc_nonzero = delay_nonzero = None
    if X is not None:
        try:
            enc_nonzero = int(np.count_nonzero(np.abs(X[:, enc_idx]) > 1e-8))
            delay_nonzero = int(np.count_nonzero(np.abs(X[:, dly_idx]) > 1e-8))
        except Exception:
            pass

    # c^T (X^T X)^{-1} c — a scalar; if ~0, contrast is not estimable
    cXtXc = float(c @ (XtX_inv @ c))
    if verbose:
        print(f"[diag] file={h5_path}")
        print(f"       dof={dof}, cXtXc={cXtXc:.3e}, enc_nonzero={enc_nonzero}, delay_nonzero={delay_nonzero}")

    P = betas.shape[0]
    if not np.isfinite(cXtXc) or cXtXc <= 0:
        nan = np.full((P,), np.nan, dtype=np.float64)
        return {
            "beta_c": nan.astype(np.float32),
            "t_vals": nan.astype(np.float32),
            "p_vals": nan,
            "z_vals": nan,
            "sig_mask": np.zeros((P,), dtype=bool),
            "fdr_mask": np.zeros((P,), dtype=bool),
            "fdr_corrected_p": nan,
            "diag": {"dof": dof, "cXtXc": cXtXc, "enc_nonzero": enc_nonzero, "delay_nonzero": delay_nonzero},
        }

    # beta contrast value per parcel: c^T beta_full, but c is nonzero only on task cols
    # and betas contains only [encoding, delay] in that order.
    # Map to the 2-vector: [b_enc, b_delay]
    # Figure positions of task columns within the task slice
    enc_pos = enc_idx - task_col_start
    dly_pos = dly_idx - task_col_start
    # Build reduced contrast for the task betas
    if contrast_key == "enc":
        beta_c = betas[:, enc_pos]
    elif contrast_key == "delay":
        beta_c = betas[:, dly_pos]
    elif contrast_key == "enc-minus-delay":
        beta_c = betas[:, enc_pos] - betas[:, dly_pos]
    elif contrast_key == "delay-minus-enc":
        beta_c = betas[:, dly_pos] - betas[:, enc_pos]
    else:
        raise ValueError("Unexpected contrast key")

    var_c = sigma2 * cXtXc                         # (P,)
    var_c = np.where(var_c > 0, var_c, np.nan)
    se_c = np.sqrt(var_c)                          # (P,)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = beta_c / se_c

    p_vals = np.full_like(t_vals, np.nan, dtype=np.float64)
    valid_t = np.isfinite(t_vals)
    if dof > 0 and np.any(valid_t):
        p_vals[valid_t] = 2.0 * t_dist.sf(np.abs(t_vals[valid_t]), df=dof)

    # Convert to z (two-tailed)
    p_safe = np.clip(p_vals, 1e-300, 1.0)
    z_vals = norm.isf(p_safe / 2.0)
    z_vals[~np.isfinite(p_vals)] = np.nan

    # Uncorrected threshold
    sig_mask = np.zeros_like(valid_t, dtype=bool)
    sig_mask[valid_t] = p_vals[valid_t] < p_thresh

    # FDR on finite p only
    fdr_mask = np.zeros_like(sig_mask, dtype=bool)
    fdr_corrected_p = np.full_like(p_vals, np.nan, dtype=np.float64)
    finite_p = np.isfinite(p_vals)
    if np.any(finite_p):
        rej, p_fdr = fdrcorrection(p_vals[finite_p], alpha=p_thresh)
        fdr_mask[finite_p] = rej
        fdr_corrected_p[finite_p] = p_fdr

    return {
        "beta_c": beta_c.astype(np.float32),
        "t_vals": t_vals.astype(np.float32),
        "p_vals": p_vals.astype(np.float64),
        "z_vals": z_vals.astype(np.float64),
        "sig_mask": sig_mask,
        "fdr_mask": fdr_mask,
        "fdr_corrected_p": fdr_corrected_p.astype(np.float64),
        "diag": {"dof": dof, "cXtXc": cXtXc, "enc_nonzero": enc_nonzero, "delay_nonzero": delay_nonzero,
                  "enc_idx": int(enc_idx), "delay_idx": int(dly_idx)},
    }


# -----------------------
# Batch processing
# -----------------------

def process_folder(folder, stats_output_dir, output_csv=None, p_thresh=0.05, contrast_key="enc-minus-delay", verbose=False):
    folder = Path(folder)
    stats_output_dir = Path(stats_output_dir)
    stats_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for h5_path in folder.rglob("*_betas.h5"):
        res = compute_t_p_z(h5_path, p_thresh=p_thresh, contrast_key=contrast_key, verbose=verbose)

        stats_file = _rename_to_stats_path(folder, h5_path, stats_output_dir, contrast_key)
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        beta_c = res["beta_c"]
        t_vals = res["t_vals"]
        p_vals = res["p_vals"]
        z_vals = res["z_vals"]
        sig_mask = res["sig_mask"]
        fdr_mask = res["fdr_mask"]
        fdr_corrected_p = res["fdr_corrected_p"]
        diag = res["diag"]

        # summary row
        finite_t = t_vals[np.isfinite(t_vals)]
        finite_p = p_vals[np.isfinite(p_vals)]
        finite_z = z_vals[np.isfinite(z_vals)]
        rows.append({
            "file": str(stats_file),
            "contrast": contrast_key,
            "n_sig_voxels": int(np.sum(sig_mask)),
            "n_fdr_sig_voxels": int(np.sum(fdr_mask)),
            "max_t": float(np.max(finite_t)) if finite_t.size else np.nan,
            "min_p": float(np.min(finite_p)) if finite_p.size else np.nan,
            "max_z": float(np.max(finite_z)) if finite_z.size else np.nan,
            "dof": int(diag["dof"]),
            "cXtXc": float(diag["cXtXc"]),
            "enc_nonzero": (int(diag["enc_nonzero"]) if diag["enc_nonzero"] is not None else np.nan),
            "delay_nonzero": (int(diag["delay_nonzero"]) if diag["delay_nonzero"] is not None else np.nan),
        })

        # write stats h5
        with h5py.File(stats_file, "w") as f:
            f.create_dataset("beta_c", data=beta_c.astype(np.float32))
            f.create_dataset("t_vals", data=t_vals.astype(np.float32))
            f.create_dataset("p_vals", data=p_vals.astype(np.float64))
            f.create_dataset("z_vals", data=z_vals.astype(np.float64))
            f.create_dataset("sig_mask", data=sig_mask.astype(np.uint8))
            f.create_dataset("fdr_mask", data=fdr_mask.astype(np.uint8))
            f.create_dataset("fdr_corrected_p", data=fdr_corrected_p.astype(np.float64))
            # pass through diag for later QA
            f.attrs["dof"] = int(diag["dof"])
            f.attrs["cXtXc"] = float(diag["cXtXc"])
            if diag["enc_nonzero"] is not None:
                f.attrs["enc_nonzero"] = int(diag["enc_nonzero"])
            if diag["delay_nonzero"] is not None:
                f.attrs["delay_nonzero"] = int(diag["delay_nonzero"])
            f.attrs["contrast"] = contrast_key

    df = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved summary CSV: {output_csv}")
    return df


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Path to folder containing *_betas.h5 files")
    ap.add_argument("--stats_output", required=True, help="Path to save stats-only HDF5 outputs")
    ap.add_argument("--p_thresh", type=float, default=0.05, help="Significance/FDR alpha")
    ap.add_argument("--output_csv", default=None, help="Path to save summary CSV")
    ap.add_argument("--contrast", default="enc-minus-delay", choices=["enc","delay","enc-minus-delay","delay-minus-enc"],
                    help="Which contrast to compute")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    df = process_folder(args.folder, args.stats_output, args.output_csv, args.p_thresh,
                        contrast_key=args.contrast, verbose=args.verbose)
    print(df)


if __name__ == "__main__":
    main()
