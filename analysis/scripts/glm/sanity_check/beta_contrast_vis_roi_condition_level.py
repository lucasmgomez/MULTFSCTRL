#!/usr/bin/env python3
"""
ROI-level contrast stats (Glasser atlas) for GLM outputs with two task regressors: 'encoding' and 'delay'.

Default contrast: encoding - delay. You can also test single regressors (encoding vs 0, delay vs 0).
Two ROI aggregation methods are supported via --roi_method:
  • stouffer  — combine voxelwise signed z-scores using Stouffer's Z (default)
  • ivw       — inverse-variance–weighted (fixed-effect) average of voxelwise contrast betas (β_c) within ROI,
                using weights w_i = 1/Var(β_c,i) where Var(β_c,i) = sigma2_i * c^T(X^TX)^{-1}c

Outputs per run under --roi_output, mirroring input tree:
  - *_roi_stats_<contrast>_<tail>.h5 with ROI-level z/p/masks and diagnostics
Optionally aggregates a per-run CSV summary via --output_csv.

Examples
--------
# Encoding - Delay, two-sided
python beta_contrast_vis_roi_condition_level.py \
  --folder /project/def-pbellec/xuan/fmri_dataset_project/data/condition_level_betas \
  --roi_output /project/def-pbellec/xuan/fmri_dataset_project/data/condition_level_betas/roi_stats \
  --atlas /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --roi_method ivw \
  --output_csv /project/def-pbellec/xuan/fmri_dataset_project/data/condition_level_betas/roi_stats/summary_enc_minus_delay.csv

# "Higher during encoding but NOT during delay" (one-sided >0): run two passes and compare
python roi_contrast_glasser.py --contrast enc --tail greater --roi_method ivw --folder ... --roi_output ... --atlas ... \
  --output_csv .../roi_stats/summary_enc_greater.csv
python roi_contrast_glasser.py --contrast delay --tail greater --roi_method ivw --folder ... --roi_output ... --atlas ... \
  --output_csv .../roi_stats/summary_delay_greater.csv

"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
from scipy.stats import norm, t as t_dist
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


def _rename_to_roi_stats_path(folder: Path, h5_path: Path, roi_output_dir: Path, contrast_key: str, tail: str) -> Path:
    rel_path = h5_path.relative_to(folder)
    name = rel_path.name
    tag = f"_roi_stats_{contrast_key}_{tail}.h5"
    if name.endswith("_betas.h5"):
        out_name = name[:-10] + tag  # strip "_betas.h5"
    else:
        out_name = rel_path.stem + tag
    return roi_output_dir / rel_path.parent / out_name


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


def _voxelwise_contrast(h5_path: Path, contrast_key: str):
    """Return per-voxel beta_c, t, p(two-sided), z(two-sided signed), var_c, enc_idx, dly_idx, dof, cXtXc."""
    with h5py.File(h5_path, "r") as f:
        betas = f["betas"][()]                   # (P x 2) expected: [encoding, delay]
        XtX_inv = f["XtX_inv"][()]               # (K x K)
        sigma2 = f["sigma2"][()]                 # (P,)
        dof = int(f.attrs["dof"])               # scalar
        task_col_start = int(f.attrs["task_col_start"])  # index in X
        task_col_end = int(f.attrs["task_col_end"])      # exclusive end
        col_names = None
        if "design_col_names" in f:
            col_names = [ _safe_decode(x) for x in f["design_col_names"][()] ]
        # task names if present
        task_names = None
        if "task_regressor_names" in f.attrs:
            tn = f.attrs["task_regressor_names"]
            task_names = [ _safe_decode(x) for x in (tn if isinstance(tn, np.ndarray) else [tn]) ]

    # determine enc/delay indices in full design
    if col_names is not None and task_names is not None and len(task_names) >= 2:
        enc_idx = None
        dly_idx = None
        for j in range(task_col_start, task_col_end):
            nm = col_names[j].lower()
            if nm == "encoding":
                enc_idx = j
            elif nm == "delay":
                dly_idx = j
        if enc_idx is None:
            enc_idx = task_col_start
        if dly_idx is None:
            dly_idx = task_col_start + 1
    else:
        enc_idx = task_col_start
        dly_idx = task_col_start + 1

    K = XtX_inv.shape[0]
    c = _build_contrast_vector(K, enc_idx, dly_idx, contrast_key)
    cXtXc = float(c @ (XtX_inv @ c))

    # map to task betas positions
    enc_pos = enc_idx - task_col_start
    dly_pos = dly_idx - task_col_start
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

    # voxelwise t and p (two-sided)
    var_c = sigma2 * cXtXc
    var_c = np.where(var_c > 0, var_c, np.nan)
    se_c = np.sqrt(var_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = beta_c / se_c
    p_two = np.full_like(t_vals, np.nan, dtype=np.float64)
    valid_t = np.isfinite(t_vals)
    if dof > 0 and np.any(valid_t):
        p_two[valid_t] = 2.0 * t_dist.sf(np.abs(t_vals[valid_t]), df=dof)

    # two-sided signed z: sign(beta) * Phi^{-1}(1 - p/2)
    p_safe = np.clip(p_two, 1e-300, 1.0)
    z_abs = norm.isf(p_safe / 2.0)
    signs = np.sign(beta_c)
    z_signed = signs * z_abs

    return beta_c.astype(np.float32), t_vals.astype(np.float32), p_two.astype(np.float64), z_signed.astype(np.float64), var_c.astype(np.float64), enc_idx, dly_idx, dof, cXtXc


# -----------------------
# Main processing
# -----------------------

def process_folder(folder, roi_output_dir, atlas_path, output_csv=None, p_thresh=0.05, contrast_key="enc-minus-delay", tail="two-sided", roi_method="stouffer", num_regions=360, verbose=False):
    folder = Path(folder)
    roi_output_dir = Path(roi_output_dir)
    roi_output_dir.mkdir(parents=True, exist_ok=True)

    # Load Glasser atlas labels vector (P,)
    atlas = nib.load(str(atlas_path)).get_fdata()
    if atlas.ndim == 2:
        atlas = atlas[0]
    atlas = atlas.astype(int)
    P_atlas = atlas.shape[0]

    rows = []

    for h5_path in folder.rglob("*_betas.h5"):
        beta_c, t_vox, p_two_vox, z_signed_vox, var_c_vox, enc_idx, dly_idx, dof, cXtXc = _voxelwise_contrast(h5_path, contrast_key)

        # sanity check atlas length vs betas length
        if beta_c.shape[0] != P_atlas:
            raise RuntimeError(f"Atlas length ({P_atlas}) != betas length ({beta_c.shape[0]}) for {h5_path}")

        # ROI aggregation
        roi_mean_beta   = np.full((num_regions,), np.nan, dtype=np.float64)
        roi_median_beta = np.full((num_regions,), np.nan, dtype=np.float64)
        roi_beta_ivw    = np.full((num_regions,), np.nan, dtype=np.float64)
        roi_se_ivw      = np.full((num_regions,), np.nan, dtype=np.float64)
        roi_n_vox       = np.zeros((num_regions,), dtype=int)
        roi_z           = np.full((num_regions,), np.nan, dtype=np.float64)
        roi_p_two       = np.full((num_regions,), np.nan, dtype=np.float64)
        roi_p_greater   = np.full((num_regions,), np.nan, dtype=np.float64)
        roi_p_less      = np.full((num_regions,), np.nan, dtype=np.float64)

        for roi in range(num_regions):
            idx = np.where(atlas == (roi + 1))[0]
            if idx.size == 0:
                continue
            roi_n_vox[roi] = int(idx.size)

            # Descriptive means
            vals_beta_all = beta_c[idx]
            vb_ok = np.isfinite(vals_beta_all)
            if np.any(vb_ok):
                roi_mean_beta[roi]   = float(np.nanmean(vals_beta_all[vb_ok]))
                roi_median_beta[roi] = float(np.nanmedian(vals_beta_all[vb_ok]))

            if roi_method == "stouffer":
                vals_z = z_signed_vox[idx]
                valid = np.isfinite(vals_z)
                if not np.any(valid):
                    continue
                zc = float(np.nansum(vals_z[valid]) / np.sqrt(valid.sum()))
                p2 = 2.0 * (1.0 - norm.cdf(abs(zc)))
                pg = 1.0 - norm.cdf(zc)
                roi_z[roi] = zc
                roi_p_two[roi] = p2
                roi_p_greater[roi] = pg
                roi_p_less[roi] = 1.0 - pg

            elif roi_method == "ivw":
                vals_beta = beta_c[idx]
                vals_var  = var_c_vox[idx]
                valid = np.isfinite(vals_beta) & np.isfinite(vals_var) & (vals_var > 0)
                if not np.any(valid):
                    continue
                w = 1.0 / vals_var[valid]
                sw = float(np.sum(w))
                if sw <= 0:
                    continue
                beta_ivw = float(np.sum(w * vals_beta[valid]) / sw)
                se_ivw   = float(np.sqrt(1.0 / sw))
                zc = beta_ivw / se_ivw if se_ivw > 0 else np.nan
                p2 = 2.0 * (1.0 - norm.cdf(abs(zc))) if np.isfinite(zc) else np.nan
                pg = 1.0 - norm.cdf(zc) if np.isfinite(zc) else np.nan
                roi_beta_ivw[roi] = beta_ivw
                roi_se_ivw[roi]   = se_ivw
                roi_z[roi] = zc
                roi_p_two[roi] = p2
                roi_p_greater[roi] = pg
                roi_p_less[roi] = 1.0 - pg if np.isfinite(pg) else np.nan

        # choose which p to threshold
        if tail == "two-sided":
            p_use = roi_p_two
        elif tail == "greater":
            p_use = roi_p_greater
        elif tail == "less":
            p_use = roi_p_less
        else:
            raise ValueError("tail must be one of: two-sided, greater, less")

        finite = np.isfinite(p_use)
        sig_mask = np.zeros_like(p_use, dtype=bool)
        sig_mask[finite] = p_use[finite] < p_thresh

        fdr_mask = np.zeros_like(sig_mask, dtype=bool)
        fdr_corrected_p = np.full_like(p_use, np.nan, dtype=np.float64)
        if np.any(finite):
            rej, p_fdr = fdrcorrection(p_use[finite], alpha=p_thresh)
            fdr_mask[finite] = rej
            fdr_corrected_p[finite] = p_fdr

        # write ROI stats HDF5
        out_path = _rename_to_roi_stats_path(folder, h5_path, roi_output_dir, contrast_key, tail)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("roi_mean_beta", data=roi_mean_beta.astype(np.float32))
            f.create_dataset("roi_median_beta", data=roi_median_beta.astype(np.float32))
            f.create_dataset("roi_beta_ivw", data=roi_beta_ivw.astype(np.float32))
            f.create_dataset("roi_se_ivw",   data=roi_se_ivw.astype(np.float32))
            f.create_dataset("roi_n_vox", data=roi_n_vox.astype(np.int32))
            f.create_dataset("roi_z", data=roi_z.astype(np.float64))
            f.create_dataset("roi_p_two", data=roi_p_two.astype(np.float64))
            f.create_dataset("roi_p_greater", data=roi_p_greater.astype(np.float64))
            f.create_dataset("roi_p_less", data=roi_p_less.astype(np.float64))
            f.create_dataset("sig_mask", data=sig_mask.astype(np.uint8))
            f.create_dataset("fdr_mask", data=fdr_mask.astype(np.uint8))
            f.create_dataset("fdr_corrected_p", data=fdr_corrected_p.astype(np.float64))
            f.attrs["contrast"] = contrast_key
            f.attrs["tail"] = tail
            f.attrs["roi_method"] = roi_method
            f.attrs["p_thresh"] = float(p_thresh)
            f.attrs["dof"] = int(dof)
            f.attrs["cXtXc"] = float(cXtXc)
            f.attrs["num_regions"] = int(num_regions)

        # summary row per run
        rows.append({
            "file": str(out_path),
            "contrast": contrast_key,
            "tail": tail,
            "roi_method": roi_method,
            "n_sig_rois": int(np.sum(sig_mask)),
            "n_fdr_sig_rois": int(np.sum(fdr_mask)),
            "min_p": float(np.nanmin(p_use)) if np.any(finite) else np.nan,
            "max_|z|": float(np.nanmax(np.abs(roi_z))) if np.any(np.isfinite(roi_z)) else np.nan,
            "mean_beta_sign": float(np.nanmean(np.sign(roi_mean_beta))) if np.any(np.isfinite(roi_mean_beta)) else np.nan,
        })

        if verbose:
            print(f"[✓] Saved ROI stats -> {out_path}")

    # write summary CSV
    df = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved ROI summary CSV: {output_csv}")
    return df


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="ROI-level contrast stats (Glasser atlas)")
    ap.add_argument("--folder", required=True, help="Folder containing *_betas.h5")
    ap.add_argument("--roi_output", required=True, help="Output directory for ROI HDF5 stats")
    ap.add_argument("--atlas", required=True, help="Path to Glasser dlabel.nii (Dense64k)")
    ap.add_argument("--output_csv", default=None, help="Path to write per-run ROI summary CSV")
    ap.add_argument("--p_thresh", type=float, default=0.05, help="Alpha for significance/FDR")
    ap.add_argument("--contrast", default="enc-minus-delay", choices=["enc","delay","enc-minus-delay","delay-minus-enc"], help="Contrast to compute")
    ap.add_argument("--tail", default="two-sided", choices=["two-sided","greater","less"], help="Tail for ROI significance thresholding")
    ap.add_argument("--roi_method", default="stouffer", choices=["stouffer","ivw"], help="ROI aggregation method")
    ap.add_argument("--num_regions", type=int, default=360, help="Number of Glasser ROIs (default 360)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    process_folder(
        folder=args.folder,
        roi_output_dir=args.roi_output,
        atlas_path=Path(args.atlas),
        output_csv=args.output_csv,
        p_thresh=args.p_thresh,
        contrast_key=args.contrast,
        tail=args.tail,
        roi_method=args.roi_method,
        num_regions=args.num_regions,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
