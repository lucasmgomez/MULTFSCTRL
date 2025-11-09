#!/usr/bin/env python3
"""
make_contrast_map.py

Compute per-run and fixed-effects combined contrast maps (encoding - delay)
from condition-level GLM outputs saved per run.

Assumes each per-run HDF5 file contains:
- betas: (P, K_task) with task columns in the order given by either:
    * HDF5 attr "task_regressor_names" (preferred), OR
    * a sibling design CSV listing "condition" in the same order.
- XtX_inv: (K, K) for the FULL design X (confounds + drifts + task)
- sigma2: (P,) residual variance per parcel
- attrs: task_col_start, task_col_end (slice of task block in X), dof

Outputs per run:
  <results_root>/<subj>/condition_level_contrasts/task-<task>/
      ses-XXX_run-YY_effect.npy
      ses-XXX_run-YY_t.npy
      ses-XXX_run-YY_z.npy
      ses-XXX_run-YY_var.npy
      ses-XXX_run-YY_surface_effect.npy
      ses-XXX_run-YY_surface_t.npy
      ses-XXX_run-YY_surface_z.npy

Outputs fixed effects (across runs):
  .../task-<task>/fixed_effects_effect.npy
  .../task-<task>/fixed_effects_t.npy
  .../task-<task>/fixed_effects_z.npy
  .../task-<task>/fixed_effects_var.npy
  .../task-<task>/fixed_effects_surface_*.npy
"""

from pathlib import Path
import argparse
import re
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats


# -----------------------
# CLI
# -----------------------
def get_args():
    p = argparse.ArgumentParser(description="Compute contrast (encoding - delay) maps from per-run condition GLMs.")
    p.add_argument("--subj", required=True, help="e.g., sub-01")
    p.add_argument("--task", required=True, help="base task name used in filenames, e.g., interdms, ctxdm, 1back")
    p.add_argument("--out_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data",
                   help="Base data root containing condition_level_betas/")
    p.add_argument("--results_root", default="/project/def-pbellec/xuan/fmri_dataset_project/results",
                   help="Where to save contrast outputs")
    p.add_argument("--atlas_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii")
    p.add_argument("--n_parcels", type=int, default=360)
    p.add_argument("--correct_only", action="store_true", help="Read from correct_condition_level_betas/")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# -----------------------
# IO helpers
# -----------------------
def betas_root(base: Path, subj: str, correct_only: bool) -> Path:
    return base / ("correct_condition_level_betas" if correct_only else "condition_level_betas") / subj

RUN_RE = re.compile(r"run-(\d+)")

def find_run_files(root: Path, subj: str, task: str):
    """Find all per-run HDF5 files for this task."""
    # Pattern: **/sub-01_ses-XXX_task-<task>_run-*_betas.h5
    patt = f"**/{subj}_ses-*_task-{task}_run-*_betas.h5"
    return sorted(root.glob(patt))

def parse_ses_run(name: str):
    """
    Extract ses-XXX and run-YY from a filename like sub-01_ses-003_task-ctxdm_run-1_betas.h5
    """
    m = re.search(r"(ses-\d+).*?(run-\d+)", name)
    if not m:
        return "ses-UNK", "run-UNK"
    return m.group(1), m.group(2)

def read_design_order_from_csv(h5_path: Path):
    """Fallback if HDF5 lacks task_regressor_names attr."""
    csv = h5_path.with_name(h5_path.name.replace("_betas.h5", "_design.csv"))
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
        if "condition" in df.columns:
            return [str(x).lower() for x in df["condition"].tolist()]
    except Exception:
        pass
    return None

def project_to_surface(region_vals: np.ndarray, atlas_img, n_parcels: int) -> np.ndarray:
    atlas_data = atlas_img.get_fdata().squeeze().astype(int)
    surf = np.zeros_like(atlas_data, dtype=float)
    for region_idx in range(1, n_parcels + 1):
        surf[atlas_data == region_idx] = region_vals[region_idx - 1]
    return surf

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------
# Core contrast per run
# -----------------------
def compute_contrast_per_run(h5_path: Path, atlas_img, n_parcels: int, verbose=False):
    """Return dict with effect, var, t, z, and some metadata for one run."""
    with h5py.File(h5_path, "r") as f:
        betas_task = f["betas"][:]            # (P, K_task)
        XtX_inv    = f["XtX_inv"][:]          # (K_full, K_full)
        sigma2     = f["sigma2"][:]           # (P,)
        dof        = int(f.attrs["dof"])
        tstart     = int(f.attrs["task_col_start"])
        tend       = int(f.attrs["task_col_end"])   # exclusive
        names_attr = f.attrs.get("task_regressor_names", [])
        if len(names_attr) > 0:
            task_names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in names_attr]
        else:
            task_names = None

    # Determine order of task columns (encoding/delay)
    if task_names is None or len(task_names) != betas_task.shape[1]:
        # try CSV fallback
        task_names = read_design_order_from_csv(h5_path)
    if task_names is None or len(task_names) != betas_task.shape[1]:
        # final fallback: assume [encoding, delay] if K_task==2
        if betas_task.shape[1] == 2:
            task_names = ["encoding", "delay"]
        else:
            raise RuntimeError(f"Can't determine task regressor names for {h5_path}")

    # Build contrast c over the *task block*: encoding - delay
    try:
        enc_idx = task_names.index("encoding")
        del_idx = task_names.index("delay")
    except ValueError:
        raise RuntimeError(f"'encoding' and/or 'delay' not found in task_regressor_names for {h5_path}: {task_names}")

    K_task = betas_task.shape[1]
    c_task = np.zeros((K_task,), dtype=float)
    c_task[enc_idx] = 1.0
    c_task[del_idx] = -1.0

    # Pull task block of (X'X)^-1 to get variance of the contrast
    S_task = XtX_inv[tstart:tend, tstart:tend]  # (K_task, K_task)
    var_c_scalar = float(c_task @ S_task @ c_task)  # same across parcels; depends on X only
    var_per_parcel = np.maximum(var_c_scalar, 1e-12) * sigma2  # (P,)

    # Effect per parcel: c' * betas_task
    effect = betas_task @ c_task  # (P,)

    # t (and z) per parcel
    se = np.sqrt(np.maximum(var_per_parcel, 1e-12))
    tvals = effect / se
    # two-sided t->z
    zvals = stats.norm.isf(stats.t.sf(np.abs(tvals), dof)) * np.sign(tvals)

    if verbose:
        print(f"[run] {h5_path.name} | K_task={K_task} | names={task_names} | dof={dof}")

    # Surfaces
    surf_effect = project_to_surface(effect, atlas_img, n_parcels)
    surf_t = project_to_surface(tvals, atlas_img, n_parcels)
    surf_z = project_to_surface(zvals, atlas_img, n_parcels)

    return dict(
        effect=effect, var=var_per_parcel, t=tvals, z=zvals,
        surf_effect=surf_effect, surf_t=surf_t, surf_z=surf_z
    )


# -----------------------
# Fixed-effects combine
# -----------------------
def fixed_effects(effects: list[np.ndarray], variances: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inverse-variance weighted fixed effects across runs (per parcel).
    Returns effect_FE, var_FE, t_FE. (z_FE ~ t_FE for large dof)
    """
    E = np.stack(effects, axis=1)    # (P, R)
    V = np.stack(variances, axis=1)  # (P, R)
    W = 1.0 / np.maximum(V, 1e-12)
    sumW = W.sum(axis=1)             # (P,)
    effect_FE = (E * W).sum(axis=1) / np.maximum(sumW, 1e-12)
    var_FE = 1.0 / np.maximum(sumW, 1e-12)
    t_FE = effect_FE / np.sqrt(np.maximum(var_FE, 1e-12))
    return effect_FE, var_FE, t_FE


# -----------------------
# Main
# -----------------------
def main():
    args = get_args()

    root = betas_root(Path(args.out_root), args.subj, args.correct_only)
    out_dir = Path(args.results_root) / args.subj / "condition_level_contrasts" / f"task-{args.task}"
    ensure_dir(out_dir)
    print(f"output dir: {out_dir}")

    atlas_img = nib.load(str(args.atlas_path))

    # Find per-run files
    run_files = find_run_files(root, args.subj, args.task)
    if len(run_files) == 0:
        print(f"[⚠️] No per-run condition-level HDF5 found under {root} for task={args.task}")
        return

    per_run_effects = []
    per_run_vars = []

    for h5_path in run_files:
        ses, run = parse_ses_run(h5_path.name)
        try:
            res = compute_contrast_per_run(h5_path, atlas_img, args.n_parcels, verbose=args.verbose)
        except Exception as e:
            print(f"[⚠️] Skipping {h5_path.name}: {e}")
            continue

        # Save per-run
        tag = f"{ses}_{run}"
        np.save(out_dir / f"{tag}_effect.npy", res["effect"])
        np.save(out_dir / f"{tag}_t.npy", res["t"])
        np.save(out_dir / f"{tag}_z.npy", res["z"])
        np.save(out_dir / f"{tag}_var.npy", res["var"])
        np.save(out_dir / f"{tag}_surface_effect.npy", res["surf_effect"])
        np.save(out_dir / f"{tag}_surface_t.npy", res["surf_t"])
        np.save(out_dir / f"{tag}_surface_z.npy", res["surf_z"])
        print(f"range of surface z: {res['surf_z'].max()}, {res['surf_z'].min()}")

        print(f"[✅] Saved per-run contrast arrays for {tag}")

        per_run_effects.append(res["effect"])
        per_run_vars.append(res["var"])

    if len(per_run_effects) == 0:
        print("[⚠️] No valid runs to combine.")
        return

    # Fixed-effects combine across runs
    effect_FE, var_FE, t_FE = fixed_effects(per_run_effects, per_run_vars)
    # FE z ~ t for large dof; you can compute a heuristic dof if needed, but t≈z is common here.
    z_FE = t_FE

    # Save FE
    np.save(out_dir / "fixed_effects_effect.npy", effect_FE)
    np.save(out_dir / "fixed_effects_t.npy", t_FE)
    np.save(out_dir / "fixed_effects_z.npy", z_FE)
    np.save(out_dir / "fixed_effects_var.npy", var_FE)

    surf_effect_FE = project_to_surface(effect_FE, atlas_img, args.n_parcels)
    surf_t_FE = project_to_surface(t_FE, atlas_img, args.n_parcels)
    surf_z_FE = project_to_surface(z_FE, atlas_img, args.n_parcels)

    np.save(out_dir / "fixed_effects_surface_effect.npy", surf_effect_FE)
    np.save(out_dir / "fixed_effects_surface_t.npy", surf_t_FE)
    np.save(out_dir / "fixed_effects_surface_z.npy", surf_z_FE)

    print(f"[🎯] Saved fixed-effects contrast arrays under {out_dir}")


if __name__ == "__main__":
    main()
