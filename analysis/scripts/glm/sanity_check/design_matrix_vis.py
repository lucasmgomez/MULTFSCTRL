#!/usr/bin/env python3
"""
Visualize GLM design matrices saved by the per-run scripts (condition-level *or* trial-level).

Features
- Loads `design_matrix` and `design_col_names` from each HDF5 (preferred), or from a CSV.
- Uses exact task slice when available via HDF5 attrs (`task_col_start`, `task_col_end`).
- Renders a heatmap (T x K) with grouped columns: Confounds | Drifts+Intercept | Tasks.
- Handles large K by sub-sampling tick labels and capping #task labels.
- If --outdir is not given, auto-saves under:
    <ANCHOR>/stats/design_matrix_visualization/<same subpath>/..._design_matrix.png
  where ANCHOR is either `condition_level_betas` or `trial_level_betas` if present in the path;
  otherwise, saves next to the input file in a sibling `design_matrix_visualization/` folder.

Usage examples
--------------
# Visualize a single HDF5
python design_matrix_vis.py --input \
  /project/def-pbellec/xuan/fmri_dataset_project/data/condition_level_betas/sub-01/ses-001/func/sub-01_ses-001_task-ctxdm_run-01_betas.h5

# Visualize all runs under a directory (recursively)
python design_matrix_vis.py --input \
  /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/sub-01 --recursive

# Visualize from a CSV design matrix (with header as column names)
python design_matrix_vis.py --design-csv \
  /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/sub-01/ses-001/func/sub-01_ses-001_task-ctxdm_run-01_design_matrix.csv --title "sub-01_ses-001_task-ctxdm_run-01"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# -----------------------
# IO helpers
# -----------------------

def _safe_decode_attr(x):
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    return x


def load_design_from_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        X = f["design_matrix"][...]
        names_raw = f["design_col_names"][...]
        names = [_safe_decode_attr(n) for n in names_raw]
        order_hint = _safe_decode_attr(f.attrs.get("order_in_X", ""))
        tstart = f.attrs.get("task_col_start", None)
        tend   = f.attrs.get("task_col_end", None)
        task_slice = (int(tstart), int(tend)) if (tstart is not None and tend is not None) else None
    return X, names, order_hint, task_slice


def load_design_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    names = list(df.columns)
    X = df.values
    return X, names, "", None

# -----------------------
# Column grouping & annotation
# -----------------------

def group_columns(names, task_slice=None):
    """Return lists of indices for confounds, drifts+intercept, and tasks.
    Prefer explicit task_slice when present (from HDF5 attrs). Fallback uses name heuristics.
    """
    if task_slice and all(x is not None for x in task_slice):
        t0, t1 = task_slice
        idx_all = list(range(len(names)))
        task_idx = list(range(max(0, t0), min(len(names), t1)))
        pre_task = idx_all[:t0]
        conf_idx, drift_idx = [], []
        for i in pre_task:
            ln = names[i].lower()
            if ln.startswith("cosine") or ln == "constant":
                drift_idx.append(i)
            elif ln.startswith("conf_") or ln.startswith("trans_") or ln in {"csf", "white_matter", "global_signal"}:
                conf_idx.append(i)
            else:
                # conservative: treat unknown pre-task as confounds
                conf_idx.append(i)
        return conf_idx, drift_idx, task_idx

    # fallback heuristic (CSV case or old HDF5)
    conf_idx, drift_idx, task_idx = [], [], []
    for i, n in enumerate(names):
        ln = n.lower()
        if ln.startswith("cosine") or ln == "constant":
            drift_idx.append(i)
        elif ln.startswith("conf_") or ln.startswith("trans_") or ln in {"csf", "white_matter", "global_signal"}:
            conf_idx.append(i)
        else:
            task_idx.append(i)
    return conf_idx, drift_idx, task_idx


def pick_tick_positions(K: int, max_ticks: int = 40):
    if K <= max_ticks:
        return list(range(K))
    step = int(np.ceil(K / max_ticks))
    return list(range(0, K, step))

# -----------------------
# Plotting
# -----------------------

def plot_design_matrix(X: np.ndarray, names: list[str], title: str, out_file: Path, task_slice=None, max_task_labels: int = 20):
    T, K = X.shape
    conf_idx, drift_idx, task_idx = group_columns(names, task_slice=task_slice)

    # Normalize columns (z-score per column) for visibility
    Xn = X.astype(float).copy()
    col_mean = Xn.mean(axis=0)
    col_std = Xn.std(axis=0, ddof=0)
    nz = col_std > 0
    Xn[:, nz] = (Xn[:, nz] - col_mean[nz]) / col_std[nz]

    fig_h = min(18, 4 + 0.008 * T)
    fig_w = min(24, 8 + 0.25 * (K / 5))
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    im = plt.imshow(Xn, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.02, pad=0.02, label="z-scored regressor amplitude")

    # X-axis ticks/labels
    xticks = pick_tick_positions(K, max_ticks=40)
    xticklabels = [names[i] for i in xticks]

    # Ensure we label up to max_task_labels task columns additionally
    missing_task = [j for j in task_idx if j not in xticks]
    if missing_task:
        step = max(1, int(np.ceil(len(missing_task) / max_task_labels)))
        for j in missing_task[::step][:max_task_labels]:
            xticks.append(j)
            xticklabels.append(names[j])

    if xticks:
        xticks, xticklabels = zip(*sorted(zip(xticks, xticklabels), key=lambda x: x[0]))
        plt.xticks(xticks, xticklabels, rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel("Regressors (columns)")
    plt.ylabel("Time (TRs)")
    plt.title(f"{title}\nT={T}, K={K}", fontsize=11)

    # Draw separators between groups
    def draw_sep(after_idx, color="white"):
        if after_idx:
            plt.axvline(max(after_idx) + 0.5, color=color, linewidth=1.2, alpha=0.9)

    draw_sep(conf_idx, color="white")
    draw_sep(conf_idx + drift_idx, color="white")

    # Group labels above
    for idxs, lbl in [(conf_idx, "Confounds"), (drift_idx, "Drifts+Const"), (task_idx, "Tasks")]:
        if idxs:
            x0, x1 = min(idxs) - 0.5, max(idxs) + 0.5
            plt.text((x0 + x1) / 2.0, -2, lbl, ha="center", va="top", fontsize=9)

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

# -----------------------
# Output path logic
# -----------------------

def derive_out_path_for_input(inp_path: Path, explicit_outdir: Path | None, stem_title: str) -> Path:
    if explicit_outdir is not None:
        return explicit_outdir / f"{stem_title}_design_matrix.png"

    # Try to detect anchor and mirror
    parts = inp_path.resolve().parts
    anchors = ("condition_level_betas", "trial_level_betas")
    for anchor in anchors:
        if anchor in parts:
            idx = parts.index(anchor)
            base = Path(*parts[:idx+1])  # include anchor
            rel = Path(*parts[idx+1:])
            out_root = base / "stats" / "design_matrix_visualization"
            return out_root / rel.parent / f"{rel.stem}_design_matrix.png"

    # Fallback: sibling folder next to file
    return inp_path.parent / "design_matrix_visualization" / f"{inp_path.stem}_design_matrix.png"

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize GLM design matrices (condition-level or trial-level)")
    ap.add_argument("--input", type=str, help="HDF5 file or directory containing *_betas.h5")
    ap.add_argument("--design-csv", type=str, default=None, help="Optional: visualize from a CSV design matrix (full path)")
    ap.add_argument("--title", type=str, default=None, help="Optional title override (used with --design-csv)")
    ap.add_argument("--recursive", action="store_true", help="Search recursively for *_betas.h5 when --input is a directory")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory for figures (overrides auto placement)")
    ap.add_argument("--max_task_labels", type=int, default=20, help="Max # extra task column labels to add to x-axis")
    args = ap.parse_args()

    explicit_outdir = Path(args.outdir) if args.outdir else None

    if args.design_csv:
        csv_path = Path(args.design_csv)
        X, names, _, _ = load_design_from_csv(csv_path)
        title = args.title or csv_path.stem
        out_file = derive_out_path_for_input(csv_path, explicit_outdir, stem_title=title)
        plot_design_matrix(X, names, title, out_file, task_slice=None, max_task_labels=args.max_task_labels)
        print(f"[✓] Saved {out_file}")
        return

    if not args.input:
        raise SystemExit("--input is required when not using --design-csv")

    inp = Path(args.input)
    if inp.is_file() and inp.suffix == ".h5":
        targets = [inp]
    elif inp.is_dir():
        pattern = "**/*_betas.h5" if args.recursive else "*_betas.h5"
        targets = list(inp.glob(pattern))
    else:
        raise SystemExit(f"Input not found or not supported: {inp}")

    if not targets:
        print("No HDF5 betas files found.")
        return

    for h5_path in sorted(targets):
        try:
            X, names, order_hint, task_slice = load_design_from_h5(h5_path)
            title = args.title or h5_path.stem
            out_file = derive_out_path_for_input(h5_path, explicit_outdir, stem_title=title)
            plot_design_matrix(X, names, title, out_file, task_slice=task_slice, max_task_labels=args.max_task_labels)
            print(f"[✓] Saved {out_file}")
        except Exception as e:
            print(f"[!] Failed on {h5_path}: {e}")


if __name__ == "__main__":
    main()
