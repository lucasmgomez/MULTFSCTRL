#!/usr/bin/env python3
"""
Noise ceiling with three granularities: voxel, Glasser region, or network (12 networks).

python noise_ceiling_tradition.py \
  --subject sub-01 \
  --input_pickle /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/sub-01_task_condition_betas.pkl \
  --atlas_path /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --network_assignments_path /project/def-pbellec/xuan/fmri_dataset_project/data/cortex_parcel_network_assignments.txt \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/noise_ceiling/traditional_way \
  --unit network --n_bootstrap 2 --min_reps 4 --apply_spearman_brown 1


Input (same as before):
  - A pickle: { condition_key: [beta_vec, beta_vec, ...] }, one beta_vec per repetition
    beta_vec length = #voxels (MNI) or #surface vertices (Glasser).

New:
  - --unit {voxel,region,network}
  - --apply_spearman_brown (default: on)
  - --standardize_halves (optional, like Method A)
  - --network_assignments_path for region->network (1..12)
  - --network_mapping_json optional: pretty names for networks; or baked default.

Computation:
  - Per condition with >= min repetitions:
      * Randomly split reps into halves
      * Average each half -> A, B (vector in chosen unit space)
  - Stack across conditions -> H1, H2 (n_conditions x n_units)
  - (Optional) z-score halves per unit (column-wise)
  - Pearson r per unit across conditions
  - (Optional) Spearman–Brown correction (k=2) per unit
  - Bootstrap over splits; average across bootstraps

Outputs:
  - Histogram figure
  - Pickle of per-(voxel|region|network) noise ceiling values (after SB if applied)
  - If unit=network: JSON summary with names & stats
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import nibabel as nib


# ---------- helpers ----------

def colwise_correlation(X, Y):
    """Column-wise Pearson r between matrices X and Y (same shape). Returns (n_cols,) with NaNs for constant cols."""
    assert X.shape == Y.shape, f"Shape mismatch: {X.shape} vs {Y.shape}"
    n = X.shape[1]
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        x = X[:, i]
        y = Y[:, i]
        # guard against constants
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            continue
        r, _ = pearsonr(x, y)
        out[i] = r
    return out


def spearman_brown(r, k=2.0):
    """Apply Spearman–Brown prophecy to a correlation r for test-length factor k."""
    return (k * r) / (1.0 + (k - 1.0) * r)


def load_glasser_labels(atlas_path: str):
    """Load Glasser .dlabel.nii/.nii labels -> (labels_vector, region_ids_sorted)."""
    lab_img = nib.load(atlas_path)
    lab = np.asarray(lab_img.get_fdata()).squeeze()
    lab = lab.reshape(-1).astype(int)
    region_ids = np.unique(lab)
    region_ids = region_ids[region_ids > 0]
    return lab, region_ids


def make_region_averager(labels_vec: np.ndarray, region_ids: np.ndarray):
    """Return a function avg_by_region(trials x features) -> (trials x n_regions)."""
    idxs = [np.where(labels_vec == rid)[0] for rid in region_ids]

    def avg_by_region(A):
        # A: (n_trials, n_features)
        out = np.full((A.shape[0], len(region_ids)), np.nan, dtype=np.float32)
        for j, idx in enumerate(idxs):
            if idx.size == 0:
                continue
            out[:, j] = np.nanmean(A[:, idx], axis=1)
        return out

    return avg_by_region


def load_region_to_network_map(assign_path: str, max_region_id: int):
    """
    Load cortex_parcel_network_assignments.txt
    Expect one network index (1..12) per line, where line number == region index.
    Returns an array map_net[region_id] -> network_id (1..12), length max_region_id+1.
    """
    txt = Path(assign_path).read_text().strip().splitlines()
    arr = np.zeros(max_region_id + 1, dtype=int)
    # file is 1-based (row i corresponds to region i)
    for i, line in enumerate(txt, start=1):
        if i <= max_region_id:
            arr[i] = int(line.strip())
    return arr


def make_network_averager(region_ids: np.ndarray, region_to_net: np.ndarray, n_networks: int = 12):
    """
    Build avg_by_network that consumes (n_trials x n_regions) and outputs (n_trials x n_networks).
    We map each region_id in region_ids to a network id via region_to_net[region_id].
    """
    nets = [[] for _ in range(n_networks)]
    # Build column indices per network for the *region-averaged matrix* order.
    for j, rid in enumerate(region_ids):
        nid = region_to_net[rid]  # 1..12
        if nid >= 1 and nid <= n_networks:
            nets[nid - 1].append(j)  # 0-based columns for region matrix

    def avg_by_network(R):
        # R: (n_trials, n_regions) in the same region_ids order
        out = np.full((R.shape[0], n_networks), np.nan, dtype=np.float32)
        for k, cols in enumerate(nets):
            if len(cols) == 0:
                continue
            out[:, k] = np.nanmean(R[:, cols], axis=1)
        return out

    return avg_by_network


def zscore_columns(X, eps=1e-8):
    """Z-score each column independently (center & scale)."""
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--input_pickle", required=True,
                    help="Path to {condition: [beta_vec,...]}")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_bootstrap", type=int, default=50)
    ap.add_argument("--min_reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    # Unit selection
    ap.add_argument("--unit", choices=["voxel", "region", "network"], default="voxel",
                    help="Granularity of NC computation")

    # Glasser options (needed for region/network)
    ap.add_argument("--atlas_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii",
                    help="Glasser atlas labels path (.dlabel.nii or NIfTI)")

    # Network mapping (needed for network)
    ap.add_argument("--network_assignments_path",
                    help="Text file mapping region index (line number) -> network index (1..12)")

    # Method-A goodies
    ap.add_argument("--apply_spearman_brown", type=int, default=1,
                    help="Apply SB correction (k=2) to split-half r (1=yes, 0=no)")
    ap.add_argument("--standardize_halves", type=int, default=0,
                    help="Z-score each column of H1 and H2 before r (0/1)")

    # Optional JSON mapping for pretty network names
    ap.add_argument("--network_mapping_json", default="",
                    help="JSON file mapping {1:'primary visual', ...}. If not provided, a default dict is used.")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ----- load grouped betas -----
    input_path = Path(args.input_pickle)
    with open(input_path, "rb") as f:
        condition_betas = pickle.load(f)

    # filter conditions with enough reps
    valid_conditions = [k for k, v in condition_betas.items() if len(v) >= args.min_reps]
    if not valid_conditions:
        raise SystemExit(f"No valid conditions with at least {args.min_reps} reps found in {input_path}")

    # infer feature length from first condition
    first_vec = np.asarray(condition_betas[valid_conditions[0]][0]).reshape(-1)
    n_features = first_vec.size

    # ----- prep unit transforms -----
    avg_by_region = None
    avg_by_network = None
    region_ids = None

    if args.unit in ("region", "network"):
        labels_vec, region_ids = load_glasser_labels(args.atlas_path)
        if labels_vec.size != n_features:
            raise SystemExit(
                f"Atlas feature count ({labels_vec.size}) != beta vector length ({n_features}). "
                "Ensure beta vectors and atlas share the same space and indexing."
            )
        avg_by_region = make_region_averager(labels_vec, region_ids)

    if args.unit == "network":
        if not args.network_assignments_path:
            raise SystemExit("--network_assignments_path is required when --unit=network")
        max_region_id = int(region_ids.max())
        region_to_net = load_region_to_network_map(args.network_assignments_path, max_region_id)
        avg_by_network = make_network_averager(region_ids, region_to_net, n_networks=12)

        # network name mapping
        if args.network_mapping_json and Path(args.network_mapping_json).exists():
            with open(args.network_mapping_json, "r") as f:
                network_name_map = {int(k): v for k, v in json.load(f).items()}
        else:
            network_name_map = {
                1: "primary visual",
                2: "secondary visual",
                3: "somatomotor",
                4: "cingulo-opercular",
                5: "dorsal attention",
                6: "language",
                7: "frontparietal",
                8: "auditory",
                9: "default mode",
                10: "posterior multimodal",
                11: "ventral multimodal",
                12: "orbito-affective",
            }

    # unit meta
    if args.unit == "voxel":
        n_units = n_features
        unit_name = "voxel"
    elif args.unit == "region":
        n_units = len(region_ids)
        unit_name = "region"
    else:
        n_units = 12
        unit_name = "network"

    # ----- bootstrap split-half NC -----
    boot_nc = []

    for b in range(args.n_bootstrap):
        half1_rows = []
        half2_rows = []

        for cond in valid_conditions:
            # stack repetitions for this condition: (n_reps, n_features)
            reps = np.stack([np.asarray(v).reshape(-1) for v in condition_betas[cond]], axis=0)
            if reps.shape[0] < args.min_reps:
                continue

            # (optional) average per trial into regions
            if avg_by_region is not None:
                reps = avg_by_region(reps)  # (n_reps, n_regions)

            # (optional) average per trial into networks (from regions)
            if avg_by_network is not None:
                reps = avg_by_network(reps)  # (n_reps, 12)

            # split reps into two halves
            perm = rng.permutation(reps.shape[0])
            mid = reps.shape[0] // 2
            A = reps[perm[:mid]].mean(axis=0)   # (n_units,)
            B = reps[perm[mid:]].mean(axis=0)   # (n_units,)

            # require both halves non-empty
            if A.size == 0 or B.size == 0:
                continue

            half1_rows.append(A)
            half2_rows.append(B)

        if not half1_rows or not half2_rows:
            # nothing usable this bootstrap
            continue

        H1 = np.stack(half1_rows, axis=0)   # (n_conditions_kept, n_units)
        H2 = np.stack(half2_rows, axis=0)   # (n_conditions_kept, n_units)

        if args.standardize_halves:
            H1 = zscore_columns(H1)
            H2 = zscore_columns(H2)

        r_cols = colwise_correlation(H1, H2)  # (n_units,)

        # Spearman–Brown correction per unit (like Method A)
        if args.apply_spearman_brown:
            # clip to avoid division edge cases
            r_cols = np.clip(r_cols, -0.9999, 0.9999)
            r_cols = spearman_brown(r_cols, k=2.0)

        boot_nc.append(r_cols)

    if not boot_nc:
        raise SystemExit("No bootstrap iterations produced valid splits — not enough data per condition.")

    noise_ceiling = np.nanmean(np.stack(boot_nc, axis=0), axis=0)  # (n_units,)

    # ----- save & plot -----
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    clipped_nc = np.clip(noise_ceiling, -1, 1)
    plt.figure(figsize=(10, 6))
    plt.hist(clipped_nc[np.isfinite(clipped_nc)], bins=60, edgecolor='black')
    plt.title(f"Noise Ceiling Distribution ({args.subject}, per-{unit_name})")
    plt.xlabel("Noise Ceiling (Pearson r{}{})".format(
        " (SB-corrected)" if args.apply_spearman_brown else "",
        ", zscored" if args.standardize_halves else ""
    ))
    plt.ylabel(f"Number of {unit_name}s")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = outdir / f"{args.subject}_noise_ceiling_distribution_per_{unit_name}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    nc_save_path = outdir / f"{args.subject}_noise_ceiling_per_{unit_name}.pkl"
    with open(nc_save_path, "wb") as f:
        pickle.dump(noise_ceiling, f)

    print(f"📊 Plot saved: {fig_path}")
    print(f"💾 Noise ceiling saved: {nc_save_path}")

    # If network mode, also dump a friendly JSON summary
    if args.unit == "network":
        # Per-bootstrap matrix -> (n_bootstrap, 12)
        boots = np.stack(boot_nc, axis=0)
        # Aggregate stats
        net_stats = []
        for k in range(12):
            vals = boots[:, k]
            net_stats.append({
                "network_id": k + 1,
                "name": network_name_map.get(k + 1, f"network-{k+1}"),
                "mean": float(np.nanmean(vals)),
                "median": float(np.nanmedian(vals)),
                "std": float(np.nanstd(vals)),
                "p25": float(np.nanpercentile(vals, 25)),
                "p75": float(np.nanpercentile(vals, 75)),
            })
        json_path = outdir / f"{args.subject}_noise_ceiling_per_network_summary.json"
        with open(json_path, "w") as f:
            json.dump({"subject": args.subject, "sb_corrected": bool(args.apply_spearman_brown),
                       "zscored": bool(args.standardize_halves), "networks": net_stats}, f, indent=2)
        print(f"🧠 Network summary saved: {json_path}")


if __name__ == "__main__":
    main()
