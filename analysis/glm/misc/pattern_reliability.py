#!/usr/bin/env python3
"""
Split-half noise ceiling for the decoding ROIs, matched to what the decoder sees.

Same events, alignment and ROIs as feature_roi_sweep.py. For each ROI and phase:
  - the 4 sessions are split 2 vs 2 in all 3 ways (as in split_half_fisherz.py);
  - for every vertex, the event-wise betas of the two half-averages are correlated
    (the decoder works on per-vertex event-by-event variation, not the ROI mean);
  - vertex r's are averaged in Fisher-z space, Spearman-Brown corrected per split,
    then Fisher-averaged across splits -> reliability of the 4-session-averaged patterns.

Also reported for the bilateral ROI mean and per-hemisphere ROI means.
Reported raw, after removing task x position mean patterns, and after removing position-only means
(each within each half; what is left over for stimulus-feature decoding).
"""
import os
import io
import argparse
import contextlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from feature import collect_one_session_dict
from feature_roi_sweep import ROI_GROUPS, bilateral, roi_columns


SPLITS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def load_session(phase, ses, cfg, union_names):
    with contextlib.redirect_stdout(io.StringIO()):
        d, _, _ = collect_one_session_dict(
            block_logs_dir=os.path.join(cfg["block_logs_base"], ses), session=ses, roi_names=union_names,
            feature_to_decode="loc", phase_to_decode=phase, verbose=False,
            **{k: v for k, v in cfg.items() if k != "block_logs_base"},
        )
    return phase, ses, {k: v[0] for k, v in d.items()}


def demean_cells(X, cells):
    X = X.copy()
    for c in np.unique(cells):
        X[cells == c] -= X[cells == c].mean(axis=0)
    return X


def vertex_split_r(A, B):
    """Per-vertex Pearson r across events between half-averages A and B (events x vertices)."""
    A = (A - A.mean(0)) / (A.std(0) + 1e-12)
    B = (B - B.mean(0)) / (B.std(0) + 1e-12)
    return (A * B).mean(0)


def fisher_mean(r):
    return float(np.tanh(np.mean(np.arctanh(np.clip(r, -0.999999, 0.999999)))))


def spearman_brown(r):
    return 2 * r / (1 + r)


REDUCTIONS = ["vertex", "mean", "hemi_mean"]


def roi_matrix(M, col_sets, reduce):
    """events x features for one ROI: every vertex, the bilateral ROI mean, or one mean per hemisphere."""
    if reduce == "vertex":
        return M[:, np.concatenate(col_sets)]
    if reduce == "mean":
        return M[:, np.concatenate(col_sets)].mean(axis=1, keepdims=True)
    return np.column_stack([M[:, c].mean(axis=1) for c in col_sets])


def reliability(S, cells, col_sets, reduce="vertex"):
    """S: list of 4 session arrays (events x union vertices), rows aligned. col_sets: [left cols, right cols]."""
    S = [roi_matrix(M, col_sets, reduce) for M in S]
    r_splits, r_sb = [], []
    for a, b in SPLITS:
        A = np.mean([S[i] for i in a], axis=0)
        B = np.mean([S[i] for i in b], axis=0)
        if cells is not None:
            A, B = demean_cells(A, cells), demean_cells(B, cells)
        r = fisher_mean(vertex_split_r(A, B))
        r_splits.append(r)
        r_sb.append(spearman_brown(r))
    r_sb_mean = fisher_mean(np.array(r_sb))
    return {
        "r_split_mean": fisher_mean(np.array(r_splits)),
        "r_sb": r_sb_mean,
        "noise_ceiling": float(np.sqrt(max(r_sb_mean, 0.0))),
        "split_rs": ",".join(f"{r:.4f}" for r in r_splits),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-vertex split-half pattern reliability for decoding ROIs")
    ap.add_argument("--sub", default="sub-01")
    ap.add_argument("--sessions", nargs=4, default=["ses-01", "ses-02", "ses-03", "ses-04"])
    ap.add_argument("--glm_type", default="lss", choices=["lsa", "lss", "glmsingle", "glmsingleB", "glmsingleC"])
    ap.add_argument("--glm_variant", default="wofdelay", choices=["wfdelay", "wofdelay"])
    ap.add_argument("--phases", nargs="+", default=["Encoding", "Delay"])
    ap.add_argument("--rois", nargs="+", default=list(ROI_GROUPS), choices=list(ROI_GROUPS))
    ap.add_argument("--levels", nargs="+", default=["group", "parcel"], choices=["group", "parcel"])
    ap.add_argument("--tasks", default=None, help="regex on task dir names (default: pool all)")
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--dlabel", default="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii")
    ap.add_argument("--out_dir", default=None, help="default: <glm_base_dir>/<sub>/_reliability_pattern")
    args = ap.parse_args()

    root = "/mnt/tempdata/lucas/fmri/recordings/TR"
    glm_base_dir = f"{root}/neural/fmriprep_outs/first_run/glm_runs/{args.glm_type}_{args.glm_variant}/64kDense"
    cfg = dict(
        glm_base_dir=glm_base_dir,
        behav_base=f"{root}/behav",
        events_subdir=f"events_{args.glm_variant}",
        block_logs_base=f"{root}/behav/{args.sub}",
        sub=args.sub,
        dlabel_path=args.dlabel,
        only_tasks_regex=args.tasks,
    )

    # (level, roi) -> (group, glasser labels); keyed by level because some parcels share a group's name (OFC)
    rois = {}
    if "group" in args.levels:
        rois.update({("group", g): (g, bilateral(ROI_GROUPS[g])) for g in args.rois})
    if "parcel" in args.levels:
        rois.update({("parcel", p): (g, bilateral([p])) for g in args.rois for p in ROI_GROUPS[g]})
    union_names = sorted({n for _, names in rois.values() for n in names})
    col_sets = {key: [roi_columns(args.dlabel, union_names, [n for n in names if n.startswith(h)]) for h in ("L_", "R_")]
                for key, (_, names) in rois.items()}

    print(f"[INFO] loading {len(args.phases) * 4} phase x session datasets ...")
    loaded = Parallel(n_jobs=min(args.n_jobs, len(args.phases) * 4))(
        delayed(load_session)(ph, ses, cfg, union_names) for ph in args.phases for ses in args.sessions
    )
    by_phase = {}
    for ph, ses, d in loaded:
        by_phase.setdefault(ph, {})[ses] = d

    rows = []
    for ph in args.phases:
        per_ses = by_phase[ph]
        keys = sorted(set.intersection(*(set(per_ses[s]) for s in args.sessions)))
        S = [np.stack([per_ses[s][k] for k in keys]).astype(np.float64) for s in args.sessions]
        # key = (task, acq, run, phase, pos, tc)
        cell_sets = {"none": None,
                     "task_pos": np.array([f"{k[0]}|{k[1]}|{k[4]}" for k in keys]),
                     "pos": np.array([str(k[4]) for k in keys])}
        print(f"[INFO] {ph}: {len(keys)} events common to all sessions")
        for (level, roi), (group, _) in rois.items():
            cs = col_sets[(level, roi)]
            for reduce in REDUCTIONS:
                for demean, cells in cell_sets.items():
                    rows.append({"roi": roi, "group": group, "level": level, "phase": ph, "reduce": reduce,
                                 "demean": demean, "n_events": len(keys), "n_vertices": int(sum(len(c) for c in cs)),
                                 **reliability(S, cells, cs, reduce)})

    df = pd.DataFrame(rows)
    out_dir = args.out_dir or os.path.join(glm_base_dir, args.sub, "_reliability_pattern")
    os.makedirs(out_dir, exist_ok=True)
    out_tsv = os.path.join(out_dir, "pattern_split_half_reliability" + (f"_tasks-{args.tasks}" if args.tasks else "") + ".tsv")
    df.to_csv(out_tsv, sep="\t", index=False)

    pd.set_option("display.width", 200)
    g = df[df["level"] == "group"]
    for reduce in REDUCTIONS:
        print(f"\n=== reliability (r_sb), reduce={reduce} ===")
        gr = g[g["reduce"] == reduce]
        print(gr.pivot(index="roi", columns=["phase", "demean"], values="r_sb").round(3).reindex(args.rois).to_string())
    print(f"\n[SAVED] {out_tsv}")
