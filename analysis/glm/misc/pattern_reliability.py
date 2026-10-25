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

--split_mode repeats (datasets without a fixed 4-session repeat structure, e.g. CNM): every condition with
>= --min_repeats repeats (across and within sessions) has its repeats shuffled and split into two equal halves
(an odd one out is dropped); the half-averages are correlated as above, over --n_splits random splits.
Conditions are --cond_key trial (task, acq, phase, pos, tc: the same trial position, as for the GLM) or
stimulus (task, phase, pos, loc, obj: the same stimulus at the same position, pooled over trials and task
variants). Spearman-Brown then estimates the reliability of the mean of the halves' size x 2 repeats, which
varies across conditions, so r_sb is approximate there.
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
from temporal_leakage import N_POS, DIGIT


SPLITS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def load_session(phase, ses, cfg, union_names, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        d, _, _ = collect_one_session_dict(
            block_logs_dir=os.path.join(cfg["block_logs_base"], ses), session=ses, roi_names=union_names,
            feature_to_decode="loc", phase_to_decode=phase, verbose=False,
            **{k: v for k, v in cfg.items() if k != "block_logs_base"}, **kw,
        )
    return phase, ses, {k: v[0] for k, v in d.items()}


def cond_of(key, cond_key):
    """Condition of a unique_key event key (task, acq, run, phase, pos, tc, trial_int)."""
    task, acq, _, phase, pos, tc, _ = key
    if cond_key == "trial":
        return (task, acq, phase, pos, tc)
    code = str(tc).zfill(2 * N_POS[task])
    loc, obj = (code[2 * (pos - 1) + DIGIT[f]] for f in ("loc", "obj"))
    return (task, phase, pos, loc, obj)


def repeat_halves(by_cond, n_splits, seed=0):
    """[(A, B)] per random split: condition x vertex half-averages of each condition's shuffled repeats."""
    rng = np.random.default_rng(seed)
    conds = sorted(by_cond)
    pairs = []
    for _ in range(n_splits):
        A, B = [], []
        for c in conds:
            V = by_cond[c]
            idx = rng.permutation(len(V))
            h = len(V) // 2
            A.append(V[idx[:h]].mean(0))
            B.append(V[idx[h:2 * h]].mean(0))
        pairs.append((np.stack(A), np.stack(B)))
    return conds, pairs


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


def reliability_pairs(pairs, cells, col_sets, reduce="vertex"):
    """Same statistics as reliability(), from precomputed (A, B) half-average pairs (one per split)."""
    r_splits, r_sb = [], []
    for A, B in pairs:
        A, B = roi_matrix(A, col_sets, reduce), roi_matrix(B, col_sets, reduce)
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
        "split_rs": ",".join(f"{r:.4f}" for r in r_splits[:10]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-vertex split-half pattern reliability for decoding ROIs")
    ap.add_argument("--sub", default="sub-01")
    ap.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02", "ses-03", "ses-04"],
                    help="exactly 4 for --split_mode sessions")
    ap.add_argument("--glm_type", default="lss", help="model folder prefix: lsa, lss, glmsingle, glmsingleB, glmsingleC, ...")
    ap.add_argument("--glm_variant", default="wofdelay", help="events variant, e.g. wfdelay / wofdelay")
    ap.add_argument("--root", default="/mnt/tempdata/lucas/fmri/recordings/TR", help="dataset root (behav/, neural/)")
    ap.add_argument("--prep_run", default="first_run", help="folder under <root>/neural/fmriprep_outs")
    ap.add_argument("--split_mode", default="sessions", choices=["sessions", "repeats"],
                    help="sessions: the 3 2-vs-2 splits of 4 sessions (TR); repeats: random half-splits of each "
                         "condition's repeats")
    ap.add_argument("--cond_key", default="trial", choices=["trial", "stimulus"], help="repeats mode only")
    ap.add_argument("--min_repeats", type=int, default=2, help="repeats mode only")
    ap.add_argument("--n_splits", type=int, default=20, help="repeats mode only: random half-splits")
    ap.add_argument("--exclude_tasks", nargs="+", default=None, help="task names to leave out, e.g. dms")
    ap.add_argument("--phases", nargs="+", default=["Encoding", "Delay"])
    ap.add_argument("--rois", nargs="+", default=list(ROI_GROUPS), choices=list(ROI_GROUPS))
    ap.add_argument("--levels", nargs="+", default=["group", "parcel"], choices=["group", "parcel"])
    ap.add_argument("--tasks", default=None, help="regex on task dir names (default: pool all)")
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--dlabel", default="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii")
    ap.add_argument("--out_dir", default=None, help="default: <glm_base_dir>/<sub>/_reliability_pattern")
    args = ap.parse_args()

    if args.split_mode == "sessions" and len(args.sessions) != 4:
        ap.error("--split_mode sessions needs exactly 4 --sessions")
    root = args.root
    glm_base_dir = f"{root}/neural/fmriprep_outs/{args.prep_run}/glm_runs/{args.glm_type}_{args.glm_variant}/64kDense"
    cfg = dict(
        glm_base_dir=glm_base_dir,
        behav_base=f"{root}/behav",
        events_subdir=f"events_{args.glm_variant}",
        block_logs_base=f"{root}/behav/{args.sub}",
        sub=args.sub,
        dlabel_path=args.dlabel,
        only_tasks_regex=args.tasks,
    )
    if args.exclude_tasks:
        cfg["exclude_tasks"] = args.exclude_tasks

    # (level, roi) -> (group, glasser labels); keyed by level because some parcels share a group's name (OFC)
    rois = {}
    if "group" in args.levels:
        rois.update({("group", g): (g, bilateral(ROI_GROUPS[g])) for g in args.rois})
    if "parcel" in args.levels:
        rois.update({("parcel", p): (g, bilateral([p])) for g in args.rois for p in ROI_GROUPS[g]})
    union_names = sorted({n for _, names in rois.values() for n in names})
    col_sets = {key: [roi_columns(args.dlabel, union_names, [n for n in names if n.startswith(h)]) for h in ("L_", "R_")]
                for key, (_, names) in rois.items()}

    n_load = len(args.phases) * len(args.sessions)
    print(f"[INFO] loading {n_load} phase x session datasets ...")
    kw = {"unique_key": True} if args.split_mode == "repeats" else {}
    loaded = Parallel(n_jobs=min(args.n_jobs, n_load))(
        delayed(load_session)(ph, ses, cfg, union_names, **kw) for ph in args.phases for ses in args.sessions
    )
    by_phase = {}
    for ph, ses, d in loaded:
        by_phase.setdefault(ph, {})[ses] = d

    rows = []
    for ph in args.phases if args.split_mode == "repeats" else []:
        by_cond = {}
        for ses in args.sessions:
            for k, v in by_phase[ph][ses].items():
                by_cond.setdefault(cond_of(k, args.cond_key), []).append(v)
        by_cond = {c: np.stack(v).astype(np.float64) for c, v in by_cond.items() if len(v) >= args.min_repeats}
        conds, pairs = repeat_halves(by_cond, args.n_splits)
        n_rep = np.array([len(by_cond[c]) for c in conds])
        # cond = (task, acq, phase, pos, tc) or (task, phase, pos, loc, obj); cells never include the stimulus
        tp = [f"{c[0]}|{c[1]}|{c[3]}" if args.cond_key == "trial" else f"{c[0]}|{c[2]}" for c in conds]
        pos = [str(c[3] if args.cond_key == "trial" else c[2]) for c in conds]
        cell_sets = {"none": None, "task_pos": np.array(tp), "pos": np.array(pos)}
        print(f"[INFO] {ph}: {len(conds)} conditions with >= {args.min_repeats} repeats "
              f"(repeats: median {np.median(n_rep):.0f}, max {n_rep.max()})")
        for (level, roi), (group, _) in rois.items():
            cs = col_sets[(level, roi)]
            for reduce in REDUCTIONS:
                for demean, cells in cell_sets.items():
                    rows.append({"roi": roi, "group": group, "level": level, "phase": ph, "reduce": reduce,
                                 "demean": demean, "n_events": int(n_rep.sum()), "n_conditions": len(conds),
                                 "n_vertices": int(sum(len(c) for c in cs)),
                                 **reliability_pairs(pairs, cells, cs, reduce)})
    for ph in args.phases if args.split_mode == "sessions" else []:
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
    tag = (f"_tasks-{args.tasks}" if args.tasks else "") + (f"_excl-{'-'.join(args.exclude_tasks)}" if args.exclude_tasks else "")
    if args.split_mode == "repeats":
        tag += f"_split-repeats_key-{args.cond_key}" + (f"_min{args.min_repeats}" if args.min_repeats != 2 else "")
    out_tsv = os.path.join(out_dir, "pattern_split_half_reliability" + tag + ".tsv")
    df.to_csv(out_tsv, sep="\t", index=False)

    pd.set_option("display.width", 200)
    g = df[df["level"] == "group"]
    for reduce in REDUCTIONS:
        print(f"\n=== reliability (r_sb), reduce={reduce} ===")
        gr = g[g["reduce"] == reduce]
        print(gr.pivot(index="roi", columns=["phase", "demean"], values="r_sb").round(3).reindex(args.rois).to_string())
    print(f"\n[SAVED] {out_tsv}")
