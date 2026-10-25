#!/usr/bin/env python3
"""
Temporal leakage test: how cleanly does a GLM separate neighbouring events?

Events are ~1.5-3.5 s apart, far shorter than the BOLD response, so a single-event beta can absorb
signal from the events around it. This decodes a stimulus feature of a NEIGHBOURING position:
  - Delay p    -> feature of the NEXT stimulus (p+1)   (forward leakage; the subject cannot know it yet)
  - Encoding p -> feature of the PREVIOUS stimulus (p-1) (backward leakage / carry-over)
next to the reference (the current stimulus, p).

Consecutive features are correlated in some tasks (1-back match rates), so neighbour decoding is done
within strata of the CURRENT feature value (current value constant -> it cannot drive the prediction);
balanced accuracy and permutation nulls are averaged across strata.

Leakage near chance = events are well separated; above chance = the beta mixes in its neighbour.
Note the backward test also picks up genuine carry-over (e.g. the 1-back comparison) - the forward
test is the clean one.
"""
import os
import io
import argparse
import contextlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from feature import collect_repeat_averaged_across_sessions, trial_groups, task_pos_cells, ridge_decode_5fold
from feature_roi_sweep import ROI_GROUPS, ALPHA_GRID, bilateral, roi_columns


N_POS = {"1back": 6, "interdms": 4, "ctxdm": 3, "dms": 2}
DIGIT = {"loc": 0, "obj": 1}  # tc = per position (loc, obj) digits, leading zeros dropped


def feature_at(meta, feature, offset):
    """Feature value at position pos+offset of the same trial, from the tc code (NaN if out of range)."""
    out = np.full(len(meta), np.nan)
    for i, (task, pos, tc) in enumerate(zip(meta["task"], meta["pos"], meta["tc"])):
        n = N_POS[task]
        p = int(pos) + offset
        if 1 <= p <= n:
            v = int(str(tc).zfill(2 * n)[2 * (p - 1) + DIGIT[feature]])
            out[i] = v // 2 if feature == "ctg" else v
    return out


def decode_stratified(X, y, strata, groups, cells, n_perm):
    """Mean balanced accuracy over strata of the current feature; null averaged across strata."""
    baccs, nulls, chance = [], [], []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if len(np.unique(y[idx])) < 2:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            res = ridge_decode_5fold(X[idx], y[idx], groups=groups[idx], alphas=ALPHA_GRID,
                                     nuisance=cells[idx], n_perm=n_perm, random_state=0)
        baccs.append(res["mean_bacc"])
        nulls.append(res["null_baccs"])
        chance.append(res["chance"])
    bacc, null = float(np.mean(baccs)), np.mean(nulls, axis=0)
    return bacc, float(np.mean(chance)), float(np.percentile(null, 95)), float((1 + np.sum(null >= bacc)) / (len(null) + 1))


def run_test(test, phase, roi, X, meta, feature, n_perm):
    groups, cells = trial_groups(meta), task_pos_cells(meta)
    cur = feature_at(meta, feature, 0)
    if test == "current":
        ok = ~np.isnan(cur)
        with contextlib.redirect_stdout(io.StringIO()):
            res = ridge_decode_5fold(X[ok], cur[ok].astype(int), groups=groups[ok], alphas=ALPHA_GRID,
                                     nuisance=cells[ok], n_perm=n_perm, random_state=0)
        null = res["null_baccs"]
        bacc, chance, null95 = res["mean_bacc"], res["chance"], float(np.percentile(null, 95))
        p = res["p_value"]
        n = int(ok.sum())
    else:
        nb = feature_at(meta, feature, +1 if test == "next" else -1)
        ok = ~np.isnan(nb) & ~np.isnan(cur)
        bacc, chance, null95, p = decode_stratified(X[ok], nb[ok].astype(int), cur[ok].astype(int),
                                                    groups[ok], cells[ok], n_perm)
        n = int(ok.sum())
    return {"phase": phase, "roi": roi, "feature": feature, "test": test, "n_events": n,
            "bacc": bacc, "chance": chance, "null_95": null95, "p": p}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decode neighbouring-event features to measure GLM temporal leakage")
    ap.add_argument("--sub", default="sub-01")
    ap.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02", "ses-03", "ses-04"])
    ap.add_argument("--glm_type", default="lss", help="model folder prefix: lsa, lss, glmsingle, glmsingleB, glmsingleC, ...")
    ap.add_argument("--glm_variant", default="wfdelay", help="events variant, e.g. wfdelay / wofdelay")
    ap.add_argument("--root", default="/mnt/tempdata/lucas/fmri/recordings/TR", help="dataset root (behav/, neural/)")
    ap.add_argument("--prep_run", default="first_run", help="folder under <root>/neural/fmriprep_outs")
    ap.add_argument("--repeat_mode", default="intersect", choices=["intersect", "average", "single"],
                    help="as in feature_roi_sweep.py")
    ap.add_argument("--min_repeats", type=int, default=1)
    ap.add_argument("--exclude_tasks", nargs="+", default=None, help="task names to leave out, e.g. dms")
    ap.add_argument("--tasks", default=None, help="regex on task dir names, e.g. '1back' (default: pool all)")
    ap.add_argument("--feature", default="loc", choices=["loc", "obj", "ctg"])
    ap.add_argument("--rois", nargs="+", default=["EVC", "VTC"], choices=list(ROI_GROUPS))
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--dlabel", default="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii")
    ap.add_argument("--out_dir", default=None, help="default: <glm_base_dir>/<sub>/_decoding_roi")
    args = ap.parse_args()

    root = args.root
    glm_base_dir = f"{root}/neural/fmriprep_outs/{args.prep_run}/glm_runs/{args.glm_type}_{args.glm_variant}/64kDense"
    cfg = dict(glm_base_dir=glm_base_dir, behav_base=f"{root}/behav", events_subdir=f"events_{args.glm_variant}",
               block_logs_base=f"{root}/behav/{args.sub}", sub=args.sub, sessions=args.sessions,
               dlabel_path=args.dlabel, only_tasks_regex=args.tasks)
    if args.repeat_mode != "intersect" or args.exclude_tasks:
        cfg.update(repeat_mode=args.repeat_mode, min_repeats=args.min_repeats, exclude_tasks=args.exclude_tasks)

    names = {r: bilateral(ROI_GROUPS[r]) for r in args.rois}
    union = sorted({n for v in names.values() for n in v})
    cols = {r: roi_columns(args.dlabel, union, v) for r, v in names.items()}

    jobs = []
    for phase, tests in (("Delay", ["current", "next"]), ("Encoding", ["current", "previous"])):
        with contextlib.redirect_stdout(io.StringIO()):
            X, _, meta, _ = collect_repeat_averaged_across_sessions(
                feature_to_decode=args.feature, phase_to_decode=phase, roi_names=union, verbose=False, **cfg)
        for roi in args.rois:
            for test in tests:
                jobs.append(delayed(run_test)(test, phase, roi, X[:, cols[roi]], meta, args.feature, args.n_perm))
    df = pd.DataFrame(Parallel(n_jobs=args.n_jobs)(jobs))
    df.insert(0, "glm", f"{args.glm_type}_{args.glm_variant}")

    out_dir = args.out_dir or os.path.join(glm_base_dir, args.sub, "_decoding_roi")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.glm_type}_{args.glm_variant}_{args.feature}" + (f"_tasks-{args.tasks}" if args.tasks else "")
    if args.exclude_tasks:
        tag += f"_excl-{'-'.join(args.exclude_tasks)}"
    if args.repeat_mode != "intersect":
        tag += f"_rep-{args.repeat_mode}" + (f"{args.min_repeats}" if args.min_repeats > 1 else "")
    out_tsv = os.path.join(out_dir, f"temporal_leakage_{tag}.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print(f"\n[SAVED] {out_tsv}")
