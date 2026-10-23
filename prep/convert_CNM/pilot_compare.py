#!/usr/bin/env python3
"""
Decision #3 pilot (HANDOFF.md): does modelling CNM's 0.5 s pos-1 delay hurt the encoding betas?

Compares GLMsingle fits of the same runs with two event sets (e.g. wfdelay = pos-1 delay dropped,
wfdelaypos1 = kept), per model type (B, D), on encoding betas at positions 1-3:
  - single-trial decoding of loc and obj (task x position means removed, tuned ridge, grouped by trial);
  - split-half reliability across repeats of the same stimulus (task, pos, loc, obj), per vertex, task means removed;
  - GLMsingle R2 (summary map) in the same ROIs.
dms is excluded (loc and obj are confounded there). Writes one TSV.

  python pilot_compare.py --root <dataset root> --glm_runs <folder with glmsingle*_<variant>/> \
      --variants wfdelay wfdelaypos1 --sessions ses-01 ... --out pilot_compare.tsv
"""
import argparse
import contextlib
import io
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

MISC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../analysis/glm/misc")
sys.path.insert(0, MISC)
from feature import collect_one_session_dict, ridge_decode_5fold, make_roi_mask  # noqa: E402
from feature_roi_sweep import ROI_GROUPS, ALPHA_GRID, bilateral  # noqa: E402
from pattern_reliability import fisher_mean, spearman_brown, vertex_split_r, demean_cells  # noqa: E402
from temporal_leakage import N_POS, DIGIT  # noqa: E402

TYPES = {"B": "glmsingleB", "D": "glmsingle"}


def load(glm_runs, gtype, variant, root, sub, sessions, dlabel, roi):
    """Encoding single-trial betas of one ROI -> X (events x vertices), meta."""
    X, rows = [], []
    for ses in sessions:
        with contextlib.redirect_stdout(io.StringIO()):
            d, _, _ = collect_one_session_dict(
                glm_base_dir=f"{glm_runs}/{TYPES[gtype]}_{variant}/64kDense", behav_base=f"{root}/behav",
                events_subdir=f"events_{variant}", block_logs_dir=f"{root}/behav/{sub}/{ses}", sub=sub, session=ses,
                dlabel_path=dlabel, roi_names=bilateral(ROI_GROUPS[roi]), feature_to_decode="loc",
                phase_to_decode="Encoding", exclude_tasks=["dms"], unique_key=True)
        for k in sorted(d):
            task, acq, run, phase, pos, tc, trial = k
            code = str(tc).zfill(2 * N_POS[task])
            loc, obj = (int(code[2 * (pos - 1) + DIGIT[f]]) for f in ("loc", "obj"))
            X.append(d[k][0])
            rows.append(dict(ses=ses, task=task, acq=acq, pos=pos, tc=tc, loc=loc, obj=obj))
    return np.stack(X).astype(np.float64), pd.DataFrame(rows)


def stim_reliability(X, meta, n_splits=20, seed=0):
    """Split-half r over stimulus conditions (task, loc, obj) at one position; task means removed per half."""
    conds = meta.groupby(["task", "loc", "obj"]).indices
    conds = {c: i for c, i in conds.items() if len(i) >= 2}
    rng = np.random.default_rng(seed)
    task = np.array([c[0] for c in conds])
    rs = []
    for _ in range(n_splits):
        A, B = [], []
        for idx in conds.values():
            p = rng.permutation(idx)
            h = len(p) // 2
            A.append(X[p[:h]].mean(0))
            B.append(X[p[h:2 * h]].mean(0))
        A, B = demean_cells(np.stack(A), task), demean_cells(np.stack(B), task)
        rs.append(fisher_mean(vertex_split_r(A, B)))
    r = fisher_mean(np.array(rs))
    return r, spearman_brown(r), len(conds)


def one(gtype, variant, roi, args):
    X, meta = load(args.glm_runs, gtype, variant, args.root, args.sub, args.sessions, args.dlabel, roi)
    out = []
    for pos in (1, 2, 3):
        sel = (meta.pos == pos).to_numpy()
        Xp, mp = X[sel], meta[sel].reset_index(drop=True)
        groups = (mp.task + "|" + mp.acq + "|" + mp.tc.astype(str)).to_numpy()
        cells = (mp.task + "|" + mp.acq).to_numpy()
        row = dict(type=gtype, variant=variant, roi=roi, pos=pos, n_events=len(mp))
        for feat in ("loc", "obj"):
            with contextlib.redirect_stdout(io.StringIO()):
                res = ridge_decode_5fold(Xp, mp[feat].to_numpy(), groups=groups, alphas=ALPHA_GRID, nuisance=cells,
                                         n_perm=0, random_state=0)
            row[f"bacc_{feat}"] = res["mean_bacc"]
        row["rel_r"], row["rel_sb"], row["rel_n_cond"] = stim_reliability(Xp, mp)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--glm_runs", required=True)
    ap.add_argument("--variants", nargs="+", default=["wfdelay", "wfdelaypos1"])
    ap.add_argument("--types", nargs="+", default=["B", "D"], choices=list(TYPES))
    ap.add_argument("--sub", default="sub-01")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--rois", nargs="+", default=["EVC", "VTC", "IT"])
    ap.add_argument("--dlabel", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "../fmriprep/Glasser_LR_Dense64k.dlabel.nii"))
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = Parallel(n_jobs=args.n_jobs)(delayed(one)(t, v, r, args) for t in args.types for v in args.variants
                                       for r in args.rois)
    df = pd.DataFrame([row for rows in res for row in rows])
    # GLMsingle R2 (type-D fit's summary map) per ROI
    r2 = {}
    for v in args.variants:
        s = nib.load(f"{args.glm_runs}/glmsingle_{v}/64kDense/{args.sub}/_glmsingle/glmsingle_summary.dscalar.nii")
        names = list(s.header.get_axis(0).name)
        R2 = s.get_fdata()[names.index("R2")]
        for roi in args.rois:
            with contextlib.redirect_stdout(io.StringIO()):
                mask, _ = make_roi_mask(args.dlabel, bilateral(ROI_GROUPS[roi]))
            r2[(v, roi)] = float(np.median(R2[mask]))
    df["R2_median_typeD"] = [r2[(v, r)] for v, r in zip(df.variant, df.roi)]
    df.to_csv(args.out, sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
