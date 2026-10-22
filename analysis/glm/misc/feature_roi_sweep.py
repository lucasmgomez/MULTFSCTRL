#!/usr/bin/env python3
import os
import io
import argparse
import contextlib
import numpy as np
import pandas as pd
import nibabel as nib
from joblib import Parallel, delayed

from feature import collect_repeat_averaged_across_sessions, trial_groups, task_pos_cells, ridge_decode_5fold


# ----------------------------
# ROI groups (bilateral, Glasser 2016 sections)
# ----------------------------
ROI_GROUPS = {
    "DLPFC":       ["8C", "8Av", "i6-8", "s6-8", "SFL", "8BL", "9p", "9a", "8Ad", "p9-46v", "a9-46v", "46", "9-46d"],
    "IFC":         ["44", "45", "IFJp", "IFJa", "IFSp", "IFSa", "47l", "p47r"],
    "FrontalPole": ["10r", "10d", "10v", "a10p", "p10p", "10pp"],
    "OFC":         ["47s", "47m", "a47r", "11l", "13l", "OFC", "pOFC"],
    "EVC":         ["V1", "V2", "V3"],
    "VTC":         ["V8", "VVC", "PIT", "FFC", "VMV1", "VMV2", "VMV3"],
    "IT":          ["TE2p", "TE2a", "TE1p", "TE1m", "TF", "PHT"],
}


def bilateral(names):
    return [f"{h}_{n}_ROI" for h in ("L", "R") for n in names]


def roi_columns(dlabel_path: str, union_names, roi_names):
    """Column indices of roi_names within X built from the union_names mask (vertex order)."""
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    id_to_name = {k: name for k, (name, _) in dl.header.get_axis(0).label[0].items()}
    union_ids = [k for k, n in id_to_name.items() if n in set(union_names)]
    col_names = np.array([id_to_name[k] for k in data[np.isin(data, union_ids)]])
    return np.flatnonzero(np.isin(col_names, roi_names))


def load_one(feature, phase, cfg, union_names):
    with contextlib.redirect_stdout(io.StringIO()):
        X, y, meta_avg, _ = collect_repeat_averaged_across_sessions(
            feature_to_decode=feature, phase_to_decode=phase, roi_names=union_names, verbose=False, **cfg,
        )
    return feature, phase, X, y, meta_avg


# demean: which nuisance cells' mean patterns are removed (None, task x position, or position only)
CONFIGS = {
    "base":             dict(demean=None,       tune=False),
    "demean":           dict(demean="task_pos", tune=False),
    "demean_pos":       dict(demean="pos",      tune=False),
    "tuned":            dict(demean=None,       tune=True),
    "demean+tuned":     dict(demean="task_pos", tune=True),
    "demean_pos+tuned": dict(demean="pos",      tune=True),
}
ALPHA_GRID = np.logspace(-1, 8, 19)


def reduce_roi(X, col_sets, reduce):
    """Decoder input for one ROI: every vertex, the bilateral ROI mean (1 feature), or per-hemisphere means (2)."""
    if reduce == "vertex":
        return X[:, np.concatenate(col_sets)]
    if reduce == "mean":
        return X[:, np.concatenate(col_sets)].mean(axis=1, keepdims=True)
    return np.column_stack([X[:, c].mean(axis=1) for c in col_sets])


def decode_one(feature, phase, roi, roi_group, level, config, X, y, groups, cells, alpha, n_perm, reduce="vertex"):
    opts = CONFIGS[config]
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        res = ridge_decode_5fold(
            X, y, groups=groups, alpha=alpha,
            alphas=ALPHA_GRID if opts["tune"] else None,
            nuisance=cells[opts["demean"]] if opts["demean"] else None,
            n_perm=n_perm, random_state=0,
        )
    null = res.get("null_baccs")
    return {
        "feature": feature, "phase": phase, "roi": roi, "group": roi_group, "level": level, "config": config,
        "reduce": reduce, "n_events": len(y), "n_features": X.shape[1], "n_classes": len(np.unique(y)),
        "chance": res["chance"], "bacc": res["mean_bacc"], "bacc_sd": float(np.std(res["fold_baccs"])),
        "acc": res["mean_acc"], "alpha_median": float(np.median(res["fold_alphas"])),
        "null_mean": float(null.mean()) if null is not None else np.nan,
        "null_95": float(np.percentile(null, 95)) if null is not None else np.nan,
        "p": res.get("p_value", np.nan),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decode stimulus features from LSA/LSS betas across ROI groups")
    ap.add_argument("--sub", default="sub-01")
    ap.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02", "ses-03", "ses-04"])
    ap.add_argument("--glm_type", default="lsa", choices=["lsa", "lss", "glmsingle", "glmsingleB", "glmsingleC"])
    ap.add_argument("--glm_variant", default="wfdelay", choices=["wfdelay", "wofdelay"])
    ap.add_argument("--features", nargs="+", default=["loc", "obj"])
    ap.add_argument("--phases", nargs="+", default=["Encoding", "Delay"])
    ap.add_argument("--rois", nargs="+", default=list(ROI_GROUPS), choices=list(ROI_GROUPS))
    ap.add_argument("--levels", nargs="+", default=["group"], choices=["group", "parcel"],
                    help="decode whole ROI groups and/or each bilateral Glasser parcel of the --rois groups")
    ap.add_argument("--per_parcel", action="store_true", help="shortcut for --levels parcel")
    ap.add_argument("--configs", nargs="+", default=["base"], choices=list(CONFIGS),
                    help="demean: remove task x position mean patterns (train-fold estimate); "
                         "demean_pos: remove position-only mean patterns; "
                         "tuned: pick ridge alpha by nested grouped CV")
    ap.add_argument("--reduce", default="vertex", choices=["vertex", "mean", "hemi_mean"],
                    help="decode from every vertex, the bilateral ROI mean (1 feature) or per-hemisphere means (2)")
    ap.add_argument("--tasks", default=None, help="regex on task dir names, e.g. '1back' (default: pool all)")
    ap.add_argument("--alpha", type=float, default=1.0, help="fixed alpha for untuned configs")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--n_jobs", type=int, default=32)
    ap.add_argument("--dlabel", default="/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii")
    ap.add_argument("--out_dir", default=None, help="default: <glm_base_dir>/<sub>/_decoding_roi")
    args = ap.parse_args()
    if args.per_parcel:
        args.levels = ["parcel"]

    root = "/mnt/tempdata/lucas/fmri/recordings/TR"
    glm_base_dir = f"{root}/neural/fmriprep_outs/first_run/glm_runs/{args.glm_type}_{args.glm_variant}/64kDense"
    cfg = dict(
        glm_base_dir=glm_base_dir,
        behav_base=f"{root}/behav",
        events_subdir=f"events_{args.glm_variant}",
        block_logs_base=f"{root}/behav/{args.sub}",
        sub=args.sub,
        sessions=args.sessions,
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

    # betas are loaded once per (feature, phase) over the ROI union; each ROI is a column subset
    print(f"[INFO] loading {len(args.features) * len(args.phases)} feature x phase datasets ...")
    loaded = Parallel(n_jobs=min(args.n_jobs, len(args.features) * len(args.phases)))(
        delayed(load_one)(f, p, cfg, union_names) for f in args.features for p in args.phases
    )

    jobs = []
    for feature, phase, X, y, meta_avg in loaded:
        print(f"[INFO] {feature}/{phase}: X={X.shape}")
        groups = trial_groups(meta_avg)
        cells = {"task_pos": task_pos_cells(meta_avg), "pos": meta_avg["pos"].astype(str).to_numpy()}
        for (level, roi), (roi_group, _) in rois.items():
            for config in args.configs:
                Xr = reduce_roi(X, col_sets[(level, roi)], args.reduce)
                jobs.append(delayed(decode_one)(feature, phase, roi, roi_group, level, config, Xr, y,
                                                groups, cells, args.alpha, args.n_perm, args.reduce))

    print(f"[INFO] decoding {len(jobs)} feature x phase x ROI x config combinations (n_perm={args.n_perm}) ...")
    df = pd.DataFrame(Parallel(n_jobs=args.n_jobs)(jobs))

    out_dir = args.out_dir or os.path.join(glm_base_dir, args.sub, "_decoding_roi")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.glm_type}_{args.glm_variant}_{'-'.join(args.features)}" + (f"_tasks-{args.tasks}" if args.tasks else "")
    if args.levels != ["group"]:
        tag += f"_{'-'.join(args.levels)}s-{'-'.join(args.rois)}"
    if args.reduce != "vertex":
        tag += f"_reduce-{args.reduce}"
    if args.configs != ["base"]:
        tag += f"_cfg-{'-'.join(c.replace('+', '') for c in args.configs)}"
    out_tsv = os.path.join(out_dir, f"roi_decoding_{tag}.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)

    pd.set_option("display.width", 200)
    df["sig"] = np.where(df["p"] < 0.001, "***", np.where(df["p"] < 0.01, "**", np.where(df["p"] < 0.05, "*", "")))
    df["v"] = df["bacc"].map("{:.3f}".format) + df["sig"]
    for feature in args.features:
        sub_df = df[(df["feature"] == feature) & (df["level"] == "group")]
        if sub_df.empty:
            continue
        print(f"\n=== {feature}: group-level balanced accuracy (chance={sub_df['chance'].iloc[0]:.2f}) ===")
        print(sub_df.pivot(index="roi", columns=["phase", "config"], values="v")
              .reindex([r for lv, r in rois if lv == "group"]).to_string())
    print(f"\n[SAVED] {out_tsv}")
