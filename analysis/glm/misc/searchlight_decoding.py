#!/usr/bin/env python3
"""
Surface searchlight decoding of stimulus features within a set of ROI groups (default DLPFC + IFC).

Same events and decoder as feature_roi_sweep.py's demean+tuned config: task x position means removed (training-fold
estimate), z-scoring, ridge classifier (±1 targets, intercept) with alpha from an inner grouped 5-fold CV on the ±1
squared error, outer 5-fold StratifiedGroupKFold by trial, balanced accuracy averaged over folds. Each centre vertex
decodes from its neighbourhood: mask vertices within --radius mm along the edges of the fsLR 32k midthickness
(template, both hemispheres separately). Searchlights are far smaller than the event count, so the ridge is solved
in primal form (eigendecomposition of X'X); this equals ridge_decode_5fold's kernel form (checked by --check).

Permutation null: --n_perm label shuffles, the same at every vertex, each re-running the alpha choice; per vertex the
null 95th percentile and p = (1 + #null >= real) / (n_perm + 1).

  python searchlight_decoding.py --sub sub-01 --sessions ses-01 ... --root <cnm_root> --prep_run fmriprep_20.2.6 \
      --surf_dir <fsLR_32k dir> --dlabel <Glasser_LR_Dense64k.dlabel.nii> --n_perm 1000 --n_jobs 192
Writes <glm_base_dir>/<sub>/_searchlight/searchlight_<groups>_r<radius>_..._perm<n>.tsv (one row per vertex x feature x
phase).
"""
import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from feature import trial_groups, task_pos_cells, remove_cell_means, ridge_decode_5fold
from feature_roi_sweep import ROI_GROUPS, ALPHA_GRID, bilateral, load_one

N_HEMI = 32492  # fsLR 32k vertices per hemisphere; the 64k dense grid is left then right


def mask_vertices(dlabel_path, names):
    """Dense indices (sorted: the column order of the loaded betas) and Glasser label of every mask vertex."""
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    id_to_name = {k: n for k, (n, _) in dl.header.get_axis(0).label[0].items()}
    ids = [k for k, n in id_to_name.items() if n in set(names)]
    dense = np.flatnonzero(np.isin(data, ids))
    return dense, np.array([id_to_name[k] for k in data[dense]])


def neighbourhoods(dense, surf_dir, radius):
    """Per mask vertex: column indices of the mask vertices within `radius` mm (mesh-edge paths, same hemisphere)."""
    out = [None] * len(dense)
    for h, off in (("L", 0), ("R", N_HEMI)):
        g = nib.load(f"{surf_dir}/tpl-fsLR_den-32k_hemi-{h}_midthickness.surf.gii")
        v, f = g.darrays[0].data.astype(np.float64), g.darrays[1].data
        e = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]]), axis=1), axis=0)
        G = coo_matrix((np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1), (e[:, 0], e[:, 1])), shape=(len(v),) * 2).tocsr()
        cols = np.flatnonzero((dense >= off) & (dense < off + N_HEMI))
        D = dijkstra(G, directed=False, indices=dense[cols] - off, limit=radius)[:, dense[cols] - off]
        for i, c in enumerate(cols):
            out[c] = cols[np.isfinite(D[i])]
    return out


def targets(L, n_classes):
    """±1 targets for label columns L (events x K): events x K (binary) or events x (K * n_classes) (one-vs-rest)."""
    if n_classes == 2:
        return np.where(L == 1, 1.0, -1.0)
    return np.concatenate([np.where(L == c, 1.0, -1.0) for c in range(n_classes)], axis=1)


def decide(S, n_classes, K):
    if n_classes == 2:
        return (S > 0).astype(int)
    return S.reshape(len(S), n_classes, K).argmax(axis=1)


def balanced_acc(L, P, n_classes):
    """Per column balanced accuracy of predictions P against labels L (events x K)."""
    return np.mean([((P == c) & (L == c)).sum(0) / np.maximum((L == c).sum(0), 1) for c in range(n_classes)], axis=0)


def parts(A, B):
    """Primal ridge pieces for train A / test B (centred on A's mean): eigenvalues of A'A, B V and A V."""
    mu = A.mean(axis=0)
    A, B = A - mu, B - mu
    ev, V = np.linalg.eigh(A.T @ A)
    return np.maximum(ev, 0.0), B @ V, A @ V


def searchlight(idx, folds, n_classes, K):
    """Balanced accuracy for each of the K label columns (column 0 = real labels, the rest permutations).
    Ridge scores for alpha a: B V diag(1 / (ev + a)) V'A'Y + mean(Y); A V has centred columns, so V'A'Y needs no centring."""
    bacc = np.zeros(K)
    for Xtr, Xte, Ytr, ybar, Lte, inner in folds:
        A, B = Xtr[:, idx], Xte[:, idx]
        # alpha per label column: inner-CV squared error on the ±1 targets, ties -> stronger penalty
        sse = np.zeros((len(ALPHA_GRID), Ytr.shape[1]))
        for itr, ite, R in inner:
            ev, BV, AV = parts(A[itr], A[ite])
            Z = AV.T @ Ytr[itr]
            Gm, S = BV.T @ BV, BV.T @ R
            for k, a in enumerate(ALPHA_GRID):
                DZ = Z / (ev + a)[:, None]
                sse[k] += (DZ * (Gm @ DZ)).sum(0) - 2 * (DZ * S).sum(0)  # + ||R||^2, the same for every alpha
        if n_classes > 2:
            sse = sse.reshape(len(ALPHA_GRID), n_classes, K).sum(axis=1)
        best = ALPHA_GRID[len(ALPHA_GRID) - 1 - np.argmin(sse[::-1], axis=0)]
        if n_classes > 2:
            best = np.tile(best, n_classes)
        ev, BV, AV = parts(A, B)
        S = BV @ ((AV.T @ Ytr) / (ev[:, None] + best[None, :])) + ybar
        bacc += balanced_acc(Lte, decide(S, n_classes, K), n_classes)
    return bacc / len(folds)


def run_chunk(nbrs, folds, n_classes, K):
    return [searchlight(idx, folds, n_classes, K) for idx in nbrs]


def prepare(X, y, groups, cells, n_perm, seed=0):
    """Outer folds with demeaned, z-scored train/test betas (vertex-wise, so shared by every searchlight) and the
    ±1 target matrices of every label column (real + permutations), which don't depend on the searchlight either."""
    y_enc = LabelEncoder().fit_transform(y)
    n_classes = len(np.unique(y_enc))
    rng = np.random.default_rng(seed)
    Lall = np.column_stack([y_enc] + [rng.permutation(y_enc) for _ in range(n_perm)])
    cv = lambda: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    folds = []
    for tr, te in cv().split(X, y_enc, groups):
        Xtr, Xte = remove_cell_means(X[tr], X[te], cells[tr], cells[te])
        sc = StandardScaler().fit(Xtr)
        Ytr = targets(Lall[tr], n_classes)
        inner = []
        for itr, ite in cv().split(Xtr, y_enc[tr], groups[tr]):
            inner.append((itr, ite, targets(Lall[tr][ite], n_classes) - Ytr[itr].mean(axis=0)))
        folds.append((sc.transform(Xtr), sc.transform(Xte), Ytr, Ytr.mean(axis=0), Lall[te], inner))
    return folds, n_classes, Lall.shape[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--glm_type", default="glmsingle")
    ap.add_argument("--glm_variant", default="wfdelay")
    ap.add_argument("--root", required=True, help="dataset root (behav/, neural/)")
    ap.add_argument("--prep_run", default="fmriprep_20.2.6")
    ap.add_argument("--dlabel", required=True)
    ap.add_argument("--surf_dir", required=True, help="folder with tpl-fsLR_den-32k_hemi-{L,R}_midthickness.surf.gii")
    ap.add_argument("--rois", nargs="+", default=["DLPFC", "IFC"], choices=list(ROI_GROUPS))
    ap.add_argument("--radius", type=float, default=8.0, help="searchlight radius in mm")
    ap.add_argument("--features", nargs="+", default=["loc", "ctg", "obj"])
    ap.add_argument("--phases", nargs="+", default=["Encoding", "Delay"])
    ap.add_argument("--exclude_tasks", nargs="+", default=["dms"])
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--max_centres", type=int, default=0, help="testing: only the first N centres")
    ap.add_argument("--check", type=int, default=0, help="also run ridge_decode_5fold on N centres and compare")
    ap.add_argument("--out_dir", default=None, help="default: <glm_base_dir>/<sub>/_searchlight")
    args = ap.parse_args()

    glm_base_dir = f"{args.root}/neural/fmriprep_outs/{args.prep_run}/glm_runs/{args.glm_type}_{args.glm_variant}/64kDense"
    cfg = dict(glm_base_dir=glm_base_dir, behav_base=f"{args.root}/behav", events_subdir=f"events_{args.glm_variant}",
               block_logs_base=f"{args.root}/behav/{args.sub}", sub=args.sub, sessions=args.sessions,
               dlabel_path=args.dlabel, only_tasks_regex=None,
               repeat_mode="single", min_repeats=1, exclude_tasks=args.exclude_tasks)
    names = bilateral([p for g in args.rois for p in ROI_GROUPS[g]])
    dense, parcel = mask_vertices(args.dlabel, names)
    nbr = neighbourhoods(dense, args.surf_dir, args.radius)
    group_of = {f"{h}_{p}_ROI": g for g in args.rois for p in ROI_GROUPS[g] for h in ("L", "R")}
    centres = np.arange(len(dense))[: args.max_centres or None]
    sizes = np.array([len(n) for n in nbr])
    print(f"[INFO] {len(dense)} mask vertices; searchlight size median {np.median(sizes):.0f} "
          f"(5-95%: {np.percentile(sizes, 5):.0f}-{np.percentile(sizes, 95):.0f})", flush=True)

    print(f"[INFO] loading {len(args.features) * len(args.phases)} feature x phase datasets ...", flush=True)
    loaded = Parallel(n_jobs=min(args.n_jobs, len(args.features) * len(args.phases)))(
        delayed(load_one)(f, p, cfg, names) for f in args.features for p in args.phases)
    rows = []
    for feature, phase, X, y, meta in loaded:
        assert X.shape[1] == len(dense), (X.shape, len(dense))
        n_bad = int((~np.isfinite(X)).any(axis=0).sum())
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        groups, cells = trial_groups(meta), task_pos_cells(meta)
        folds, n_classes, K = prepare(X, y, groups, cells, args.n_perm)
        print(f"[INFO] {feature}/{phase}: {X.shape[0]} events, {n_classes} classes, {n_bad} non-finite vertices "
              f"set to 0; {len(centres)} centres x {args.n_perm} perms", flush=True)
        chunks = np.array_split(centres, max(1, min(len(centres), args.n_jobs * 8)))
        res = Parallel(n_jobs=args.n_jobs)(
            delayed(run_chunk)([nbr[c] for c in cs], folds, n_classes, K) for cs in chunks)
        B = np.vstack([b for r in res for b in r])
        real, null = B[:, 0], B[:, 1:]
        for i, c in enumerate(centres):
            d = dense[c]
            rows.append({"feature": feature, "phase": phase, "dense": int(d), "hemi": "L" if d < N_HEMI else "R",
                         "vertex": int(d % N_HEMI), "parcel": parcel[c][2:-4], "roi": group_of[parcel[c]],
                         "n_vertices": int(sizes[c]), "bacc": float(real[i]), "chance": 1 / n_classes,
                         "null_95": float(np.percentile(null[i], 95)) if args.n_perm else np.nan,
                         "p": float((1 + (null[i] >= real[i]).sum()) / (args.n_perm + 1)) if args.n_perm else np.nan})
        for c in centres[: args.check]:
            ref = ridge_decode_5fold(X[:, nbr[c]], y, groups=groups, alphas=ALPHA_GRID, nuisance=cells)["mean_bacc"]
            print(f"[CHECK] centre {c}: searchlight {real[list(centres).index(c)]:.6f} vs ridge_decode_5fold {ref:.6f}")

    out_dir = args.out_dir or os.path.join(glm_base_dir, args.sub, "_searchlight")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{'-'.join(args.rois)}_r{args.radius:g}_{args.glm_type}_{args.glm_variant}_{'-'.join(args.features)}"
    if args.exclude_tasks:
        tag += f"_excl-{'-'.join(args.exclude_tasks)}"
    tag += "_rep-single_cfg-demeantuned" + (f"_perm{args.n_perm}" if args.n_perm else "")
    if args.max_centres:
        tag += f"_first{args.max_centres}"
    out = os.path.join(out_dir, f"searchlight_{tag}.tsv")
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"[SAVED] {out}")
    print(df.groupby(["feature", "phase", "roi"])[["bacc", "null_95"]].agg(["median", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
