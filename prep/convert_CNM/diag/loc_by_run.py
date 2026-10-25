#!/usr/bin/env python3
"""Positive-control breakdown: cross-validated EVC location decoding (single trials, encoding) per task, position,
session and run. A timing/label problem confined to some runs, sessions or tasks shows up as accuracy dropping there.
  python loc_by_run.py <sub> <glm_type> [variant] [roi]"""
import io, contextlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/lucasg/projects/MULTFSCTRL/analysis/glm/misc")
from feature import collect_one_session_dict
from feature_roi_sweep import ROI_GROUPS, ALPHA_GRID, bilateral
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sub, gt = sys.argv[1], sys.argv[2]
var = sys.argv[3] if len(sys.argv) > 3 else "wfdelay"
roi = sys.argv[4] if len(sys.argv) > 4 else "EVC"
R = "/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM"
X, rows = [], []
for s in range(1, 17):
    ses = f"ses-{s:02d}"
    with contextlib.redirect_stdout(io.StringIO()):
        d, meta, _ = collect_one_session_dict(
            glm_base_dir=f"{R}/neural/fmriprep_outs/fmriprep_20.2.6/glm_runs/{gt}_{var}/64kDense", behav_base=f"{R}/behav",
            events_subdir=f"events_{var}", block_logs_dir=f"{R}/behav/{sub}/{ses}", sub=sub, session=ses,
            dlabel_path="/home/lucasg/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii",
            roi_names=bilateral(ROI_GROUPS[roi]), feature_to_decode="loc", phase_to_decode="Encoding",
            exclude_tasks=["dms"], unique_key=True)
    for k in sorted(d):
        X.append(d[k][0]); rows.append(dict(ses=ses, task=k[0], acq=k[1], run=k[2], pos=k[4], tc=k[5], y=int(d[k][1])))
X, m = np.stack(X).astype(np.float64), pd.DataFrame(rows)
groups = (m.task + "|" + m.acq + "|" + m.tc.astype(str)).to_numpy()
cells = (m.task + "|" + m.acq + "|" + m.pos.astype(str)).to_numpy()
pred = np.zeros(len(m), int)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, m.y, groups):
    Xtr, Xte = X[tr].copy(), X[te].copy()
    for c in np.unique(cells):          # remove task x position means (training estimate), as demean+tuned
        mu = Xtr[cells[tr] == c].mean(0) if (cells[tr] == c).any() else Xtr.mean(0)
        Xtr[cells[tr] == c] -= mu; Xte[cells[te] == c] -= mu
    sc = StandardScaler().fit(Xtr)
    clf = RidgeClassifierCV(alphas=ALPHA_GRID).fit(sc.transform(Xtr), m.y[tr])
    pred[te] = clf.predict(sc.transform(Xte))
m["ok"] = pred == m.y
pd.set_option("display.width", 200)
print(f"{sub} {gt}_{var} {roi} loc encoding, overall accuracy {m.ok.mean():.3f} (n={len(m)})")
print(m.groupby("task").ok.mean().round(3).to_dict())
print("by pos x task:\n", m.pivot_table(index="task", columns="pos", values="ok").round(2).to_string())
print("by session:", m.groupby("ses").ok.mean().round(2).to_dict())
r = m.groupby(["ses", "task", "acq", "run"]).ok.agg(["mean", "size"]).reset_index()
print("worst runs:\n", r.sort_values("mean").head(10).round(2).to_string(index=False))
print("run accuracy quantiles:", r["mean"].quantile([0.05, 0.25, 0.5, 0.75]).round(2).to_dict())
