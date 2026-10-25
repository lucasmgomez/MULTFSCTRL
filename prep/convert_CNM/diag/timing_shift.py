#!/usr/bin/env python3
"""Per-run timing check without the GLM: project each run's EVC time series onto the location pattern (beta
difference loc1 - loc0 from all OTHER runs), correlate with the predicted location signal (encoding events coded
+-1, canonical HRF, frames at i*TR + 0.686 s) for onset shifts of -8..+8 s. A correctly timed run peaks near 0 s.
  python timing_shift.py <sub> [glm_type] [root] [prep_run] [start_time]   (defaults: CNM, fmriprep_20.2.6, 0.686)"""
import os, sys, glob, numpy as np, pandas as pd, nibabel as nib
from nilearn.glm.first_level.hemodynamic_models import glover_hrf
sys.path.insert(0, "/home/lucasg/projects/MULTFSCTRL/analysis/glm/misc")
from feature import make_roi_mask
from feature_roi_sweep import ROI_GROUPS, bilateral
a = sys.argv[1:] + [None] * 5
sub, gt = a[0], a[1] or "glmsingle"
C = a[2] or "/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM"
PR = a[3] or "fmriprep_20.2.6"
G = f"{C}/neural/fmriprep_outs/{PR}/glm_runs/{gt}_wfdelay/64kDense/{sub}"
TR, ST, DT = 1.49, float(a[4] or 0.686), 0.05
mask, _ = make_roi_mask("/home/lucasg/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii", bilateral(ROI_GROUPS["EVC"]))
man = pd.read_csv(f"{G}/_glmsingle/glmsingle_manifest.tsv", sep="\t")
man = man[(man.phase == "Encoding") & (man.task != "dms")]
# location label of each encoding event from tc (loc digit of its position)
NP = {"1back": 6, "interdms": 4, "ctxdm": 3}
man["loc"] = [int(str(int(float(tc))).zfill(2 * NP[t])[2 * (p - 1)]) for tc, t, p in zip(man.tc, man.task, man.pos)]
B = np.stack([nib.load(f"{G}/{s}/{td}/{f}").get_fdata().squeeze()[mask]
              for s, td, f in zip(man.session, man.taskdir, man.beta_dscalar)])
key = (man.session + "|" + man.taskdir).to_numpy()
hrf = glover_hrf(DT, oversampling=1, time_length=32.0)
shifts = np.arange(-8, 8.01, 0.5)
rows = []
for k in pd.unique(key):
    tr_ix, te = key != k, man[key == k]
    lab = man.loc[tr_ix, "loc"].to_numpy(); Bt = B[tr_ix]
    w = Bt[lab == 1].mean(0) - Bt[lab == 0].mean(0)
    ses, td = k.split("|"); task, acq, run = td.split("_acq-")[0][5:], td.split("_acq-")[1].split("_run-")[0], td.split("_run-")[1]
    f = glob.glob(f"{C}/neural/fmriprep_outs/{PR}/64kDense/{sub}/{ses}/{sub}_{ses}_task-{task}_acq-{acq.replace('_', '')}_run-{run}_*dtseries.nii")[0]
    Y = nib.load(f).get_fdata(dtype=np.float32)[:, mask]
    n = len(Y); t = np.arange(n) * TR + ST
    P = np.vander(t, 4); Y = Y - P @ np.linalg.lstsq(P, Y, rcond=None)[0]
    Y /= Y.std(0) + 1e-9
    s = Y @ w
    hi = np.arange(0, n * TR + 40, DT)
    r = []
    for sh in shifts:
        x = np.zeros_like(hi)
        for on, du, lo in zip(te.onset + sh, te.duration, te["loc"]):
            x[(hi >= on) & (hi < on + du)] = 1 if lo else -1
        xp = np.interp(t, hi, np.convolve(x, hrf)[:len(hi)])
        xp = xp - P @ np.linalg.lstsq(P, xp, rcond=None)[0]
        r.append(np.corrcoef(s, xp)[0, 1])
    r = np.array(r)
    rows.append(dict(ses=ses, run=td, r_at_0=r[shifts == 0][0], best_shift=shifts[np.argmax(r)], r_best=r.max()))
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"{sub}: best shift (s) distribution:", d.best_shift.value_counts().sort_index().to_dict())
print("r at 0 s quantiles:", d.r_at_0.quantile([.05, .25, .5, .75]).round(2).to_dict())
print(d.sort_values("r_at_0").head(12).round(2).to_string(index=False))
d.to_csv(f"/home/lucasg/scratch/CNM/diag/timing_shift_{os.path.basename(C)}_{sub}_{gt}.tsv", sep="\t", index=False)
