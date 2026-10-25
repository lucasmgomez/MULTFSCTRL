#!/usr/bin/env python3
"""Is the imported old (Xuan) beta set aligned with our events? (1) per run, correlate each old EVC encoding beta with
our type-B beta of the same event and of events k positions away; (2) in the old design matrix, where does each
encoding regressor peak relative to its logged onset?"""
import glob, sys, numpy as np, pandas as pd, nibabel as nib, h5py
sys.path.insert(0, "/home/lucasg/projects/MULTFSCTRL/analysis/glm/misc")
from feature import make_roi_mask
from feature_roi_sweep import ROI_GROUPS, bilateral
mask, _ = make_roi_mask("/home/lucasg/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii", bilateral(ROI_GROUPS["EVC"]))
G = "/home/lucasg/scratch/CNM/cmp_root/neural/fmriprep_outs/fmriprep_20.2.6/glm_runs"
X = "/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/fmri_dataset_project_data/encoding_delay/trial_level_betas"
imp = pd.read_csv(f"{G}/xuanols_wfdelay/64kDense/sub-01/_import/manifest.tsv", sep="\t")
res = {k: [] for k in range(-3, 4)}
for (ses, td), g in list(imp.groupby(["session", "taskdir"]))[:24]:
    g = g[g.event_id.str.startswith("Enc")].sort_values("event_id")
    load = lambda m, eid: nib.load(glob.glob(f"{G}/{m}_wfdelay/64kDense/sub-01/{ses}/{td}/*_{eid}_beta.dscalar.nii")[0]).get_fdata().squeeze()[mask]
    A = np.stack([load("xuanols", e) for e in g.event_id]); B = np.stack([load("glmsingleB", e) for e in g.event_id])
    A = (A - A.mean(0)); B = (B - B.mean(0))
    for k in res:
        rs = [np.corrcoef(A[i], B[i + k])[0, 1] for i in range(max(0, -k), len(g) - max(0, k))]
        res[k].append(np.mean(rs))
print("mean corr(old beta of event i, our B beta of event i+k), EVC, encoding, 24 runs:")
print({k: round(float(np.mean(v)), 3) for k, v in res.items()})
# design-matrix peaks
f = sorted(glob.glob(f"{X}/sub-01/ses-001/func/*interdms_run-1_design_matrix.csv"))[0]
dm = pd.read_csv(f); des = pd.read_csv(f.replace("_design_matrix", "_design"))
with h5py.File(f.replace("_design_matrix.csv", "_betas.h5")) as h:
    print("attrs:", {k: v for k, v in h.attrs.items() if k != "task_regressor_names"})
task_cols = [c for c in dm.columns if c.startswith("trial")]
for j in range(6):
    col, row = task_cols[j], des.iloc[j]
    pk = int(np.argmax(dm[col].to_numpy()))
    print(f"{col:22s} {row.type:8s} onset {row.onset_time:7.2f}s  regressor peak at row {pk} "
          f"(= {pk * 1.49:.2f}s if row 0 is volume 0; {(pk + 1) * 1.49:.2f}s if the first volume was dropped)")
