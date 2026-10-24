import glob, numpy as np, pandas as pd, nibabel as nib
from collections import Counter
dl = nib.load("/home/lucasg/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii")
lab = dl.get_fdata().squeeze().astype(int); names = {k: v[0] for k, v in dl.header.get_axis(0).label[0].items()}
def region(ix): return Counter(names[lab[i]] for i in ix).most_common(8)
out = {}
for name, d in [("CNM", "/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM/neural/fmriprep_outs/fmriprep_20.2.6/64kDense/sub-01"),
                ("TR", "/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/TR/neural/fmriprep_outs/first_run/64kDense/sub-01")]:
    files = sorted(glob.glob(f"{d}/ses-*/*dtseries.nii"))
    nz = []
    for f in files:
        Y = nib.load(f).get_fdata(dtype=np.float32)
        nz.append(Y.std(0) > 0)
    nz = np.array(nz); valid = nz.all(0); out[name] = valid
    common = nz.sum(0) >= len(files) * 0.5          # vertices with signal in most runs
    lost = common & ~valid
    print(f"{name}: {len(files)} runs; valid in all runs {valid.sum()}; with signal in >= half the runs {common.sum()}; "
          f"lost to some run {lost.sum()}")
    per_run = (common & ~nz).sum(1)
    bad = np.argsort(per_run)[::-1][:8]
    for b in bad:
        if per_run[b]:
            print(f"   {per_run[b]:5d} missing in {files[b].split('/')[-1]}")
    print("   lost vertices by parcel:", region(np.flatnonzero(lost)))
    # where are the top-FIR vertices
    raw = {"CNM": "/home/lucasg/scratch/CNM/glmsingle_raw/sub-01_wfdelay/RUNWISEFIR.npy",
           "TR": "/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/TR/neural/fmriprep_outs/first_run/glm_runs/glmsingle_wfdelay/64kDense/sub-01/_glmsingle/glmsingle_raw/RUNWISEFIR.npy"}[name]
    R2 = np.asarray(np.load(raw, allow_pickle=True).item()["firR2"]).squeeze()
    R2mn = np.nanmean(R2, 0)
    vidx = np.flatnonzero(valid)
    assert len(vidx) == len(R2mn), (len(vidx), len(R2mn))
    top = vidx[R2mn > np.nanpercentile(R2mn, 99)]
    print("   top-1% FIR-R2 vertices by parcel:", region(top))

import sys; sys.path.insert(0, "/home/lucasg/projects/MULTFSCTRL/analysis/glm/misc")
from feature_roi_sweep import ROI_GROUPS, bilateral
lost = ~out["CNM"] & out["TR"]
for g, ps in ROI_GROUPS.items():
    m = np.isin(lab, [k for k, v in names.items() if v in bilateral(ps)])
    print(f"   {g:12s} vertices {m.sum():5d}  dropped in CNM sub-01 {int((m & lost).sum())}")
