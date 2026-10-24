import glob, sys, numpy as np, nibabel as nib
sys.path.insert(0, "/home/lucasg/projects/MULTFSCTRL/analysis/glm/misc")
from feature_roi_sweep import ROI_GROUPS, bilateral
dl = nib.load("/home/lucasg/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii")
lab = dl.get_fdata().squeeze().astype(int); names = {k: v[0] for k, v in dl.header.get_axis(0).label[0].items()}
for sub in sys.argv[1:]:
    files = sorted(glob.glob(f"/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM/neural/fmriprep_outs/fmriprep_20.2.6/64kDense/{sub}/ses-*/*dtseries.nii"))
    nz = np.array([nib.load(f).get_fdata(dtype=np.float32).std(0) > 0 for f in files])
    lost = (nz.sum(0) >= len(files) / 2) & ~nz.all(0)
    worst = np.argmax((lost & ~nz).sum(1))
    lost_wo = (nz.sum(0) >= len(files) / 2) & ~np.delete(nz, worst, 0).all(0)
    row = []
    for g, ps in ROI_GROUPS.items():
        m = np.isin(lab, [k for k, v in names.items() if v in bilateral(ps)])
        row.append(f"{g} {int((m & lost).sum())}/{m.sum()}")
    print(sub, "lost per ROI:", ", ".join(row), f"| total lost {lost.sum()}, without worst run {lost_wo.sum()}")
