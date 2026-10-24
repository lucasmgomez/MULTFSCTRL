import glob, sys, numpy as np, nibabel as nib
for sub in sys.argv[1:]:
    files = sorted(glob.glob(f"/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM/neural/fmriprep_outs/fmriprep_20.2.6/64kDense/{sub}/ses-*/*dtseries.nii"))
    nz = np.array([nib.load(f).get_fdata(dtype=np.float32).std(0) > 0 for f in files])
    common = nz.sum(0) >= len(files) / 2
    per_run = (common & ~nz).sum(1)
    print(f"{sub}: fitted vertices {nz.all(0).sum()} of {common.sum()} with signal in most runs; runs missing > 10:",
          [(files[i].split('/')[-1].split('_space')[0], int(per_run[i])) for i in np.flatnonzero(per_run > 10)])
