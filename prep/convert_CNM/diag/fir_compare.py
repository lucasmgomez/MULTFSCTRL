import numpy as np
dt = 1.49 / 3
for name, f in [("TR", "/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/TR/neural/fmriprep_outs/first_run/glm_runs/glmsingle_wfdelay/64kDense/sub-01/_glmsingle/glmsingle_raw/RUNWISEFIR.npy"),
                ("CNM sub-01", "/home/lucasg/scratch/CNM/glmsingle_raw/sub-01_wfdelay/RUNWISEFIR.npy")]:
    d = np.load(f, allow_pickle=True).item()
    R2, tcs = np.asarray(d["firR2"]).squeeze(), np.asarray(d["firtcs"]).squeeze()   # runs x vox, runs x vox x lags
    print(name, "firR2", R2.shape, "firtcs", tcs.shape, "firavg", np.shape(d["firavg"]), "grand", np.shape(d["firgrandavg"]))
    R2mn = np.nanmean(R2, 0)
    for pct in (99, 95):
        ix = R2mn > np.nanpercentile(R2mn, pct)
        g = np.nanmean(tcs[:, ix, :], axis=(0, 1))
        print(f"  top {100-pct}% vox (n={ix.sum()}, mean FIR R2 {R2mn[ix].mean():.2f}): grand peak at {np.argmax(g)*dt:.1f} s;"
              " timecourse (every 2 lags):", np.round(g[::2], 2).tolist())
    ix = R2mn > np.nanpercentile(R2mn, 99)
    per_run = np.nanmean(tcs[:, ix, :], axis=1)
    pk = np.argmax(per_run[:, :int(20 / dt)], axis=1) * dt
    print("  per-run peak time (s): median %.1f, IQR %.1f-%.1f, n runs peaking > 10 s: %d / %d" %
          (np.median(pk), *np.percentile(pk, [25, 75]), (pk > 10).sum(), len(pk)))
    print("  median FIR R2 over vertices %.3f; 99th pct %.2f" % (np.nanmedian(R2mn), np.nanpercentile(R2mn, 99)))
