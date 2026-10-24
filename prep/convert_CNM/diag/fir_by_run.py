import numpy as np, pandas as pd
dt = 1.49 / 3
d = np.load("/home/lucasg/scratch/CNM/glmsingle_raw/sub-01_wfdelay/RUNWISEFIR.npy", allow_pickle=True).item()
R2, tcs = np.asarray(d["firR2"]).squeeze(), np.asarray(d["firtcs"]).squeeze()
man = pd.read_csv("/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM/neural/fmriprep_outs/fmriprep_20.2.6/glm_runs/glmsingle_wfdelay/64kDense/sub-01/_glmsingle/glmsingle_manifest.tsv", sep="\t")
runs = man.groupby("run_index")[["session", "taskdir", "task"]].first()
ix = np.nanmean(R2, 0) > np.nanpercentile(np.nanmean(R2, 0), 99)
pr = np.nanmean(tcs[:, ix, :], axis=1)
runs["peak_s"] = np.argmax(pr[:, :int(20 / dt)], axis=1) * dt
runs["amp_4s"] = pr[:, int(round(4 / dt))]
runs["amp_12s"] = pr[:, int(round(12 / dt))]
runs["firR2"] = np.nanmean(R2[:, ix], 1)
pd.set_option("display.width", 200)
print(runs.groupby("task")[["peak_s", "amp_4s", "amp_12s", "firR2"]].median().round(2))
print((runs.assign(late=runs.peak_s > 8).groupby("task").late.agg(["sum", "size"])))
for t in runs.task.unique():
    print(t, "mean FIR (every 2 lags):", np.round(pr[runs.task.to_numpy() == t].mean(0)[::2], 2).tolist())
print(runs.groupby("session").peak_s.median().round(1).to_dict())
