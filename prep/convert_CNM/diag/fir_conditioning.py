import glob, numpy as np, pandas as pd
dt, nlag = 1.49 / 3, 61
def fir_cond(ev_path, nvol):
    ev = pd.read_csv(ev_path, sep="\t")
    n = (nvol - 1) * 3 + 1
    s = np.zeros(n); s[np.clip(np.rint((ev.onset - 0.686) / dt).astype(int), 0, n - 1)] = 1
    X = np.column_stack([np.r_[np.zeros(k), s[:n - k]] for k in range(nlag)])
    X = X - X.mean(0)
    sv = np.linalg.svd(X, compute_uv=False)
    vif = np.diag(np.linalg.inv(np.corrcoef(X.T)))
    return sv[0] / sv[-1], np.median(vif)
CNM = "/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM/behav/sub-01/ses-01/events_wfdelay"
TR = "/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/TR/behav/sub-01/ses-01/events_wfdelay"
rows = []
for name, d, vols in [("CNM", CNM, {"1back": 188, "ctxdm": 184, "dms": 128, "interdms": 208}),
                      ("TR", TR, {"1back": 206, "ctxdm": 137, "interdms": 93})]:
    for f in sorted(glob.glob(f"{d}/*_base-events.tsv")):
        task = f.split("task-")[1].split("_")[0]
        c, v = fir_cond(f, vols[task])
        rows.append(dict(data=name, run=f.split("/")[-1].replace("_base-events.tsv", ""), cond=c, median_vif=v))
print(pd.DataFrame(rows).round(1).to_string(index=False))
