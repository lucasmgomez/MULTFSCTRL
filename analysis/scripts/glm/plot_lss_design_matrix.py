import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn.glm.first_level import make_first_level_design_matrix


TR = 1.49

# ---- paths ----
events_tsv = "/mnt/tempdata/lucas/fmri/recordings/TR/behav/sub-01/ses-1/events/task-ctxdm_col_run-01_LSS/task-ctxdm_col_run-01_lss-EncTarget_Enc0001_events.tsv"   # one LSS events file
confounds_tsv = "/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/confounds/sub-01/ses-01/sub-01_ses-01_task-ctxdm_acq-col_run-01_desc-confounds_timeseries.tsv"

# ---- settings (match your GLM) ----
hrf_model = "glover"
high_pass = 0.01  # Hz
confound_cols = ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z"]  # add FD if you used it
add_fd = True


# ---------- load ----------
events = pd.read_csv(events_tsv, sep="\t")[["onset","duration","trial_type"]]
conf = pd.read_csv(confounds_tsv, sep="\t")

cols = confound_cols.copy()
if add_fd and "framewise_displacement" in conf.columns:
    cols.append("framewise_displacement")

confounds = conf[cols].fillna(0.0)

n_scans = confounds.shape[0]
frame_times = np.arange(n_scans) * TR

# ---------- build design matrix ----------
X = make_first_level_design_matrix(
    frame_times=frame_times,
    events=events,
    hrf_model=hrf_model,
    drift_model="cosine",
    high_pass=high_pass,
    add_regs=confounds.to_numpy(),
    add_reg_names=cols,
)

print("Design matrix shape:", X.shape)
print("Columns:", list(X.columns))

t = frame_times

# ---------- pick columns ----------
# Task regressors for LSS (covers both encoding-target and delay-target variants)
task_cols = [c for c in X.columns if any(k in c for k in ["EncTarget","EncOther","EncAll","DelayTarget","DelayOther","DelayAll"])]

# Confounds you added
conf_cols = [c for c in X.columns if c in cols]

# Drift terms (cosine / constant)
drift_cols = [c for c in X.columns if "cosine" in c.lower() or c.lower() == "constant"]


# ---------- plot 1: task regressors ----------
plt.figure()
for c in task_cols:
    plt.plot(t, X[c].values, label=c)
plt.xlabel("Time (s)")
plt.ylabel("Regressor value (HRF-convolved)")
plt.title("LSS task regressors over time")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# ---------- plot 2: confounds ----------
plt.figure()
for c in conf_cols:
    plt.plot(t, X[c].values, label=c)
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("Confounds over time")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# ---------- plot 3: heatmap of full design matrix ----------
# (standardize columns just for visualization, so big confounds don't dominate)
Xv = X.values.copy()
Xv = (Xv - Xv.mean(axis=0, keepdims=True)) / (Xv.std(axis=0, keepdims=True) + 1e-8)

plt.figure(figsize=(10, 6))
plt.imshow(Xv, aspect="auto", interpolation="nearest")
plt.yticks(np.arange(0, n_scans, max(1, n_scans // 10)), [f"{t[i]:.0f}" for i in range(0, n_scans, max(1, n_scans // 10))])
plt.xticks(np.arange(X.shape[1]), X.columns, rotation=90, fontsize=7)
plt.xlabel("Regressors")
plt.ylabel("Time (s)")
plt.title("Design matrix (z-scored columns for display)")
plt.tight_layout()
plt.show()

# ---------- optional: show event timing (sticks) ----------
# shows onsets of each trial_type as vertical lines
plt.figure()
for tt in sorted(events["trial_type"].unique()):
    onsets = events.loc[events["trial_type"] == tt, "onset"].values
    for o in onsets:
        plt.axvline(o, linewidth=0.8, label=tt)
# prevent legend duplicates
handles, labels = plt.gca().get_legend_handles_labels()
uniq = {}
for h, l in zip(handles, labels):
    if l not in uniq:
        uniq[l] = h
plt.legend(uniq.values(), uniq.keys(), fontsize=8)
plt.xlabel("Time (s)")
plt.title("Event onsets by trial_type (sticks)")
plt.tight_layout()
plt.show()