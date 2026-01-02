import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ========== Config ==========
subject = 'sub-06'
data_root = '/project/def-pbellec/xuan/cneuromod.multfs.fmriprep'
save_root = '/project/def-pbellec/xuan/fmri_dataset_project/results/fd_distribution'
os.makedirs(save_root, exist_ok=True)
fd_threshold = 1.0  # Threshold in mm
# ============================

subject_path = os.path.join(data_root, subject)

# Initialize
all_fd = []
mean_fd_per_run = []
fd_exceeding_runs = {}  # { "ses-001_run-01": [indices of frames > threshold] }

confound_files = glob.glob(os.path.join(subject_path, 'ses-*', 'func', f'{subject}_ses-*_task-*_run-*_desc-confounds_timeseries.tsv'))
print(f"Found {len(confound_files)} confound files.")

for file in sorted(confound_files):
    df = pd.read_csv(file, sep='\t')
    
    if 'framewise_displacement' not in df.columns:
        print(f"No FD in {file}")
        continue
    
    fd = df['framewise_displacement'].fillna(0).values
    all_fd.extend(fd)
    mean_fd_per_run.append(fd.mean())
    
    # Extract session/run name for tracking
    filename = os.path.basename(file)
    session_run = filename.split('_desc-')[0].replace(f"{subject}_", "")  # e.g. ses-001_task-xxx_run-01

    # Check for any frame above threshold
    exceed_indices = list((fd > fd_threshold).nonzero()[0])
    if exceed_indices:
        fd_exceeding_runs[session_run] = exceed_indices

# Convert to Series for plotting
all_fd = pd.Series(all_fd)
mean_fd_per_run = pd.Series(mean_fd_per_run)

# ========== Save histograms ==========
plt.figure(figsize=(10,4))
plt.hist(all_fd, bins=100, color='steelblue')
plt.xlabel('Framewise Displacement (mm)')
plt.ylabel('Number of frames')
plt.title(f'{subject} FD per frame distribution')
plt.tight_layout()
plt.savefig(os.path.join(save_root, f'{subject}_FD_per_frame_hist.png'))
plt.close()

plt.figure(figsize=(10,4))
plt.hist(mean_fd_per_run, bins=50, color='darkorange')
plt.xlabel('Mean FD per run (mm)')
plt.ylabel('Number of runs')
plt.title(f'{subject} Mean FD per run distribution')
plt.tight_layout()
plt.savefig(os.path.join(save_root, f'{subject}_Mean_FD_per_run_hist.png'))
plt.close()

# ========== Print + Save Summary ==========
summary_path = os.path.join(save_root, f'{subject}_FD_summary.txt')
with open(summary_path, 'w') as f:
    f.write(f"Overall mean FD: {all_fd.mean():.4f} mm\n")
    f.write(f"Overall median FD: {all_fd.median():.4f} mm\n")
    f.write(f"Number of frames with FD > 0.5mm: {(all_fd>0.5).sum()} ({(all_fd>0.5).mean()*100:.2f}%)\n")
    f.write(f"\nRuns with any frame FD > {fd_threshold} mm:\n")
    for key, indices in fd_exceeding_runs.items():
        f.write(f"{key}: {len(indices)} frames exceed threshold at indices {indices}\n")

