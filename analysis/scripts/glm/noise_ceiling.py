
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# === Load saved betas dictionary ===
subject = "sub-01"
input_path = f"/project/def-pbellec/xuan/fmri_dataset_project/data/betas/grouped_betas/task_relevant_only/{subject}_task_condition_betas.pkl"
with open(input_path, "rb") as f:
    condition_betas = pickle.load(f)

# === Initialize output dictionary ===
voxelwise_signal_var = []
voxelwise_noise_var = []

# === Process all conditions ===
for condition, betas_list in condition_betas.items():
    n_reps = len(betas_list)

    if n_reps < 2:
        print(f"⚠️ Skipping {condition} due to insufficient repetitions ({n_reps})")
        continue

    betas_array = np.stack(betas_list)  # shape: (n_reps, n_voxels)
    print("shape of betas_array:", betas_array.shape)
    # === Z-score across conditions for each repetition (standardize each repetition across images)
    # Here, "images" are conditions, and standardization is per voxel across conditions
    # First, transpose to (n_voxels, n_reps) for zscoring per voxel
    means = np.mean(betas_array, axis=0, keepdims=True)
    stds = np.std(betas_array, axis=0, ddof=1, keepdims=True)
    stds[stds == 0] = 1  # Avoid division by zero
    betas_zscored = (betas_array - means) / stds

    # === Calculate noise variance per voxel
    # Variance across repetitions, normalized by (n-1) for unbiased estimate
    noise_var = np.var(betas_zscored, axis=0, ddof=1)
    voxelwise_noise_var.append(noise_var)

    # === Calculate signal variance per voxel
    # Since standardized variance is 1, signal variance = 1 - noise variance
    signal_var = 1 - noise_var
    signal_var[signal_var < 0] = 0  # Half-wave rectification
    voxelwise_signal_var.append(signal_var)

# === Aggregate across all conditions (average) ===
voxelwise_noise_var = np.array(voxelwise_noise_var)
voxelwise_signal_var = np.array(voxelwise_signal_var)

mean_signal_var = np.mean(voxelwise_signal_var, axis=0)
mean_noise_var = np.mean(voxelwise_noise_var, axis=0)

# === Calculate ncsnr per voxel ===
ncsnr = np.sqrt(mean_signal_var) / np.sqrt(mean_noise_var)

# === Number of repetitions (assuming same for all conditions processed) ===
n = n_reps

# === Calculate noise ceiling per voxel ===
noise_ceiling = (100 * ncsnr ** 2) / (ncsnr ** 2 + (1 / n))

# === Save results ===
output_path = f"/project/def-pbellec/xuan/fmri_dataset_project/results/noise_ceiling/things_way/{subject}_voxelwise_noise_ceiling.pkl"
with open(output_path, "wb") as f:
    pickle.dump(noise_ceiling, f)

print(f"\n✅ Noise ceiling calculation done. Saved to: {output_path}")

# === Print summary ===
print(f"Noise ceiling shape: {noise_ceiling.shape}")
print(f"Mean noise ceiling: {np.nanmean(noise_ceiling):.3f}")


# === Plot distribution of noise ceilings ===
plt.figure(figsize=(8, 5))
plt.hist(noise_ceiling, bins=50, edgecolor='black')
plt.xlabel("Noise Ceiling (%)")
plt.ylabel("Number of Voxels (Log scale)")
plt.yscale('log')
plt.xscale('linear')
plt.title("Distribution of Voxel-wise Noise Ceilings")
plt.tight_layout()

# Save plot
plot_path = f"/project/def-pbellec/xuan/fmri_dataset_project/results/noise_ceiling/things_way/{subject}_voxelwise_noise_ceiling_distribution.png"
plt.savefig(plot_path)
plt.close()

print(f"✅ Saved noise ceiling distribution plot to: {plot_path}")



###### below is the code with stimulus type as task conditions.
# import os
# import glob
# import h5py
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import nibabel as nib
# import matplotlib.pyplot as plt

# # ✅ ======== Step 1. File paths ===========

# subj = "sub-03" 
# data_dir = f"/mnt/tempdata/Project_fMRI_analysis_data/data/sub-03/glm_betas/sub-03/glm_betas_encoding_delay_full_TR_betas/{subj}"

# h5_files = sorted(glob.glob(os.path.join(data_dir, "glmmethod1*.h5")))
# csv_files = sorted(glob.glob(os.path.join(data_dir, "glmmethod1*.csv")))

# assert len(h5_files) == len(csv_files), "Mismatch between h5 and csv files."

# print(f"Found {len(h5_files)} h5 files and {len(csv_files)} csv files for {subj}.")

# # ✅ ======== Step 2. Load all betas and metadata ===========

# all_betas = []
# all_stimuli = []

# for h5_file, csv_file in tqdm(zip(h5_files, csv_files), total=len(h5_files)):
#     with h5py.File(h5_file, 'r') as f:
#         betas = f['betas'][:]  # shape: (n_voxels, n_regressors), n_voxels = 64984

#     df = pd.read_csv(csv_file)
    
#     # Check consistency
#     assert betas.shape[1] == df.shape[0], f"Regressor mismatch in {h5_file}"
    
#     all_betas.append(betas)  # list of arrays
#     all_stimuli.append(df['stimulus'].values)

# # ✅ ======== Step 3. Concatenate across all runs/sessions ===========

# # betas: list of (n_voxels, n_regressors) → (n_voxels, n_total_trials)
# betas_all = np.concatenate(all_betas, axis=1)  # (n_voxels, n_total_trials)
# stimuli_all = np.concatenate(all_stimuli)      # (n_total_trials,)

# print("Combined betas shape:", betas_all.shape) # (64984, 9120)
# print("Total number of trials:", stimuli_all.shape[0]) # (9120)

# # ✅ ======== Step 4. Organize betas per stimulus with variable repetitions ===========

# # Get unique stimuli and repetition counts
# unique_stimuli, counts = np.unique(stimuli_all, return_counts=True)
# print("Number of unique stimuli:", len(unique_stimuli))

# n_voxels = betas_all.shape[0]
# stimulus_beta_dict = {}

# for stim in unique_stimuli:
#     inds = np.where(stimuli_all == stim)[0]
#     if len(inds) >= 2:  # need at least 2 repetitions to calculate variance
#         stimulus_beta_dict[stim] = betas_all[:, inds].T  # shape: (n_reps, n_voxels)

# print(f"Number of stimuli with >=2 repetitions:", len(stimulus_beta_dict))

# # ✅ ======== Step 5. Calculate noise ceiling per voxel (variable n) ===========

# noise_ceilings = []

# for v in tqdm(range(n_voxels)):
#     nc_per_stim = []

#     for stim, beta_arr in stimulus_beta_dict.items():
#         n_reps = beta_arr.shape[0]
#         if n_reps < 2:
#             continue  # skip if only 1 repetition, cannot calculate variance

#         # Extract beta values for this voxel and stimulus
#         beta_vals = beta_arr[:, v]  # (n_reps,)

#         # Z-score across repetitions
#         beta_z = (beta_vals - np.mean(beta_vals)) / np.std(beta_vals) if np.std(beta_vals) != 0 else np.zeros_like(beta_vals)

#         # Noise variance: variance across repetitions for this stimulus
#         var_noise = np.var(beta_z, ddof=1)
#         print(f"var_noise: {var_noise:.4f} for stimulus '{stim}' with {n_reps} repetitions")

#         # Signal variance
#         signal_var = max(0, 1 - var_noise)

#         # ncsnr and noise ceiling for this stimulus
#         if var_noise == 0:
#             ncsnr = 0
#         else:
#             ncsnr = np.sqrt(signal_var) / np.sqrt(var_noise)

#         nc = (100 * ncsnr ** 2) / (ncsnr ** 2 + (1 / n_reps))
#         print(f"nc: {nc:.4f} for stimulus '{stim}' with {n_reps} repetitions")
#         nc_per_stim.append(nc)

#     # Average noise ceiling across stimuli for this voxel
#     if len(nc_per_stim) > 0:
#         voxel_nc = np.mean(nc_per_stim)
#     else:
#         voxel_nc = 0

#     noise_ceilings.append(voxel_nc)

# noise_ceilings = np.array(noise_ceilings)  # (n_voxels,)

# print("Noise ceiling calculation complete with variable repetitions.")
# print("Noise ceiling shape:", noise_ceilings.shape)




# # ✅ ======== Step 5.5. Plot distribution of noise ceilings ===========

# plt.figure(figsize=(8,5))
# plt.hist(noise_ceilings, bins=100, color='skyblue', edgecolor='k')
# plt.xlabel('Noise ceiling (%)')
# plt.ylabel('Number of voxels')
# plt.title(f'Distribution of noise ceilings for {subj}')
# plt.grid(True)

# # Show mean and median
# mean_nc = np.mean(noise_ceilings)
# median_nc = np.median(noise_ceilings)
# plt.axvline(mean_nc, color='r', linestyle='dashed', linewidth=1.5, label=f"Mean: {mean_nc:.2f}%")
# plt.axvline(median_nc, color='g', linestyle='dashed', linewidth=1.5, label=f"Median: {median_nc:.2f}%")
# plt.legend()

# plt.tight_layout()

# # ✅ Save figure
# output_fig_dir = "/home/xiaoxuan/projects/Project_fMRI_flatmaps/results"
# os.makedirs(output_fig_dir, exist_ok=True)  # Create directory if not exists

# fig_path = os.path.join(output_fig_dir, f"{subj}_noise_ceiling_distribution.png")
# plt.savefig(fig_path, dpi=300)
# print(f"✅ Saved figure to {fig_path}")







# # # ✅ ======== Step 6. Save output for TkSurfer visualization ===========

# # # Split into left and right hemispheres (assuming standard fs_LR ordering)
# # nc_lh = noise_ceilings[:32492]
# # nc_rh = noise_ceilings[32492:]

# # # Cast to float32
# # nc_lh = nc_lh.astype(np.float32)
# # nc_rh = nc_rh.astype(np.float32)

# # # Create GIFTI images for each hemisphere
# # gii_lh = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(nc_lh)])
# # gii_rh = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(nc_rh)])

# # # Save
# # output_dir = f'/mnt/tempdata/xiaoxuan/Project_fmri_flatmap/freesurfer_data/{subj}'
# # os.makedirs(output_dir, exist_ok=True)

# # nib.save(gii_lh, os.path.join(output_dir, 'lh.nc.func.gii'))
# # nib.save(gii_rh, os.path.join(output_dir, 'rh.nc.func.gii'))

# # print(f"Saved GIFTI files to {output_dir}")