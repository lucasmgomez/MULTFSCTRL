# MULTFSCTRL

Single-subject (`sub-01`) multi-task working-memory fMRI study. Pipeline: DICOM → BIDS → fMRIPrep →
Glasser-64k CIFTI → single-trial GLMs → ROI decoding / model-to-brain prediction.

State as of 2026-10-22. Branch `ctrl`.

## Repo copies
- This checkout: `/home/lucas/projects/trash/MULTFSCTRL`. The GLMsingle and decoding work lives here
  (commit `180e2f6`).
- Main checkout with the GitHub remote: `/home/lucas/projects/MULTFSCTRL`. Many shell scripts and
  defaults hard-code paths into it, e.g. `runs/run_lsa.sh` and `--dlabel` →
  `/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii` (the same file is in
  this repo at `prep/fmriprep/`). `runs/run_glmsingle.sh` resolves the script relative to itself.
- `.gitignore` ignores `*.tsv`; result tables are not committed.

## Environment
- Use `/home/lucas/miniconda3/envs/fmri_glm/bin/python`. It has nilearn, nibabel, sklearn, joblib and
  GLMsingle 1.2. The base python has no sklearn.
- The machine (`mirzakhani`, 128 cores) is shared with other users. Cap BLAS threads
  (`OMP_NUM_THREADS` etc.), and use 1 thread per worker inside joblib sweeps.
- Run long jobs in tmux (`runs/*.sh` spawn tmux windows) or with `setsid nohup`, so they survive the
  session ending.

## Layout
```
prep/
  dicom2bids/   map_dcms.py, cnmod/ (heudiconv conversion, defacing, fmap fixes)
  fmriprep/     docker_run.sh, cifti_resample.sh (→ Glasser64k), cifti_parcellate.sh, confounds.sh,
                Glasser_LR_Dense64k.dlabel.nii, Glasser_2016_Table.xlsx
  behav/        score_responses(.py|_ctrl.py) (→ *_scored.tsv), get_accuracy.py, map_events.py
analysis/
  task_search/  stimulus/trial generation via iWISDM (/home/lucas/projects/iWISDM), trial selection
  glm/
    create_events_lsa.py / create_events_lss.py   block logs → base-events + LSA/LSS event files
    lsa.py / lss.py                               nilearn single-trial GLMs (one run at a time)
    glmsingle_fit.py                              GLMsingle joint fit across sessions (recommended)
    split_half_fisherz.py                         ROI-mean split-half reliability (older)
    runs/run_{lsa,lsa_ctrl,lss,glmsingle,split_half,beta_avgs}.sh   tmux launchers
    misc/
      feature.py              beta loading/alignment + ridge decoder (core library)
      feature_roi_sweep.py    feature × phase × ROI (group/parcel) decoding sweeps
      plot_roi_decoding.py    per-parcel bar plots (+ --reliability mode)
      pattern_reliability.py  per-vertex split-half noise ceiling
      temporal_leakage.py     decodes neighbouring events' features to measure GLM blending
      beta_save.py, beta_averages.py, betas_aggregator.py   ROI beta tables for analysis/predict
  predict/      main.py, fit.py, regression.py, infer.py: regress model-layer activations (.pth)
                onto ROI betas; visualize/ plotting scripts
```

## Task design (sessions 1–4)
- 13 blocks per session = 52 runs. TR 1.49 s, multiband 4, slice-timing corrected with
  **StartTime 0.685 s** (scan i is at i·TR + 0.685 s).
- Tasks (number of positions per trial):
  - `1back` (6), acq `loc|ctg|obj`;
  - `interdms` (4), acq `{loc,ctg,obj}_{ABAB,ABBA}`;
  - `ctxdm` (3), acq `col|lco`, run-01 and run-02.
- Each position is an Encoding event (~1.49 s) followed by a Delay event: ~1.74 s inside a trial,
  3.48 s for the final delay.
- Every trial repeats once per session (same `tc`). ctxdm trials can switch between run-01 and run-02
  across sessions, so align repeats on (task, acq, phase, pos, tc) and never on run.
- Session 5 is a control session with different tasks: `1back` alo/olo, `2back` ctg/loc, `dms`,
  `ctxdm` lol/olc, `interdms2` ABBCCA/ABCABC.

## Behaviour and labels
- Scored block logs: `behav/sub-01/ses-0X/sub-01_ses-X_<date>_task-<task>_<acq>_block_<k>_events_<time>_scored.tsv`
  (block k = run k+1).
- Columns `loc{p}`, `obj{p}`, `ctg{p}`, `ref{p}`, `ang{p}`, `tc`, `expected_ans{k}`, `ans{k}`.
- Label meanings (checked against the data):
  - `loc`: binary location;
  - `obj`: 4 objects;
  - `ctg = obj // 2` (bus vs face; which code is which is undocumented);
  - `expected_ans`: 2 = match, 3 = non-match.
- `tc` is the per-position (loc, obj) digits concatenated, with leading zeros lost to float. Recover
  them with `zfill(2 * n_positions)`.
- Event files: `behav/sub-01/ses-0X/events_{wfdelay,wofdelay}/task-<task>_<acq>_run-<run>_base-events.tsv`
  (columns onset, duration, trial_type, trial, pos, phase, event_id), plus `_LSA/` and `_LSS/` folders.
  `wfdelay` models the final post-trial delay, `wofdelay` does not.
- Acq label differs between behaviour and fMRIPrep: `obj_ABAB` (behaviour) ↔ `objABAB` (fMRIPrep).

## Neural data (`/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/`)
- `first_run/`: fMRIPrep 23.2 for sessions 1–4. **All current GLM and decoding results use this.**
  - `64kDense/sub-01/ses-0X/*_space-Glasser64k_bold.dtseries.nii`: 64,984 vertices; 6,175
    medial-wall vertices are all zero.
  - `confounds/`, `derivatives/` (fMRIPrep, including StartTime JSONs), `bids/`.
  - `glm_runs/<type>_<variant>/64kDense/sub-01/ses-0X/task-<task>_acq-<acq>_run-<run>/`: one
    dscalar per event.
- `ctrl_run/`: a separate fMRIPrep run over sessions 1–5. Its data are not identical to `first_run`
  (r ≈ 0.99 for the same run). It has `64kDense/`, `confounds/` and `glasser_parcellated/` (session 5),
  plus `glm_runs/lsa/` (LSA betas for sessions 1–5, made with the older `lsa.py`).
- `ctrl_run_s5_only/`: BIDS for session 5 only.

### GLM variants in `first_run/glm_runs/` (all `wfdelay` and `wofdelay`)
| folder | method | notes |
|---|---|---|
| `lsa_*`, `lss_*` | nilearn, Glover HRF, frame times start at 0 (ignore StartTime) | timing is off by 0.685 s; delay betas strongly decode the NEXT stimulus. Superseded |
| `glmsingleB_*` | GLMsingle type B: per-vertex HRF from library, OLS | least temporal leakage. **Use for delay-phase analyses** |
| `glmsingleC_*` | + GLMdenoise | 0 PCs selected, so identical to B. Ignore |
| `glmsingle_*` | type D: + per-vertex fractional ridge | best SNR/decoding. **Use for encoding**; strong ridge (median frac 0.015) blends neighbouring events (EVC delay→next-stimulus decoding 0.93) |

Recommended fit: `glmsingle*_wfdelay` (ties wofdelay for encoding and keeps 504 delay events vs 380).
GLMsingle betas are in percent signal change. `<model>/64kDense/sub-01/_glmsingle/` holds the
manifest (trial ↔ event ↔ condition), summary maps (R2, HRFindex, FRACvalue, noisepool, meanvol),
the fit-summary JSON and figures.

Caveat for every GLM: delays are mostly 1.74 s, so delay betas mostly carry the tail of the encoding
response. Delay decoding ≠ maintenance unless shown otherwise (e.g. long vs short delays).

## glmsingle_fit.py
- One joint fit over all selected runs/sessions. Conditions = (task, acq, phase, pos, tc), so each
  repeats 4×.
- `xvalscheme` = leave-one-session-out; `sessionindicator` = session.
- Data upsampled ×3 (0.497 s grid, onset error ≤ 0.25 s); stimdur 1.5 s for all events.
- Ridge fractions: log-spaced from 1 to 0.005 (20 values); chunklen 65000 (one chunk).
- Extra regressors: motion + framewise displacement (`--add_fd`).
- GLMsingle **deletes** its `outputdir`/`figuredir`. The script points them at dedicated
  `_glmsingle/glmsingle_raw` and `_glmsingle/figures` folders only.
- Monkeypatches a GLMsingle 1.2 bug: its run-wise FIR diagnostic used run 0's extra regressors for
  every run and crashed on mixed run lengths.
- Full fit: about 3–3.7 h per variant at 16 threads with both variants running together; budget
  ≥ 30 GB RAM each (seen with the earlier 2-chunk setting). Launch with
  `bash analysis/glm/runs/run_glmsingle.sh` (one tmux window per variant). `--dry_run` checks the
  design only.

## Decoding pipeline (analysis/glm/misc)
- `collect_repeat_averaged_across_sessions` averages each event over the 4 sessions → 504 events per
  phase (1back 180, ctxdm 156, interdms 168).
- `ridge_decode_5fold`:
  - outer StratifiedGroupKFold(5), grouped by trial (task|acq|run|tc);
  - per fold, optional nuisance removal (mean pattern of task×position or position cells, from
    training events only), then StandardScaler;
  - kernel ridge classifier, identical to sklearn `RidgeClassifier`;
  - optional nested alpha tuning (19 values, 0.1–1e8, inner grouped 5-fold);
  - balanced accuracy, with a permutation null (alpha re-tuned per permutation).
- Sweep configs:
  - `tuned` (nothing removed);
  - `demean_pos+tuned`;
  - `demean+tuned` (task×position removed; the main reported setting);
  - untuned versions `base`, `demean`, `demean_pos`.
- ROI groups (bilateral Glasser): DLPFC, IFC, FrontalPole, OFC, EVC (V1–3), VTC (V8, VVC, PIT, FFC,
  VMV1–3), IT (TE2p, TE2a, TE1p, TE1m, TF, PHT).
  - `--levels parcel` decodes each bilateral parcel.
  - ROIs are keyed by (level, name), because Glasser also has a parcel called "OFC".
- Outputs go to `<glm>/64kDense/sub-01/_decoding_roi/` and `_reliability_pattern/`. Plots use
  BH-FDR across all bars.
- Typical run:
  ```
  cd analysis/glm/misc
  OMP_NUM_THREADS=1 python feature_roi_sweep.py --glm_type glmsingle --glm_variant wfdelay \
    --features loc ctg obj --rois DLPFC EVC VTC IT --levels group parcel \
    --configs tuned demean_pos+tuned demean+tuned --n_perm 1000 --n_jobs 16
  python plot_roi_decoding.py <tsv> --config demean+tuned --level parcel --out <png>
  ```

## Current findings (sub-01, GLMsingle, wfdelay)
- Visual cortex decodes everything near ceiling during encoding: EVC location ≈ 1.0; VTC category
  ≈ 0.99; object in EVC/PIT ≈ 0.6–0.7 (chance 0.25).
- IT (TE2p, TF, PHT) decodes category and object.
- DLPFC effects are weak (about 0.55–0.65 for 2-class features).
- DLPFC object/category drop when task/position structure is removed; DLPFC location does not.
- Noise ceiling tracks decoding across parcels (Spearman ρ ≈ 0.77, measured on LSS betas). DLPFC
  reliability is low (about 0.05 after task×position removal).
