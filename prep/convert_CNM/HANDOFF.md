# CNM conversion: handoff (state as of 2026-10-23)

Converting the **CNeuroMod multi-task working-memory dataset** ("CNM", short for CNeuroMod; the name is ours, not the
dataset's) into the same state as subject TR, so `analysis/glm/glmsingle_fit.py` and `analysis/glm/misc/*` run on it with
minimal code changes. Read `CLAUDE.md` first, especially **Converting another dataset to the TR layout**: it defines the
target tree, the per-file contracts and the TR assumptions in the code.

The work moved from the lab machine (`mirzakhani`) to a Compute Canada cluster. Paths for both are given below; the
cluster paths come from the old project's readme and **have not been checked**.

## Status
| Phase | State |
|---|---|
| 1. Inventory (read-only) | **Done.** Findings below; reproducible with `inventory.py`. |
| Decisions | Partly discussed with the user (see **Decisions**). **Confirm the open ones with the user before Phase 2.** |
| 2. Conversion | Not started. |
| 3. Pipeline changes | Not started. |
| 4. Fit + validate | Not started. |

The user's original instructions, which still apply:
- Never write into the TR tree or the original data; build a parallel root with symlinks/copies plus a mapping table.
- Keep conversion code in `prep/convert_CNM/`.
- Generalise code with CLI arguments whose defaults keep TR's behaviour byte-for-byte; don't fork scripts.
- After changing shared code, re-run a quick TR check: `glmsingle_fit.py --dry_run` and one small decoding run.
- Run fMRIPrep only with explicit approval. Cap threads; the user prefers long jobs to survive the session (SLURM on the cluster).
- Keep `CLAUDE.md` up to date with the layout, the decisions and the code changes.
- Commit in logical steps on branch `cnm-conversion` (made off `ctrl`).

## Data sources
| What | mirzakhani | Cluster (unverified, from the old project's readme) |
|---|---|---|
| Raw iWISDM block logs (**primary behaviour source**) | `/mnt/store1/shared/fmri_dataset_project/data/behavior/` | `/project/def-pbellec/xuan/fmri_dataset_project/data/behavior/` |
| Run → block file (**acq labels**) | `.../data/study_designs/sub-XX_design_design_with_converted.tsv` | `.../fmri_dataset_project/data/study_designs/` |
| Glasser-64k dtseries + confounds (all 560 runs) | `.../data/glasser_resampled/<sub>/<ses>/` | `.../fmri_dataset_project/data/glasser_resampled/` |
| BIDS + fMRIPrep 20.2.6 | `/mnt/store1/lucas/cneuromod_multfs_bids/sub-XX.tar` (5 × ~120 GB uncompressed tars + `metadata.tar`) | `/project/def-pbellec/xuan/cneuromod.multfs.fmriprep` (datalad; raw BIDS under `sourcedata/cneuromod.multfs.raw`) |
| Reformatted behaviour (**not needed**) | `.../data/reformated_behavior/` | – |
| TR reference data | `/mnt/tempdata/lucas/fmri/recordings/TR` | **not on the cluster** (see **TR checks**) |

What these sources contain:
- **`behavior/`**: 560 logs (5 subjects × 16 sessions × 7 runs) in nearly TR's raw-log format:
  - BOM, `TrialNumber`;
  - per position: `stim{p}, loc{p}, locmod{p}, ctgmod{p}, objmod{p}, angmod{p}, objlocmod{p}, ref{p}`;
  - `stimulus_{i}_onset/offset`, `response_{i}`, `response_{i}_time`, `all_responses_{i}`, `trial_end`;
  - iWISDM ground truth: `action2..6` (1back), `action1/2` (interdms), `action` (ctxdm, dms).
- **`study_designs/`**: names like `interdms_obj_ABAB_block_2`, with `nback` = `1back`. dms has no row.
- **`glasser_resampled/`**: made with the same `wb_command -cifti-resample ... ADAP_BARY_AREA CUBIC` and the same dlabel as TR (byte-identical).
  - Cortex values equal the tar's fsLR-91k data exactly (max abs diff 0, checked on one run).
  - All 560 confound files are byte-identical to the tar's.
  - **Reuse these; no resampling or CIFTI extraction is needed.**
- **The tars**: raw `anat/` holds only localizers and scouts, **no T1w**. fMRIPrep used a separate sMRIPrep anatomy that isn't included. `derivatives/glm` holds an old 4 mm-smoothed OLS fit; ignore it.
- **`reformated_behavior/`**: a reformat of the BIDS events with no responses and no `dms`.

## Inventory findings
Reproduce them with `python prep/convert_CNM/inventory.py --project_data <.../fmri_dataset_project/data> [--bids_root <BIDS>] --out <dir>`.
`--bids_root` must be an extracted BIDS tree with `derivatives/fmriprep`, and is only used for the onset cross-check and the `SliceTimingCorrected` flag.

### Layout
- **Subjects:** sub-01, sub-02, sub-03, sub-05, sub-06. Sessions `ses-001`..`ses-016`.
- **Runs per session (BIDS names, no acq entity):** `1back` run-01; `ctxdm` run-01..02; `dms` run-01; `interdms` run-01..03.
- **fMRIPrep derivative names use `run-1`; logs and BIDS use `run-01`.**

| Task | Trials/run | Positions | Volumes | acq labels (runs per subject) |
|---|---|---|---|---|
| 1back | 9 | 6 | 188 (1 run: 194) | loc 12, ctg 2, obj 2 |
| interdms | 16 | 4 | 208 | loc_ABAB 14, loc_ABBA 14, {ctg,obj}×{ABAB,ABBA} 5 each |
| ctxdm | 20 | 3 | 184 | col 16, lco 16 |
| dms (new; TR's sessions 1–4 have no dms) | 16 | 2 | 128 | location match; **the same 16 trials every session** |

### Imaging
- **Acquisition:** Prisma, TR 1.49 s, MB4, 60 slices; SliceTiming 0–1.3725 s (mean 0.686 s), the same scheme as TR.
- **fMRIPrep 20.2.6 for all subjects:**
  - `--ignore slicetiming`, so no slice-timing correction and no `StartTime`;
  - `--cifti-output 91k`; SyN SDC.
- **Confounds:** `trans_*`, `rot_*`, `framewise_displacement` present; rows = volumes in every run.
- **64k mask:** the zero (medial-wall) mask has 5,572 vertices (TR: 6,175). `glmsingle_fit.py` drops vertices that are zero in any run, so this is automatic there; check anything else that hard-codes a mask.

### Behaviour and labels (all checked, not assumed)
- **Raw logs vs BIDS:** onsets agree within 0.01 s and features are identical, except the two stubs below. Use the raw logs; the BIDS `delay` rows (fixed 2.98 s) are nominal and overlap the next stimulus.
- **Feature coding (TR conventions hold for 1back, interdms, ctxdm):**
  - use `locmod/ctgmod/objmod` as TR's `loc/ctg/obj`;
  - loc ∈ {0,1}, obj ∈ {0..3}, `ctg == obj // 2` everywhere;
  - `tc` is not in the logs; build it TR-style (loc digit then obj digit per position);
  - responses are **`x` = match, `b` = non-match**, confirmed by rule fits; convert to TR's `2`/`3` for `score_responses.py`.
- **dms:**
  - logs have only `loc{p}` (binary) and `objmod{p}` (0–7; 6 never observed); no category;
  - it is a **location-match task** (`action` = loc match in 100% of trials, responses agree 99.7%);
  - an object match only ever happens together with a location match, so loc and obj are confounded.
- **acq labels:** design label + TR's scoring rule reproduces iWISDM `action` in 100% of 1back/interdms/dms runs, except sub-06 ses-007 (below).
- **ctxdm caveat:**
  - its `action` column (and the BIDS `accuracy` column) only implements the same-context branch: `obj2==obj3` for col, `ctg2==ctg3` for lco;
  - subjects follow TR's full col/lco rule (~100% of answered trials);
  - **score ctxdm with TR's `score_responses.py`, never with `action` or BIDS `accuracy`**.
- **Low-accuracy runs:** about 20 runs, mostly sub-03 and sub-05, have correct labels but the subject applied the wrong rule (e.g. sub-05 ses-005 interdms run-3 at 28%). Report them; don't relabel.

### Data problems
| Subject / session | Problem | Proposed handling (not yet confirmed) |
|---|---|---|
| sub-06 ses-007 | Every run's trial set equals sub-06 ses-006's. Responses fit **ses-006's** rules perfectly (1back obj; interdms loc_ABBA, loc_ABBA, loc_ABAB; ctxdm col). The design sheet and partly `action` give ses-007's planned labels. | Label with ses-006's acq labels. It then becomes a full repeat of ses-006. |
| sub-05 ses-012 | `dms` run-01 and `interdms` run-01 logs are empty BIDS template stubs; BIDS events are empty too. Trial order is shuffled per subject, so they can't be rebuilt from other subjects. | Drop these 2 runs (the BOLD exists but has no events). |
| sub-05 ses-002 | No response columns in any of the 7 logs. | Keep; accuracy is NaN. |

### Timing (from stimulus onsets/offsets)
| Interval | CNM | TR (for comparison) |
|---|---|---|
| Encoding | 1.49 s | 1.49 s |
| Gap after pos 1 | **0.50 s** (dms: 3.98 s) | 1.74 s (interdms: pos 1–2) |
| Gap after later positions | **3.98 s** | 3.48 s |
| Final gap (to next trial) | **3.98 s** (no extra ITI) | 4.73 s (modelled as 3.48 s) |
| First stimulus | 6.0 s | ≈3.0 s |
| Scan after last stimulus | ≥ 13.5 s | – |

- Minimum gap between stimulus onsets is 1.98 s.
- `CLAUDE.md`'s "~1.74 s inside a trial" for TR is only true for the first gap; fix it when you next edit `CLAUDE.md`.

### Repeats (the main difference from TR)
- Every subject saw the **same trial sets per session/run, in shuffled order**. Within a subject each block file is used once.
- With TR's condition key (task, acq, phase, pos, tc), pos-1 delays dropped and sub-06 ses-007 relabelled:

| | Conditions | Seen once | Repeated, same session only | Repeated across sessions | Events in cross-session repeats |
|---|---|---|---|---|---|
| sub-01/02/03 | 9,320 | 8,477 | 27 | 816 | 2,653 / 11,184 (24%) |
| sub-05 | 9,208 | 8,365 | 27 | 816 | 2,589 / 11,008 |
| sub-06 | 8,799 | 7,509 | 32 | 1,258 | 3,611 / 11,184 |

- By task, for sub-01, events in cross-session repeats are: 1back 0/1,584; interdms 224/5,376; ctxdm 1,405/3,200; dms 1,024/1,024.
- With a stimulus key (task, phase, pos, loc, obj), sub-01 has 246 conditions and every event repeats across sessions.

## Decisions
Discussed with the user. **Confirm everything not marked "agreed" before Phase 2.**

1. **ID and root: open.** Proposed `CNM`; the user asked what it meant and didn't confirm.
   - On mirzakhani the plan was `/mnt/tempdata/lucas/fmri/recordings/CNM/{behav,neural}`.
   - **On the cluster, ask where the root should live.** `$SCRATCH` is purged on Compute Canada, so it can't be the only copy of results.
   - Keep subject labels; rename sessions to `ses-01..16` (filenames `ses-X`), as in TR.
   - Suggest doing sub-01 end to end first, then the other four.
2. **fMRIPrep: open.** Recommended: reuse the 20.2.6 outputs (via `glasser_resampled`) and pass `--start_time 0.686` to `glmsingle_fit.py`.
   - 0.686 s is the mean slice time, matching TR's slice-timing reference (0.685 s). Without correction, per-vertex timing error is ≤ ±0.69 s.
   - `--start_time` already exists and overrides the JSON lookup.
   - A rerun with 23.2.0 would need the CNeuroMod T1w/sMRIPrep data, which isn't in the BIDS tars, and 560 runs of compute. **Rerun only with explicit approval.**
3. **pos-1 delay (0.5 s): open; the user pushed back.** Recommendation: don't model the 0.5 s pos-1 delay, but model all 3.98 s delays including the final one (`--final_delay_duration 3.98`); one `stimdur` of 1.5 s.
   - The user worried this "loses an important piece of each trial". Answer given: no data are removed. That signal stays in the time series and is absorbed by the encoding regressors; what's lost is a separate beta, and that beta isn't estimable.
   - Evidence from `vif_check.py` (median VIF, canonical HRF):

     | | Delay pos 1 | Encoding pos 2 | Encoding pos 1 |
     |---|---|---|---|
     | 1back, pos-1 delay modelled | **1,731** | **1,751** | 80 |
     | 1back, pos-1 delay dropped | – | 28 | 16 |
     | TR 1back as modelled | 64 | 75 | 28 |

     interdms is the same: 1,785 / 1,770 when modelled.
   - Keeping the regressor also ruins the stim-2 encoding beta. TR's pos-1 delay therefore has no CNM counterpart; compare datasets on encoding betas and delays from pos 2 onward.
   - Offered but not done: fit sub-01 type B both ways on a few sessions and compare how reproducible the stim-2 encoding betas are across repeats, plus `temporal_leakage.py`. **Ask the user whether they want this before deciding.**
   - Optional extra: a "late delay" event at delay onset + 2 s.
4. **GLMsingle condition key: agreed in direction.** The user suggested choosing the noise components and ridge fraction only on the data that have repeats. That is GLMsingle's default with TR's key: `glmsingle/ssq/calcbadness.py` skips any held-out trial whose condition isn't in the training folds, while every event stays in the model and gets a beta. Plan:
   - Keep TR's key and fit B, C and D. **Don't drop singletons from the design**: their responses would leak into neighbouring betas and residuals, and type C would treat them as noise.
   - Dry run: report how many trials per task actually count toward tuning (sub-01: none from 1back, few from interdms).
   - Check D with `temporal_leakage.py`, focusing on 1back and pos-1/pos-2 pairs. D's tuning comes from dms and ctxdm, whose event spacing differs from 1back's.
   - Fallback if D blends heavily: tune on the stimulus key (`--cond_key stimulus`, still to write).
   - **Open:** a `--n_folds` argument grouping the 16 sessions into e.g. 4 folds, to cut runtime. Default stays one fold per session.
5. **Repeats in evaluation: open.** Proposed:
   - decode single-trial betas (no averaging, minimum repeats 1);
   - reliability by splitting each condition's repeats in half, plus a stimulus-key reliability, because TR-key repeats are almost only dms/ctxdm.
6. **dms: open.** Proposed: include it in the GLM fit (its 16×16 repeats help tuning), exclude it from feature decoding by default.
7. **Data problems: open.** Proposed handling is in the table above.

## Phase 2 plan: conversion (`prep/convert_CNM/`)
1. **`build_tree.py`** writes the mapping table and the tree. One row per run, with columns:
   - original log, design block, original BIDS name;
   - final acq (with the sub-06 ses-007 fix);
   - new session (`ses-XX`), new run (counted within session × task × acq, in original run order);
   - new log name, new dtseries/confounds paths, include flag, notes.

   Note: `.gitignore` ignores `*.tsv`, so force-add the mapping table or commit it as `.csv`.
2. **Neural tree:** `<root>/neural/fmriprep_outs/<prep_run>/{64kDense,confounds}/<sub>/ses-XX/<sub>_ses-XX_task-<task>_acq-<acqfp>_run-<rr>_...` as **symlinks** into `glasser_resampled`.
   - `<acqfp>` is the acq without underscores, e.g. `locABAB`.
   - dms gets `acq-loc`.
   - Also copy the raw BOLD JSONs (SliceTiming) into `bids/`.
3. **Behaviour tree:** for each run write `<root>/behav/<sub>/ses-XX/<sub>_ses-<n>_<date>_task-<task>_<acq>_block_<run-1>_events_<HH-MM-SS>.tsv`:
   - strip padding, keep the BOM;
   - rename `locmod/ctgmod/objmod/angmod` to `loc/ctg/obj/ang` (dms: keep `loc`, set `ctg` empty);
   - add `tc` as a float, as in TR;
   - map responses x → 2, b → 3;
   - date and time come from `scans.tsv` `acq_time` in the BIDS session folder.
4. **Scoring:** run `prep/behav/score_responses.py <dir>`. Its filename regex only knows `interdms|ctxdm|1back`; add `dms` with spec `loc` (location match, response at position 2), keeping TR's behaviour unchanged.
5. **Events:**
   - Run `analysis/glm/create_events_lsa.py -i <behav ses dir> -o .../events_wfdelay --final_delay_duration 3.98`, plus a new option to skip short delays (e.g. `--min_delay_s 1.0`, default 0 keeps TR byte-identical).
   - The script globs every `*.tsv`, so raw and scored logs both get processed (as for TR).
   - **Check how TR's `events_wofdelay` was produced.** The current `build_base_events` always appends the final delay, even though its docstring says it doesn't, and nothing in the script mentions `wofdelay`.

## Phase 3 plan: code changes (defaults must keep TR identical)
- **Hard-coded TR root and sessions** in `feature_roi_sweep.py`, `pattern_reliability.py`, `temporal_leakage.py` (and `feature.py`'s `__main__`): add `--root`; sessions are already arguments in most of them.
- **`feature.collect_repeat_averaged_across_sessions`:** add a mode that averages the repeats that exist with a minimum count, and a single-trial mode. The default must still intersect all sessions.
- **`pattern_reliability.py`:** `--sessions nargs=4` and fixed `SPLITS`; generalise to a half-split of each condition's repeats.
- **`temporal_leakage.py`:** add `dms: 2` to `N_POS`. The tc digit layout is the same as TR's.
- **`misc/beta_*.py`:** `tc_format` length maps, likewise.
- **`glmsingle_fit.py`:**
  - `--n_folds` (see decision #4);
  - print the tuning trials per task;
  - optional `--cond_key`;
  - `find_block_log` expects exactly one scored log per run, which the CNM tree must satisfy;
  - `--upsample 3` is fine (gaps ≥ 0.5 s);
  - `--start_time 0.686` is given on the command line, no code change.
- **TR checks:** after each shared-code change, run TR's `glmsingle_fit.py --dry_run` and compare its manifest with the saved one, and run one small decoding job and compare with the saved `_decoding_roi` TSVs.

## Phase 4 plan: fit and validate
- **Dry run:**
  ```
  python analysis/glm/glmsingle_fit.py --sub sub-01 --sessions ses-01 ... ses-16 \
    --neural_base <root>/neural/fmriprep_outs/<prep_run> --behav_base <root>/behav \
    --events_variant wfdelay --add_fd --start_time 0.686 --dry_run
  ```
  Check run, condition and repeat counts, the maximum onset error, and that no two events share a frame.
- **Fit:** `--save_types B C D`.
  - Scale vs TR: ~11k events and 112 runs per subject (TR: ~4k events, 52 runs).
  - TR took 3–3.7 h per variant at 16 threads with ≥30 GB RAM. **Guess** 10+ h and ~100 GB per subject per variant; measure on sub-01 first.
  - GLMsingle **deletes** its output and figure directories; the script already points them at dedicated folders.
- **Evaluate:** `temporal_leakage.py`, `pattern_reliability.py`, `feature_roi_sweep.py`, then `plot_roi_decoding.py` in three versions (task×position removed, position removed, nothing removed).
- **Positive control:** EVC location decoding during encoding should be near 1.0. If it isn't, suspect timing (StartTime), alignment or labels first.

## Running on the cluster
- **Environment** (mirzakhani's `fmri_glm`): Python 3.11, numpy 2.4.0, scipy 1.16.3, pandas 2.3.3, scikit-learn 1.8.0, nilearn 0.12.1, nibabel 5.3.3, joblib 1.5.3, **glmsingle 1.2** (its FIR bug is monkeypatched in `glmsingle_fit.py` for exactly this version), fracridge 2.0, matplotlib 3.10.8. `wb_command` isn't needed.
- **Jobs:** run them through SLURM, not tmux. Set `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` to the CPUs requested, and use 1 thread per joblib worker.
- **TR checks:** TR exists only on mirzakhani. Either run the TR checks there, or ask the user to copy (~6.3 GB):
  - `TR/behav/sub-01/ses-0[1-4]` (40 MB);
  - `TR/neural/fmriprep_outs/first_run/{64kDense,confounds}/sub-01/ses-0[1-4]` (1.8 GB);
  - `TR/neural/fmriprep_outs/first_run/glm_runs/glmsingle_wfdelay` (4.5 GB; reference manifest, betas and decoding TSVs).

  Several shell scripts and defaults also hard-code mirzakhani paths (see `CLAUDE.md` → Repo copies).

## Files here
- `inventory.py`: the read-only inventory (run summary, labels, repeats, timing).
- `vif_check.py`: regressor collinearity for decision #3.
- `HANDOFF.md`: this file. Replace it with a proper `CLAUDE.md` section once the conversion is done.
