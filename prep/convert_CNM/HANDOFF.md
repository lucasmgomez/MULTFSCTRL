# CNM conversion: handoff (state as of 2026-10-23)

Converting the **CNeuroMod multi-task working-memory dataset** ("CNM", short for CNeuroMod; the name is ours, not the
dataset's) into the same state as subject TR, so `analysis/glm/glmsingle_fit.py` and `analysis/glm/misc/*` run on it with
minimal code changes. Read `CLAUDE.md` first, especially **Converting another dataset to the TR layout**: it defines the
target tree, the per-file contracts and the TR assumptions in the code.

The work moved from the lab machine (`mirzakhani`) to the Compute Canada cluster Rorqual. Paths for both are given below;
the Rorqual paths were checked on 2026-10-23 (see **Rorqual verification**).

## Status
| Phase | State |
|---|---|
| 1. Inventory (read-only) | **Done**, and re-run on Rorqual (2026-10-23) with identical results. Findings below; reproducible with `inventory.py`. |
| Decisions | **All agreed** (2026-10-23, see **Decisions**). |
| 2. Conversion | **Done** for all 5 subjects (558 runs); `check_tree.py` passes. See **Phase 2–3 results**. |
| 3. Pipeline changes | **Done**; TR checks byte-identical. See **Phase 2–3 results**. |
| 4. Fit + validate | Pilot done (decision #3 confirmed); TR refit reproduces TR's betas; full sub-01 fit submitted (job 25064630). |

The user's original instructions, which still apply:
- Never write into the TR tree or the original data; build a parallel root with symlinks/copies plus a mapping table.
- Keep conversion code in `prep/convert_CNM/`.
- Generalise code with CLI arguments whose defaults keep TR's behaviour byte-for-byte; don't fork scripts.
- After changing shared code, re-run a quick TR check: `glmsingle_fit.py --dry_run` and one small decoding run.
- Run fMRIPrep only with explicit approval. Cap threads; the user prefers long jobs to survive the session (SLURM on the cluster).
- Keep `CLAUDE.md` up to date with the layout, the decisions and the code changes.
- Commit in logical steps on branch `cnm-conversion` (made off `ctrl`).

## Data sources
| What | mirzakhani | Rorqual (verified 2026-10-23; `S=/home/lucasg/links/projects/def-bashivan/shared`) |
|---|---|---|
| Raw iWISDM block logs (**primary behaviour source**) | `/mnt/store1/shared/fmri_dataset_project/data/behavior/` | `$S/fmri_dataset_project/data/behavior/` (560 logs) |
| Run → block file (**acq labels**) | `.../data/study_designs/sub-XX_design_design_with_converted.tsv` | `$S/fmri_dataset_project/data/study_designs/` |
| Glasser-64k dtseries | `.../data/glasser_resampled/<sub>/<ses>/` (+ confounds) | `$S/fmri_dataset_project/data/glasser_resampled/` (560 dtseries, **no confounds**) |
| fMRIPrep 20.2.6 confounds + sidecar JSONs | in the tars / `glasser_resampled` | `$S/cneuromod.multfs.fmriprep/<sub>/<ses>/func/` (datalad tree; only small files have content) |
| Raw BOLD JSONs (SliceTiming) | in the tars | `$S/cneuromod.multfs.fmriprep/sourcedata/cneuromod.multfs.raw/<sub>/<ses>/func/*_bold.json` |
| BIDS + fMRIPrep 20.2.6 | `/mnt/store1/lucas/cneuromod_multfs_bids/sub-XX.tar` (5 × ~120 GB uncompressed tars + `metadata.tar`) | `$S/cneuromod_multfs_bids/sub-XX.tar` (same tars, still packed) |
| Reformatted behaviour (**not needed**) | `.../data/reformated_behavior/` | `$S/fmri_dataset_project/data/reformated_behavior/` |
| TR reference data | `/mnt/tempdata/lucas/fmri/recordings/TR` | `/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/TR` (same layout; holds the files listed under **TR checks**) |

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
Reproduce them with `python prep/convert_CNM/inventory.py --project_data <.../fmri_dataset_project/data> [--confounds_root <fMRIPrep derivatives>] [--bids_root <BIDS>] --out <dir>`.
`--bids_root` must be an extracted BIDS tree with `derivatives/fmriprep`, and is only used for the onset cross-check and the `SliceTimingCorrected` flag.
On Rorqual: `--confounds_root $S/cneuromod.multfs.fmriprep`, and for `--bids_root` extract the small members from the tars
(`tar -xf sub-XX.tar --wildcards '*_events.tsv' '*_scans.tsv' '*desc-confounds_timeseries.tsv' '*_bold.json'`, about 1 min per tar).

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
All agreed with the user on 2026-10-23 (the evidence behind #3 is kept below).

1. **ID and root: agreed.** ID `CNM`. Root `/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM` (next to
   `TR`; backed-up project space). Regenerable bulk (GLMsingle raw outputs, pilots, job logs) goes to
   `/home/lucasg/scratch/CNM`; never the only copy of results. Subject labels kept; sessions renamed `ses-01..16`
   (filenames `ses-X`). sub-01 end to end first.
   - The behaviour/design/dtseries source is `trevor_ctrl/fmri_dataset_project_data`, a symlink (made with the user's
     OK) to `$S/fmri_dataset_project/data`.
   - **Neural files are copied** into the root (not symlinked): the sources are someone else's folders on a
     `def-bashivan` space at its file quota. `mapping.csv` records every original path.
2. **fMRIPrep: agreed, reuse 20.2.6** with `--start_time 0.686`. A 23.2.0 rerun is blocked until the CNeuroMod
   anatomy is available: the user may ask for read access to `cneuromod.anat.smriprep` (sMRIPrep + FreeSurfer
   subjects, the anatomy 20.2.6 used), or failing that the raw `cneuromod.anat` T1w. Never launch fMRIPrep without approval.
3. **pos-1 delay: agreed.** Don't model the 0.5 s pos-1 delay (`create_events_lsa.py --min_delay_s 1.0`); model all
   3.98 s delays including the final one (`--final_delay_duration 3.98`); one `stimdur` of 1.5 s. dms's pos-1 delay
   is 3.98 s and is kept. Run the comparison pilot (sub-01, 4 sessions, both designs; `events_wfdelaypos1` keeps the
   pos-1 delays) and report it.
   - Evidence from `vif_check.py` (median VIF, canonical HRF):

     | | Delay pos 1 | Encoding pos 2 | Encoding pos 1 |
     |---|---|---|---|
     | 1back, pos-1 delay modelled | **1,731** | **1,751** | 80 |
     | 1back, pos-1 delay dropped | – | 28 | 16 |
     | TR 1back as modelled | 64 | 75 | 28 |

     interdms is the same: 1,785 / 1,770 when modelled.
   - No data are removed: that signal stays in the time series and is absorbed by the encoding regressors; what's
     lost is a separate beta, which isn't estimable. TR's pos-1 delay therefore has no CNM counterpart; compare
     datasets on encoding betas and delays from pos 2 onward.
4. **GLMsingle condition key: agreed.** Keep TR's key (task, acq, phase, pos, tc); every event stays in the model
   (GLMsingle's `calcbadness` already ignores held-out trials whose condition isn't in the training folds). Fit B, C, D;
   write C's betas only if GLMdenoise selects > 0 PCs (for TR C = B). Check D with `temporal_leakage.py`; fallback
   if D blends heavily: tune on a stimulus key (`--cond_key`, not written).
   - `--n_folds`: default one fold per session; measure runtime first and use fewer (interleaved) folds only if
     the full fit doesn't fit the walltime.
5. **Repeats in evaluation: agreed.** Decode single-trial betas (`--repeat_mode single`, minimum repeats 1;
   grouped CV keeps all repeats of a trial in one fold). Reliability: half-splits of each condition's repeats
   (`pattern_reliability.py --split_mode repeats`), with the TR key and with a stimulus key.
6. **dms: agreed.** In the GLM fit; out of feature decoding (`--exclude_tasks dms`).
7. **Data problems: agreed** as in the table above (sub-06 ses-007 relabelled; sub-05 ses-012's 2 stub runs dropped;
   sub-05 ses-002 kept with no responses; low-accuracy runs kept and reported).

## Phase 2–3 results (2026-10-23)
**Tree** (`CNM=/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM`, TR's layout):
- `mapping.csv` (copy in the repo): one row per original run, with original log, design block, final acq, new
  session/run, include flag, notes, acq_time and every source -> destination path.
- `behav/<sub>/ses-XX/`: TR-format block logs + `_scored.tsv` + `events_wfdelay/` (sub-01 also `events_wfdelaypos1/`).
- `neural/fmriprep_outs/fmriprep_20.2.6/{64kDense,confounds,derivatives,bids}/`: **copies** (24 GB) of the Glasser-64k
  dtseries, confounds, fMRIPrep 91k sidecars (no `StartTime`; pass `--start_time 0.686`) and raw BOLD sidecars.
- `sourcedata/scans/`: the 80 `scans.tsv` extracted from the BIDS tars (source of the dates/times in the log names).
- Scratch for regenerable outputs: `/home/lucasg/scratch/CNM` (`logs/`, `dry/`, `pilot/`).
- Steps: `sbatch slurm_build_tree.sh` (9.5 min) -> `sbatch slurm_make_events.sh` (2 min) -> `check_tree.py`.

**Checks** (`check_tree.py`, all 558 runs): logs keep every trial, timing and feature; tc decodes to loc/obj;
ctg == obj // 2; copies byte-identical; volumes = confound rows; base events as planned (no delay < 1 s; dms keeps its 3.98 s pos-1 delay; final delays 3.98 s). Scored `expected_ans` = iWISDM `action` everywhere
except sub-06 ses-07 interdms loc_ABBA/loc_ABAB run-01 (expected: `action` follows the planned labels there).

**Block-log format:** loc/ref/obj/ctg/ang per position (from `locmod/objmod/ctgmod/angmod`; the 1back/ctxdm `loc{p}`
columns equal `locmod{p}`); dms: `loc{p}`, `objmod{p}`, empty ctg/ang. `tc` written as in TR (float). Responses
x -> 2, b -> 3, `--` -> empty (see CLAUDE.md -> Scored log for why not `--`). Other original columns are kept.

**sub-01 dry run** (16 sessions, `events_wfdelay`): 112 runs, 11,184 trials, 9,320 conditions (as in the repeat
table), max onset error 0.224 s, no shared frames. Trials that can inform tuning: 2,653 with 16 folds (1back 0,
interdms 224, ctxdm 1,405, dms 1,024), 2,575 with 8, 2,409 with 4.

**Code changes** (all defaults keep TR; see CLAUDE.md -> TR assumptions): `score_responses.py` (dms),
`create_events_lsa.py --min_delay_s`, `glmsingle_fit.py --n_folds/--raw_dir` + tuning report + free
`--events_variant`, `feature.py` repeat modes / `exclude_tasks`, `--root/--prep_run/--repeat_mode/...` in the sweep,
leakage and reliability scripts, `pattern_reliability.py --split_mode repeats`.

**TR checks on Rorqual** (TR folder above; scripts in the session scratchpad, rerun as described here):
- `glmsingle_fit.py --dry_run` manifest is byte-identical to the saved one (before and after the changes).
- Baseline = unchanged scripts with only the root string swapped: the small decoding run (loc/obj, DLPFC/EVC group,
  demean+tuned, 1000 perms) and `temporal_leakage.py` are byte-identical to the saved TSVs;
  `pattern_reliability.py` differs from the saved TSV by ≤ 2e-16 (numpy/BLAS). The new code reproduces this
  baseline byte for byte for all three.
- The saved `..._cfg-demeantuned.tsv` (09:40, 456 events) is stale: it doesn't match the current betas; the
  10:47 TSVs (504 events) do.
- New modes on TR: `--repeat_mode average --min_repeats 4` is byte-identical to the default; repeat-split
  reliability with the trial key matches the 2-vs-2 session split closely (e.g. EVC encoding r 0.245 vs 0.246).
  Single-trial TR decoding (reference for CNM): EVC loc encoding 0.975 (0.996 averaged), obj 0.615 (0.688).

## Phase 4 results so far (2026-10-24)
**TR refit on Rorqual** (`glmsingle_fit.py`, TR's settings, type D, outputs in scratch): fit summary equals the saved
one (pcnum 0, median frac identical, frac_at_floor 0.27489 vs 0.27495); all 4,032 betas correlate ≥ 0.999996 with
the saved ones, mean |diff| ~1e-5 of mean |beta|. So the Rorqual environment reproduces TR's GLM.
(`CNM/pilot/tr_refit_compare.py`; 2 h 35 min, 64 GB peak at 16 CPUs.)

**Decision #3 pilot** (sub-01, ses-01..04, `pilot_compare.py`, table in `CNM/pilot/pilot_compare.tsv`): encoding
betas, single-trial decoding (balanced acc., demean task×acq, tuned) and stimulus-key repeat reliability:

| | EVC loc pos 1 / 2 / 3 | VTC loc pos 1 / 2 / 3 | EVC reliability pos 2 |
|---|---|---|---|
| B, pos-1 delay dropped (`wfdelay`) | 0.92 / **0.88** / 0.94 | 0.82 / **0.74** / 0.84 | 0.057 |
| B, pos-1 delay modelled (`wfdelaypos1`) | 0.91 / **0.58** / 0.94 | 0.75 / **0.61** / 0.86 | −0.032 |
| D, dropped | 0.88 / 0.90 / 0.97 | 0.84 / 0.87 / 0.92 | 0.114 |
| D, modelled | 0.89 / 0.91 / 0.96 | 0.82 / 0.88 / 0.91 | 0.115 |

- Type B confirms the VIF prediction: modelling the 0.5 s delay ruins the pos-2 encoding beta. Type D's ridge
  absorbs it (± 0.01) by shrinking the collinear pair: 58% of vertices at the ridge floor (vs 45%), median frac
  0.005 (vs 0.012), and type D took 2.6× longer. Decision #3 stands (`wfdelay`).
- GLMdenoise picked 2 PCs (`wfdelay`) / 4 (`wfdelaypos1`), unlike TR's 0, so type C ≠ B and is saved.
- IT is at chance in this 4-session single-trial pilot (loc and obj).

**Runtime/memory.** GLMsingle refits the FULL model for every HRF bin × setting (B: 20 HRFs; GLMdenoise CV:
20 × 11 PC counts; D: 20 × 21 fractions); folds only enter `calcbadness`, which compares betas across folds. So
`--n_folds` saves little time: **keep one fold per session**. Measured stage times:

| fit | trials | CPUs | B | PC CV | D | total | peak RSS |
|---|---|---|---|---|---|---|---|
| pilot `wfdelay` | 2,796 | 32 | 10 min | 28 min | 56 min | 1 h 58 | 47 GB |
| pilot `wfdelaypos1` | 3,184 | 32 | 16 min | 47 min | 2 h 27 | 3 h 57 | 49 GB |
| TR refit | 4,032 | 16 | 20 min | 47 min | 1 h 10 | 2 h 35 | 64 GB |

Full sub-01 (11,184 trials, ~63k frames, ~20× TR's per-fit cost): estimated 15–45 h, 150–250 GB; submitted with
64 CPUs, 300 GB, 3 days.

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

## Rorqual verification (2026-10-23)
- `inventory.py` on the Rorqual copies gives byte-identical `runs/trials/timing.tsv` to the run with the scratch view, and
  every number above (run counts, acq counts, stubs, label checks, BIDS onsets within 0.0001 s, `SliceTimingCorrected=False`,
  volumes = confound rows, gaps, sub-06 ses-007 trial sets = ses-006, repeat table). Low-accuracy runs: 11 with response
  fit < 0.6 (4 of them sub-06 ses-007), 21 below 0.75.
- The 560 confound files in the datalad tree are byte-identical to the tars'. One `glasser_resampled` run
  (sub-06 ses-010 ctxdm run-2) equals the tar's fsLR-91k cortex exactly; the repo's dlabel and `fmri_dataset_project/data`'s
  have the same md5.
- `cneuromod.multfs.fmriprep` is a datalad dataset: all NIfTI/CIFTI content is absent (annex pointers only). Its
  `sourcedata/smriprep` submodule is empty; raw `events.tsv`/`scans.tsv` there are missing too.
- `$S/sourcedata/cneuromod.multfs.raw` holds unpacked raw BOLD + events + `scans.tsv` for sub-01/02/05/06 (no sub-03).
- **No CNM T1w or sMRIPrep/FreeSurfer output exists on any storage this account can read** (searched `$S`, all
  `ctb-bashivan/lucasg`, scratch). The raw `anat/` folders hold only localizers/scouts. 20.2.6 took its anatomy from
  CNeuroMod's `cneuromod.anat.smriprep` (`anat_derivatives` in `fmriprep.toml`), whose RIA store
  (`/project/rrg-pbellec/ria-rorqual`) is not readable from this account.
- Quotas (`diskusage_report`): `def-bashivan` is at 498K/500K files, so nothing can be written there; `ctb-bashivan`
  409 GB/1 TB, 277K/500K files; `rrg-bashivan` 514 GB/1 TB, 155K/500K files.

## Running on the cluster
- **Environment** (mirzakhani's `fmri_glm`): Python 3.11, numpy 2.4.0, scipy 1.16.3, pandas 2.3.3, scikit-learn 1.8.0, nilearn 0.12.1, nibabel 5.3.3, joblib 1.5.3, **glmsingle 1.2** (its FIR bug is monkeypatched in `glmsingle_fit.py` for exactly this version), fracridge 2.0, matplotlib 3.10.8. `wb_command` isn't needed.
  - Rorqual: venv `~/envs/fmri_glm` (`module load python/3.11`, then `~/envs/fmri_glm/bin/python`). Same versions except
    numpy 2.4.2 and scipy 1.16.2 (the closest cluster wheels; PyPI's manylinux numpy wheels don't install on the cluster
    Python). nilearn, nibabel, joblib, fracridge come from PyPI and GLMsingle from GitHub `main` at `1ab54a6` (Python code as
    of 2025-08-28; its setup.py still says 1.2), installed with `PIP_CONFIG_FILE=/dev/null pip install --no-deps ...`
    to bypass the cluster constraints file.
  - **Not the `1.2` tag** (2023): it calls `np.Inf` (crashes in GLMdenoise PC selection under NumPy 2) and passes
    `'extraregressors'` (typo) to the run-wise FIR, i.e. it lacks the bug `glmsingle_fit.py` monkeypatches. TR's
    `DESIGNINFO.npy` params have `firdelay`/`firpct` (added 2024-10-25), so TR used a `main`-era install; the Python
    code of `main` has not changed since 2025-08-28 (pcnum fix in the type-D GLMdenoise refit).
- **Jobs:** run them through SLURM, not tmux. Set `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` to the CPUs requested, and use 1 thread per joblib worker.
- **TR checks:** TR data are copied to Rorqual at `/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/TR`
  (that folder also holds ~100k other files from an earlier copy). The checks need these files (12,262 files, 6.7 GB),
  in the same relative layout. Everything comes from `first_run`; `ctrl_run` is only used by
  `runs/run_lsa_ctrl.sh` and isn't needed.
  - `TR/behav/sub-01/ses-0[1-4]` (40 MB);
  - `TR/neural/fmriprep_outs/first_run/{64kDense,confounds}/sub-01/ses-0[1-4]` (1.8 GB);
  - `TR/neural/fmriprep_outs/first_run/derivatives/sub-01/ses-0[1-4]/func/*bold.json` (39 KB). These hold the
    StartTime (0.685 s); without them the dry run falls back to 0 and every onset frame shifts;
  - `TR/neural/fmriprep_outs/first_run/glm_runs/glmsingle_wfdelay` (4.5 GB; reference manifest, betas and decoding TSVs).

  Several shell scripts and defaults also hard-code mirzakhani paths (see `CLAUDE.md` → Repo copies).

## Files here
- `inventory.py`: the read-only inventory (run summary, labels, repeats, timing).
- `vif_check.py`: regressor collinearity for decision #3.
- `HANDOFF.md`: this file. Replace it with a proper `CLAUDE.md` section once the conversion is done.
