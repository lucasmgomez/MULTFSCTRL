# CNM GLMsingle fits: portable package

Everything needed to fit single-trial GLMsingle betas for the remaining CNM subjects (sub-02, sub-03, sub-05,
sub-06) on another cluster, with the same code, inputs and settings as sub-01 on Rorqual. Made by
`prep/convert_CNM/make_package.sh` (branch `cnm-conversion`; commit in `code/COMMIT`).

## Contents
```
run_fit.sh                 runs one subject (see its header for options and resources)
requirements.txt           Python packages (versions used on Rorqual)
code/MULTFSCTRL/           the repo at the packaged commit (glmsingle_fit.py is analysis/glm/glmsingle_fit.py)
code/GLMsingle-1ab54a6/    GLMsingle source, GitHub main @ 1ab54a6 (setup.py says 1.2; NOT the 1.2 tag)
data/behav/<sub>/ses-XX/   block logs, scored logs, events_wfdelay/ base events
data/neural/fmriprep_outs/fmriprep_20.2.6/{64kDense,confounds}/<sub>/   Glasser-64k dtseries, fMRIPrep confounds
data/mapping.csv           original -> CNM path table for the packaged runs
FILES.tsv                  every data file with its size
```
sub-02 ses-10 `1back` loc run-01 is **left out on purpose** (decision 2026-10-26, see CLAUDE.md): it has 2,787
empty vertices that would otherwise be dropped from all of sub-02, including 361 DLPFC vertices.

## Setup
```
python3.11 -m venv env && . env/bin/activate
pip install -r requirements.txt
pip install --no-deps ./code/GLMsingle-1ab54a6
```
Python 3.11 was used. Exact versions matter most for GLMsingle (use the included source: the `1.2` tag crashes
under numpy 2 and lacks fixes the fit relies on) and numpy 2.x.

## Run
1. Dry run (seconds; checks inputs and design, writes only the manifest):
   `bash run_fit.sh sub-03 4 --dry_run`
   Expected: sub-03/sub-06 112 runs and 11,184 trials; sub-05 110 runs, 11,008 trials; sub-02 111 runs, 11,085
   trials. Max onset error ≤ 0.23 s.
2. Fit, one job per subject (independent; run them in parallel):
   `sbatch --cpus-per-task=16 --mem=220G --time=2-00:00:00 --wrap "bash run_fit.sh sub-03 16"`
   sub-01 took 24 h 53 min at 64 CPUs with ~15 busy on average; peak RAM 188 GB. Start with ≥ 200 GB.

## Outputs
`out/glm_runs/{glmsingleB,glmsingleC,glmsingle}_wfdelay/64kDense/<sub>/`: one dscalar per event per model type
(~33.5k files, ~9 GB per subject), plus `_glmsingle/` (manifest, summary maps, fit summary, figures).
`out/glmsingle_raw/<sub>/` holds GLMsingle's own files (~11 GB; regenerable, no need to copy back).

To bring results back, copy `out/glm_runs/*_wfdelay/64kDense/<sub>` into
`CNM/neural/fmriprep_outs/fmriprep_20.2.6/glm_runs/*_wfdelay/64kDense/` on Rorqual; the evaluation there
(`prep/convert_CNM/slurm_evaluate.sh`) reads that layout.
