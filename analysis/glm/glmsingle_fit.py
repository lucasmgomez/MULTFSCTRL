#!/usr/bin/env python3
"""
Single-trial betas with GLMsingle (Prince et al., 2022) on CIFTI dtseries (vertex-wise).

ONE joint fit across all sessions/runs selected, because GLMsingle tunes its hyperparameters
(GLMdenoise PCs, per-vertex ridge fraction) by cross-validating REPEATED conditions, and in this
design every event repeats once per session. Conditions are (task, acq, phase, pos, tc), i.e. the
same trial position of the same stimulus sequence; run is deliberately left out so ctxdm trials
that swap between run-01/run-02 across sessions still count as repeats.

Differences vs lsa.py / lss.py:
  - per-vertex HRF chosen from GLMsingle's library of 20 HRFs (type B),
    GLMdenoise noise regressors (type C), per-vertex fractional ridge (type D);
  - cross-validation folds = sessions (leave-one-session-out), session-wise beta z-scoring
    for hyperparameter selection;
  - slice-timing reference (fMRIPrep StartTime) is used for frame times (lsa.py assumes 0);
  - data are upsampled in time so sub-TR event onsets land on the model grid;
  - betas are in percent BOLD change (no time-series z-scoring: GLMsingle needs the mean signal);
  - medial-wall (all-zero) vertices are excluded from the fit and written back as 0.

Encoding and delay events are separate single-trial regressors, so both timescales are kept.
GLMsingle uses ONE event duration (--stimdur) for all events: the default 1.5 s matches encoding
(~1.49 s); delays (1.7-3.5 s) are modelled from their onset with the same duration.

Outputs (per saved model type) mirror the LSA layout so the decoding scripts work unchanged:
  <glm_runs>/glmsingle[<T>]_<events_variant>/64kDense/<sub>/<ses>/task-<task>_acq-<acq>_run-<run>/
      glmsingle_<task>_<acq>_<Enc|Del>_<event_id>_beta.dscalar.nii
  <...>/<sub>/_glmsingle/  manifest, summary maps (R2, HRF index, ridge fraction, noise pool),
                           fit summary JSON, GLMsingle figures
Type D (final model) is written to glmsingle_<variant>; other types to glmsingle<T>_<variant>.
"""
import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.interpolate import CubicSpline


def parse_args():
    p = argparse.ArgumentParser(
        "Run ONE joint GLMsingle fit across sessions on CIFTI dtseries (vertex-wise)."
    )

    p.add_argument("--sub", default="sub-01")
    p.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02", "ses-03", "ses-04"])
    p.add_argument("--neural_base", default="/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run",
                   help="Contains 64kDense/, confounds/, derivatives/ and glm_runs/.")
    p.add_argument("--behav_base", default="/mnt/tempdata/lucas/fmri/recordings/TR/behav")
    p.add_argument("--events_variant", default="wfdelay",
                   help="Which base-events to model (events_<variant>/, e.g. wfdelay or wofdelay). Default wfdelay: "
                        "the post-trial period is modelled instead of leaking into the baseline and the noise PCs.")
    p.add_argument("--tasks_regex", default=None,
                   help="Only runs whose 'task-<task>_<acq>_run-<run>' matches (e.g. 'interdms_loc'). Default: all.")
    p.add_argument("--tr", type=float, default=1.49, help="TR in seconds (default 1.49).")
    p.add_argument("--start_time", type=float, default=None,
                   help="Slice-timing reference in s (fMRIPrep 'StartTime'). Default: read from the run's "
                        "derivatives JSON, else 0.")
    p.add_argument("--upsample", type=int, default=3,
                   help="Temporal upsampling factor (cubic spline). 3 -> 0.497 s grid, max onset error 0.25 s.")
    p.add_argument("--stimdur", type=float, default=1.5, help="Event duration (s) for the HRF library.")

    p.add_argument("--confounds_cols", default="trans_x,trans_y,trans_z,rot_x,rot_y,rot_z",
                   help="Comma-separated confound columns to include as extra regressors.")
    p.add_argument("--add_fd", action="store_true", help="Also include framewise_displacement.")

    p.add_argument("--n_folds", type=int, default=None,
                   help="Cross-validation folds. Default: one per session (leave-one-session-out). With k < n "
                        "sessions, session i (sorted) goes to fold i %% k, so neighbouring sessions are split up. "
                        "Fewer folds run faster; a condition whose repeats all fall in one fold cannot inform tuning.")
    p.add_argument("--n_pcs", type=int, default=10, help="Max GLMdenoise PCs (GLMsingle default 10).")
    p.add_argument("--fracs_min", type=float, default=0.005,
                   help="Smallest ridge fraction evaluated. GLMsingle's default floor (0.05) was chosen by ~90%% "
                        "of vertices in test fits on this design, so the default is lower.")
    p.add_argument("--n_fracs", type=int, default=20,
                   help="Number of ridge fractions, log-spaced from 1 to --fracs_min. Type-D runtime scales "
                        "with this (one full refit per fraction x HRF x chunk).")
    p.add_argument("--chunklen", type=int, default=65000,
                   help="Vertices processed at once. >= fitted vertices (58809) = one chunk, which halves "
                        "runtime (every refit is per chunk) for ~5 GB more RAM.")
    p.add_argument("--save_types", nargs="+", default=["D"], choices=["B", "C", "D"],
                   help="Model types whose single-trial betas are written as dscalars "
                        "(B: library HRF, C: + GLMdenoise, D: + ridge; default D).")

    p.add_argument("--output_root", default=None,
                   help="Root for outputs (default <neural_base>/glm_runs). Model folders are created inside.")
    p.add_argument("--raw_dir", default=None,
                   help="GLMsingle's own outputdir (deleted and rewritten by GLMsingle; ~2 GB for TR). "
                        "Default <model>/_glmsingle/glmsingle_raw. Must be a dedicated folder.")
    p.add_argument("--save_multidscalar", action="store_true",
                   help="Also save one multi-scalar dscalar per run with all its event betas.")
    p.add_argument("--dry_run", action="store_true",
                   help="Build and check the design, write the manifest, but do not fit.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ----------------------------
# Same helpers as lsa.py
# ----------------------------
def load_confounds(confounds_tsv, confounds_cols, add_fd):
    conf = pd.read_csv(confounds_tsv, sep="\t")
    cols = [c.strip() for c in confounds_cols.split(",") if c.strip()]
    if add_fd and "framewise_displacement" in conf.columns:
        cols.append("framewise_displacement")

    missing = [c for c in cols if c not in conf.columns]
    if missing:
        raise ValueError(f"Missing confound columns in {confounds_tsv}: {missing}")

    Xc = conf[cols].copy().fillna(0.0)
    return Xc, cols


def load_dtseries_as_time_by_nodes(dt_path):
    dt = nib.load(dt_path)
    Y = dt.get_fdata(dtype=np.float32)
    if Y.ndim != 2:
        raise ValueError(f"Expected 2D dtseries data. Got shape {Y.shape}")

    n_time_hdr = len(dt.header.get_axis(0))
    n_nodes_hdr = len(dt.header.get_axis(1))
    if Y.shape == (n_time_hdr, n_nodes_hdr):
        return dt, Y
    if Y.shape == (n_nodes_hdr, n_time_hdr):
        return dt, Y.T
    raise ValueError(
        f"dtseries data shape {Y.shape} does not match header axes "
        f"(time={n_time_hdr}, nodes={n_nodes_hdr})."
    )


def save_dscalar_from_dtseries(dt_img, beta_vec, out_path, scalar_name="beta"):
    save_multidscalar_from_dtseries(dt_img, beta_vec[np.newaxis, :], [scalar_name], out_path)


def save_multidscalar_from_dtseries(dt_img, beta_2d, scalar_names, out_path):
    """beta_2d: (n_scalars, n_nodes)"""
    beta_2d = np.asarray(beta_2d, dtype=np.float32)
    if beta_2d.ndim != 2 or len(scalar_names) != beta_2d.shape[0]:
        raise ValueError("beta_2d must be (n_scalars, n_nodes) with one name per scalar")

    bm_axis = dt_img.header.get_axis(1)
    scalar_axis = nib.cifti2.ScalarAxis(list(scalar_names))
    hdr = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
    nib.save(nib.cifti2.Cifti2Image(beta_2d, hdr, dt_img.nifti_header), out_path)


# ----------------------------
# Run discovery
# ----------------------------
BASE_EVENTS_RE = re.compile(r"^task-(?P<task>[^_]+)_(?P<acq>.+)_run-(?P<run>\d+)_base-events\.tsv$")


def find_block_log(block_logs_dir, task, acq, run):
    """Scored raw block log for this run (block index = run - 1), as in misc/feature.py."""
    block0 = int(run) - 1
    hits = glob.glob(os.path.join(block_logs_dir, f"*task-{task}_{acq}_block_{block0}_*scored*.tsv"))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected 1 scored block log for task-{task}_{acq} block {block0} "
                                f"in {block_logs_dir}, found {len(hits)}")
    return hits[0]


def read_start_time(neural_base, sub, ses, task, acq_fp, run, fallback):
    if fallback is not None:
        return float(fallback)
    pat = os.path.join(neural_base, "derivatives", sub, ses, "func",
                       f"{sub}_{ses}_task-{task}_acq-{acq_fp}_run-{run}_*bold.json")
    for js in sorted(glob.glob(pat)):
        meta = json.load(open(js))
        if "StartTime" in meta:
            return float(meta["StartTime"])
    print(f"[WARN] No StartTime found for {pat}; assuming 0.")
    return 0.0


def discover_runs(args):
    runs = []
    for ses in args.sessions:
        events_dir = os.path.join(args.behav_base, args.sub, ses, f"events_{args.events_variant}")
        for ev in sorted(glob.glob(os.path.join(events_dir, "*_base-events.tsv"))):
            m = BASE_EVENTS_RE.match(os.path.basename(ev))
            if not m:
                continue
            task, acq, run = m.group("task"), m.group("acq"), m.group("run")
            if args.tasks_regex and not re.search(args.tasks_regex, f"task-{task}_{acq}_run-{run}"):
                continue
            acq_fp = acq.replace("_", "")  # fMRIPrep acq label: obj_ABAB -> objABAB
            stem = f"{args.sub}_{ses}_task-{task}_acq-{acq_fp}_run-{run}"
            runs.append({
                "ses": ses, "task": task, "acq": acq, "run": run,
                "taskdir": f"task-{task}_acq-{acq}_run-{run}",
                "base_events": ev,
                "block_log": find_block_log(os.path.join(args.behav_base, args.sub, ses), task, acq, run),
                "dtseries": os.path.join(args.neural_base, "64kDense", args.sub, ses,
                                         f"{stem}_space-Glasser64k_bold.dtseries.nii"),
                "confounds": os.path.join(args.neural_base, "confounds", args.sub, ses,
                                          f"{stem}_desc-confounds_timeseries.tsv"),
                "start_time": read_start_time(args.neural_base, args.sub, ses, task, acq_fp, run, args.start_time),
            })
    for r in runs:
        for k in ("dtseries", "confounds"):
            if not os.path.isfile(r[k]):
                raise FileNotFoundError(f"Missing {k}: {r[k]}")
    return runs


# ----------------------------
# Events -> conditions
# ----------------------------
def run_events_with_conditions(run):
    """Base events + condition key (task, acq, phase, pos, tc) from the scored block log."""
    ev = pd.read_csv(run["base_events"], sep="\t")
    needed = {"onset", "duration", "phase", "trial", "pos", "event_id"}
    if needed - set(ev.columns):
        raise ValueError(f"Missing columns {needed - set(ev.columns)} in {run['base_events']}")

    log = pd.read_csv(run["block_log"], sep="\t")
    log["TrialNumber"] = pd.to_numeric(log["TrialNumber"], errors="raise").astype(int)
    tc_by_trial = {int(t): str(int(float(tc))) for t, tc in zip(log["TrialNumber"], log["tc"])}

    ev = ev.copy()
    ev["trial_int"] = ev["trial"].str.extract(r"Trial(\d+)", expand=False).astype(int)
    missing = sorted(set(ev["trial_int"]) - set(tc_by_trial))
    if missing:
        raise ValueError(f"Trials {missing} of {run['base_events']} not in block log {run['block_log']}")
    ev["tc"] = ev["trial_int"].map(tc_by_trial)
    ev["cond_key"] = [f"{run['task']}|{run['acq']}|{ph}|{int(pos)}|{tc}"
                      for ph, pos, tc in zip(ev["phase"], ev["pos"], ev["tc"])]
    return ev


# ----------------------------
# Upsampling
# ----------------------------
def upsample_run(Y, conf, k):
    """Cubic-spline upsampling by integer factor k on [0, (T-1)] samples -> (T-1)*k + 1 frames."""
    T = Y.shape[0]
    t = np.arange(T, dtype=np.float64)
    t_up = np.arange((T - 1) * k + 1, dtype=np.float64) / k
    Y_up = CubicSpline(t, Y, axis=0)(t_up).astype(np.float32)
    C_up = np.column_stack([np.interp(t_up, t, conf[:, j]) for j in range(conf.shape[1])])
    return Y_up, C_up


def patch_runwise_fir(glmsingle_module, n_runs):
    """
    GLMsingle 1.2 bug: the diagnostic run-wise FIR fits ONE run at a time but passes the extra
    regressors / polynomial degrees of ALL runs, so every run is fitted with run 0's regressors
    (and runs of different length crash). The FIR calls are made in run order, so hand each call
    its own run's entries. Only the FIR diagnostic is affected; all other calls pass through.
    """
    orig = glmsingle_module.glm_estimatemodel
    state = {"run": 0}

    def wrapped(design, data, stimdur, tr, hrfmodel, hrfknobs, resampling, opt=None, *a, **kw):
        if (hrfmodel == "fir" and not isinstance(design, list) and isinstance(opt, dict)
                and isinstance(opt.get("extra_regressors"), list) and len(opt["extra_regressors"]) == n_runs):
            p = state["run"] % n_runs
            state["run"] += 1
            opt = dict(opt, extra_regressors=[opt["extra_regressors"][p]], maxpolydeg=[opt["maxpolydeg"][p]])
        return orig(design, data, stimdur, tr, hrfmodel, hrfknobs, resampling, opt, *a, **kw)

    glmsingle_module.glm_estimatemodel = wrapped


def main():
    args = parse_args()
    out_root = args.output_root or os.path.join(args.neural_base, "glm_runs")
    model_dirs = {T: os.path.join(out_root, f"glmsingle{'' if T == 'D' else T}_{args.events_variant}",
                                  "64kDense", args.sub) for T in args.save_types}
    work_dir = os.path.join(model_dirs["D" if "D" in model_dirs else args.save_types[0]], "_glmsingle")
    os.makedirs(work_dir, exist_ok=True)

    runs = discover_runs(args)
    if not runs:
        raise RuntimeError("No runs found (check --sessions / --tasks_regex / --events_variant).")
    print(f"Runs: {len(runs)} across {len(set(r['ses'] for r in runs))} sessions")

    # ---- events / conditions ----
    for r in runs:
        r["events"] = run_events_with_conditions(r)
    cond_keys = sorted({k for r in runs for k in r["events"]["cond_key"]})
    cond_index = {k: i for i, k in enumerate(cond_keys)}
    reps = pd.Series([k for r in runs for k in r["events"]["cond_key"]]).value_counts()
    print(f"Conditions: {len(cond_keys)}  events: {int(reps.sum())}  "
          f"repeats per condition: {reps.value_counts().sort_index().to_dict()}")

    # ---- cross-validation folds (by session) ----
    sessions = sorted(set(r["ses"] for r in runs))
    n_folds = args.n_folds or len(sessions)
    if not 2 <= n_folds <= len(sessions):
        raise ValueError(f"--n_folds must be in [2, {len(sessions)}] (number of sessions)")
    for r in runs:
        r["fold"] = sessions.index(r["ses"]) % n_folds
    # GLMsingle only scores held-out trials whose condition also occurs in the training folds
    ev_all = pd.concat([r["events"].assign(task=r["task"], fold=r["fold"]) for r in runs], ignore_index=True)
    n_fold_per_cond = ev_all.groupby("cond_key")["fold"].nunique()
    ev_all["tunes"] = ev_all["cond_key"].map(n_fold_per_cond) > 1
    print(f"Folds: {n_folds} ({'one per session' if n_folds == len(sessions) else 'interleaved sessions'}); "
          f"trials that inform tuning (condition in >= 2 folds), per task:")
    print(ev_all.groupby(["task", "phase"])["tunes"].agg(["sum", "size"])
          .rename(columns={"sum": "tuning", "size": "all"}).to_string())

    # ---- design on the upsampled grid ----
    dt_up = args.tr / args.upsample
    manifest_rows = []
    designs = []
    for run_i, r in enumerate(runs):
        n_scans = len(nib.load(r["dtseries"]).header.get_axis(0))
        n_up = (n_scans - 1) * args.upsample + 1
        ev = r["events"]
        frame = np.rint((ev["onset"].to_numpy() - r["start_time"]) / dt_up).astype(int)
        late = frame >= n_up
        if late.any():
            print(f"[WARN] {r['ses']} {r['taskdir']}: {late.sum()} events start after the last frame; dropped")
        frame = np.clip(frame, 0, None)
        ev = ev.assign(frame=frame)[~late].sort_values("frame", kind="stable")
        if ev["frame"].duplicated().any():
            raise RuntimeError(f"{r['ses']} {r['taskdir']}: two events share an onset frame; increase --upsample")

        D = np.zeros((n_up, len(cond_keys)), dtype=np.float32)
        D[ev["frame"].to_numpy(), ev["cond_key"].map(cond_index).to_numpy()] = 1.0
        designs.append(D)
        r["events"] = ev
        r["n_scans"] = n_scans
        for _, e in ev.iterrows():
            manifest_rows.append({
                "trial_index": len(manifest_rows), "session": r["ses"], "run_index": run_i,
                "taskdir": r["taskdir"], "task": r["task"], "acq": r["acq"], "run": r["run"],
                "event_id": e["event_id"], "phase": e["phase"], "trial": e["trial"], "pos": int(e["pos"]),
                "tc": e["tc"], "onset": float(e["onset"]), "duration": float(e["duration"]),
                "onset_frame": int(e["frame"]), "onset_error_s": float(e["frame"] * dt_up + r["start_time"] - e["onset"]),
                "condition": cond_index[e["cond_key"]], "cond_key": e["cond_key"],
                "beta_dscalar": f"glmsingle_{r['task']}_{r['acq']}_{e['phase'][:3]}_{e['event_id']}_beta.dscalar.nii",
            })
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(work_dir, "glmsingle_manifest.tsv"), sep="\t", index=False)
    print(f"Design: {len(designs)} runs, {len(manifest)} trials, grid {dt_up:.3f} s, "
          f"max |onset error| {manifest['onset_error_s'].abs().max():.3f} s")
    if args.dry_run:
        print(f"[DRY RUN] wrote {os.path.join(work_dir, 'glmsingle_manifest.tsv')}")
        return

    # ---- data + confounds ----
    data, extra = [], []
    valid = None
    dt_ref = None
    for r in runs:
        dt_img, Y = load_dtseries_as_time_by_nodes(r["dtseries"])
        conf, conf_cols = load_confounds(r["confounds"], args.confounds_cols, args.add_fd)
        if len(conf) != Y.shape[0]:
            raise ValueError(f"Confounds rows ({len(conf)}) != dtseries timepoints ({Y.shape[0]}): {r['confounds']}")
        nz = Y.std(axis=0) > 0
        valid = nz if valid is None else (valid & nz)
        Y_up, C_up = upsample_run(Y, conf.to_numpy(np.float64), args.upsample)
        C_up = (C_up - C_up.mean(0)) / np.where(C_up.std(0) > 0, C_up.std(0), 1.0)
        data.append(Y_up)
        extra.append(C_up)
        dt_ref = dt_ref or dt_img
        print(f"Loaded {r['ses']} {r['taskdir']}: {Y.shape[0]} -> {Y_up.shape[0]} frames, StartTime={r['start_time']}")
    n_nodes = data[0].shape[1]
    data = [np.ascontiguousarray(Y[:, valid].T) for Y in data]  # GLMsingle wants vertices x time
    print(f"Vertices fitted: {int(valid.sum())} / {n_nodes} (all-zero medial wall excluded)")

    # ---- GLMsingle ----
    import glmsingle.glmsingle as glmsingle_module
    from glmsingle.glmsingle import GLM_single
    patch_runwise_fir(glmsingle_module, len(runs))

    ses_num = [sessions.index(r["ses"]) + 1 for r in runs]
    # held-out runs per fold (default one fold per session); GLMsingle needs each fold as an array
    fold_of_run = np.array([r["fold"] for r in runs])
    xvalscheme = [np.flatnonzero(fold_of_run == k) for k in range(n_folds)]
    want = [0, int("B" in args.save_types), int("C" in args.save_types), 1]
    params = {
        "wantlibrary": 1, "wantglmdenoise": 1, "wantfracridge": 1,
        "xvalscheme": xvalscheme, "sessionindicator": np.array([ses_num]),
        "extra_regressors": extra, "n_pcs": args.n_pcs, "chunklen": args.chunklen,
        "fracs": np.geomspace(1, args.fracs_min, args.n_fracs),
        "wantfileoutputs": [0, 0, 0, 0], "wantmemoryoutputs": want,
        "seed": 0,
    }
    if len(sessions) < 2:
        raise RuntimeError("Need >= 2 sessions: repeats across sessions drive GLMsingle's cross-validation.")
    print(f"Fitting GLMsingle: stimdur={args.stimdur}s, tr={dt_up:.4f}s, folds={len(xvalscheme)} (sessions)")
    raw_dir = args.raw_dir or os.path.join(work_dir, "glmsingle_raw")
    glm = GLM_single(params)
    results = glm.fit(designs, data, args.stimdur, dt_up,
                      outputdir=raw_dir,
                      figuredir=os.path.join(work_dir, "figures"))

    def full(vec):
        out = np.zeros(n_nodes, dtype=np.float32)
        out[valid] = np.asarray(vec, dtype=np.float32).reshape(-1)
        return out

    key = {"B": "typeb", "C": "typec", "D": "typed"}
    summary = {"n_runs": len(runs), "n_trials": int(len(manifest)), "n_conditions": len(cond_keys),
               "upsample": args.upsample, "tr_model": dt_up, "stimdur": args.stimdur,
               "events_variant": args.events_variant, "vertices_fitted": int(valid.sum())}
    res_d = results[key["D"]]
    if "pcnum" in res_d:
        summary["pcnum"] = int(np.asarray(res_d["pcnum"]).squeeze())
    if "FRACvalue" in res_d:
        fr = np.asarray(res_d["FRACvalue"]).reshape(-1)
        summary["fracvalue_median"] = float(np.median(fr))
        summary["frac_at_floor"] = float(np.mean(fr <= args.fracs_min + 1e-6))

    # summary maps (type D fit)
    maps, names = [], []
    for name in ("R2", "HRFindex", "FRACvalue", "noisepool", "meanvol"):
        src = res_d if name in res_d else results.get("typeb", {})
        if name in src:
            maps.append(full(src[name]))
            names.append(name)
    save_multidscalar_from_dtseries(dt_ref, np.vstack(maps), names, os.path.join(work_dir, "glmsingle_summary.dscalar.nii"))
    json.dump(summary, open(os.path.join(work_dir, "glmsingle_fit_summary.json"), "w"), indent=2)
    print("Fit summary:", summary)

    # ---- single-trial betas -> per-event dscalars ----
    for T in args.save_types:
        betas = np.asarray(results[key[T]]["betasmd"], dtype=np.float32).reshape(int(valid.sum()), -1)
        if betas.shape[1] != len(manifest):
            raise RuntimeError(f"type {T}: {betas.shape[1]} betas but {len(manifest)} trials in manifest")
        n_written = 0
        for (ses, taskdir), rows in manifest.groupby(["session", "taskdir"], sort=False):
            out_dir = os.path.join(model_dirs[T], ses, taskdir)
            os.makedirs(out_dir, exist_ok=True)
            multi = []
            for _, m in rows.iterrows():
                vec = full(betas[:, m["trial_index"]])
                out_beta = os.path.join(out_dir, m["beta_dscalar"])
                if args.overwrite or not os.path.exists(out_beta):
                    save_dscalar_from_dtseries(dt_ref, vec, out_beta, scalar_name=m["beta_dscalar"][:-len("_beta.dscalar.nii")])
                    n_written += 1
                multi.append(vec)
            rows.to_csv(os.path.join(out_dir, "glmsingle_manifest.tsv"), sep="\t", index=False)
            if args.save_multidscalar:
                save_multidscalar_from_dtseries(dt_ref, np.vstack(multi), rows["event_id"].tolist(),
                                                os.path.join(out_dir, "glmsingle_betas_all_events.dscalar.nii"))
        print(f"type {T}: wrote {n_written} beta dscalars under {model_dirs[T]}")

    print(f"All done. Manifest + summary in {work_dir}")


if __name__ == "__main__":
    main()

"""
Example (all runs, final-delay events, type D betas):
python ./glmsingle_fit.py \
  --sub sub-01 --sessions ses-01 ses-02 ses-03 ses-04 \
  --events_variant wfdelay \
  --add_fd \
  --save_types D
"""
