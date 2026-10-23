#!/usr/bin/env python3
"""
Build the CNM tree in TR's layout (see CLAUDE.md -> Converting another dataset, and HANDOFF.md -> Phase 2).

One row per original run in <root>/mapping.csv (also written to --mapping_copy):
  original log / design block / BIDS name -> final acq, new session (ses-XX), new run, new paths, include flag.

  behav/<sub>/ses-XX/<sub>_ses-<n>_<YYYYMMDD-HHMMSS>_task-<task>_<acq>_block_<run-1>_events_<HH-MM-SS>.tsv
      raw block log in TR's format: loc/ref/obj/ctg/ang per position (from locmod/objmod/ctgmod/angmod; dms has
      loc/objmod only, so ctg/ang are empty), tc (TR-style, as float), responses x -> 2, b -> 3, missing -> empty.
      Other columns are kept. Date = first scan of the session, time = the run's BOLD acq_time (scans.tsv).
  neural/fmriprep_outs/<prep_run>/
      64kDense/<sub>/ses-XX/<sub>_ses-XX_task-<task>_acq-<acqfp>_run-<rr>_space-Glasser64k_bold.dtseries.nii
      confounds/<sub>/ses-XX/..._desc-confounds_timeseries.tsv
      derivatives/<sub>/ses-XX/func/..._space-fsLR_den-91k_bold.json   (fMRIPrep sidecar; no StartTime)
      bids/<sub>/ses-XX/func/..._bold.json                             (raw sidecar with SliceTiming)
  Neural files are copied (--link_mode copy, default) or symlinked.

Fixes (HANDOFF.md -> Data problems, agreed 2026-10-23):
  - sub-06 ses-007 takes ses-006's acq labels (same trial sets; responses follow ses-006's rules);
  - sub-05 ses-012 dms run-01 / interdms run-01 are empty template logs: include=0, nothing written;
  - dms has no design row: acq = loc (location-match task).
"""
import argparse
import os
import re
import shutil

import pandas as pd

NPOS = {"1back": 6, "interdms": 4, "ctxdm": 3, "dms": 2}
SCORED_RESP = {"1back": [1, 2, 3, 4, 5], "interdms": [2, 3], "ctxdm": [2], "dms": [1]}  # response_{i} the scorer reads
RESP_CODE = {"x": "2", "b": "3"}  # x = match, b = non-match (HANDOFF.md); anything else (e.g. '--') -> empty
RUN_RE = re.compile(r"(sub-\d+)_(ses-\d+)_task-(\w+?)_run-(\d+)_events\.tsv")
RELABEL = {("sub-06", "ses-007"): "ses-006"}   # session whose acq labels to use

CNM_SHARED = "/home/lucasg/links/projects/def-bashivan/shared"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project_data", default="/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/"
                                              "fmri_dataset_project_data",
                    help="behavior/, study_designs/, glasser_resampled/")
    ap.add_argument("--fmriprep_root", default=f"{CNM_SHARED}/cneuromod.multfs.fmriprep",
                    help="fMRIPrep 20.2.6 derivatives: <sub>/<ses>/func/ confounds + sidecars")
    ap.add_argument("--raw_root", default=f"{CNM_SHARED}/cneuromod.multfs.fmriprep/sourcedata/cneuromod.multfs.raw",
                    help="raw BIDS: <sub>/<ses>/func/*_bold.json")
    ap.add_argument("--scans_root", required=True, help="<sub>/<ses>/<sub>_<ses>_scans.tsv (acq_time)")
    ap.add_argument("--root", required=True, help="new dataset root (behav/, neural/)")
    ap.add_argument("--prep_run", default="fmriprep_20.2.6")
    ap.add_argument("--subs", nargs="+", default=["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"])
    ap.add_argument("--link_mode", default="copy", choices=["copy", "symlink"])
    ap.add_argument("--mapping_copy", default=None, help="also write the mapping table here (e.g. the repo)")
    ap.add_argument("--dry_run", action="store_true", help="mapping table only")
    return ap.parse_args()


def read_log(path):
    """Raw iWISDM log as strings (values kept verbatim), or None for empty BIDS-template stubs."""
    raw = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]                         # some logs are space-padded
    raw = raw.apply(lambda c: c.str.strip())
    return raw if "TrialNumber" in raw.columns else None


def load_designs(project_data):
    rows = []
    for sub in ("sub-01", "sub-02", "sub-03", "sub-05", "sub-06"):
        d = pd.read_csv(f"{project_data}/study_designs/{sub}_design_design_with_converted.tsv", sep="\t", dtype=str)
        d.columns = [c.strip() for c in d.columns]
        rows.append(d.apply(lambda c: c.str.strip()))
    des = pd.concat(rows)
    des = pd.concat([des, des.block_file_name.str.extract(r"^(?P<dtask>[a-z0-9]+)_(?P<acq>.+)_block_(?P<block>\d+)$")],
                    axis=1)
    assert des.converted_file_name.is_unique
    return des.set_index("converted_file_name")


def to_tr_log(raw, task):
    """CNM log (strings) -> TR raw-log columns first, then every other original column."""
    P = NPOS[task]
    out, used = {"TrialNumber": raw["TrialNumber"]}, {"TrialNumber"}
    empty = pd.Series([""] * len(raw), index=raw.index)
    for p in range(1, P + 1):
        loc_src = f"loc{p}" if task == "dms" else f"locmod{p}"
        if task != "dms" and f"loc{p}" in raw.columns:
            assert (raw[f"loc{p}"] == raw[f"locmod{p}"]).all(), f"loc{p} != locmod{p}"
            used.add(f"loc{p}")
        out[f"loc{p}"] = raw[loc_src]
        out[f"ref{p}"] = raw[f"ref{p}"]
        out[f"obj{p}"] = raw[f"objmod{p}"]
        out[f"ctg{p}"] = empty if task == "dms" else raw[f"ctgmod{p}"]
        out[f"ang{p}"] = empty if task == "dms" else raw[f"angmod{p}"]
        used |= {loc_src, f"ref{p}", f"objmod{p}", f"ctgmod{p}", f"angmod{p}"}
    # tc as in TR: per position the loc digit then the obj digit, stored as a float (leading zeros drop)
    tc = ["".join(f"{int(r[f'loc{p}'])}{int(r[f'obj{p}'])}" for p in range(1, P + 1))
          for _, r in pd.DataFrame(out).iterrows()]
    out["tc"] = pd.Series([f"{float(int(t)):.1f}" for t in tc], index=raw.index)
    df = pd.DataFrame(out)
    for c in raw.columns:
        if c not in used:
            df[c] = raw[c]
    for i in SCORED_RESP[task]:
        col = f"response_{i}"
        df[col] = df[col].map(lambda v: RESP_CODE.get(v, "")) if col in df.columns else ""
    return df


def acq_times(scans_root, sub, ses):
    s = pd.read_csv(f"{scans_root}/{sub}/{ses}/{sub}_{ses}_scans.tsv", sep="\t")
    s["t"] = pd.to_datetime(s["acq_time"], format="ISO8601")
    return s.set_index("filename")["t"], s["t"].min()


def put(src, dst, mode):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    des = load_designs(args.project_data)
    nb = f"{args.root}/neural/fmriprep_outs/{args.prep_run}"
    rows = []
    for sub in args.subs:
        logs = sorted(f for f in os.listdir(f"{args.project_data}/behavior/{sub}") if f.startswith("ses-"))
        for ses in logs:
            fdir = f"{args.project_data}/behavior/{sub}/{ses}/func"
            new_ses = f"ses-{int(ses.split('-')[1]):02d}"
            times, ses_start = acq_times(args.scans_root, sub, ses)
            for fname in sorted(os.listdir(fdir)):
                m = RUN_RE.match(fname)
                if not m:
                    continue
                _, _, task, run = m.groups()
                row = dict(sub=sub, orig_ses=ses, task=task, orig_run=int(run), orig_log=f"{fdir}/{fname}",
                           new_ses=new_ses, include=1, notes="")
                raw = read_log(row["orig_log"])
                label_ses = RELABEL.get((sub, ses), ses)
                key = fname.replace(ses, label_ses)
                row["design_block"] = des.block_file_name.get(fname, "")
                if task == "dms":
                    row["acq"] = "loc"
                else:
                    row["acq"] = des.acq[key]
                if label_ses != ses and task != "dms":
                    row["notes"] = f"acq from {label_ses} (design said {des.acq[fname]})"
                if raw is None:
                    row.update(include=0, notes="empty template log (no trials); run dropped")
                bold = f"func/{sub}_{ses}_task-{task}_run-{run}_bold.nii.gz"
                row["acq_time"] = times[bold].isoformat()
                row["_raw"] = raw
                row["_ses_start"] = ses_start
                rows.append(row)

    mp = pd.DataFrame(rows)
    # new run = order within (sub, new_ses, task, acq), by original run number, included runs only
    mp = mp.sort_values(["sub", "new_ses", "task", "orig_run"]).reset_index(drop=True)
    mp["new_run"] = 0
    inc = mp.include == 1
    mp.loc[inc, "new_run"] = mp[inc].groupby(["sub", "new_ses", "task", "acq"]).cumcount() + 1
    for i, r in mp.iterrows():
        n = int(r.new_ses.split("-")[1])
        acqfp = r.acq.replace("_", "")
        o = f"{r['sub']}_{r.orig_ses}_task-{r.task}_run-{r.orig_run}"           # fMRIPrep names use run-1
        mp.loc[i, "dtseries_src"] = f"{args.project_data}/glasser_resampled/{r['sub']}/{r.orig_ses}/{o}_space-Glasser64k_bold.dtseries.nii"
        mp.loc[i, "confounds_src"] = f"{args.fmriprep_root}/{r['sub']}/{r.orig_ses}/func/{o}_desc-confounds_timeseries.tsv"
        mp.loc[i, "deriv_json_src"] = f"{args.fmriprep_root}/{r['sub']}/{r.orig_ses}/func/{o}_space-fsLR_den-91k_bold.json"
        mp.loc[i, "bids_json_src"] = f"{args.raw_root}/{r['sub']}/{r.orig_ses}/func/{r['sub']}_{r.orig_ses}_task-{r.task}_run-{r.orig_run:02d}_bold.json"
        if not r.include:
            continue
        rr = f"{int(r.new_run):02d}"
        stem = f"{r['sub']}_{r.new_ses}_task-{r.task}_acq-{acqfp}_run-{rr}"
        t = pd.Timestamp(r.acq_time)
        mp.loc[i, "new_log"] = (f"{args.root}/behav/{r['sub']}/{r.new_ses}/{r['sub']}_ses-{n}_"
                                f"{r._ses_start:%Y%m%d-%H%M%S}_task-{r.task}_{r.acq}_block_{int(r.new_run) - 1}_"
                                f"events_{t:%H-%M-%S}.tsv")
        mp.loc[i, "dtseries_dst"] = f"{nb}/64kDense/{r['sub']}/{r.new_ses}/{stem}_space-Glasser64k_bold.dtseries.nii"
        mp.loc[i, "confounds_dst"] = f"{nb}/confounds/{r['sub']}/{r.new_ses}/{stem}_desc-confounds_timeseries.tsv"
        mp.loc[i, "deriv_json_dst"] = f"{nb}/derivatives/{r['sub']}/{r.new_ses}/func/{stem}_space-fsLR_den-91k_bold.json"
        mp.loc[i, "bids_json_dst"] = f"{nb}/bids/{r['sub']}/{r.new_ses}/func/{stem}_bold.json"

    dst_cols = ["new_log", "dtseries_dst", "confounds_dst", "deriv_json_dst", "bids_json_dst"]
    for c in dst_cols:
        assert mp[c].dropna().is_unique, f"duplicate {c}"
    cols = ["sub", "orig_ses", "task", "orig_run", "design_block", "acq", "new_ses", "new_run", "include", "notes",
            "acq_time", "orig_log", "new_log", "dtseries_src", "dtseries_dst", "confounds_src", "confounds_dst",
            "deriv_json_src", "deriv_json_dst", "bids_json_src", "bids_json_dst"]
    table = mp[cols]
    os.makedirs(args.root, exist_ok=True)
    for out in [f"{args.root}/mapping.csv"] + ([args.mapping_copy] if args.mapping_copy else []):
        table.to_csv(out, index=False)
    print(table.groupby(["sub", "include"]).size().to_string())
    print(pd.crosstab([mp.task, mp.acq], mp.new_run).to_string())
    if args.dry_run:
        return

    for _, r in mp[mp.include == 1].iterrows():
        os.makedirs(os.path.dirname(r.new_log), exist_ok=True)
        to_tr_log(r._raw, r.task).to_csv(r.new_log, sep="\t", index=False, encoding="utf-8-sig")
        for k in ("dtseries", "confounds", "deriv_json", "bids_json"):
            put(r[f"{k}_src"], r[f"{k}_dst"], args.link_mode)
    print(f"wrote {int(mp.include.sum())} runs under {args.root}")


if __name__ == "__main__":
    main()
