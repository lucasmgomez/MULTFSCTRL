#!/usr/bin/env python3
"""
Read-only inventory of the CNeuroMod multfs dataset (CNM) against the TR contract (see HANDOFF.md).

Inputs
  --project_data  fmri_dataset_project/data folder, containing
                    behavior/<sub>/<ses>/func/*_events.tsv       raw iWISDM block logs (primary behaviour source)
                    study_designs/<sub>_design_design_with_converted.tsv   run -> block file (acq label)
                    glasser_resampled/<sub>/<ses>/*Glasser64k_bold.dtseries.nii + *confounds_timeseries.tsv
  --confounds_root  optional fMRIPrep derivatives root (<sub>/<ses>/func/*desc-confounds_timeseries.tsv); default:
                  the confounds next to the dtseries in glasser_resampled (the Rorqual copy has none there).
  --bids_root     optional; BIDS dataset with <sub>/<ses>/func/*_events.tsv and derivatives/fmriprep/...
                  Used only to cross-check event onsets/features and read fMRIPrep sidecars (SliceTimingCorrected).

Outputs (in --out): runs.tsv (one row per run), trials.tsv (one row per trial), timing.tsv (one row per stimulus).

Label checks use the TR scoring rules (prep/behav/score_responses.py); responses: x = match, b = non-match.
"""
import argparse
import glob
import json
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd

F = ["loc", "ctg", "obj"]
NPOS = {"1back": 6, "interdms": 4, "ctxdm": 3, "dms": 2}
# action column holding iWISDM's expected match flag for each scored position (1-based)
ACTION_COL = {"1back": {p: f"action{p}" for p in range(2, 7)}, "interdms": {3: "action1", 4: "action2"},
              "ctxdm": {3: "action"}, "dms": {2: "action"}}
RUN_RE = re.compile(r"(sub-\d+)_(ses-\d+)_task-(\w+?)_run-(\d+)_events\.tsv")


def rule_truth(tr, task, acq):
    """{pos: is_match} under TR's scoring rules."""
    g = lambda f, p: tr[f"{f}{p}"]
    if task == "1back":
        return {p: g(acq, p) == g(acq, p - 1) for p in range(2, 7)}
    if task == "interdms":
        f, order = acq.split("_")
        pairs = [(3, 1), (4, 2)] if order == "ABAB" else [(3, 2), (4, 1)]
        return {p: g(f, p) == g(f, q) for p, q in pairs}
    if task == "ctxdm":
        if acq == "col":
            f = "obj" if g("ctg", 1) == g("ctg", 2) else "loc"
        else:  # lco
            f = "ctg" if g("loc", 1) == g("loc", 2) else "obj"
        return {3: g(f, 2) == g(f, 3)}
    if task == "dms":
        return {2: g(acq, 2) == g(acq, 1)}
    raise ValueError(task)


def candidates(task):
    return {"1back": F, "interdms": [f"{f}_{o}" for f in F for o in ("ABAB", "ABBA")],
            "ctxdm": ["col", "lco"], "dms": ["loc", "obj"]}[task]


def as_bool(v):
    if pd.isna(v):
        return np.nan
    return str(v).strip().lower() in ("true", "1", "1.0")


def read_raw_log(path, task):
    """Raw block log -> DataFrame with loc{p}/ctg{p}/obj{p}, or None for empty BIDS-template stubs."""
    raw = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]                     # some logs are space-padded
    raw = raw.apply(lambda c: c.str.strip() if c.dtype == object else c)
    if "TrialNumber" not in raw.columns:
        return None
    raw = raw.sort_values("TrialNumber").reset_index(drop=True)
    for p in range(1, NPOS[task] + 1):
        if task == "dms":           # dms logs only have loc{p} and objmod{p}; no category
            raw[f"ctg{p}"] = np.nan
        else:
            raw[f"loc{p}"] = raw[f"locmod{p}"]
            raw[f"ctg{p}"] = raw[f"ctgmod{p}"]
        raw[f"obj{p}"] = raw[f"objmod{p}"]
    return raw


def load_designs(project_data):
    rows = []
    for f in sorted(glob.glob(f"{project_data}/study_designs/sub-*_design_design_with_converted.tsv")):
        d = pd.read_csv(f, sep="\t")
        d.columns = [c.strip() for c in d.columns]
        rows.append(d.apply(lambda c: c.str.strip() if c.dtype == object else c))
    des = pd.concat(rows)
    des = pd.concat([des, des.block_file_name.str.extract(r"^(?P<dtask>[a-z0-9]+)_(?P<acq>.+)_block_(?P<block>\d+)$")],
                    axis=1)
    assert des.converted_file_name.is_unique
    return des.set_index("converted_file_name")


def score_run(raw, task):
    """For each candidate acq: (fit of rule to iWISDM action, fit of rule to subject responses, n responses)."""
    out = {}
    for cand in candidates(task):
        a_ok = a_n = r_ok = r_n = 0
        for _, r in raw.iterrows():
            for p, match in rule_truth(r, task, cand).items():
                act = as_bool(r.get(ACTION_COL[task][p]))
                if act == act:
                    a_ok += act == match
                    a_n += 1
                resp = r.get(f"response_{p - 1}")
                if resp in ("x", "b"):
                    r_ok += (resp == "x") == match
                    r_n += 1
        out[cand] = (a_ok / max(a_n, 1), r_ok / max(r_n, 1) if r_n else np.nan, r_n)
    return out


def bids_check(bids_root, sub, ses, fname, raw, P):
    ev_path = f"{bids_root}/{sub}/{ses}/func/{fname}"
    if not os.path.exists(ev_path):
        return "no-bids-events"
    e = pd.read_csv(ev_path, sep="\t", na_values="n/a")
    enc = e[e.trial_type == "encoding"].sort_values(["trial_number", "stim_order"])
    on_raw = np.array([raw[f"stimulus_{p - 1}_onset"].values for p in range(1, P + 1)]).T.ravel()
    if len(enc) != len(on_raw):
        return f"n_enc {len(enc)} vs {len(on_raw)}"
    dt = np.abs(enc.onset.values - on_raw).max()
    feats_ok = all((enc[c].values == np.array([raw[f"{k}{p}"].values for p in range(1, P + 1)]).T.ravel()).all()
                   for k, c in (("loc", "location"), ("obj", "object")))
    return f"ok(dt={dt:.4f})" if dt < 0.01 and feats_ok else f"MISMATCH(dt={dt:.3f},feats={feats_ok})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project_data", required=True)
    ap.add_argument("--confounds_root", default=None)
    ap.add_argument("--bids_root", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    des = load_designs(args.project_data)
    runs, trials, timing = [], [], []
    for f in sorted(glob.glob(f"{args.project_data}/behavior/sub-*/ses-*/func/*_events.tsv")):
        fname = os.path.basename(f)
        sub, ses, task, run = RUN_RE.match(fname).groups()
        P = NPOS[task]
        row = dict(sub=sub, ses=ses, task=task, run=int(run), log=f)
        raw = read_raw_log(f, task)
        if raw is None:
            runs.append(dict(row, status="STUB-LOG", ntrials=0))
            continue
        acq = des.acq.get(fname, "loc" if task == "dms" else None)   # dms has no design row; it is a loc-match task
        block = des.block.get(fname)
        sc = score_run(raw, task)
        best_action = max(sc, key=lambda c: sc[c][0])
        best_resp = max(sc, key=lambda c: -1 if np.isnan(sc[c][1]) else sc[c][1])
        raw["tc"] = raw.apply(lambda r: "".join(f"{int(r[f'loc{p}'])}{int(r[f'obj{p}'])}" for p in range(1, P + 1)),
                              axis=1)
        on = np.array([raw[f"stimulus_{p}_onset"].values for p in range(P)]).T
        off = np.array([raw[f"stimulus_{p}_offset"].values for p in range(P)]).T
        for i in range(len(raw)):
            for p in range(P):
                nxt = on[i, p + 1] if p + 1 < P else (on[i + 1, 0] if i + 1 < len(raw) else np.nan)
                timing.append(dict(sub=sub, task=task, pos=p + 1, last=p + 1 == P, enc=off[i, p] - on[i, p],
                                   gap=nxt - off[i, p]))
        # neural files from the existing Glasser64k resample (fMRIPrep names use run-1, logs use run-01)
        stem = f"{args.project_data}/glasser_resampled/{sub}/{ses}/{sub}_{ses}_task-{task}_run-{int(run)}"
        dts, conf = stem + "_space-Glasser64k_bold.dtseries.nii", stem + "_desc-confounds_timeseries.tsv"
        if args.confounds_root:
            conf = f"{args.confounds_root}/{sub}/{ses}/func/{sub}_{ses}_task-{task}_run-{int(run)}_desc-confounds_timeseries.tsv"
        nvol = nib.load(dts).header.get_axis(0).size if os.path.exists(dts) else np.nan
        nconf = len(pd.read_csv(conf, sep="\t", usecols=[0])) if os.path.exists(conf) else np.nan
        stc = None
        if args.bids_root:
            js = (f"{args.bids_root}/derivatives/fmriprep/{sub}/{ses}/func/"
                  f"{sub}_{ses}_task-{task}_run-{int(run)}_space-fsLR_den-91k_bold.json")
            stc = json.load(open(js)).get("SliceTimingCorrected") if os.path.exists(js) else None
        runs.append(dict(
            row, status="ok", ntrials=len(raw), acq=acq, block=block,
            action_fit=round(sc[acq][0], 3) if acq in sc else np.nan, best_by_action=best_action,
            resp_fit=round(sc[acq][1], 3) if acq in sc else np.nan, best_by_resp=best_resp,
            best_resp_fit=round(sc[best_resp][1], 3), nresp=sc[best_resp][2],
            ctg_is_obj2=bool(task == "dms" or all((raw[f"ctg{p}"] == raw[f"obj{p}"] // 2).all()
                                                  for p in range(1, P + 1))),
            bids=bids_check(args.bids_root, sub, ses, fname, raw, P) if args.bids_root else None,
            nvol=nvol, nconf=nconf, stc=stc, first_on=on[0, 0], last_off=off[-1, -1]))
        t = raw[["TrialNumber", "tc"] + [f"{k}{p}" for p in range(1, P + 1) for k in F]].copy()
        for k, v in reversed(list(dict(sub=sub, ses=ses, task=task, run=int(run), acq=acq, block=block).items())):
            t.insert(0, k, v)
        trials.append(t)

    os.makedirs(args.out, exist_ok=True)
    runs, trials, timing = pd.DataFrame(runs), pd.concat(trials, ignore_index=True), pd.DataFrame(timing)
    runs.to_csv(f"{args.out}/runs.tsv", sep="\t", index=False)
    trials.to_csv(f"{args.out}/trials.tsv", sep="\t", index=False)
    timing.to_csv(f"{args.out}/timing.tsv", sep="\t", index=False)

    pd.set_option("display.width", 200)
    ok = runs[runs.status == "ok"]
    print("runs per subject x task:\n", pd.crosstab(runs["sub"], runs.task))
    print("\nstub logs:\n", runs[runs.status != "ok"][["sub", "ses", "task", "run"]].to_string(index=False))
    print("\nruns where design label != iWISDM action (ctxdm excluded: its action column is wrong):")
    print(ok[(ok.task != "ctxdm") & (ok.action_fit < 1)][["sub", "ses", "task", "run", "acq", "best_by_action",
                                                          "best_by_resp", "best_resp_fit"]].to_string(index=False))
    print("\nruns with no responses logged:\n", ok[ok.nresp == 0][["sub", "ses", "task", "run"]].to_string(index=False))
    print("\nnvol == confound rows:", (ok.nvol == ok.nconf).all(), " nvol:",
          ok.groupby("task").nvol.agg(lambda s: s.value_counts().to_dict()).to_dict())
    d = timing.dropna(subset=["gap"]).assign(gap=lambda x: x.gap.round(1))
    print("\ngap after each position (s), counts:\n", d.groupby(["task", "pos"]).gap.value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
