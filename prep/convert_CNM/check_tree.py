#!/usr/bin/env python3
"""
Validate the CNM tree written by build_tree.py + slurm_make_events.sh (read-only).

Per included run in <root>/mapping.csv:
  - the new block log has the original's trials, stimulus timing and features (loc/obj/ctg from the *mod columns);
  - tc decodes back to loc{p}/obj{p}; ctg == obj // 2 (not dms);
  - the scored copy's expected_ans{k} equals iWISDM's action columns (ctxdm skipped: its action column only
    implements the same-context branch, see HANDOFF.md);
  - dtseries volumes == confound rows; copied files are byte-identical to their sources;
  - base events (events_wfdelay): one Encoding per stimulus, Delays only >= 1 s (dms keeps its pos-1 delay),
    final delay 3.98 s, onsets equal the log's stimulus onsets/offsets.
Prints one line per problem and a summary; exit code 1 if anything failed.
"""
import argparse
import filecmp
import glob
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

NPOS = {"1back": 6, "interdms": 4, "ctxdm": 3, "dms": 2}
# expected_ans index -> iWISDM action column (True = match)
ACTION = {"1back": {k: f"action{k + 1}" for k in range(1, 6)}, "interdms": {1: "action1", 2: "action2"},
          "dms": {1: "action"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM")
    ap.add_argument("--subs", nargs="+", default=None)
    ap.add_argument("--variant", default="wfdelay")
    ap.add_argument("--no_cmp", action="store_true", help="skip byte comparison of copied neural files")
    args = ap.parse_args()

    mp = pd.read_csv(f"{args.root}/mapping.csv")
    mp = mp[(mp.include == 1) & (mp["sub"].isin(args.subs) if args.subs else True)]
    bad = []
    for _, r in mp.iterrows():
        tag = f"{r['sub']} {r.new_ses} {r.task}_{r.acq} run-{int(r.new_run):02d}"
        P = NPOS[r.task]
        orig = pd.read_csv(r.orig_log, sep="\t", encoding="utf-8-sig")
        orig.columns = [c.strip() for c in orig.columns]
        new = pd.read_csv(r.new_log, sep="\t")
        scored = glob.glob(r.new_log[:-4] + "_scored.tsv")
        if len(new) != len(orig) or not (new.TrialNumber.values == orig.TrialNumber.values).all():
            bad.append(f"{tag}: trials differ")
            continue
        for p in range(1, P + 1):
            src = {"loc": "loc" if r.task == "dms" else "locmod", "obj": "objmod", "ctg": "ctgmod"}
            for f, o in src.items():
                if f == "ctg" and r.task == "dms":
                    continue
                if not (new[f"{f}{p}"].values == orig[f"{o}{p}"].values).all():
                    bad.append(f"{tag}: {f}{p} != {o}{p}")
            if r.task != "dms" and not (new[f"ctg{p}"] == new[f"obj{p}"] // 2).all():
                bad.append(f"{tag}: ctg{p} != obj{p} // 2")
        for c in [c for c in orig.columns if c.startswith("stimulus_")]:
            if not np.allclose(new[c].values, orig[c].values):
                bad.append(f"{tag}: {c} changed")
        code = new["tc"].map(lambda v: str(int(float(v))).zfill(2 * P))
        for p in range(1, P + 1):
            for f, d in (("loc", 0), ("obj", 1)):
                if not (code.str[2 * (p - 1) + d].astype(int).values == new[f"{f}{p}"].values).all():
                    bad.append(f"{tag}: tc does not decode to {f}{p}")
        if len(scored) != 1:
            bad.append(f"{tag}: {len(scored)} scored logs")
        else:
            sc = pd.read_csv(scored[0], sep="\t")
            for k, col in ACTION.get(r.task, {}).items():
                act = orig[col].astype(str).str.strip().str.lower().map({"true": 2, "false": 3})
                if not (sc[f"expected_ans{k}"].values == act.values).all():
                    n = int((sc[f"expected_ans{k}"].values != act.values).sum())
                    bad.append(f"{tag}: expected_ans{k} != {col} in {n} trials")
        # neural
        nvol = nib.load(r.dtseries_dst).header.get_axis(0).size
        nconf = len(pd.read_csv(r.confounds_dst, sep="\t", usecols=[0]))
        if nvol != nconf:
            bad.append(f"{tag}: {nvol} volumes vs {nconf} confound rows")
        if not args.no_cmp:
            for k in ("dtseries", "confounds", "deriv_json", "bids_json"):
                if not filecmp.cmp(r[f"{k}_src"], r[f"{k}_dst"], shallow=False):
                    bad.append(f"{tag}: {k} copy differs from source")
        # base events
        ev_path = os.path.join(os.path.dirname(r.new_log), f"events_{args.variant}",
                               f"task-{r.task}_{r.acq}_run-{int(r.new_run):02d}_base-events.tsv")
        ev = pd.read_csv(ev_path, sep="\t")
        enc, dl = ev[ev.phase == "Encoding"], ev[ev.phase == "Delay"]
        on = np.sort(new[[f"stimulus_{i}_onset" for i in range(P)]].values.ravel())
        if len(enc) != len(new) * P or not np.allclose(np.sort(enc.onset.values), on, atol=1e-3):
            bad.append(f"{tag}: encoding events don't match stimulus onsets")
        n_del = len(new) * (P if r.task == "dms" else P - 1)
        if len(dl) != n_del or (dl.duration < 1.0).any():
            bad.append(f"{tag}: {len(dl)} delays (expected {n_del}), min duration {dl.duration.min():.2f}")
        last = dl[dl.pos == P]
        if not np.allclose(last.duration, 3.98):
            bad.append(f"{tag}: final delay durations {sorted(last.duration.unique())}")
        if ev.onset.duplicated().any():
            bad.append(f"{tag}: duplicate onsets")

    for b in bad:
        print("[FAIL]", b)
    print(f"checked {len(mp)} runs: {len(bad)} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
