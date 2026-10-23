#!/usr/bin/env python3
"""
Collinearity of single-trial regressors: CNM vs TR event timing (decision #3 in HANDOFF.md).

Each event = 1.5 s boxcar (GLMsingle stimdur) * Glover HRF, sampled at TR 1.49 s with frame times
i*TR + 0.686 s; intercept + cubic drift projected out. Prints the median variance inflation factor (VIF) per
phase/position (1 = independent; a beta's standard error scales with sqrt(VIF)).

  python vif_check.py --cnm_log_dir <project_data>/behavior/sub-01/ses-001/func \
                      --tr_events_dir <TR root>/behav/sub-01/ses-01/events_wfdelay    (optional)
"""
import argparse
import glob

import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import glover_hrf

TR, START, DT = 1.49, 0.686, 0.01


def median_vif(ev, nvol):
    hrf = glover_hrf(TR, oversampling=int(TR / DT), time_length=32.0)
    t_hi = np.arange(0, nvol * TR + 40, DT)
    frames = np.arange(nvol) * TR + START
    X = np.column_stack([np.interp(frames, t_hi, np.convolve(((t_hi >= on) & (t_hi < on + 1.5)).astype(float),
                                                             hrf)[: len(t_hi)]) for on in ev.onset])
    tt = np.linspace(-1, 1, nvol)
    N = np.column_stack([tt ** k for k in range(4)])
    Xr = X - N @ np.linalg.lstsq(N, X, rcond=None)[0]
    vif = np.diag(np.linalg.pinv(Xr.T @ Xr)) * (Xr ** 2).sum(0)
    s = ev.assign(vif=vif).groupby(["phase", "pos"]).vif.median().round(1)
    return "  ".join(f"{ph[:3]}{p}={v}" for (ph, p), v in s.items())


def cnm_events(raw_tsv, n_pos):
    x = pd.read_csv(raw_tsv, sep="\t", encoding="utf-8-sig")
    x.columns = [c.strip() for c in x.columns]
    rows = [dict(onset=x[f"stimulus_{p}_{edge}"][i], phase=ph, pos=p + 1)
            for i in range(len(x)) for p in range(n_pos)
            for edge, ph in (("onset", "Encoding"), ("offset", "Delay"))]
    return pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnm_log_dir", required=True, help="func/ folder of one CNM session's raw block logs")
    ap.add_argument("--tr_events_dir", default=None, help="TR events_wfdelay folder for comparison")
    args = ap.parse_args()
    for task, n_pos, nvol, tr_stem in (("1back", 6, 188, "task-1back_ctg_run-01"),
                                       ("interdms", 4, 208, "task-interdms_ctg_ABAB_run-01")):
        ev = cnm_events(glob.glob(f"{args.cnm_log_dir}/*task-{task}_run-01_events.tsv")[0], n_pos)
        print(f"--- {task}")
        print("CNM all delays      ", median_vif(ev, nvol))
        print("CNM pos-1 delay out ", median_vif(ev[~((ev.phase == "Delay") & (ev.pos == 1))], nvol))
        if args.tr_events_dir:
            tr = pd.read_csv(f"{args.tr_events_dir}/{tr_stem}_base-events.tsv", sep="\t")
            print("TR as modelled      ", median_vif(tr, int(np.ceil((tr.onset.max() + 18) / TR))))


if __name__ == "__main__":
    main()
