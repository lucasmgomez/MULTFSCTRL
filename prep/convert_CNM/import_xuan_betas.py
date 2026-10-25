#!/usr/bin/env python3
"""
Import the earlier CNM trial-level betas (fmri_dataset_project/data/encoding_delay/trial_level_betas: per-run OLS,
one regressor per encoding/delay event, made by Xuan) into our glm_runs layout, so the evaluation scripts can run
on them unchanged and be compared with our GLMsingle betas.

For each included run in <root>/mapping.csv, every event in our events_<variant> base events is matched to the
regressor of the same type (encoding/delay) whose onset in <run>_design.csv is within 0.02 s (both come from the
same raw logs); our events without a counterpart are reported. Written as
  <out_root>/<name>_<variant>/64kDense/<sub>/<ses>/task-<task>_acq-<acq>_run-<rr>/lsa_<task>_<acq>_<Enc|Del>_<eid>_beta.dscalar.nii
(the "lsa_" prefix is what feature.py's filename parser expects). Uses the run-N files (OLS, 2025-09-30), not
the older run-0N duplicates (WLS-AR1, 2025-09-04).
"""
import argparse
import os

import h5py
import nibabel as nib
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xuan_dir", default="/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/"
                                          "fmri_dataset_project_data/encoding_delay/trial_level_betas")
    ap.add_argument("--root", default="/home/lucasg/links/projects/ctb-bashivan/lucasg/trevor_ctrl/CNM")
    ap.add_argument("--out_root", required=True, help="glm_runs folder to write into")
    ap.add_argument("--name", default="xuanols")
    ap.add_argument("--variant", default="wfdelay")
    ap.add_argument("--sub", default="sub-01")
    args = ap.parse_args()

    mp = pd.read_csv(f"{args.root}/mapping.csv")
    mp = mp[(mp.include == 1) & (mp["sub"] == args.sub)]
    base = f"{args.out_root}/{args.name}_{args.variant}/64kDense/{args.sub}"
    n_ok, unmatched, manifest = 0, [], []
    for _, r in mp.iterrows():
        rr = f"{int(r.new_run):02d}"
        stem = f"{args.xuan_dir}/{r['sub']}/{r.orig_ses}/func/{r['sub']}_{r.orig_ses}_task-{r.task}_run-{r.orig_run}"
        des = pd.read_csv(f"{stem}_design.csv")
        with h5py.File(f"{stem}_betas.h5") as h:
            betas = h["betas"][:]
            assert betas.shape[1] == len(des) == len(h.attrs["task_regressor_names"]), stem
        ev = pd.read_csv(f"{args.root}/behav/{r['sub']}/{r.new_ses}/events_{args.variant}/"
                         f"task-{r.task}_{r.acq}_run-{rr}_base-events.tsv", sep="\t")
        bm = nib.load(r.dtseries_dst).header.get_axis(1)
        out = f"{base}/{r.new_ses}/task-{r.task}_acq-{r.acq}_run-{rr}"
        os.makedirs(out, exist_ok=True)
        for _, e in ev.iterrows():
            typ = "encoding" if e.phase == "Encoding" else "delay"
            hit = np.flatnonzero((des.type == typ).to_numpy() & (np.abs(des.onset_time - e.onset) < 0.02).to_numpy())
            if len(hit) != 1:
                unmatched.append((r.new_ses, r.task, r.acq, rr, e.event_id, len(hit)))
                continue
            name = f"lsa_{r.task}_{r.acq}_{e.phase[:3]}_{e.event_id}_beta"
            img = nib.cifti2.Cifti2Image(betas[:, hit[0]][None].astype(np.float32),
                                         nib.cifti2.Cifti2Header.from_axes((nib.cifti2.ScalarAxis([name]), bm)))
            nib.save(img, f"{out}/{name}.dscalar.nii")
            manifest.append(dict(session=r.new_ses, taskdir=os.path.basename(out), event_id=e.event_id,
                                 xuan_file=os.path.basename(stem), xuan_column=int(hit[0])))
            n_ok += 1
    os.makedirs(f"{base}/_import", exist_ok=True)
    pd.DataFrame(manifest).to_csv(f"{base}/_import/manifest.tsv", sep="\t", index=False)
    print(f"{args.sub}: wrote {n_ok} betas; unmatched events: {len(unmatched)}", unmatched[:10])


if __name__ == "__main__":
    main()
