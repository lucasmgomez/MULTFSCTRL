#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser("QC: ROI-averaged sanity checks for LSS beta-series (save figures).")

    p.add_argument("--lss_beta_dir", required=True,
                   help="Directory containing *_beta-target.dscalar.nii from one run.")
    p.add_argument("--dlabel", required=True,
                   help="Glasser/HCP-MMP dlabel in the SAME vertex space as your dtseries/dscalars.")

    # NEW: strict ROI selection by exact label names
    p.add_argument("--roi_names", required=True,
                   help=("Comma-separated list of exact label names to include, e.g. "
                         "'L_V1_ROI,R_V1_ROI' or 'L_FEF_ROI'."))

    p.add_argument("--fig_dir", required=True,
                   help="Directory where figures will be saved.")
    p.add_argument("--out_prefix", default="qc_roi",
                   help="Prefix for output files.")
    p.add_argument("--lss_beta_dir_2", default=None,
                   help="Optional second LSS beta directory for test–retest QC.")

    return p.parse_args()


def make_roi_mask(dlabel_path, roi_names):
    """
    Strict mask: include ONLY the provided roi_names (exact string match).
    roi_names: list[str]
    """
    dl = nib.load(dlabel_path)

    # label id per vertex
    data = dl.get_fdata().squeeze().astype(int)

    # dict: label_id -> (label_name, rgba)
    label_dict = dl.header.get_axis(0).label[0]

    roi_names_set = set([r.strip() for r in roi_names if r.strip()])
    if not roi_names_set:
        raise ValueError("roi_names list is empty after parsing.")

    # find keys whose label_name exactly matches one of roi_names
    keys = [k for k, (name, _) in label_dict.items() if name in roi_names_set]

    found_names = [label_dict[k][0] for k in keys]
    missing = sorted(list(roi_names_set - set(found_names)))

    if missing:
        # Help user debug by showing a few close matches
        all_names = sorted([name for _, (name, _) in label_dict.items()])
        suggestions = [n for n in all_names if any(m.lower() in n.lower() for m in missing)]
        suggestions = suggestions[:30]
        raise ValueError(
            f"Missing ROI labels in dlabel: {missing}\n"
            f"Found: {found_names}\n"
            f"Suggestions (partial contains): {suggestions}"
        )

    mask = np.isin(data, keys)

    print("Matched ROI labels:", found_names)
    print("Total ROI vertices:", int(mask.sum()))

    if mask.sum() == 0:
        raise ValueError("ROI mask has 0 vertices. Likely space mismatch or wrong label names.")

    return mask, found_names


def parse_event(fname):
    m = re.search(r"lss-(EncTarget|DelayTarget)_(Enc\d{4}|Del\d{4})", fname)
    if not m:
        return None
    target, eid = m.groups()
    phase = "Enc" if eid.startswith("Enc") else "Del"
    return phase, target, int(eid[3:])


def collect_roi_betas(beta_dir, roi_mask, roi_colname="roi_beta_mean"):
    rows = []
    for f in sorted(glob.glob(os.path.join(beta_dir, "*_beta-target.dscalar.nii"))):
        parsed = parse_event(os.path.basename(f))
        if parsed is None:
            continue
        phase, target, idx = parsed
        beta = nib.load(f).get_fdata().squeeze().astype(float)

        rows.append({
            "file": os.path.basename(f),
            "phase": phase,
            "target": target,
            "event_index": idx,
            roi_colname: float(beta[roi_mask].mean())
        })

    if not rows:
        raise ValueError("No valid LSS beta files found (or naming pattern mismatch).")

    return pd.DataFrame(rows).sort_values(["phase", "event_index"])


def save_distribution_plot(df, fig_dir, prefix, roi_label, roi_col):
    enc = df[df.phase == "Enc"][roi_col]
    dele = df[df.phase == "Del"][roi_col]

    plt.figure(figsize=(10, 4))
    plt.hist(enc, bins=30, alpha=0.6, label=f"Encoding (n={len(enc)})")
    plt.hist(dele, bins=30, alpha=0.6, label=f"Delay (n={len(dele)})")
    if len(enc) > 0:
        plt.axvline(enc.mean(), ls="--")
    if len(dele) > 0:
        plt.axvline(dele.mean(), ls=":")
    plt.legend()
    plt.xlabel(f"Mean {roi_label} beta")
    plt.ylabel("Count")
    plt.title(f"{roi_label} LSS beta distributions")

    out = os.path.join(fig_dir, f"{prefix}_dist_enc_vs_del.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def save_betaseries_plot(df, fig_dir, prefix, roi_label, roi_col):
    plt.figure(figsize=(12, 4))
    for phase in ["Enc", "Del"]:
        d = df[df.phase == phase]
        if not d.empty:
            plt.plot(d.event_index, d[roi_col], marker="o", label=phase)

    plt.xlabel("Event index")
    plt.ylabel(f"Mean {roi_label} beta")
    plt.title(f"{roi_label} LSS beta-series")
    plt.legend()

    out = os.path.join(fig_dir, f"{prefix}_betaseries.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def save_test_retest(df1, df2, fig_dir, prefix, roi_label, roi_col):
    for phase in ["Enc", "Del"]:
        a = df1[df1.phase == phase].set_index("event_index")[roi_col]
        b = df2[df2.phase == phase].set_index("event_index")[roi_col]
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 5:
            continue

        r = np.corrcoef(a.loc[common], b.loc[common])[0, 1]

        plt.figure(figsize=(5, 5))
        plt.scatter(a.loc[common], b.loc[common], alpha=0.7)
        plt.xlabel(f"Run 1 {roi_label} beta")
        plt.ylabel(f"Run 2 {roi_label} beta")
        plt.title(f"{phase} test–retest r={r:.2f}")

        out = os.path.join(fig_dir, f"{prefix}_testretest_{phase}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        print("Saved:", out)


def main():
    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    roi_names = [x.strip() for x in args.roi_names.split(",") if x.strip()]
    roi_mask, found_names = make_roi_mask(args.dlabel, roi_names)

    # pretty label for plots
    roi_label = "+".join(found_names)
    # column name in outputs
    roi_col = "roi_beta_mean"

    df = collect_roi_betas(args.lss_beta_dir, roi_mask, roi_colname=roi_col)

    df_out = os.path.join(args.fig_dir, f"{args.out_prefix}_{'_'.join(found_names)}_betas.tsv")
    df.to_csv(df_out, sep="\t", index=False)
    print("Saved:", df_out)

    save_distribution_plot(df, args.fig_dir, args.out_prefix, roi_label, roi_col)
    save_betaseries_plot(df, args.fig_dir, args.out_prefix, roi_label, roi_col)

    if args.lss_beta_dir_2:
        df2 = collect_roi_betas(args.lss_beta_dir_2, roi_mask, roi_colname=roi_col)
        save_test_retest(df, df2, args.fig_dir, args.out_prefix, roi_label, roi_col)


if __name__ == "__main__":
    main()

"""
python qc_lss_betaseries_save.py \
  --lss_beta_dir /mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lss/64kDense/sub-01/ses-01/task-ctxdm_acq-col_run-01 \
  --dlabel /home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii \
  --roi_names "L_46_ROI" \
  --fig_dir /mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lss/64kDense/sub-01/ses-01/task-ctxdm_acq-col_run-01/figs \
  --out_prefix qc_46
"""