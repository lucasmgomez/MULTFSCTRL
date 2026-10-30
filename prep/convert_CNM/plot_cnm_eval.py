#!/usr/bin/env python3
"""Cross-subject figures for the CNM evaluation (outputs of slurm_evaluate.sh and diag/loc_by_run.py).
  python plot_cnm_eval.py --glm_runs <.../glm_runs> --subs sub-02 sub-03 sub-05 sub-06 --out_dir <dir>
Figures (one panel per subject; ROI groups DLPFC, IFC (= VLPFC), EVC, VTC, IT, whichever have results; extra ROI
groups from slurm_trillium_rois.sh are picked up automatically):
  noise_ceiling_<key>_<demean>.png      split-half noise ceiling, per-vertex vs ROI-mean signal (Encoding, Delay)
  noise_ceiling_parcels_<key>.png       per-vertex noise ceiling of every bilateral parcel (heatmap)
  decoding_<phase>_<config>.png         balanced accuracy for loc / ctg / obj, vertex patterns vs ROI mean
  leakage.png                           neighbour-stimulus decoding, GLMsingle D vs B (temporal leakage)
  loc_by_run.png                        EVC location decoding per run (positive control), by session
"""
import argparse
import glob
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../analysis/glm/misc"))
from plot_roi_decoding import PALETTE, INK, INK_2, GRID, fdr_bh, stars, FEATURE_LABELS
from feature_roi_sweep import ROI_GROUPS

GROUPS = ["DLPFC", "IFC", "EVC", "VTC", "IT"]  # display order; a figure shows the groups present in its data
PHASES = ["Encoding", "Delay"]
# two-series slots (blue / orange): validated all-pairs; ROI groups use the first four slots on adjacent bars only
SERIES = {"vertex": PALETTE[0], "mean": PALETTE[1]}
SERIES_LABEL = {"vertex": "Vertex pattern (per-vertex)", "mean": "ROI mean (average vertex)"}
TASK_COLOR = {"1back": PALETTE[0], "ctxdm": PALETTE[1], "interdms": PALETTE[2]}


def setup():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})


def style(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, path, title, caption, handles=None):
    """Fixed header (title, then legend) and footer (wrapped caption) in inches; panels fill the rest."""
    w, h = fig.get_size_inches()
    lines = textwrap.wrap(caption, width=int(w * 15))
    header, footer = (0.75 if handles else 0.45) / h, (0.16 * len(lines) + 0.12) / h
    fig.text(0.01, 1 - 0.1 / h, title, ha="left", va="top", fontsize=11, fontweight="bold")
    if handles:
        fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, fontsize=8.5,
                   bbox_to_anchor=(0.005, 1 - 0.36 / h), handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, footer, 1, 1 - header))
    fig.text(0.01, 0.06 / h, "\n".join(lines), ha="left", va="bottom", fontsize=7.5, color=INK_2)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {path}")


def read_all(pattern_by_sub, exclude=None):
    """Every file matching each subject's glob(s) (main ROI set and extra ROI runs), minus names containing exclude."""
    frames = []
    for sub, pats in pattern_by_sub.items():
        pats = [pats] if isinstance(pats, str) else pats
        files = sorted({f for p in pats for f in glob.glob(p) if not (exclude and exclude in os.path.basename(f))})
        if not files:
            print(f"[WARN] {sub}: no files for {pats}")
        frames += [pd.read_csv(f, sep="\t").assign(sub=sub) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def present(df, col="roi"):
    """ROI groups in display order that have rows in df (group level, or the 'group' column for parcels)."""
    have = set(df[col])
    return [g for g in GROUPS if g in have]


def panel_width(groups):
    return 0.55 * len(groups) + 0.4


def paired_bars(ax, d, value, groups, top, err=None, chance=None, q=None):
    """Bars per ROI group for the two reductions (vertex, mean), side by side with a 2px gap."""
    x = np.arange(len(groups))
    w = 0.38
    for k, (red, dx) in enumerate((("vertex", -w / 2 - 0.01), ("mean", w / 2 + 0.01))):
        r = d[d["reduce"] == red].set_index("roi").reindex(groups)
        ax.bar(x + dx, r[value].clip(lower=0), width=w, color=SERIES[red], zorder=2)
        if err is not None:
            ax.errorbar(x + dx, r[value], yerr=r[err], fmt="none", ecolor=INK_2, elinewidth=0.9, zorder=3)
        if chance is not None:
            ax.hlines(r["chance"], x + dx - w / 2, x + dx + w / 2, colors=INK, linestyles=(0, (3, 1.5)),
                      linewidth=1.2, zorder=4)
        if q is not None:
            for xi, v, e, qq in zip(x + dx, r[value], r[err] if err else 0 * r[value], r[q]):
                ax.text(xi, v + e + 0.01, stars(qq), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, top)
    style(ax)


def fig_noise_ceiling(rel, subs, key, demean, out_dir):
    d0 = rel[(rel["key"] == key) & (rel["demean"] == demean) & (rel["level"] == "group")]
    if d0.empty:
        return
    groups = present(d0)
    fig, axes = plt.subplots(len(PHASES), len(subs), figsize=(panel_width(groups) * len(subs) + 0.6,
                                                               2.4 * len(PHASES) + 0.9), sharey=True, squeeze=False)
    top = min(1.0, d0["noise_ceiling"].max() + 0.08)
    for i, ph in enumerate(PHASES):
        for j, sub in enumerate(subs):
            ax = axes[i, j]
            d = d0[(d0["phase"] == ph) & (d0["sub"] == sub)]
            paired_bars(ax, d, "noise_ceiling", groups, top)
            if i == 0:
                ax.set_title(sub, fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{ph}\nnoise ceiling (√r$_{{SB}}$)")
            if not d.empty:
                ax.text(0.99, 0.97, f"{int(d['n_conditions'].max())} cond.", transform=ax.transAxes, ha="right",
                        va="top", fontsize=7, color=INK_2)
    dm = {"none": "nothing removed", "task_pos": "task × position means removed", "pos": "position means removed"}
    kd = {"stimulus": "stimulus key (task, phase, position, loc, obj)", "trial": "trial key (task, acq, phase, position, tc)"}
    save(fig, os.path.join(out_dir, f"noise_ceiling_{key}_{demean}.png"),
         f"Split-half noise ceiling · {kd[key]} · {dm[demean]}",
         "Repeats of each condition split in random halves (20 splits); per-vertex (or ROI-mean) correlation of the "
         "half-averages across conditions, Fisher-averaged, Spearman-Brown corrected; ceiling = √r_SB (negative r → 0). "
         "GLMsingle type D, single trials, dms excluded.",
         [Patch(color=SERIES[r], label=SERIES_LABEL[r]) for r in ("vertex", "mean")])


def fig_noise_ceiling_parcels(rel, subs, key, out_dir):
    d0 = rel[(rel["key"] == key) & (rel["demean"] == "none") & (rel["level"] == "parcel") & (rel["reduce"] == "vertex")]
    if d0.empty:
        return
    groups = present(d0, "group")
    parcels = [p for g in groups for p in ROI_GROUPS[g]]
    cols = [(ph, s) for ph in PHASES for s in subs]
    M = np.array([[d0[(d0["roi"] == p) & (d0["phase"] == ph) & (d0["sub"] == s)]["noise_ceiling"].mean()
                   for ph, s in cols] for p in parcels])
    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 2.6, 0.24 * len(parcels) + 1.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=np.nanmax(M), aspect="auto")
    for (r, c), v in np.ndenumerate(M):
        if np.isfinite(v):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.3,
                    color="white" if v > 0.6 * np.nanmax(M) else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([s for _, s in cols], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(parcels)))
    ax.set_yticklabels(parcels, fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    edges = np.cumsum([len(ROI_GROUPS[g]) for g in groups])[:-1] - 0.5
    for e in edges:
        ax.axhline(e, color="white", linewidth=3)
    ax.axvline(len(subs) - 0.5, color="white", linewidth=4)
    for k, ph in enumerate(PHASES):
        ax.text(k * len(subs) + (len(subs) - 1) / 2, -0.9, ph, ha="center", va="bottom", fontsize=9, fontweight="bold")
    starts = np.r_[0, edges + 0.5]
    for g, st in zip(groups, starts):
        ax.annotate(g, xy=(0, st + (len(ROI_GROUPS[g]) - 1) / 2), xycoords=("axes fraction", "data"), xytext=(-48, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=9, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("noise ceiling (√r$_{SB}$)")
    cb.outline.set_visible(False)
    save(fig, os.path.join(out_dir, f"noise_ceiling_parcels_{key}.png"),
         f"Per-vertex noise ceiling by parcel · {key} key", "Bilateral Glasser parcels; per-vertex split-half "
         "reliability Fisher-averaged within the parcel; nothing removed. GLMsingle type D.")


def fig_decoding(dec, subs, phase, config, out_dir):
    d0 = dec[(dec["phase"] == phase) & (dec["config"] == config) & (dec["level"] == "group")].copy()
    if d0.empty:
        return
    d0["q"] = np.nan
    for sub in subs:  # BH-FDR within each subject's figure cells (both reductions, all features)
        m = dec["sub"] == sub
        allp = dec[m & (dec["config"] == config) & (dec["level"] == "group")]
        q = pd.Series(fdr_bh(allp["p"].to_numpy()), index=allp.index)
        d0.loc[d0.index.intersection(q.index), "q"] = q
    feats = [f for f in ("loc", "ctg", "obj") if f in set(d0["feature"])]
    groups = present(d0)
    fig, axes = plt.subplots(len(feats), len(subs), figsize=(panel_width(groups) * len(subs) + 0.6,
                                                              2.3 * len(feats) + 0.9), sharey="row", squeeze=False)
    for i, f in enumerate(feats):
        df = d0[d0["feature"] == f]
        top = min(1.0, (df["bacc"] + df["bacc_sd"]).max() + 0.1)
        for j, sub in enumerate(subs):
            ax = axes[i, j]
            paired_bars(ax, df[df["sub"] == sub], "bacc", groups, top, err="bacc_sd", chance=True, q="q")
            if i == 0:
                ax.set_title(sub, fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{FEATURE_LABELS[f]}\nbalanced acc.")
    handles = [Patch(color=SERIES[r], label=SERIES_LABEL[r]) for r in ("vertex", "mean")] + \
              [Line2D([], [], color=INK, linestyle=(0, (3, 1.5)), linewidth=1.2, label="Chance")]
    save(fig, os.path.join(out_dir, f"decoding_{phase}_{config.replace('+', '')}.png"),
         f"Feature decoding · {phase} · {config}",
         "Single-trial GLMsingle D betas, dms excluded; kernel ridge classifier, 5-fold CV grouped by trial, alpha tuned "
         "by nested CV. Error bars: SD over folds. * q<.05 ** q<.01 *** q<.001 (1000-perm p, BH-FDR within subject).",
         handles)


LEAK_PANELS = [("Delay", "next", "Delay → next stim."), ("Encoding", "previous", "Encoding → prev. stim."),
               ("Encoding", "current", "Encoding → own stim.")]


def fig_leakage(lk, subs, out_dir, feature="loc"):
    d0 = lk[lk["feature"] == feature]
    if d0.empty:
        return
    rois = [r for r in GROUPS if r in set(d0["roi"])]
    glm_color = {"glmsingle": PALETTE[0], "glmsingleB": PALETTE[1]}
    glm_label = {"glmsingle": "Type D (ridge)", "glmsingleB": "Type B (OLS)"}
    fig, axes = plt.subplots(len(LEAK_PANELS), len(subs), figsize=(2.5 * len(subs) + 0.6, 2.3 * len(LEAK_PANELS) + 0.9),
                             sharey="row", squeeze=False)
    for i, (ph, test, label) in enumerate(LEAK_PANELS):
        dt = d0[(d0["phase"] == ph) & (d0["test"] == test)]
        top = 1.0 if test == "current" else min(1.0, dt["bacc"].max() + 0.08) if not dt.empty else 1.0
        for j, sub in enumerate(subs):
            ax = axes[i, j]
            x = np.arange(len(rois))
            w = 0.38
            for gt, dx in (("glmsingle", -w / 2 - 0.01), ("glmsingleB", w / 2 + 0.01)):
                r = dt[(dt["sub"] == sub) & (dt["glm"] == f"{gt}_wfdelay")].set_index("roi").reindex(rois)
                ax.bar(x + dx, r["bacc"], width=w, color=glm_color[gt], zorder=2)
                ax.hlines(r["chance"], x + dx - w / 2, x + dx + w / 2, colors=INK, linestyles=(0, (3, 1.5)),
                          linewidth=1.2, zorder=4)
                for xi, v, p in zip(x + dx, r["bacc"], r["p"]):
                    ax.text(xi, v + 0.01, stars(p), ha="center", va="bottom", fontsize=7, color=INK)
            ax.set_xticks(x)
            ax.set_xticklabels(rois)
            ax.set_ylim(0.3 if test != "current" else 0, top)
            style(ax)
            if i == 0:
                ax.set_title(sub, fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{label}\nbalanced acc.", fontsize=8)
    handles = [Patch(color=glm_color[g], label=glm_label[g]) for g in glm_color] + \
              [Line2D([], [], color=INK, linestyle=(0, (3, 1.5)), linewidth=1.2, label="Chance")]
    save(fig, os.path.join(out_dir, f"leakage_{feature}.png"),
         f"Temporal leakage · {FEATURE_LABELS[feature]}",
         "Rows 1-2: decoding the neighbouring stimulus's feature from an event's beta, within strata of the current "
         "stimulus (so the current feature cannot drive it); near chance = events well separated, and the forward "
         "(delay → next) test is the clean one. Row 3: the event's own stimulus, for reference. Single trials, dms "
         "excluded; * p<.05 ** p<.01 *** p<.001 (uncorrected perm.).",
         handles)


def fig_loc_by_run(lbr, subs, out_dir):
    if lbr.empty:
        return
    fig, axes = plt.subplots(len(subs), 1, figsize=(9.0, 1.9 * len(subs) + 0.9), sharex=True, squeeze=False)
    for ax, sub in zip(axes[:, 0], subs):
        m = lbr[lbr["sub"] == sub]
        r = m.groupby(["ses", "task", "acq", "run"]).ok.agg(["mean", "size"]).reset_index()
        r["s"] = r["ses"].str[-2:].astype(int)
        r["x"] = r["s"] + np.random.default_rng(0).uniform(-0.22, 0.22, len(r))
        for t, c in TASK_COLOR.items():
            rt = r[r["task"] == t]
            ax.scatter(rt["x"], rt["mean"], s=22, color=c, edgecolor="white",
                       linewidth=0.8, zorder=3, label=t)
        sm = m.groupby("ses").ok.mean()
        ax.plot([int(s[-2:]) for s in sm.index], sm.values, color=INK, linewidth=2, zorder=4)
        ax.axhline(0.5, color=INK_2, linestyle=(0, (3, 1.5)), linewidth=1.0, zorder=1)
        weak = r[r["mean"] < 0.7].sort_values("mean")
        ax.scatter(weak["x"], weak["mean"], s=90, facecolor="none", edgecolor=INK, linewidth=1.1, zorder=5)
        if len(weak):
            ax.set_title("< 0.7: " + "; ".join(f"ses-{w.s:02d} {w.task} {w.acq} r{int(w.run)} ({w['mean']:.2f})"
                                               for _, w in weak.iterrows()), loc="right", fontsize=7, color=INK_2)
        ax.set_ylim(0.2, 1.02)
        ax.set_ylabel("EVC loc acc.")
        ax.set_title(f"{sub} · overall {m.ok.mean():.2f} (n = {len(m)})", loc="left", fontsize=9.5, fontweight="bold")
        style(ax)
    axes[-1, 0].set_xticks(range(1, 17))
    axes[-1, 0].set_xlabel("Session")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=c, markeredgecolor="white",
                      label=t) for t, c in TASK_COLOR.items()] + \
              [Line2D([], [], color=INK, linewidth=2, label="Session mean"),
               Line2D([], [], color=INK_2, linestyle=(0, (3, 1.5)), label="Chance")]
    save(fig, os.path.join(out_dir, "loc_by_run.png"), "Positive control by run · EVC location decoding (encoding)",
         "Cross-validated (5-fold grouped by trial, task × position means removed, RidgeClassifierCV) prediction of "
         "each encoding event's location; dots = runs (jittered), line = session mean. Ringed runs are below 0.7 and "
         "listed above their panel.", handles)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glm_runs", required=True)
    ap.add_argument("--subs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", default="wfdelay")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    setup()
    G = lambda gt, sub: f"{args.glm_runs}/{gt}_{args.variant}/64kDense/{sub}"

    rel = pd.concat([read_all({s: [f"{G('glmsingle', s)}/_reliability_pattern/{d}pattern_split_half_reliability_excl-dms"
                                   f"_split-repeats_key-{k}.tsv" for d in ("", "rois-*/")] for s in args.subs}).assign(key=k)
                     for k in ("stimulus", "trial")], ignore_index=True)
    if not rel.empty:
        for k in ("stimulus", "trial"):
            for dm in ("none", "task_pos"):
                fig_noise_ceiling(rel, args.subs, k, dm, args.out_dir)
            fig_noise_ceiling_parcels(rel, args.subs, k, args.out_dir)

    # main ROI set and extra ROI runs (group-parcels-<groups>); vertex files are the ones without _reduce-
    dec = pd.concat([read_all({s: f"{G('glmsingle', s)}/_decoding_roi/roi_decoding_glmsingle_{args.variant}_{f}"
                                  f"_excl-dms_rep-single_group-parcels-*{red}_cfg-*.tsv" for s in args.subs},
                              exclude=None if red else "_reduce-")
                     for f in ("loc", "ctg", "obj") for red in ("", "_reduce-mean")], ignore_index=True)
    if not dec.empty:
        for ph in PHASES:
            for cfg in ("demean+tuned", "tuned"):
                fig_decoding(dec, args.subs, ph, cfg, args.out_dir)

    lk = pd.concat([read_all({s: f"{G(gt, s)}/_decoding_roi/temporal_leakage_{gt}_{args.variant}_{f}_excl-dms_rep-single.tsv"
                              for s in args.subs}) for gt in ("glmsingle", "glmsingleB") for f in ("loc", "obj")],
                   ignore_index=True)
    if not lk.empty:
        for f in ("loc", "obj"):
            fig_leakage(lk, args.subs, args.out_dir, f)

    lbr = read_all({s: f"{G('glmsingle', s)}/_decoding_roi/loc_by_run_EVC.tsv" for s in args.subs})
    fig_loc_by_run(lbr, args.subs, args.out_dir)
