#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


FEATURE_LABELS = {"loc": "Location", "ctg": "Category (bus / face)", "obj": "Object identity"}
GROUP_ORDER = ["DLPFC", "IFC", "FrontalPole", "OFC", "EVC", "VTC", "IT"]

# categorical slots in fixed order (validated adjacent-pair CVD / normal-vision)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def fdr_bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def bar_layout(df):
    """ROIs ordered group by group with a gap between groups; one colour per group."""
    groups = [g for g in GROUP_ORDER if g in set(df["group"])] + \
             [g for g in pd.unique(df["group"]) if g not in GROUP_ORDER]
    rois = [r for g in groups for r in pd.unique(df.loc[df["group"] == g, "roi"])]
    roi_group = dict(zip(df["roi"], df["group"]))
    gap = 0.8
    xs = np.array([i + gap * groups.index(roi_group[r]) for i, r in enumerate(rois)], dtype=float)
    group_color = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    colors = [group_color[roi_group[r]] for r in rois]
    return groups, rois, roi_group, xs, group_color, colors


def new_figure(n_rows, n_bars):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig_w = max(7.0, 0.36 * n_bars + 1.8)
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, 2.0 * n_rows + 0.8), sharex=True, squeeze=False)
    return fig, axes[:, 0]


def style_axis(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def finish(fig, axes, layout, extra_handles, title, caption, out_path):
    groups, rois, roi_group, xs, group_color, _ = layout
    bottom = axes[-1]
    bottom.set_xticks(xs)
    bottom.set_xticklabels(rois, rotation=60, ha="right", fontsize=7.5)
    bottom.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)

    # group names under each block (identity is never colour alone)
    for g in groups:
        gx = xs[[roi_group[r] == g for r in rois]]
        bottom.annotate(g, xy=(gx.mean(), 0), xycoords=("data", "axes fraction"), xytext=(0, -46),
                        textcoords="offset points", ha="center", va="top", fontsize=9, fontweight="bold", color=INK)

    handles = [Patch(color=group_color[g], label=g) for g in groups] + extra_handles
    # fixed header in inches: title, then legend, then panels
    h = fig.get_figheight()
    header = (0.75 if title else 0.4) / h
    fig.tight_layout(rect=(0, 0.02, 1, 1 - header))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1 - header + 0.18 / h))
    if title:
        fig.suptitle(title, y=1 - 0.12 / h, va="top", fontsize=11)
    fig.text(0.5, 0.0, caption, ha="center", va="top", fontsize=7.5, color=INK_2)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[SAVED] {out_path}")


def plot(df, out_path, title=None):
    if "group" not in df.columns:
        df["group"] = df["roi"]
    df["q"] = fdr_bh(df["p"])

    features = [f for f in ["loc", "ctg", "obj"] if f in set(df["feature"])]
    phases = [p for p in ["Encoding", "Delay"] if p in set(df["phase"])]
    layout = bar_layout(df)
    _, rois, _, xs, _, colors = layout
    fig, axes = new_figure(len(features) * len(phases), len(rois))

    for i, feat in enumerate(features):
        f_df = df[df["feature"] == feat]
        top = min(1.0, (f_df["bacc"] + f_df["bacc_sd"]).max() + 0.08)
        for j, phase in enumerate(phases):
            ax = axes[i * len(phases) + j]
            d = f_df[f_df["phase"] == phase].set_index("roi").reindex(rois)
            ax.bar(xs, d["bacc"], width=0.72, color=colors, edgecolor="white", linewidth=1, zorder=2)
            ax.errorbar(xs, d["bacc"], yerr=d["bacc_sd"], fmt="none", ecolor=INK_2, elinewidth=0.9, capsize=0, zorder=3)

            # chance level drawn across each bar
            ax.hlines(d["chance"], xs - 0.36, xs + 0.36, colors=INK, linestyles=(0, (3, 1.5)), linewidth=1.3, zorder=4)

            for x, b, sd, q in zip(xs, d["bacc"], d["bacc_sd"], d["q"]):
                ax.text(x, b + sd + 0.008, stars(q), ha="center", va="bottom", fontsize=7.5, color=INK)

            ax.set_ylim(0, top)
            ax.set_ylabel("Balanced acc.")
            ax.set_title(f"{FEATURE_LABELS.get(feat, feat)} · {phase}", loc="left", fontsize=9.5, fontweight="bold")
            ax.set_title(f"n = {int(d['n_events'].max())} events", loc="right", fontsize=8, color=INK_2)
            style_axis(ax)

    chance = Line2D([], [], color=INK, linestyle=(0, (3, 1.5)), linewidth=1.3, label="Chance")
    caption = ("Bilateral parcels.  Error bars: SD across 5 grouped CV folds.  "
               "* q<.05  ** q<.01  *** q<.001 (permutation p; BH-FDR across all bars)")
    finish(fig, axes, layout, [chance], title, caption, out_path)


def plot_reliability(rel, out_path, title=None):
    """Split-half reliability (Spearman-Brown) per ROI, one panel per phase, same layout as the decoding plot."""
    phases = [p for p in ["Encoding", "Delay"] if p in set(rel["phase"])]
    layout = bar_layout(rel)
    _, rois, _, xs, _, colors = layout
    fig, axes = new_figure(len(phases), len(rois))
    top = rel["r_sb"].max() + 0.05
    for ax, phase in zip(axes, phases):
        d = rel[rel["phase"] == phase].set_index("roi").reindex(rois)
        ax.bar(xs, d["r_sb"].clip(lower=0), width=0.72, color=colors, edgecolor="white", linewidth=1, zorder=2)
        ax.set_ylim(0, top)
        ax.set_ylabel("Split-half r (SB)")
        ax.set_title(f"Pattern reliability · {phase}", loc="left", fontsize=9.5, fontweight="bold")
        ax.set_title(f"n = {int(d['n_events'].max())} events", loc="right", fontsize=8, color=INK_2)
        style_axis(ax)
    caption = ("Per-vertex event-wise split-half correlation (sessions 2 vs 2, all 3 splits), Fisher-averaged over "
               "vertices, Spearman-Brown corrected = reliability of the 4-session-averaged patterns.")
    finish(fig, axes, layout, [], title, caption, out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bar plot of ROI decoding results from feature_roi_sweep.py TSVs")
    ap.add_argument("tsvs", nargs="+", help="one or more roi_decoding_*.tsv files (concatenated)")
    ap.add_argument("--out", required=True, help="output image path (.png / .pdf)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--config", default=None, help="keep only this decoder config (e.g. demean+tuned)")
    ap.add_argument("--level", default=None, choices=["group", "parcel"], help="keep only this ROI level")
    ap.add_argument("--reduce", default="vertex", choices=["vertex", "mean", "hemi_mean"],
                    help="keep only this ROI reduction (vertex patterns, ROI mean, per-hemisphere means)")
    ap.add_argument("--groups", nargs="+", default=None, help="keep only these ROI groups (e.g. DLPFC EVC VTC IT)")
    ap.add_argument("--reliability", action="store_true",
                    help="inputs are pattern_reliability.py TSVs; plots reliability instead")
    ap.add_argument("--demean", default="task_pos", choices=["none", "task_pos", "pos"],
                    help="with --reliability: which nuisance removal to plot")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(t, sep="\t") for t in args.tsvs], ignore_index=True)
    if args.groups:
        df = df[df["group"].isin(args.groups)]
    if "reduce" in df.columns:
        df = df[df["reduce"] == args.reduce]
    if args.level and "level" in df.columns:
        df = df[df["level"] == args.level]
    if args.reliability:
        plot_reliability(df[df["demean"] == args.demean].copy(), args.out, args.title)
    else:
        if args.config and "config" in df.columns:
            df = df[df["config"] == args.config]
        plot(df.copy(), args.out, args.title)
