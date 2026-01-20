#!/usr/bin/env python3
import os
import glob
import argparse
import re
import numpy as np
import pandas as pd

ANS_RE = re.compile(r"^ans(\d+)$")

def expand_inputs(path_or_glob: str) -> list[str]:
    if os.path.isdir(path_or_glob):
        return sorted(glob.glob(os.path.join(path_or_glob, "*_scored.tsv")))
    hits = sorted(glob.glob(path_or_glob))
    return hits

def find_ans_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if ANS_RE.match(c)]
    # sort ans1, ans2, ...
    cols.sort(key=lambda c: int(ANS_RE.match(c).group(1)))
    return cols

def bool_mean(series: pd.Series) -> float:
    # handle True/False, 0/1, and NaNs robustly
    s = pd.to_numeric(series, errors="coerce")
    return float(np.nanmean(s))

def summarize_file(path: str) -> dict:
    df = pd.read_csv(path, sep="\t")

    ans_cols = find_ans_cols(df)

    out = {
        "file": os.path.basename(path),
        "n_trials": int(len(df)),
    }

    # Per-answer accuracy
    for c in ans_cols:
        out[f"{c}_acc"] = bool_mean(df[c])  # fraction correct on that comparison

    # Trial accuracy
    if "trial_correct" in df.columns:
        out["trial_acc"] = bool_mean(df["trial_correct"])
        out["block_acc"] = out["trial_acc"]  # block accuracy == mean trial_correct
    else:
        out["trial_acc"] = np.nan
        out["block_acc"] = np.nan

    # Overall “per-response” pooled accuracy (across all ans columns)
    if ans_cols:
        pooled = pd.to_numeric(df[ans_cols].stack(), errors="coerce")
        out["response_acc_overall"] = float(np.nanmean(pooled))
        out["n_responses_total"] = int(pooled.notna().sum())
    else:
        out["response_acc_overall"] = np.nan
        out["n_responses_total"] = 0

    return out

def main():
    ap = argparse.ArgumentParser("Summarize accuracies from *_scored.tsv files.")
    ap.add_argument("input", help="Directory of scored TSVs, a glob, or a single file.")
    ap.add_argument("--out_tsv", default="scored_accuracy_summary.tsv",
                    help="Output TSV path (default: scored_accuracy_summary.tsv in CWD).")
    args = ap.parse_args()

    files = expand_inputs(args.input)
    if not files:
        raise SystemExit(f"No files matched: {args.input}")

    rows = []
    for f in files:
        if not os.path.isfile(f) or not f.endswith(".tsv"):
            continue
        rows.append(summarize_file(f))

    out_df = pd.DataFrame(rows)

    # Put some core cols first
    front = ["file", "n_trials", "block_acc", "trial_acc", "response_acc_overall", "n_responses_total"]
    ordered = [c for c in front if c in out_df.columns] + [c for c in out_df.columns if c not in front]
    out_df = out_df[ordered].sort_values("file").reset_index(drop=True)

    out_df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote: {args.out_tsv}  (n_files={len(out_df)})")

if __name__ == "__main__":
    main()