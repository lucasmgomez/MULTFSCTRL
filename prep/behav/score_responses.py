#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
from typing import Literal, Dict, Any

Feature = Literal["ctg", "obj", "loc"]
InterOrder = Literal["ABAB", "ABBA"]
ctxFeatOrder = Literal["col", "lco"]
TaskKind = Literal["interdms", "1back"]


def _truth_to_resp_code(is_match: bool) -> int:
    """Boolean match -> expected response code (2=yes, 3=no)."""
    return 2 if is_match else 3

def _is_missing(x) -> bool:
    """
    Treat NaN/NA and the string '--' (subject didn't respond) as missing.
    """
    if pd.isna(x):
        return True
    if isinstance(x, str) and x.strip() in {"--", ""}:
        return True
    return False

# =========================
# CtxDm scoring
# =========================

def score_ctxdm_row(
    row: pd.Series,
    forder: ctxFeatOrder,
    resp_col: str = "response_2",
) -> Dict[str, Any]:
    """
    CtxDm has 3 stimuli features per row (feature1..feature3) and 1 scored response.

    ans1: compare 3 vs 1 (Ctx2 vs Ctx1)

    Returns flat dict:
      ans1, trial_correct,
      expected_ans1,
      ans1_comp
    """
    # Pull features based on order
    if forder == "col":
        if row[f"ctg1"] == row[f"ctg2"]:
            f2 = row[f"obj2"]
            f3 = row[f"obj3"]

            ans1_comp = f"obj2==obj3"
        else:
            f2 = row[f"loc2"]
            f3 = row[f"loc3"]

            ans1_comp = f"loc2==loc3"

    elif forder == "lco":
        if row[f"loc1"] == row[f"loc2"]:
            f2 = row[f"ctg2"]
            f3 = row[f"ctg3"]

            ans1_comp = f"ctg2==ctg3"
        else:
            f2 = row[f"obj2"]
            f3 = row[f"obj3"]

            ans1_comp = f"obj2==obj3"
    

    # Compute match truth value
    match1 = bool(f3 == f2)
    exp1 = _truth_to_resp_code(match1)

    # Pull response
    r1 = row.get(resp_col, pd.NA)

    # Score answer (False if missing)
    ans1 = (not _is_missing(r1)) and (int(r1) == exp1)  

    # trial_correct requires answer present + correct
    trial_correct = bool(ans1 and (not _is_missing(r1)))

    return {
        "ans1": bool(ans1),
        "trial_correct": bool(trial_correct),
        "expected_ans1": int(exp1),
        "ans1_comp": ans1_comp,
    }

def score_ctxdm_df(
    df: pd.DataFrame,
    forder: ctxFeatOrder,
    resp_col: str = "response_2",
) -> pd.DataFrame:
    scored = df.apply(lambda r: score_ctxdm_row(r, forder=forder, resp_col=resp_col), axis=1)
    return pd.concat([df, pd.DataFrame(scored.tolist(), index=df.index)], axis=1)


# =========================
# InterDMS scoring
# =========================
def score_interdms_row(
    row: pd.Series,
    feature: Feature,
    order: InterOrder,
    resp_cols: tuple[str, str] = ("response_2", "response_3"),
) -> Dict[str, Any]:
    """
    InterDMS has 4 stimuli features per row (feature1..feature4) and 2 scored responses.

    ABAB:
      ans1: compare 3 vs 1 (A2 vs A1)
      ans2: compare 4 vs 2 (B2 vs B1)

    ABBA:
      ans1: compare 3 vs 2 (B2 vs B1)
      ans2: compare 4 vs 1 (A2 vs A1)

    Returns flat dict:
      ans1, ans2, trial_correct,
      expected_ans1, expected_ans2,
      ans1_comp, ans2_comp
    """
    # Pull features
    f1 = row[f"{feature}1"]
    f2 = row[f"{feature}2"]
    f3 = row[f"{feature}3"]
    f4 = row[f"{feature}4"]

    # Which comparisons are performed depends on order
    if order == "ABAB":
        left1, right1 = 3, 1
        left2, right2 = 4, 2
    elif order == "ABBA":
        left1, right1 = 3, 2
        left2, right2 = 4, 1
    else:
        raise ValueError(f"Unknown InterDMS order: {order}")

    # Compute match truth values
    vals = {1: f1, 2: f2, 3: f3, 4: f4}
    match1 = bool(vals[left1] == vals[right1])
    match2 = bool(vals[left2] == vals[right2])

    exp1 = _truth_to_resp_code(match1)
    exp2 = _truth_to_resp_code(match2)

    # Pull responses
    r1 = row.get(resp_cols[0], pd.NA)
    r2 = row.get(resp_cols[1], pd.NA)

    # Score answers (False if missing)
    ans1 = (not _is_missing(r1)) and (int(r1) == exp1)
    ans2 = (not _is_missing(r2)) and (int(r2) == exp2)

    # trial_correct requires all answers present + correct
    trial_correct = bool(ans1 and ans2 and (not _is_missing(r1)) and (not _is_missing(r2)))

    ans1_comp = f"{feature}{left1}=={feature}{right1}"
    ans2_comp = f"{feature}{left2}=={feature}{right2}"

    return {
        "ans1": bool(ans1),
        "ans2": bool(ans2),
        "trial_correct": bool(trial_correct),
        "expected_ans1": int(exp1),
        "expected_ans2": int(exp2),
        "ans1_comp": ans1_comp,
        "ans2_comp": ans2_comp,
    }


def score_interdms_df(
    df: pd.DataFrame,
    feature: Feature,
    order: InterOrder,
    resp_cols: tuple[str, str] = ("response_2", "response_3"),
) -> pd.DataFrame:
    scored = df.apply(lambda r: score_interdms_row(r, feature=feature, order=order, resp_cols=resp_cols), axis=1)
    return pd.concat([df, pd.DataFrame(scored.tolist(), index=df.index)], axis=1)


# =========================
# 1-back scoring
# =========================
def score_1back_trial(row, feature: str, resp_cols=None):
    """
    Score a single 1-back trial.

    feature: one of {"loc", "ctg", "obj"}
    resp_cols: optional list like ["response_1", ..., "response_5"]
    """

    if resp_cols is None:
        resp_cols = [f"response_{i}" for i in range(1, 6)]  # 5 comparisons

    out = {}
    correct_flags = []

    for k in range(2, 7):  # stimuli 2..6
        curr = k
        prev = k - 1
        ans_idx = k - 1  # ans1..ans5

        resp = row.get(resp_cols[ans_idx - 1], pd.NA)

        f_prev = row.get(f"{feature}{prev}", pd.NA)
        f_curr = row.get(f"{feature}{curr}", pd.NA)

        match = (f_prev == f_curr)
        expected = 2 if match else 3  # 2=yes, 3=no

        # correctness flag
        is_correct = (resp == expected)

        out[f"ans{ans_idx}"] = bool(is_correct)
        out[f"expected_ans{ans_idx}"] = expected
        out[f"ans{ans_idx}_comp"] = f"{feature}{curr}=={feature}{prev}"

        correct_flags.append(bool(is_correct))

    out["trial_correct"] = all(correct_flags)
    return out

def score_1back_df(
    df: pd.DataFrame,
    feature: Feature,
    resp_cols: tuple[str, str, str, str, str] = ("response_1", "response_2", "response_3", "response_4", "response_5"),
) -> pd.DataFrame:
    scored = df.apply(lambda r: score_1back_trial(r, feature=feature, resp_cols=resp_cols), axis=1)
    return pd.concat([df, pd.DataFrame(scored.tolist(), index=df.index)], axis=1)


# =========================
# Wrapper
# =========================
def score_block_df(
    df: pd.DataFrame,
    task_kind: TaskKind,
    feature: Feature,
    order: InterOrder | None = None,
) -> pd.DataFrame:
    if task_kind == "interdms":
        if order is None:
            raise ValueError("InterDMS scoring requires order='ABAB' or 'ABBA'.")
        return score_interdms_df(df, feature=feature, order=order)
    if task_kind == "1back":
        return score_1back_df(df, feature=feature)
    raise ValueError(f"Unknown task_kind: {task_kind}")

import os
import re
import glob
import argparse
import pandas as pd

# -------------------------
# Filename parsing
# -------------------------
# Examples:
# sub-01_ses-1_..._task-interdms_loc_ABBA_block_0_events_....tsv
# sub-01_ses-1_..._task-ctxdm_col_block_0_events_....tsv
# sub-01_ses-1_..._task-1back_ctg_block_0_events_....tsv

FNAME_TASK_RE = re.compile(
    r"""
    task-(?P<task>interdms|ctxdm|1back)_
    (?P<spec>.+?)
    _block_(?P<block>\d+)
    _events_.*\.tsv$
    """,
    re.VERBOSE,
)

INTERDMS_SPEC_RE = re.compile(r"^(?P<feature>ctg|obj|loc)_(?P<order>ABAB|ABBA)$")
CTXDM_SPEC_RE = re.compile(r"^(?P<forder>col|lco)$")
ONEBACK_SPEC_RE = re.compile(r"^(?P<feature>ctg|obj|loc)$")


def parse_task_from_filename(path: str) -> dict:
    """
    Returns a dict describing how to score this file:
      {
        "task": "interdms"/"ctxdm"/"1back",
        "feature": "ctg"/"obj"/"loc" (if applicable),
        "order": "ABAB"/"ABBA" (if applicable),
        "forder": "col"/"lco" (if applicable)
      }
    """
    base = os.path.basename(path)
    m = FNAME_TASK_RE.search(base)
    if not m:
        raise ValueError(
            f"Could not parse task from filename:\n  {base}\n"
            f"Expected patterns like:\n"
            f"  ..._task-interdms_loc_ABBA_block_0_events_....tsv\n"
            f"  ..._task-ctxdm_col_block_0_events_....tsv\n"
            f"  ..._task-1back_ctg_block_0_events_....tsv"
        )

    task = m.group("task")
    spec = m.group("spec")

    if task == "interdms":
        mm = INTERDMS_SPEC_RE.match(spec)
        if not mm:
            raise ValueError(f"Bad interdms spec '{spec}' in {base} (expected e.g. loc_ABBA)")
        return {"task": "interdms", "feature": mm.group("feature"), "order": mm.group("order")}

    if task == "ctxdm":
        mm = CTXDM_SPEC_RE.match(spec)
        if not mm:
            raise ValueError(f"Bad ctxdm spec '{spec}' in {base} (expected 'col' or 'lco')")
        return {"task": "ctxdm", "forder": mm.group("forder")}

    if task == "1back":
        mm = ONEBACK_SPEC_RE.match(spec)
        if not mm:
            raise ValueError(f"Bad 1back spec '{spec}' in {base} (expected ctg/obj/loc)")
        return {"task": "1back", "feature": mm.group("feature")}

    raise ValueError(f"Unhandled task='{task}' parsed from {base}")


def scored_outpath(in_path: str) -> str:
    """Same name, but append _scored before .tsv"""
    if not in_path.endswith(".tsv"):
        return in_path + "_scored.tsv"
    return in_path[:-4] + "_scored.tsv"


# -------------------------
# Router
# -------------------------
def score_tsv_file(in_path: str, overwrite: bool = False) -> str:
    """
    Read one TSV, route to correct scorer, write *_scored.tsv.
    Returns output path.
    """
    out_path = scored_outpath(in_path)
    if (not overwrite) and os.path.exists(out_path):
        print(f"[SKIP] Exists: {out_path}")
        return out_path

    info = parse_task_from_filename(in_path)
    df = pd.read_csv(in_path, sep="\t")

    if info["task"] == "interdms":
        df_scored = score_interdms_df(df, feature=info["feature"], order=info["order"])
    elif info["task"] == "ctxdm":
        df_scored = score_ctxdm_df(df, forder=info["forder"])
    elif info["task"] == "1back":
        df_scored = score_1back_df(df, feature=info["feature"])
    else:
        raise RuntimeError(f"Unexpected routing result: {info}")

    df_scored.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] {os.path.basename(in_path)} -> {os.path.basename(out_path)}  ({info})")
    return out_path


# -------------------------
# CLI
# -------------------------
def _expand_inputs(path_or_glob: str) -> list[str]:
    # directory -> *.tsv
    if os.path.isdir(path_or_glob):
        files = sorted(glob.glob(os.path.join(path_or_glob, "*.tsv")))
        return files
    # glob pattern or file
    hits = sorted(glob.glob(path_or_glob))
    return hits if hits else [path_or_glob]


def main():
    ap = argparse.ArgumentParser("Score task TSVs (interdms / ctxdm / 1back) by routing from filename.")
    ap.add_argument("input", help="TSV file path, a directory, or a glob (e.g. '/path/*.tsv').")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing *_scored.tsv.")
    args = ap.parse_args()

    inputs = _expand_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No inputs matched: {args.input}")

    for p in inputs:
        if not os.path.isfile(p):
            print(f"[SKIP] Not a file: {p}")
            continue
        if not p.endswith(".tsv"):
            print(f"[SKIP] Not a .tsv: {p}")
            continue
        score_tsv_file(p, overwrite=args.overwrite)


if __name__ == "__main__":
    main()