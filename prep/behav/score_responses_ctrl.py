#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import glob
import argparse
import pandas as pd
from typing import Literal, Dict, Any

# =========================
# Helpers
# =========================

def _truth_to_resp_code(is_match: bool) -> int:
    """Boolean match -> expected response code (2=yes, 3=no)."""
    return 2 if is_match else 3

def _is_missing(x) -> bool:
    """Treat NaN/NA and the string '--' as missing."""
    if pd.isna(x):
        return True
    if isinstance(x, str) and x.strip() in {"--", ""}:
        return True
    return False

def _safe_int_compare(val, expected) -> bool:
    """Safely compare a response value to expected code, handling NA."""
    print(val)
    return val == expected

def get_unique_outpath(in_path: str, overwrite: bool) -> str:
    """
    If overwrite is True, returns ..._scored.tsv.
    If overwrite is False and file exists, returns ..._scored_1.tsv, _scored_2.tsv, etc.
    """
    base_path = in_path.replace(".tsv", "_scored.tsv")
    if overwrite or not os.path.exists(base_path):
        return base_path
    
    counter = 1
    while True:
        new_path = in_path.replace(".tsv", f"_scored_{counter}.tsv")
        if not os.path.exists(new_path):
            return new_path
        counter += 1

# =========================
# Scorer Functions
# =========================

def score_ctxdm_row(row: pd.Series, forder: str, resp_col: str) -> Dict[str, Any]:
    ans1_comp = "unknown"
    f2, f3 = None, None

    if forder == "lol":
        if row.get("loc1") == row.get("loc2"):
            f2, f3 = row.get("obj2"), row.get("obj3")
            ans1_comp = "obj2==obj3"
        else:
            f2, f3 = row.get("loc2"), row.get("loc3")
            ans1_comp = "loc2==loc3"
    elif forder == "olc":
        if row.get("obj1") == row.get("obj2"):
            f2, f3 = row.get("loc2"), row.get("loc3")
            ans1_comp = "loc2==loc3"
        else:
            f2, f3 = row.get("ctg2"), row.get("ctg3")
            ans1_comp = "ctg2==ctg3"
    
    match1 = bool(f2 is not None and f3 is not None and f3 == f2)
    exp1 = _truth_to_resp_code(match1)
    resp = row.get(resp_col, pd.NA)
    ans1 = _safe_int_compare(resp, exp1)

    return {
        "ans1": ans1,
        "trial_correct": ans1,
        "expected_ans1": exp1,
        "ans1_comp": ans1_comp,
    }

def score_interdms2_row(row: pd.Series, feature: str, order: str, resp_cols: list[str]) -> Dict[str, Any]:
    mapping = {
        "ABCABC": [(4, 1), (5, 2), (6, 3)],
        "ABBCCA": [(3, 2), (5, 4), (6, 1)]
    }
    if order not in mapping:
        raise ValueError(f"Unknown InterDMS2 order: {order}")

    comps = mapping[order]
    out = {}
    correct_flags = []

    for i, (left, right) in enumerate(comps, 1):
        f_left = row.get(f"{feature}{left}")
        f_right = row.get(f"{feature}{right}")
        match = bool(f_left == f_right)
        exp = _truth_to_resp_code(match)
        resp = row.get(resp_cols[i-1], pd.NA)
        
        is_correct = _safe_int_compare(resp, exp)
        out[f"ans{i}"] = is_correct
        out[f"expected_ans{i}"] = exp
        out[f"ans{i}_comp"] = f"{feature}{left}=={feature}{right}"
        correct_flags.append(is_correct)

    out["trial_correct"] = all(correct_flags)
    return out

def score_1back_ao_trial(row: pd.Series, op: str, forder: str, resp_cols: list[str]) -> Dict[str, Any]:
    out = {}
    correct_flags = []

    for k in range(2, 6): 
        ans_idx = k - 1
        resp = row.get(resp_cols[ans_idx - 1], pd.NA)

        if forder == "lo":
            l_match = (row.get(f"loc{k-1}") == row.get(f"loc{k}"))
            o_match = (row.get(f"obj{k-1}") == row.get(f"obj{k}"))
            match = (l_match and o_match) if op == "a" else (l_match or o_match)
        else:
            match = (row.get(f"{forder}{k-1}") == row.get(f"{forder}{k}"))

        expected = _truth_to_resp_code(match)
        is_correct = _safe_int_compare(resp, expected)

        out[f"ans{ans_idx}"] = is_correct
        out[f"expected_ans{ans_idx}"] = expected
        out[f"ans{ans_idx}_comp"] = f"{op}_{forder}{k}=={forder}{k-1}"
        correct_flags.append(is_correct)

    out["trial_correct"] = all(correct_flags)
    return out

def score_2back_trial(row: pd.Series, feature: str, resp_cols: list[str]) -> Dict[str, Any]:
    out = {}
    correct_flags = []

    for k in range(3, 6): 
        ans_idx = k - 2 
        resp = row.get(resp_cols[ans_idx - 1], pd.NA)

        f_prev = row.get(f"{feature}{k-2}")
        f_curr = row.get(f"{feature}{k}")

        match = bool(f_prev == f_curr)
        expected = _truth_to_resp_code(match)
        is_correct = _safe_int_compare(resp, expected)

        out[f"ans{ans_idx}"] = is_correct
        out[f"expected_ans{ans_idx}"] = expected
        out[f"ans{ans_idx}_comp"] = f"{feature}{k}=={feature}{k-2}"
        correct_flags.append(is_correct)

    out["trial_correct"] = all(correct_flags)
    return out

def score_dms_ao_trial(row: pd.Series, op: str, forder: str, resp_cols: list[str]) -> Dict[str, Any]:
    if forder == "cl":
        f1_a, f1_b = row.get("ctg1"), row.get("loc1")
        f2_a, f2_b = row.get("ctg2"), row.get("loc2")
    else: 
        f1_a, f1_b = row.get("loc1"), row.get("ctg1")
        f2_a, f2_b = row.get("loc2"), row.get("ctg2")

    match_a, match_b = (f1_a == f2_a), (f1_b == f2_b)
    match = (match_a and match_b) if op == "a" else (match_a or match_b)
    expected = _truth_to_resp_code(match)
    
    resp = row.get(resp_cols[0], pd.NA)
    is_correct = _safe_int_compare(resp, expected)

    return {
        "ans1": is_correct,
        "expected_ans1": expected,
        "ans1_comp": f"{op}_{forder}_match",
        "trial_correct": is_correct,
    }

# =========================
# Routing & Input Handling
# =========================

FNAME_TASK_RE = re.compile(
    r"""task-(?P<task>interdms2|ctxdm|1back|2back|dms)_(?P<spec>.+?)_block_(?P<block>\d+)_events_.*""",
    re.VERBOSE
)

AO_SPEC_RE = re.compile(r"^(?:(?P<op>a|o)_(?P<feature>ctg|obj|loc|lo|lc)|(?P<feature_only>ctg|obj|loc|lo|lc))$")

def parse_task_from_filename(path: str) -> dict:
    base = os.path.basename(path)
    m = FNAME_TASK_RE.search(base)
    if not m: raise ValueError(f"Regex fail: {base}")

    task, spec = m.group("task"), m.group("spec")
    info = {"task": task, "op": None, "forder": None, "feature": None, "order": None}

    if task == "interdms2":
        parts = spec.split('_')
        info["feature"], info["order"] = parts[0], parts[1]
    elif task == "ctxdm":
        info["forder"] = spec
    elif task in ["1back", "2back", "dms"]:
        mm = AO_SPEC_RE.match(spec)
        if mm:
            info["op"] = mm.group("op")
            feat = mm.group("feature") or mm.group("feature_only")
            if feat in ["lo", "lc", "cl"]: info["forder"] = feat
            else: info["feature"] = feat
        if task == "dms": info["task"] = "dms_ao"
    
    return info

def score_tsv_file(in_path: str, overwrite: bool = False):
    out_path = get_unique_outpath(in_path, overwrite)
    info = parse_task_from_filename(in_path)
    df = pd.read_csv(in_path, sep="\t")

    if info["task"] == "interdms2":
        if info["order"] == "ABCABC":
            resp_cols = ["response_3", "response_4", "response_5"]
        elif info["order"] == "ABBCCA":
            resp_cols = ["response_2", "response_4", "response_5"]
        res = df.apply(score_interdms2_row, axis=1, feature=info["feature"], order=info["order"], 
                       resp_cols=resp_cols)
    elif info["task"] == "ctxdm":
        res = df.apply(score_ctxdm_row, axis=1, forder=info["forder"], resp_col="response_2")
    elif info["task"] == "1back":
        res = df.apply(score_1back_ao_trial, axis=1, op=info["op"] or "a", forder=info["forder"] or info["feature"],
                       resp_cols=[f"response_{i}" for i in range(1, 5)])
    elif info["task"] == "2back":
        res = df.apply(score_2back_trial, axis=1, feature=info["feature"], 
                       resp_cols=["response_2", "response_3", "response_4"])
    elif info["task"] == "dms_ao":
        res = df.apply(score_dms_ao_trial, axis=1, op=info["op"] or "a", forder=info["forder"] or "lc",
                       resp_cols=["response_1"])
    
    df_scored = pd.concat([df, pd.DataFrame(res.tolist(), index=df.index)], axis=1)
    df_scored.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] {os.path.basename(in_path)} -> {os.path.basename(out_path)}")

def _expand_inputs(path_or_glob: str) -> list[str]:
    """Resolves directory, glob, or single file into a list of paths."""
    if os.path.isdir(path_or_glob):
        return sorted(glob.glob(os.path.join(path_or_glob, "*.tsv")))
    hits = sorted(glob.glob(path_or_glob))
    return hits if hits else [path_or_glob]

# =========================
# Main Entry
# =========================

def main():
    ap = argparse.ArgumentParser(description="Score task TSVs based on filename routing.")
    ap.add_argument("input", help="TSV file path, directory, or glob pattern.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing results.")
    args = ap.parse_args()

    inputs = _expand_inputs(args.input)
    if not inputs:
        print(f"No inputs matched: {args.input}")
        return

    for p in inputs:
        if not os.path.isfile(p):
            print(f"[SKIP] Not a file: {p}")
            continue
        if not p.endswith(".tsv") or "_scored" in p:
            # Skip non-TSVs and already scored files
            continue
        try:
            score_tsv_file(p, overwrite=args.overwrite)
        except Exception as e:
            print(f"[ERR] Failed to process {os.path.basename(p)}: {e}")

if __name__ == "__main__":
    main()