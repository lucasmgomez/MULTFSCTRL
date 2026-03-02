import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]
DEFAULT_BEHAVIOR_ROOT = Path("/mnt/store1/lucas/newest_fmri_data/fmri_dataset_project/data/behavior")


def _normalize_value(value):
    if pd.isna(value):
        return ""

    as_text = str(value).strip()
    if as_text in {"--", "", "None", "nan", "NaN"}:
        return ""

    if re.fullmatch(r"-?\d+\.0+", as_text):
        return as_text.split(".")[0]

    return as_text


def infer_stim_count(columns):
    loc_indices = {
        int(m.group(1))
        for col in columns
        if (m := re.fullmatch(r"loc(\d+)", str(col).strip()))
    }
    locmod_indices = {
        int(m.group(1))
        for col in columns
        if (m := re.fullmatch(r"locmod(\d+)", str(col).strip()))
    }
    obj_indices = {
        int(m.group(1))
        for col in columns
        if (m := re.fullmatch(r"objmod(\d+)", str(col).strip()))
    }

    common = sorted((loc_indices | locmod_indices) & obj_indices)
    if not common:
        raise ValueError(
            "Could not infer stimulus count: no matching (locX|locmodX)/objmodX columns found."
        )

    expected = set(range(1, max(common) + 1))
    contiguous = sorted(expected & set(common))
    if contiguous != list(range(1, max(contiguous) + 1)):
        raise ValueError("Could not infer a contiguous 1..X range from locX/objmodX columns.")

    return max(contiguous)


def compute_trial_condition(df, stim_count):
    loc_col_by_index = {}
    missing = []

    for i in range(1, stim_count + 1):
        loc_candidate = f"loc{i}"
        locmod_candidate = f"locmod{i}"
        obj_candidate = f"objmod{i}"

        if loc_candidate in df.columns:
            loc_col_by_index[i] = loc_candidate
        elif locmod_candidate in df.columns:
            loc_col_by_index[i] = locmod_candidate
        else:
            missing.append(f"loc{i}|locmod{i}")

        if obj_candidate not in df.columns:
            missing.append(obj_candidate)

    if missing:
        raise ValueError(f"Missing required columns for X={stim_count}: {missing}")

    return df.apply(
        lambda row: "".join(
            f"{_normalize_value(row[loc_col_by_index[i]])}{_normalize_value(row[f'objmod{i}'])}"
            for i in range(1, stim_count + 1)
        ),
        axis=1,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add a trial condition column built from concatenated locX+objmodX values "
            "for all runs across selected subjects/sessions."
        )
    )
    parser.add_argument(
        "--behavior-root",
        type=Path,
        default=DEFAULT_BEHAVIOR_ROOT,
        help="Root behavior directory that contains sub-*/ses-*/func/*_events.tsv files.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=DEFAULT_SUBJECTS,
        help="Subjects to process (space-separated), e.g. sub-01 sub-02.",
    )
    parser.add_argument(
        "--session-start",
        type=int,
        default=1,
        help="First session index to include (default: 1 for ses-001).",
    )
    parser.add_argument(
        "--session-end",
        type=int,
        default=16,
        help="Last session index to include (default: 16 for ses-016).",
    )
    parser.add_argument(
        "-x",
        "--stim-count",
        type=int,
        default=None,
        help="Number of stimuli X. If omitted, infer from locX/objmodX columns.",
    )
    parser.add_argument(
        "-c",
        "--column-name",
        type=str,
        default="trial_condition",
        help="Name of the output condition column.",
    )
    parser.add_argument(
        "--inplace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output back to each input run file (default: True).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_with_condition",
        help=(
            "Suffix to append to output filename when --no-inplace is used, "
            "e.g. run-01_events_with_condition.tsv."
        ),
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=None,
        help="Input/output delimiter override. Auto-detected by extension if omitted.",
    )
    return parser.parse_args()


def _infer_sep(file_path, explicit_sep):
    if explicit_sep is not None:
        return explicit_sep
    return "\t" if file_path.suffix.lower() == ".tsv" else ","


def _iter_event_files(behavior_root, subjects, session_start, session_end):
    for subject in subjects:
        for session_idx in range(session_start, session_end + 1):
            session = f"ses-{session_idx:03d}"
            func_dir = behavior_root / subject / session / "func"
            if not func_dir.exists():
                continue

            for pattern in ("*_events.tsv", "*_events.csv"):
                for file_path in sorted(func_dir.glob(pattern)):
                    yield file_path


def process_file(file_path, stim_count, column_name, sep, inplace, output_suffix):
    file_sep = _infer_sep(file_path, sep)
    df = pd.read_csv(file_path, sep=file_sep)
    df.columns = df.columns.str.strip()

    used_stim_count = stim_count if stim_count is not None else infer_stim_count(df.columns)
    if used_stim_count <= 0:
        raise ValueError("--stim-count must be >= 1")

    df[column_name] = compute_trial_condition(df, used_stim_count)

    if inplace:
        output_file = file_path
    else:
        output_file = file_path.with_name(f"{file_path.stem}{output_suffix}{file_path.suffix}")

    df.to_csv(output_file, sep=file_sep, index=False)
    return output_file, used_stim_count


def main():
    args = parse_args()

    if not args.behavior_root.exists():
        raise FileNotFoundError(f"Behavior root not found: {args.behavior_root}")

    if args.session_start < 1 or args.session_end < args.session_start:
        raise ValueError("Invalid session range. Expected 1 <= session_start <= session_end.")

    files = list(
        _iter_event_files(
            behavior_root=args.behavior_root,
            subjects=args.subjects,
            session_start=args.session_start,
            session_end=args.session_end,
        )
    )
    if not files:
        raise FileNotFoundError(
            "No run files found matching *_events.tsv/csv under the provided subjects/sessions."
        )

    processed = 0
    failed = 0

    for file_path in files:
        try:
            output_file, used_stim_count = process_file(
                file_path=file_path,
                stim_count=args.stim_count,
                column_name=args.column_name,
                sep=args.sep,
                inplace=args.inplace,
                output_suffix=args.output_suffix,
            )
            processed += 1
        except Exception as error:
            failed += 1
            print(f"ERR | {file_path} | {error}")

    print("\nSummary")
    print(f"Behavior root: {args.behavior_root}")
    print(f"Subjects: {', '.join(args.subjects)}")
    print(f"Sessions: ses-{args.session_start:03d}..ses-{args.session_end:03d}")
    print(f"Column added: {args.column_name}")
    print(f"Processed files: {processed}")
    print(f"Failed files: {failed}")


if __name__ == "__main__":
    main()
