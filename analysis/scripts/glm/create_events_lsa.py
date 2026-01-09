#!/usr/bin/env python3
import pandas as pd
import glob
import os
import argparse
import sys
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw block logs to block-level LSA-style BIDS events files "
                    "(ONE events.tsv per run, one unique trial_type per event)."
    )

    parser.add_argument('--input_dir', '-i', required=True,
                        help="Directory containing the raw block .tsv files (each file = one scanned block/run).")
    parser.add_argument('--output_dir', '-o', required=True,
                        help="Directory where events outputs will be saved (base-events + LSA folder).")

    parser.add_argument('--final_delay_duration', type=float, default=3.48,
                        help="Duration (s) for the final delay after the last stimulus in a trial.")

    parser.add_argument('--lsa_mode', choices=['encoding', 'delay', 'both'], default='both',
                        help="Which event set(s) to include in the LSA events file.")

    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing outputs if present.")

    parser.add_argument('--save_base', action='store_true', default=True,
                        help="Also save a block-level base events.tsv for QC/recordkeeping (default: on).")

    return parser.parse_args()


def get_stimulus_indices(df_columns):
    """Scans column names to find all unique stimulus indices (0, 1, ...)."""
    indices = set()
    for col in df_columns:
        match = re.search(r'stimulus_(\d+)_onset', col)
        if match:
            indices.add(int(match.group(1)))
    return sorted(list(indices))


def extract_metadata(filename):
    """
    Parses filename for Task Name (including variants), Run/Block Number, and Timestamp.
    Matches your existing behavior:
      - if block/run is found, convert 0-based -> 1-based
    """
    run_match = re.search(r'(?:block|run)[-_](\d+)', filename, re.IGNORECASE)
    if run_match:
        run_num = int(run_match.group(1)) + 1
        run_start_index = run_match.start()
    else:
        run_num = 1
        run_start_index = len(filename)

    task_match = re.search(r'task-(.+)', filename[:run_start_index])
    if task_match:
        raw_name = task_match.group(1)
        task_name = raw_name.rstrip('_')
    else:
        task_name = "unknown"

    time_match = re.search(r'(\d{2}-\d{2}-\d{2})', filename)
    timestamp = time_match.group(1) if time_match else "00-00-00"

    return task_name, run_num, timestamp


def build_base_events(df, task_name, final_delay_duration):
    """
    Build a base events table with one row per Encoding/Delay event.
    Columns:
      onset, duration, trial_type, phase, trial, pos, event_id
    """
    if 'TrialNumber' in df.columns:
        df = df.sort_values('TrialNumber')

    stim_indices = get_stimulus_indices(df.columns)
    if not stim_indices:
        return pd.DataFrame()

    base = []

    for trial_idx, (_, row) in enumerate(df.iterrows()):
        trial_label = f"Trial{trial_idx + 1:02d}"

        for i, curr_idx in enumerate(stim_indices):
            pos = i + 1

            # Encoding
            s_on = row.get(f'stimulus_{curr_idx}_onset')
            s_off = row.get(f'stimulus_{curr_idx}_offset')

            if pd.notna(s_on) and pd.notna(s_off):
                base.append({
                    'onset': float(s_on),
                    'duration': float(s_off - s_on),
                    'trial_type': f'{task_name}_Encoding',
                    'phase': 'Encoding',
                    'trial': trial_label,
                    'pos': pos,
                })

            # Delay after this encoding position
            if i < len(stim_indices) - 1:
                next_idx = stim_indices[i + 1]
                next_on = row.get(f'stimulus_{next_idx}_onset')

                if pd.notna(s_off) and pd.notna(next_on):
                    dur = float(next_on - s_off)
                    if dur > 0:
                        base.append({
                            'onset': float(s_off),
                            'duration': dur,
                            'trial_type': f'{task_name}_Delay',
                            'phase': 'Delay',
                            'trial': trial_label,
                            'pos': pos,
                        })
            else:
                # Final delay
                if pd.notna(s_off):
                    base.append({
                        'onset': float(s_off),
                        'duration': float(final_delay_duration),
                        'trial_type': f'{task_name}_Delay',
                        'phase': 'Delay',
                        'trial': trial_label,
                        'pos': pos,
                    })

    if not base:
        return pd.DataFrame()

    base_df = pd.DataFrame(base).sort_values('onset').reset_index(drop=True)

    # Assign per-phase event ids in chronological order
    base_df['event_id'] = pd.NA
    enc_counter = 0
    del_counter = 0
    for idx, row in base_df.iterrows():
        if row['phase'] == 'Encoding':
            enc_counter += 1
            base_df.at[idx, 'event_id'] = f"Enc{enc_counter:04d}"
        else:
            del_counter += 1
            base_df.at[idx, 'event_id'] = f"Del{del_counter:04d}"

    base_df['onset'] = base_df['onset'].round(4)
    base_df['duration'] = base_df['duration'].round(4)

    return base_df


def write_base_events(base_df, output_dir, task_name, run_num, overwrite):
    """Save a single block/run-level base events TSV for QC."""
    run_prefix = f"task-{task_name}_run-{run_num:02d}"
    out_path = os.path.join(output_dir, f"{run_prefix}_base-events.tsv")

    if os.path.exists(out_path) and not overwrite:
        return out_path

    out_df = base_df[['onset', 'duration', 'trial_type', 'trial', 'pos', 'phase', 'event_id']].copy()
    out_df = out_df.sort_values('onset').reset_index(drop=True)
    out_df.to_csv(out_path, sep='\t', index=False)
    return out_path


def write_lsa_events_for_block(base_df, output_dir, task_name, run_num, overwrite, lsa_mode):
    """
    Creates ONE LSA events.tsv for this block/run.
    Output columns: onset, duration, trial_type
    trial_type is unique per event_id so Nilearn/GLM will build one regressor per event.
    """
    if base_df.empty:
        print("   -> No events found after parsing. Skipping.")
        return None

    run_prefix = f"task-{task_name}_run-{run_num:02d}"
    block_outdir = os.path.join(output_dir, f"{run_prefix}_LSA")
    os.makedirs(block_outdir, exist_ok=True)

    out_path = os.path.join(block_outdir, f"{run_prefix}_lsa-events.tsv")
    if os.path.exists(out_path) and not overwrite:
        print(f"   -> LSA events exists (skip): {out_path}")
        return out_path

    enc_df = base_df[base_df['phase'] == 'Encoding'].copy()
    del_df = base_df[base_df['phase'] == 'Delay'].copy()

    rows = []

    if lsa_mode in ("encoding", "both"):
        for _, r in enc_df.iterrows():
            rows.append({
                "onset": float(r["onset"]),
                "duration": float(r["duration"]),
                "trial_type": f"{task_name}_Enc_{r['event_id']}",
            })

    if lsa_mode in ("delay", "both"):
        for _, r in del_df.iterrows():
            rows.append({
                "onset": float(r["onset"]),
                "duration": float(r["duration"]),
                "trial_type": f"{task_name}_Del_{r['event_id']}",
            })

    if not rows:
        print("   -> No events selected for LSA (lsa_mode filtered everything).")
        return None

    out_df = pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
    out_df["onset"] = out_df["onset"].round(4)
    out_df["duration"] = out_df["duration"].round(4)
    out_df.to_csv(out_path, sep="\t", index=False)

    print(f"   -> Wrote LSA events: {out_path} (n_rows={len(out_df)})")
    return out_path


def process_single_file(file_path, output_dir, final_delay_duration, overwrite, lsa_mode, save_base):
    base_name = os.path.basename(file_path)
    task_name, run_num, timestamp = extract_metadata(base_name)

    print(f"Processing: {base_name}")
    print(f"   -> Detected Task/Block: '{task_name}', Run: {run_num}")

    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"   Error reading file: {e}")
        return

    base_df = build_base_events(df, task_name, final_delay_duration)
    if base_df.empty:
        print("   -> No usable events parsed. Skipping.")
        return

    if save_base:
        write_base_events(base_df, output_dir, task_name, run_num, overwrite)

    write_lsa_events_for_block(base_df, output_dir, task_name, run_num, overwrite, lsa_mode)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.input_dir, "*.tsv"))
    # exclude previously generated outputs
    files = [f for f in files if '_events.tsv' not in f and '_base-events.tsv' not in f]

    if not files:
        print(f"No raw .tsv files found in {args.input_dir}")
        sys.exit(1)

    files.sort(key=lambda x: extract_metadata(os.path.basename(x))[2])

    print(f"Found {len(files)} files. LSA mode: {args.lsa_mode}")

    for f in files:
        process_single_file(
            f,
            args.output_dir,
            final_delay_duration=args.final_delay_duration,
            overwrite=args.overwrite,
            lsa_mode=args.lsa_mode,
            save_base=args.save_base
        )


if __name__ == "__main__":
    main()

"""
Example:
python create_events_lsa.py \
  --input_dir /mnt/tempdata/lucas/fmri/recordings/TR/behav/sub-01/ses-02 \
  --output_dir /mnt/tempdata/lucas/fmri/recordings/TR/behav/sub-01/ses-02/events \
  --lsa_mode both \
  --final_delay_duration 3.48 \
  --overwrite
"""