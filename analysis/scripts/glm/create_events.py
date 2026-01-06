import pandas as pd
import glob
import os
import argparse
import sys
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw block logs to block-level LSS-style BIDS events files (one events.tsv per target event)."
    )

    parser.add_argument('--input_dir', '-i', required=True,
                        help="Directory containing the raw block .tsv files (each file = one scanned block/run).")
    parser.add_argument('--output_dir', '-o', required=True,
                        help="Directory where LSS events files will be saved.")

    parser.add_argument('--extract_responses', action='store_true',
                        help="(Kept for compatibility) Currently ignored.")

    parser.add_argument('--final_delay_duration', type=float, default=3.48,
                        help="Duration (s) for the final delay after the last stimulus in a trial.")

    parser.add_argument('--lss_mode', choices=['encoding', 'delay', 'both'], default='both',
                        help="Which LSS target-event set(s) to generate.")

    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing LSS events files if present.")

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
        return

    out_df = base_df[['onset', 'duration', 'trial_type', 'trial', 'pos', 'phase', 'event_id']].copy()
    out_df = out_df.sort_values('onset').reset_index(drop=True)
    out_df.to_csv(out_path, sep='\t', index=False)

def write_lss_files_for_block(base_df, output_dir, task_name, run_num, overwrite, lss_mode):
    """
    Creates one events.tsv per target event for LSS, within this scanned block/run.
    Output columns: onset, duration, trial_type
    trial_type values are task-specific to keep multi-task projects organized.
    """
    if base_df.empty:
        print("   -> No events found after parsing. Skipping.")
        return

    run_prefix = f"task-{task_name}_run-{run_num:02d}"
    block_outdir = os.path.join(output_dir, f"{run_prefix}_LSS")
    os.makedirs(block_outdir, exist_ok=True)

    enc_df = base_df[base_df['phase'] == 'Encoding'].copy()
    del_df = base_df[base_df['phase'] == 'Delay'].copy()

    def save_events(out_df, out_path):
        out_df = out_df.sort_values('onset').reset_index(drop=True)
        out_df.to_csv(out_path, sep='\t', index=False)

    # -------- Encoding-target LSS (within the block) --------
    if lss_mode in ('encoding', 'both') and not enc_df.empty:
        for target_id in enc_df['event_id'].tolist():
            out_name = f"{run_prefix}_lss-EncTarget_{target_id}_events.tsv"
            out_path = os.path.join(block_outdir, out_name)

            if os.path.exists(out_path) and not overwrite:
                continue

            # Target encoding event
            target = enc_df[enc_df['event_id'] == target_id][['onset', 'duration']].copy()
            target['trial_type'] = f'{task_name}_EncTarget'

            # Other encodings
            other = enc_df[enc_df['event_id'] != target_id][['onset', 'duration']].copy()
            other['trial_type'] = f'{task_name}_EncOther'

            # All delays
            delays = del_df[['onset', 'duration']].copy()
            delays['trial_type'] = f'{task_name}_DelayAll'

            out_df = pd.concat([target, other, delays], axis=0, ignore_index=True)
            out_df = out_df[['onset', 'duration', 'trial_type']]
            save_events(out_df, out_path)

        print(f"   -> Wrote encoding LSS files for block: {len(enc_df)}")

    elif lss_mode in ('encoding', 'both'):
        print("   -> No Encoding events found; skipping encoding LSS files.")

    # -------- Delay-target LSS (within the block) --------
    if lss_mode in ('delay', 'both') and not del_df.empty:
        for target_id in del_df['event_id'].tolist():
            out_name = f"{run_prefix}_lss-DelayTarget_{target_id}_events.tsv"
            out_path = os.path.join(block_outdir, out_name)

            if os.path.exists(out_path) and not overwrite:
                continue

            # Target delay event
            target = del_df[del_df['event_id'] == target_id][['onset', 'duration']].copy()
            target['trial_type'] = f'{task_name}_DelayTarget'

            # Other delays
            other = del_df[del_df['event_id'] != target_id][['onset', 'duration']].copy()
            other['trial_type'] = f'{task_name}_DelayOther'

            # All encodings
            encs = enc_df[['onset', 'duration']].copy()
            encs['trial_type'] = f'{task_name}_EncAll'

            out_df = pd.concat([target, other, encs], axis=0, ignore_index=True)
            out_df = out_df[['onset', 'duration', 'trial_type']]
            save_events(out_df, out_path)

        print(f"   -> Wrote delay LSS files for block: {len(del_df)}")

    elif lss_mode in ('delay', 'both'):
        print("   -> No Delay events found; skipping delay LSS files.")

def process_single_file(file_path, output_dir, final_delay_duration, overwrite, lss_mode, save_base):
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

    write_lss_files_for_block(base_df, output_dir, task_name, run_num, overwrite, lss_mode)

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.input_dir, "*.tsv"))
    files = [f for f in files if '_events.tsv' not in f and '_base-events.tsv' not in f]

    if not files:
        print(f"No raw .tsv files found in {args.input_dir}")
        sys.exit(1)

    files.sort(key=lambda x: extract_metadata(os.path.basename(x))[2])

    print(f"Found {len(files)} files. LSS mode: {args.lss_mode}")

    for f in files:
        process_single_file(
            f,
            args.output_dir,
            final_delay_duration=args.final_delay_duration,
            overwrite=args.overwrite,
            lss_mode=args.lss_mode,
            save_base=args.save_base
        )

if __name__ == "__main__":
    main()

"""
python create_events.py \
  --input_dir /mnt/tempdata/lucas/fmri/recordings/TR/behav/sub-01/ses-02 \
  --output_dir /mnt/tempdata/lucas/fmri/recordings/TR/behav/sub-01/ses-02/events \
  --lss_mode both \
  --final_delay_duration 3.48 \
  --overwrite
"""