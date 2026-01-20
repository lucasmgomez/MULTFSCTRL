import os
import re
import glob
import numpy as np
import pandas as pd
import nibabel as nib

# Regex to parse task info from beta files
TASKDIR_RE = re.compile(r"^task-(?P<task>[^_]+)_acq-(?P<acq>.+)_run-(?P<run>\d+)$")

# Regex for beta filename parsing (LSS + LSA)
BETA_RE = re.compile(
    r"""
    (?:
        # LSS
        lss-(?P<lss_kind>EncTarget|DelayTarget)_(?P<eid_lss>Enc\d{4}|Del\d{4})_events_beta-target
      |
        # LSA
        lsa_(?P<lsa_stub>.+)_(?P<lsa_phase>Enc|Del)_(?P<eid_lsa>Enc\d{4}|Del\d{4})_beta
    )
    \.dscalar\.nii$
    """,
    re.VERBOSE
)

def find_block_log_for_run(block_logs_dir: str, task: str, acq: str, run: str):
    """
    Finds your raw scored block log TSV.

    Example:
      sub-01_ses-1_20251106-120104_task-1back_ctg_block_0_events_12-33-56_scored.tsv
    """
    run_int = int(run)
    block0 = run_int - 1

    task_tag = f"task-{task}_{acq}"

    patterns = [
        f"*{task_tag}*block_{block0}*scored*.tsv",
        f"*{task_tag}*block-{block0}*scored*.tsv",
        f"*{task_tag}*block_{block0}*.tsv",
        f"*{task_tag}*block-{block0}*.tsv",
        f"*{task_tag}*.tsv",
    ]

    hits = []
    for pat in patterns:
        hits.extend(glob.glob(os.path.join(block_logs_dir, pat)))

    if not hits:
        return None

    def is_bids_events_output(path: str) -> bool:
        bn = os.path.basename(path)
        return bn.endswith("_events.tsv") or bn.endswith("_base-events.tsv")

    hits = [h for h in hits if not is_bids_events_output(h)]
    if not hits:
        return None

    preferred = [h for h in hits if re.search(rf"(block)[-_]{block0}\b", os.path.basename(h))]
    if preferred:
        preferred = sorted(preferred, key=lambda x: len(os.path.basename(x)))
        return preferred[0]

    hits = sorted(hits, key=lambda x: len(os.path.basename(x)))
    return hits[0]

def load_block_log(block_log_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(block_log_tsv, sep="\t")
    if "TrialNumber" not in df.columns:
        raise ValueError(f"Block log missing TrialNumber column: {block_log_tsv}")
    if "tc" not in df.columns:
        raise ValueError(f"Block log missing tc column: {block_log_tsv}")

    df = df.copy()
    df["TrialNumber"] = pd.to_numeric(df["TrialNumber"], errors="raise").astype(int)
    df = df.sort_values("TrialNumber").reset_index(drop=True)
    return df

def parse_taskdir_name(taskdir_basename: str):
    m = TASKDIR_RE.match(taskdir_basename)
    if not m:
        return None
    return m.group("task"), m.group("acq"), m.group("run")

def parse_beta_filename(bname: str):
    """
    Returns (phase, event_id) where:
      phase in {"Encoding","Delay"}
      event_id like "Enc0001"
    """
    m = BETA_RE.search(bname)
    if not m:
        return None

    if m.group("eid_lss"):
        eid = m.group("eid_lss")
        kind = m.group("lss_kind")
        phase = "Encoding" if kind.lower().startswith("enc") else "Delay"
        return phase, eid

    eid = m.group("eid_lsa")
    lsa_phase = m.group("lsa_phase")
    phase = "Encoding" if lsa_phase == "Enc" else "Delay"
    return phase, eid

def load_eventid_to_trialpos(base_events_tsv: str):
    df = pd.read_csv(base_events_tsv, sep="\t")
    needed = {"event_id", "phase", "trial", "pos"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {base_events_tsv}: {missing}")

    out = {}
    for _, row in df.iterrows():
        eid = str(row["event_id"])
        phase = str(row["phase"])
        trial = str(row["trial"])  # Trial01
        pos = int(row["pos"])      # 1..N
        out[eid] = (phase, trial, pos)
    return out

def for_tc_align(beta_file, event_map, trial_to_row):
        bname = os.path.basename(beta_file)
        parsed_beta = parse_beta_filename(bname)

        phase, eid = parsed_beta

        _, trial_label, pos = event_map[eid]

        m = re.match(r"Trial(\d+)", str(trial_label))
        trial_int = int(m.group(1))
 
        row_idx = trial_to_row[trial_int]

        return phase, row_idx

def get_betas(subj, ses, task_name, betas_dir, phase):
    # TODO: get event beta for a given subject, session, task, run, and phase (Encoding/Delay)
    return

def get_activations(task, tc, activations_dir):
    # TODO: get model activations for a given task and trial condition
    return

def study_dict(subj, sessions, tasks, behav_dir, betas_dir, activations_dir):

    betas_dict = {}
    activations_dict = {}

    for session in sessions:
        betas_dict[session] = {}

        ses_dir = os.path.join(betas_dir, subj, session)
        if not os.path.isdir(ses_dir):
            raise FileNotFoundError(f"Missing GLM session dir: {ses_dir}")

        task_dirs = sorted([parse_taskdir_name(d) for d in os.listdir(ses_dir) if d.startswith("task-")])

        # TODO: glob to get all block log files
       
        for (task, acq, run) in tasks:
            task_key = f"task-{task}_acq-{acq}_run-{run}"

            betas_dict[session][task_key] = {}
            activations_dict[task_key] = {}

            events_dir = os.path.join(behav_dir, subj, session, "events")
            base_events_tsv = os.path.join(events_dir, f"task-{task}_{acq}_run-{run}_base-events.tsv")
            event_map = load_eventid_to_trialpos(base_events_tsv)


            block_log_tsv = find_block_log_for_run(behav_dir, task, acq, run)
            block_df = pd.read_csv(block_log_tsv, sep="\t")
            block_df = block_df.copy()
            block_df["TrialNumber"] = pd.to_numeric(block_df["TrialNumber"], errors="raise").astype(int)
            block_df = block_df.sort_values("TrialNumber").reset_index(drop=True)

            trial_to_row = {int(t): i for i, t in enumerate(block_df["TrialNumber"].tolist())}



            for tc in tcs:
                betas_dict[session][task][tc] = {
                    'Encoding': get_betas(subj, session, task, betas_dir, phase='Encoding'),
                    'Delay': get_betas(subj, session, task, betas_dir, phase='Delay')
                }

                if tc not in activations_dict[task]:
                    activations_dict[task][tc] = get_activations(task, tc, activations_dir)




    return  