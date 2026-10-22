#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import nibabel as nib


# ----------------------------
# ROI mask (Glasser dlabel)
# ----------------------------
def make_roi_mask(dlabel_path: str, roi_names):
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0]  # {id: (name, rgba)}

    roi_set = {r.strip() for r in roi_names if r.strip()}
    if not roi_set:
        raise ValueError("roi_names is empty after parsing.")

    keys = [k for k, (name, _) in label_dict.items() if name in roi_set]
    found = [label_dict[k][0] for k in keys]
    missing = sorted(list(roi_set - set(found)))
    if missing:
        all_names = sorted([name for _, (name, _) in label_dict.items()])
        suggestions = [n for n in all_names if any(m.lower() in n.lower() for m in missing)][:40]
        raise ValueError(
            f"Missing ROI labels: {missing}\n"
            f"Found: {found}\n"
            f"Suggestions: {suggestions}"
        )

    mask = np.isin(data, keys)
    if mask.sum() == 0:
        raise ValueError("ROI mask has 0 vertices (space mismatch or wrong labels).")

    roi_label = "+".join(found)
    print("Matched ROI labels:", found)
    print("Total ROI vertices:", int(mask.sum()))
    return mask, roi_label


# ----------------------------
# Task dir parsing
# ----------------------------
TASKDIR_RE = re.compile(r"^task-(?P<task>[^_]+)_acq-(?P<acq>.+)_run-(?P<run>\d+)$")


def parse_taskdir_name(taskdir_basename: str):
    m = TASKDIR_RE.match(taskdir_basename)
    if not m:
        return None
    return m.group("task"), m.group("acq"), m.group("run")


def behav_baseevents_name(task: str, acq: str, run: str):
    return f"task-{task}_{acq}_run-{run}_base-events.tsv"


# ----------------------------
# Beta filename parsing (LSS + LSA)
# ----------------------------
BETA_RE = re.compile(
    r"""
    (?:
        # LSS
        lss-(?P<lss_kind>EncTarget|DelayTarget)_(?P<eid_lss>Enc\d{4}|Del\d{4})_events_beta-target
      |
        # LSA / GLMsingle (same filename layout)
        (?:lsa|glmsingle[BC]?)_(?P<lsa_stub>.+)_(?P<lsa_phase>Enc|Del)_(?P<eid_lsa>Enc\d{4}|Del\d{4})_beta
    )
    \.dscalar\.nii$
    """,
    re.VERBOSE
)


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


# ----------------------------
# base-events: event_id -> (phase, trial_label, pos)
# ----------------------------
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


def trial_label_to_int(trial_label: str) -> int:
    m = re.match(r"Trial(\d+)", str(trial_label))
    if not m:
        raise ValueError(f"Unexpected trial label: {trial_label}")
    return int(m.group(1))


# ----------------------------
# Block log utilities
# ----------------------------
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


# ----------------------------
# Collect per-session: key -> (roi_vec, feat)
# KEY IS BASED ON tc + pos (+ task/acq/phase) TO ALIGN REPEATS ACROSS SESSIONS (not run: see below)
# ----------------------------
def collect_one_session_dict(
    glm_base_dir: str,
    behav_base: str,
    events_subdir: str,     # e.g. "events_wfdelay" — must match the GLM variant
    block_logs_dir: str,
    sub: str,
    session: str,
    dlabel_path: str,
    roi_names: list[str],
    feature_to_decode: str = "loc",
    phase_to_decode: str = "Encoding",
    only_tasks_regex: str | None = None,
    verbose: bool = False,
):
    roi_mask, roi_label = make_roi_mask(dlabel_path, roi_names)
    n_roi = int(roi_mask.sum())

    ses_dir = os.path.join(glm_base_dir, sub, session)
    if not os.path.isdir(ses_dir):
        raise FileNotFoundError(f"Missing GLM session dir: {ses_dir}")

    task_dirs = sorted([d for d in os.listdir(ses_dir) if d.startswith("task-")])
    if only_tasks_regex:
        pat = re.compile(only_tasks_regex)
        task_dirs = [d for d in task_dirs if pat.search(d)]

    out = {}        # key -> (roi_vec, feat_val)
    meta_rows = []

    for taskdir in task_dirs:
        parsed = parse_taskdir_name(taskdir)
        if not parsed:
            if verbose:
                print("[WARN] Skipping unparsable task dir:", taskdir)
            continue
        task, acq, run = parsed

        glm_task_dir = os.path.join(ses_dir, taskdir)
        behav_events_dir = os.path.join(behav_base, sub, session, events_subdir)
        base_events_tsv = os.path.join(behav_events_dir, behav_baseevents_name(task, acq, run))
        if not os.path.isfile(base_events_tsv):
            if verbose:
                print("[WARN] Missing base-events:", base_events_tsv)
            continue

        block_log_tsv = find_block_log_for_run(block_logs_dir, task, acq, run)
        if block_log_tsv is None or (not os.path.isfile(block_log_tsv)):
            if verbose:
                print(f"[WARN] Could not find block log for {taskdir} in {block_logs_dir}")
            continue

        event_map = load_eventid_to_trialpos(base_events_tsv)
        block_df = load_block_log(block_log_tsv)
        trial_to_row = {int(t): i for i, t in enumerate(block_df["TrialNumber"].tolist())}

        files = sorted(glob.glob(os.path.join(glm_task_dir, "*.dscalar.nii")))
        if not files:
            if verbose:
                print("[WARN] No dscalars in:", glm_task_dir)
            continue

        for f in files:
            bname = os.path.basename(f)
            parsed_beta = parse_beta_filename(bname)
            if parsed_beta is None:
                continue

            phase, eid = parsed_beta
            if phase != phase_to_decode:
                continue

            if eid not in event_map:
                if verbose:
                    print(f"[WARN] {taskdir}: {eid} not in base-events (file={bname})")
                continue

            phase2, trial_label, pos = event_map[eid]
            if str(phase2).lower() != phase_to_decode.lower():
                continue

            trial_int = trial_label_to_int(trial_label)
            if trial_int not in trial_to_row:
                if verbose:
                    print(f"[WARN] {taskdir}: TrialNumber {trial_int} missing from block log {os.path.basename(block_log_tsv)}")
                continue

            row_idx = trial_to_row[trial_int]

            # --- tc is the repeat-alignment key ---
            tc_val = block_df.loc[row_idx, "tc"]
            if pd.isna(tc_val):
                if verbose:
                    print(f"[WARN] {taskdir}: tc is NaN for trial={trial_int}")
                continue
            # canonicalize; avoids '...0' vs '...' mismatches
            try:
                tc_key = str(int(float(tc_val)))
            except Exception:
                tc_key = str(tc_val)

            feat_col = f"{feature_to_decode}{pos}"
            if feat_col not in block_df.columns:
                if verbose:
                    print(f"[WARN] {taskdir}: missing {feat_col} in block log {os.path.basename(block_log_tsv)}")
                continue

            feat_val = block_df.loc[row_idx, feat_col]
            if pd.isna(feat_val):
                continue

            beta = nib.load(f).get_fdata().squeeze().astype(np.float32)
            roi_vec = beta[roi_mask].astype(np.float32).reshape(-1)
            if roi_vec.shape[0] != n_roi:
                raise ValueError(f"ROI vec length mismatch for {bname}: got {roi_vec.shape[0]} expected {n_roi}")

            # NEW KEY: stable across sessions even if event_id order changes.
            # Run is deliberately NOT part of it ("*" slot keeps the tuple shape): ctxdm trials swap between
            # run-01 and run-02 across sessions, and tc is unique within a task variant per session.
            key = (task, acq, "*", phase_to_decode, int(pos), tc_key)

            if key in out:
                raise RuntimeError(f"Duplicate key in session {session}: {key} (file={bname}); "
                                   "tc is no longer unique within this task variant, so run must be part of the key")

            out[key] = (roi_vec, feat_val)

            meta_rows.append({
                "session": session,
                "taskdir": taskdir,
                "task": task,
                "acq": acq,
                "run": run,
                "phase": phase_to_decode,
                "pos": int(pos),
                "tc": tc_key,
                "event_id": eid,
                "trial": trial_label,
                "trial_int": trial_int,
                "feat_col": feat_col,
                "feat": feat_val,
                "beta_file": bname,
                "block_log": os.path.basename(block_log_tsv),
                "roi": roi_label,
            })

    meta = pd.DataFrame(meta_rows)
    return out, meta, roi_label


# ----------------------------
# Multi-session repeat averaging
# ----------------------------
def collect_repeat_averaged_across_sessions(
    glm_base_dir: str,
    behav_base: str,
    events_subdir: str,     # e.g. "events_wfdelay" — must match the GLM variant
    block_logs_base: str,   # base dir: .../behav/sub-01 (contains ses-01, ses-02, ...)
    sub: str,
    sessions: list[str],
    dlabel_path: str,
    roi_names: list[str],
    feature_to_decode: str = "loc",
    phase_to_decode: str = "Encoding",
    only_tasks_regex: str | None = None,
    verbose: bool = False,
):
    """
    Returns repeat-averaged patterns across sessions, aligned by (task, acq, phase, pos, tc) — not run.

    Output:
      X_avg : (n_common_events, n_vertices)
      y     : (n_common_events,)
      meta_avg : DataFrame with one row per averaged event
      meta_per_ses : DataFrame with one row per session-event used
    """
    per_ses = {}
    meta_all = []
    roi_label = None

    for ses in sessions:
        block_logs_dir = os.path.join(block_logs_base, ses)
        d, meta, roi_label_local = collect_one_session_dict(
            glm_base_dir=glm_base_dir,
            behav_base=behav_base,
            events_subdir=events_subdir,
            block_logs_dir=block_logs_dir,
            sub=sub,
            session=ses,
            dlabel_path=dlabel_path,
            roi_names=roi_names,
            feature_to_decode=feature_to_decode,
            phase_to_decode=phase_to_decode,
            only_tasks_regex=only_tasks_regex,
            verbose=verbose,
        )
        per_ses[ses] = d
        meta_all.append(meta)
        if roi_label is None:
            roi_label = roi_label_local

        print(f"[INFO] {ses}: collected {len(d)} events")

    # Intersect keys across ALL sessions
    key_sets = [set(per_ses[ses].keys()) for ses in sessions]
    common_keys = set.intersection(*key_sets) if key_sets else set()
    common_keys = sorted(common_keys)

    if not common_keys:
        raise RuntimeError(
            "No common event keys across sessions.\n"
            "Since we align using (task,acq,phase,pos,tc), this usually means:\n"
            "  - tc differs across sessions (format mismatch), OR\n"
            "  - some sessions are missing blocks/GLM outputs, OR\n"
            "  - block log finder mismatched the run/block file.\n"
        )

    print(f"[INFO] common keys across sessions: {len(common_keys)}")

    X_rows = []
    y_rows = []
    meta_rows = []

    for key in common_keys:
        roi_vecs = []
        feat_vals = []

        for ses in sessions:
            roi_vec, feat = per_ses[ses][key]
            roi_vecs.append(roi_vec)
            feat_vals.append(feat)

        # label consistency check
        uniq = pd.unique(pd.Series(feat_vals).astype(str))
        if len(uniq) != 1:
            raise RuntimeError(
                f"Label mismatch across sessions for key={key}\n"
                f"feat_vals={feat_vals}\n"
                "This usually means your alignment is wrong."
            )

        X_avg = np.mean(np.stack(roi_vecs, axis=0), axis=0).astype(np.float32)
        y_val = feat_vals[0]

        X_rows.append(X_avg)
        y_rows.append(y_val)

        task, acq, run, phase, pos, tc = key
        meta_rows.append({
            "task": task,
            "acq": acq,
            "run": run,
            "phase": phase,
            "pos": pos,
            "tc": tc,
            "feat": y_val,
            "roi": roi_label,
            "n_sessions_averaged": len(sessions),
        })

    X = np.vstack(X_rows).astype(np.float32)
    y = np.asarray(y_rows)
    meta_avg = pd.DataFrame(meta_rows)
    meta_per_ses = pd.concat(meta_all, ignore_index=True) if meta_all else pd.DataFrame()

    return X, y, meta_avg, meta_per_ses


# ----------------------------
# Ridge decoding (5-fold CV, grouped by trial, optional permutation null)
# ----------------------------
def trial_groups(meta_avg: pd.DataFrame) -> np.ndarray:
    """One group per trial: all positions of a trial share context and must stay in the same fold."""
    return meta_avg[["task", "acq", "run", "tc"]].astype(str).agg("|".join, axis=1).to_numpy()


def task_pos_cells(meta_avg: pd.DataFrame) -> np.ndarray:
    """One nuisance cell per task variant x sequence position (e.g. 1back|loc|3)."""
    return meta_avg[["task", "acq", "pos"]].astype(str).agg("|".join, axis=1).to_numpy()


def remove_cell_means(Xtr, Xte, cells_tr, cells_te):
    """Subtract each nuisance cell's mean pattern, estimated from training events only."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    fallback = Xtr.mean(axis=0)
    for c in np.unique(np.concatenate([cells_tr, cells_te])):
        in_tr, in_te = cells_tr == c, cells_te == c
        m = Xtr[in_tr].mean(axis=0) if in_tr.any() else fallback
        Xtr[in_tr] -= m
        Xte[in_te] -= m
    return Xtr, Xte


def ridge_decode_5fold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    alpha: float = 1.0,
    alphas: np.ndarray | None = None,
    nuisance: np.ndarray | None = None,
    n_perm: int = 0,
    random_state: int = 0,
):
    """
    alphas:   if given, alpha is chosen per outer fold by inner 5-fold (grouped) CV on the training set,
              and re-chosen for every permutation so the null includes the tuning step.
    nuisance: per-event cell labels (e.g. task x position); each cell's training mean pattern is removed
              from train and test events before scaling.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    vals, counts = np.unique(y, return_counts=True)
    print("Class counts:", list(zip(vals, counts)))
    print("majority-class baseline:", counts.max() / counts.sum())
    print("balanced-acc chance:", 1 / len(vals))

    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(vals)

    def make_cv():
        if groups is None:
            return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

    if groups is not None:
        print(f"Grouped CV: {len(np.unique(groups))} groups")

    # Same model as sklearn's RidgeClassifier (±1 targets, intercept), in kernel form. Ridge is linear
    # in its targets: with K = Xtr Xtr^T = U diag(ev) U^T, the test scores for any alpha are
    # (Xte Xtr^T U) diag(1 / (ev + alpha)) U^T (Y - mean(Y)) + mean(Y). The eigendecomposition depends
    # only on X, so each permutation (and each candidate alpha) is a few matrix products, not a refit.
    def kernel_parts(Xa, Xb):
        mu = Xa.mean(axis=0)  # ridge intercept = centre on the training mean
        Xa, Xb = Xa - mu, Xb - mu
        ev, U = np.linalg.eigh(Xa @ Xa.T)
        return (Xb @ Xa.T) @ U, ev, U

    folds = []
    for tr, te in make_cv().split(X, y_enc, groups):
        Xtr, Xte = X[tr], X[te]
        if nuisance is not None:
            Xtr, Xte = remove_cell_means(Xtr, Xte, nuisance[tr], nuisance[te])
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        inner = []
        if alphas is not None:
            g_tr = None if groups is None else groups[tr]
            for itr, ite in make_cv().split(Xtr, y_enc[tr], g_tr):
                inner.append((itr, ite, kernel_parts(Xtr[itr], Xtr[ite])))
        folds.append((tr, te, kernel_parts(Xtr, Xte), inner))

    def centred_targets(ytr):
        if n_classes == 2:
            Y = np.where(ytr == 1, 1.0, -1.0)[:, None]
        else:
            Y = np.where(ytr[:, None] == np.arange(n_classes), 1.0, -1.0)
        ybar = Y.mean(axis=0)
        return Y - ybar, ybar

    def decide(S):
        return (S[:, 0] > 0).astype(int) if n_classes == 2 else S.argmax(axis=1)

    def balanced_acc(yt, yp):
        return float(np.mean([(yp[yt == c] == c).mean() for c in np.unique(yt)]))

    def ridge_predict(parts, ytr, a):
        A, ev, U = parts
        Yc, ybar = centred_targets(ytr)
        return decide(A @ ((U.T @ Yc) / (ev + a)[:, None]) + ybar)

    def choose_alpha(inner, ytr):
        """Inner-CV squared error on the ±1 targets (ridge's own loss; smoother than accuracy)."""
        if alphas is None:
            return alpha
        sse = np.zeros(len(alphas))
        for itr, ite, (A, ev, U) in inner:
            Yc, ybar = centred_targets(ytr[itr])
            Yte, _ = centred_targets(ytr[ite])
            Yte = Yte + centred_targets(ytr[ite])[1]  # uncentred ±1 targets of the inner test set
            UtY = U.T @ Yc
            for k, a in enumerate(alphas):
                sse[k] += float(((A @ (UtY / (ev + a)[:, None]) + ybar - Yte) ** 2).sum())
        return float(alphas[len(alphas) - 1 - np.argmin(sse[::-1])])  # ties -> stronger penalty

    def cv_scores(labels):
        """Per-fold (accuracy, balanced accuracy, chosen alpha)."""
        scores = []
        for tr, te, parts, inner in folds:
            a = choose_alpha(inner, labels[tr])
            yhat = ridge_predict(parts, labels[tr], a)
            scores.append((float((yhat == labels[te]).mean()), balanced_acc(labels[te], yhat), a))
        return np.array(scores)

    scores = cv_scores(y_enc)
    fold_accs, fold_baccs, fold_alphas = scores[:, 0].tolist(), scores[:, 1].tolist(), scores[:, 2].tolist()
    for fold, ((_, te, _, _), acc, bacc, a) in enumerate(zip(folds, fold_accs, fold_baccs, fold_alphas), start=1):
        print(f"[fold {fold}/5] acc={acc:.4f} bacc={bacc:.4f} alpha={a:g} (n_test={len(te)})")

    mean_acc = float(np.mean(fold_accs))
    mean_bacc = float(np.mean(fold_baccs))
    print(f"[DONE] mean CV acc={mean_acc:.4f}  bacc={mean_bacc:.4f}  folds(bacc)={['%.3f' % a for a in fold_baccs]}")
    res = {"mean_acc": mean_acc, "fold_accs": fold_accs,
           "mean_bacc": mean_bacc, "fold_baccs": fold_baccs, "fold_alphas": fold_alphas,
           "chance": 1 / len(vals), "label_encoder": le}

    if n_perm > 0:
        # null on balanced accuracy, same folds (alpha re-tuned per permutation when alphas is given)
        rng = np.random.default_rng(random_state)
        null = np.array([cv_scores(rng.permutation(y_enc))[:, 1].mean() for _ in range(n_perm)])
        p = (1 + np.sum(null >= mean_bacc)) / (n_perm + 1)
        print(f"[PERM] bacc null mean={null.mean():.4f}  95th pct={np.percentile(null, 95):.4f}  p={p:.4g} (n_perm={n_perm})")
        res.update({"null_baccs": null, "p_value": float(p)})

    return res


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    sub = "sub-01"
    sessions = ["ses-01", "ses-02", "ses-03", "ses-04"]

    glm_variant = "wfdelay"   # or "wofdelay" — selects both the GLM outputs and the matching events folder

    glm_base_dir = f"/mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/glm_runs/lsa_{glm_variant}/64kDense"
    behav_base   = "/mnt/tempdata/lucas/fmri/recordings/TR/behav"
    events_subdir = f"events_{glm_variant}"
    block_logs_base = f"/mnt/tempdata/lucas/fmri/recordings/TR/behav/{sub}"  # contains ses-01/, ses-02/, ...

    dlabel = "/home/lucas/projects/MULTFSCTRL/prep/fmriprep/Glasser_LR_Dense64k.dlabel.nii"

    roi_names = ["L_10r_ROI", "R_10r_ROI", "L_10d_ROI", "R_10d_ROI", 
                 "L_10v_ROI", "R_10v_ROI", "L_a10p_ROI", "R_a10p_ROI",
                 "L_p10p_ROI", "R_p10p_ROI", "L_10pp_ROI", "R_10pp_ROI"]

    X, y, meta_avg, meta_per_ses = collect_repeat_averaged_across_sessions(
        glm_base_dir=glm_base_dir,
        behav_base=behav_base,
        events_subdir=events_subdir,
        block_logs_base=block_logs_base,
        sub=sub,
        sessions=sessions,
        dlabel_path=dlabel,
        roi_names=roi_names,
        feature_to_decode="ctg",       # or "loc" / "obj"
        phase_to_decode="Encoding",    # or "Delay"
        only_tasks_regex=None,         # e.g. "1back"
        verbose=True,
    )

    print("Repeat-averaged X:", X.shape, "(n_events_common, n_vertices)")
    print("Repeat-averaged y:", y.shape)

    res = ridge_decode_5fold(X, y, groups=trial_groups(meta_avg), alpha=1.0, n_perm=1000, random_state=0)
    print("Mean decoding accuracy:", res["mean_acc"])
    print("Mean balanced accuracy:", res["mean_bacc"])
    print("Permutation p-value:", res["p_value"])