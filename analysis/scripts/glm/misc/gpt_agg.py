import os
import argparse
import ast
import numpy as np
import pandas as pd
import nibabel as nib
from joblib import load

def load_atlas_data(dlabel_path):
    dl = nib.load(dlabel_path)
    data = dl.get_fdata().squeeze().astype(int)
    label_dict = dl.header.get_axis(0).label[0]
    return label_dict, data

def create_roi_mask(dlabel_info, roi_name, lateralize='LR'):
    label_dict, data = dlabel_info

    if lateralize == 'LR':
        roi_list = [f'L_{roi_name}_ROI', f'R_{roi_name}_ROI']
    elif lateralize == 'L':
        roi_list = [f'L_{roi_name}_ROI']
    elif lateralize == 'R':
        roi_list = [f'R_{roi_name}_ROI']
    else:
        roi_list = [roi_name]

    keys = [k for k, (name, _) in label_dict.items() if name in roi_list]
    return np.isin(data, keys)

def main():
    ap = argparse.ArgumentParser(description="Aggregate ROI betas EXACTLY matching betas.npz via saved s_betas.npy")
    ap.add_argument("--data_cache_dir", required=True,
                    help="Same path passed to predict_arg.py --data_cache_dir (contains s_betas.npy)")
    ap.add_argument("--decode_results_dir", required=True,
                    help="Run folder containing regressors/<ROI>/betas_scalar.joblib and betas.npz")
    ap.add_argument("--dlabel_path", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--subj", default="sub-01")
    ap.add_argument("--lateralize", default="LR", choices=["LR", "L", "R"])
    ap.add_argument("--rois", required=True, help="e.g. \"['46','10pp']\"")
    ap.add_argument("--verify_against_npz", action="store_true")
    args = ap.parse_args()

    # Load the exact s_betas used to generate betas.npz
    s_betas_path = os.path.join(args.data_cache_dir, "s_betas.npy")
    if not os.path.exists(s_betas_path):
        raise FileNotFoundError(f"Missing s_betas.npy at: {s_betas_path}")

    s_betas = np.load(s_betas_path)
    print("Loaded s_betas:", s_betas.shape)  # should be (474, 64984) for your run

    atlas_info = load_atlas_data(args.dlabel_path)

    roi_list = ast.literal_eval(args.rois)
    rows = []

    for roi in roi_list:
        mask = create_roi_mask(atlas_info, roi, args.lateralize)
        if mask.sum() == 0:
            print(f"[skip] ROI {roi}: 0 vertices in mask")
            continue

        scaler_path = os.path.join(args.decode_results_dir, "regressors", roi, "betas_scalar.joblib")
        if not os.path.exists(scaler_path):
            print(f"[skip] ROI {roi}: missing {scaler_path}")
            continue

        scaler = load(scaler_path)

        # This matches pca_ridge_decode(avg_vertices=True)
        roi_raw = np.mean(s_betas[:, mask], axis=1)              # (n_events,)
        roi_z = scaler.transform(roi_raw.reshape(-1, 1)).ravel()  # (n_events,)

        print(f"[write] {roi}: n={roi_z.size} mean={roi_z.mean():.6g} std={roi_z.std():.6g}")

        rows.append({
            "subject": args.subj,
            "roi": roi,
            "n_events": int(roi_z.size),
            "betas": roi_z.tolist()
        })

    if not rows:
        print("No rows written (no valid ROIs/scalers).")
        return

    os.makedirs(args.save_path, exist_ok=True)
    out_csv = os.path.join(args.save_path, f"{args.subj}_roi_pretrained_zscored_betas_ALIGNED.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # Optional verification: this should match betas.npz exactly
    if args.verify_against_npz:
        npz_path = os.path.join(args.decode_results_dir, "betas.npz")
        if not os.path.exists(npz_path):
            print("[verify] betas.npz not found, skipping")
            return

        npz = np.load(npz_path)
        ok = True
        for r in rows:
            roi = r["roi"]
            if roi not in npz.files:
                print(f"[verify] ROI {roi} missing in betas.npz")
                ok = False
                continue

            a = np.array(r["betas"], dtype=np.float64)
            b = np.array(npz[roi], dtype=np.float64).ravel()

            if a.shape != b.shape:
                print(f"[verify] {roi}: shape mismatch csv={a.shape} npz={b.shape}")
                ok = False
                continue

            if not np.allclose(a, b, atol=0, rtol=0):
                print(f"[verify] {roi}: mismatch max_abs={np.max(np.abs(a-b))}")
                ok = False
            else:
                print(f"[verify] {roi}: OK (exact allclose)")

        print("[verify] PASS" if ok else "[verify] FAIL")

if __name__ == "__main__":
    main()