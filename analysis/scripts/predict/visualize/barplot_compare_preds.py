import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_max_r_by_roi(json_path, anatomical_order):
    with open(json_path, "r") as f:
        data = json.load(f)

    roi_to_maxr = {}
    for roi in anatomical_order:
        if roi not in data:
            continue

        # Expect data[roi] is a dict of models/settings -> metrics dict containing 'r'
        r_values = []
        for _, metrics in data[roi].items():
            if isinstance(metrics, dict) and "r" in metrics and metrics["r"] is not None:
                r_values.append(metrics["r"])

        if len(r_values) > 0:
            roi_to_maxr[roi] = max(r_values)

    return roi_to_maxr


def plot_roi_comparison(
    json_path_a,
    json_path_b,
    save_dir,
    label_a="Result A",
    label_b="Result B",
    filename="roi_rostral_caudal_comparison_lineplot.png",
):
    # 1) Define Explicit Anatomical Order (Rostral -> Caudal)
    anatomical_order = [
        # --- ROSTRAL (FRONT) ---
        # Pole / Area 10
        "10pp", "a10p", "p10p", "10r", "10d", "10v",

        # Orbital / Ventral (Moving back)
        "11l", "a47r", "47s", "47m", "47l", "p47r", "13l",

        # Dorsolateral Anterior (Mid-Front)
        "9a", "46", "a9-46v",

        # Dorsolateral Mid/Posterior
        "9-46d", "p9-46v", "9m", "9p",

        # Inferior Frontal Gyrus (Anterior -> Posterior)
        "IFSa", "45", "IFSp", "44",

        # Posterior / Superior / Junction (Area 8 & 6)
        "8Av", "8Ad", "8C", "8BL", "8BM",
        "IFJa", "IFJp",
        "SFL", "s6-8", "i6-8",
        # --- CAUDAL (BACK) ---
    ]

    # 2) Load max r per ROI for each result
    a = load_max_r_by_roi(json_path_a, anatomical_order)
    b = load_max_r_by_roi(json_path_b, anatomical_order)

    # Use intersection so both lines are comparable ROI-by-ROI
    rois = [roi for roi in anatomical_order if (roi in a and roi in b)]

    # Warn about missing ROIs
    missing_a = [roi for roi in anatomical_order if roi not in a]
    missing_b = [roi for roi in anatomical_order if roi not in b]
    if missing_a:
        print(f"[{label_a}] Missing/empty ROIs (skipped): {missing_a}")
    if missing_b:
        print(f"[{label_b}] Missing/empty ROIs (skipped): {missing_b}")
    if not rois:
        raise ValueError("No overlapping ROIs between the two JSONs in the specified anatomical order.")

    max_rs_a = [a[roi] for roi in rois]
    max_rs_b = [b[roi] for roi in rois]

    # 3) Plot
    plt.figure(figsize=(14, 7))
    x = np.arange(len(rois))

    plt.plot(x, max_rs_a, marker="o", linewidth=2.5, label=label_a, color="#2c7bb6")  # blue
    plt.plot(x, max_rs_b, marker="o", linewidth=2.5, label=label_b, color="#d7191c")  # red

    plt.xlabel("Prefrontal ROIs (Ordered Rostral → Caudal)", fontsize=12, fontweight="bold")
    plt.ylabel("Max Pearson's r", fontsize=12, fontweight="bold")
    plt.title("Decoding Performance Comparison: Rostral to Caudal Gradient", fontsize=14)

    plt.xticks(x, rois, rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    results_a = "/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay/results.json"
    results_b = "/mnt/store1/lucas/checkpoints/tf_reg/untrained/tf_medium_untrained/results/frame-only_enc+delay_delay_lsa_wfdelay/results.json"
    save_dir = "/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay"

    plot_roi_comparison(
        results_a,
        results_b,
        save_dir,
        label_a="Trained",
        label_b="Untrained",
    )