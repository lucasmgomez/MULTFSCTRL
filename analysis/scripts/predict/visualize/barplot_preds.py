import json
import matplotlib.pyplot as plt
import numpy as np

def plot_roi_performance(json_path, save_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Define Explicit Anatomical Order (Rostral -> Caudal)
    anatomical_order = [
        # --- ROSTRAL (FRONT) ---
        # Pole / Area 10
        '10pp', 'a10p', 'p10p', '10r', '10d', '10v', 
        
        # Orbital / Ventral (Moving back)
        '11l', 'a47r', '47s', '47m', '47l', 'p47r', '13l',

        # Dorsolateral Anterior (Mid-Front)
        '9a', '46', 'a9-46v', 
        
        # Dorsolateral Mid/Posterior
        '9-46d', 'p9-46v', '9m', '9p',

        # Inferior Frontal Gyrus (Anterior -> Posterior)
        'IFSa', '45', 'IFSp', '44',

        # Posterior / Superior / Junction (Area 8 & 6)
        '8Av', '8Ad', '8C', '8BL', '8BM',
        'IFJa', 'IFJp',
        'SFL', 's6-8', 'i6-8'
        # --- CAUDAL (BACK) ---
    ]

    # 2. Extract Data in this specific order
    rois = []
    max_rs = []

    for roi in anatomical_order:
        if roi in data:
            # Get max r for this ROI
            r_values = [metrics['r'] for metrics in data[roi].values()]
            max_r = max(r_values)
            
            rois.append(roi)
            max_rs.append(max_r)
        else:
            print(f"Warning: ROI {roi} found in list but not in JSON results.")

    # 3. Create the Plot
    plt.figure(figsize=(14, 7))
    
    # Create bars
    # Using a gradient or distinct color for anatomical groups could be nice, 
    # but we'll stick to a clean blue for now.
    plt.bar(rois, max_rs, color='#2c7bb6', edgecolor='black', alpha=0.9)

    # Labels
    plt.xlabel('Prefrontal ROIs (Ordered Rostral $\u2192$ Caudal)', fontsize=12, fontweight='bold')
    plt.ylabel('Max Pearson\'s r', fontsize=12, fontweight='bold')
    plt.title('Decoding Performance: Rostral to Caudal Gradient', fontsize=14)

    # Visual formatting
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(np.arange(0, 0.7, 0.05), fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save and Show
    plt.savefig(save_dir + '/roi_rostral_caudal_plot.png')
    plt.show()

if __name__ == "__main__":
    results_fp = '/mnt/store1/lucas/checkpoints/tf_reg/untrained/tf_medium_untrained/results/frame-only_enc+delay_delay_lsa_wofdelay_LH/results.json'
    save_dir = '/mnt/store1/lucas/checkpoints/tf_reg/untrained/tf_medium_untrained/results/frame-only_enc+delay_delay_lsa_wofdelay_LH'
    plot_roi_performance(results_fp, save_dir)