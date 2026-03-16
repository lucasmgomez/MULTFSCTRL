import json

def get_true_r_from_best_base_layer(true_filepath, base_filepath):
    # Load the JSON files
    with open(true_filepath, 'r') as f:
        results_true = json.load(f)
        
    with open(base_filepath, 'r') as f:
        results_base = json.load(f)

    # Iterate through each ROI in results_true
    for roi, true_layers in results_true.items():
        # Make sure the ROI also exists in results_base to avoid errors
        if roi in results_base:
            base_layers = results_base[roi]
            
            # Find the layer key ('0', '1', '2', etc.) that has the maximum 'r' value in results_base
            best_layer = max(base_layers, key=lambda layer: base_layers[layer]['r'])
            
            # Extract the corresponding 'r' value from results_true
            true_r = true_layers[best_layer]['r']
            
            print(f"ROI: {roi:<6} | Best Base Layer: {best_layer} | True 'r': {true_r}")
        else:
            print(f"ROI: {roi:<6} | Not found in results_base.json")

# Example usage (assuming your files are named results_true.json and results_base.json)
if __name__ == "__main__":
    rt_fp = '/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay_pls_nps5to50_cv/results.json'
    rb_fp = '/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/results/frame-only_enc+delay_delay_lsa_wfdelay_pls_nps5to50/results.json'
    get_true_r_from_best_base_layer(rt_fp, rb_fp)