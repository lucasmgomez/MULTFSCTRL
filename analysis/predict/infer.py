import numpy as np
import torch

def get_best_layer(decode_results, roi):
    max_r = -1.0
    best_layer = "-1"
    roi_results = decode_results[roi]
    for layer in roi_results:
        layer_r = roi_results[layer]['r']
        if layer_r > max_r:
            max_r = layer_r
            best_layer = layer
    return int(best_layer)

def get_activations(acts_fp):
    acts_dict = torch.load(acts_fp, map_location='cpu') 
    all_acts = acts_dict['layer_activations'].to(torch.float32)
    trial_idxs = acts_dict['trial_idxs']
    return all_acts.numpy(), trial_idxs

def select_acts(acts, n_images=5, phase='delay'):
    n_tokens = acts.shape[2]
    start_idx = max(0, n_tokens - n_images)
    selected_acts = acts[:, :, start_idx:] 

    encoding_idxs = [i for i in range(0, selected_acts.shape[2], 2)]
    delay_idxs = [i for i in range(1, selected_acts.shape[2], 2)]

    if phase == 'delay':
        encoding_idxs = encoding_idxs[:len(delay_idxs)]

    enc_acts = selected_acts[:, :, encoding_idxs, :]
    delay_acts = selected_acts[:, :, delay_idxs, :]
    selected_acts = np.concatenate((enc_acts, delay_acts), axis=-1)

    time_dim = selected_acts.shape[2]
    selected_acts = selected_acts.reshape(selected_acts.shape[0], selected_acts.shape[1]*selected_acts.shape[2], -1)

    return selected_acts, time_dim