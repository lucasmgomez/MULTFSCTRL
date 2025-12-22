import os
import torch
import numpy as np
from joblib import load
import json

hooked_trials_path = '/mnt/store1/lucas/checkpoints/fixed/tf_medium_full_3000eps_ubt_semifixed/activations/1back_ctg_hierarchy_stacked_tracker.pth'
trial_times_path = '...'
regressors_dir = '...'
regions = ['46', '9-46v', 'a9-46v']

### Load hooked model activations
import pdb; pdb.set_trace()
tracker = torch.load(hooked_trials_path, map_location=torch.device('cpu'))
activations = tracker['layer_activations']
n_layers = int(activations.shape[0])
n_tokens = int(activations.shape[2])
activations = activations[:, :, n_tokens-10:]
trial_times = json.load(open(trial_times_path, 'r'))

# Select and reshape activations for encoding and delay periods



### keep track of regions and highest predicted activity
predictions = {}
for region in regions:
    regressors_region_dir = os.path.join(regressors_dir, f'{region}')
    predictions[region] = []

    for trial in range(activations.shape[1]):
        activation = activations[:, trial, :, :]
        times = trial_times[str(trial)]

        activation = activation[:, times, :]

        encoding_idxs = [i for i in range(0, activation.size(1), 2)]
        delay_idxs = [i for i in range(1, activation.size(1), 2)]
        enc_acts = activation[:, encoding_idxs, :]
        delay_acts = activation[:, delay_idxs, :]
        sr_activation = torch.cat((enc_acts, delay_acts), dim=-1)

        layer_preds = []
        for layer in range(n_layers):
            regressor_path = os.path.join(regressors_region_dir, f'layer_{layer}_regressor.joblib')
            regressor = load(regressor_path)

            pred = regressor.predict(sr_activation[layer].numpy().T)
            layer_preds.append(pred.mean())
        
        predictions[region].append(max(layer_preds))



            

        


