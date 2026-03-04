import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from tqdm import tqdm


def _prepare_runwise_targets(betas_by_run, avg_vertices, standardize_betas):
    y_by_run = {}
    for run_name, curr_betas in betas_by_run.items():
        if avg_vertices:
            y_curr = np.mean(curr_betas, axis=1)
        else:
            y_curr = curr_betas
        y_by_run[run_name] = y_curr

    if standardize_betas:
        y_scaler = StandardScaler()
        y_all = np.concatenate([y_by_run[k] for k in y_by_run.keys()], axis=0)
        y_all_scaled = y_scaler.fit_transform(y_all.reshape(-1, 1)).flatten()

        offset = 0
        for run_name in y_by_run.keys():
            n = y_by_run[run_name].shape[0]
            y_by_run[run_name] = y_all_scaled[offset:offset + n]
            offset += n
    else:
        y_scaler = None

    y_concat = np.concatenate([y_by_run[k] for k in y_by_run.keys()], axis=0)
    return y_by_run, y_concat, y_scaler

def determine_npcas(X):
    """
    Determine the number of principal components to use
    """

    # Fit PCA
    pca = PCA()
    pca.fit(X)  # X is your input data

    # Compute cumulative explained variance
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # Choose the number of components explaining 95% variance
    n_components = np.argmax(explained_variance_ratio >= 0.95) + 1

    return n_components

def pca_ridge_decode(activations, betas, avg_vertices, standardize_acts, standardize_betas, ridge_alphas, n_pcs=64):
    """
    Perform Grid Search using K-Fold CV to select the best alpha based on held-out performance.
    """

    if isinstance(activations, dict):
        return pca_ridge_decode_leave_one_run_out(
            activations=activations,
            betas=betas,
            avg_vertices=avg_vertices,
            standardize_acts=standardize_acts,
            standardize_betas=standardize_betas,
            ridge_alphas=ridge_alphas,
            n_pcs=n_pcs,
        )

    # --- Preprocessing ---
    if avg_vertices:
        y = np.mean(betas, axis=1)
    else:
        y = betas

    if standardize_betas:
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        y_scaler = None

    per_layer_results = {}
    per_layer_y_preds = []
    
    final_regressors = []
    pcas = []
    scalars = []

    # Iterate through layers
    for layer in tqdm(range(activations.shape[0])):

        # --- PCA and Scaling ---
        if n_pcs is None: n_components = 0
        elif n_pcs == 0: n_components = determine_npcas(activations[layer])
        else: n_components = n_pcs

        X = activations[layer]
        
        if standardize_acts:
            acts_scaler = StandardScaler()
            X = acts_scaler.fit_transform(X)
        else:
            acts_scaler = None
            
        if n_components > 0:
            pca = PCA(n_components=n_components)
            X = pca.fit_transform(X)
        else:
            pca = None

        # --- Alpha Grid Search and Cross-validation loop ---
        alpha_preds = np.zeros((len(ridge_alphas), y.shape[0]))
        
        kf = KFold(n_splits=10, shuffle=True, random_state=5112000)

        for xidx, yidx in kf.split(X):
            X_train, X_test = X[xidx], X[yidx]
            y_train, _ = y[xidx], y[yidx]

            for i, alpha in enumerate(ridge_alphas):
                reg = Ridge(alpha=alpha) 
                reg.fit(X_train, y_train)
                
                # Predict on held-out chunk
                alpha_preds[i, yidx] = reg.predict(X_test)

        # --- Select the Best Alpha ---
        best_r = -1.0
        best_alpha = None
        best_mse = None
        best_alpha_idx = 0

        # Check which alpha row performed best across the full dataset
        for i, alpha in enumerate(ridge_alphas):
            current_r = pearsonr(alpha_preds[i].flatten(), y.flatten())[0]
            
            if current_r > best_r:
                best_r = current_r
                best_alpha = alpha
                best_alpha_idx = i
                best_mse = np.sum((alpha_preds[i] - y)**2)

        print(f"Layer {layer} - Best Alpha: {best_alpha:.2f} (r={best_r:.4f})")

        # --- Final Full Data fit ---
        final_regressor = Ridge(alpha=best_alpha)
        final_regressor.fit(X, y)
        
        # Store results for the WINNING alpha
        results = {
            'mse': float(best_mse), 
            'r': float(best_r), 
            'best_alpha': float(best_alpha),
            'nsamples': int(y.shape[0]), 
            'kf_nsplits': int(kf.get_n_splits()), 
            'npcas': int(n_components)
        }
        per_layer_results[layer] = results
        
        # Store the predictions of the winner
        per_layer_y_preds.append(alpha_preds[best_alpha_idx].tolist())
        
        final_regressors.append(final_regressor)
        pcas.append(pca)
        scalars.append(acts_scaler)

    return per_layer_results, per_layer_y_preds, final_regressors, pcas, scalars, y, y_scaler
        
def pls_decode(activations, betas, avg_vertices, standardize_acts, standardize_betas, n_components_list=[10, 20, 30, 40, 50], n_pcs=None):
    """
    Perform Grid Search using K-Fold CV to select the best n_components for PLS.
    
    Parameters:
        n_components_list (list): List of component counts to test (e.g. [10, 20, 30]). 
                                  Note: PLS usually needs fewer components than PCA.
        n_pcs: Ignored (kept for compatibility with old function calls).
    """

    if isinstance(activations, dict):
        return pls_decode_leave_one_run_out(
            activations=activations,
            betas=betas,
            avg_vertices=avg_vertices,
            standardize_acts=standardize_acts,
            standardize_betas=standardize_betas,
            n_components_list=n_components_list,
            n_pcs=n_pcs,
        )

    # --- Preprocessing (Identical to original) ---
    if avg_vertices:
        y = np.mean(betas, axis=1)
    else:
        y = betas

    if standardize_betas:
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        y_scaler = None

    per_layer_results = {}
    per_layer_y_preds = []
    
    final_regressors = []
    scalars = []

    # Iterate through layers
    for layer in tqdm(range(activations.shape[0]), desc="PLS Decoding"):

        X = activations[layer]
        
        # PLS benefits greatly from standardized X
        if standardize_acts:
            acts_scaler = StandardScaler()
            X = acts_scaler.fit_transform(X)
        else:
            acts_scaler = None
            
        # --- Grid Search for n_components ---
        comp_preds = np.zeros((len(n_components_list), y.shape[0]))
        
        kf = KFold(n_splits=10, shuffle=True, random_state=5112000)

        for xidx, yidx in kf.split(X):
            X_train, X_test = X[xidx], X[yidx]
            y_train, _ = y[xidx], y[yidx]

            # INNER LOOP: Test every n_component count
            for i, n_comps in enumerate(n_components_list):
                # Safety check: Cannot have more components than samples/features
                valid_n = min(n_comps, X_train.shape[0], X_train.shape[1])
                
                reg = PLSRegression(n_components=valid_n, scale=False) 
                reg.fit(X_train, y_train)
                
                # Predict (PLS returns shape [n_samples, 1], flatten it)
                pred = reg.predict(X_test)
                if pred.ndim > 1: pred = pred.flatten()
                
                comp_preds[i, yidx] = pred

        # --- Select the Best Component Count ---
        best_r = -1.0
        best_n = n_components_list[0]
        best_mse = None
        best_idx = 0

        for i, n_comps in enumerate(n_components_list):
            current_r = pearsonr(comp_preds[i].flatten(), y.flatten())[0]
            
            if current_r > best_r:
                best_r = current_r
                best_n = n_comps
                best_idx = i
                best_mse = np.sum((comp_preds[i] - y)**2)

        print(f"Layer {layer} - Best n_comps: {best_n} (r={best_r:.4f})")

        # --- Final Full Data Fit ---
        valid_best_n = min(best_n, X.shape[0], X.shape[1])
        final_regressor = PLSRegression(n_components=valid_best_n, scale=False)
        final_regressor.fit(X, y)
        
        # Store results
        results = {
            'mse': float(best_mse), 
            'r': float(best_r), 
            'best_n_components': int(best_n), # Renamed from best_alpha
            'nsamples': int(y.shape[0]), 
            'kf_nsplits': int(kf.get_n_splits())
        }
        per_layer_results[layer] = results
        
        # Store predictions of winner
        per_layer_y_preds.append(comp_preds[best_idx].tolist())
        
        final_regressors.append(final_regressor)
        scalars.append(acts_scaler)

    return per_layer_results, per_layer_y_preds, final_regressors, scalars, y, y_scaler


def pca_ridge_decode_leave_one_run_out(activations, betas, avg_vertices, standardize_acts, standardize_betas, ridge_alphas, n_pcs=64):
    run_names = sorted(activations.keys())
    if len(run_names) < 2:
        raise ValueError("leave_one_run_out requires at least 2 runs/tasks.")

    y_by_run, y_concat, y_scaler = _prepare_runwise_targets(betas, avg_vertices, standardize_betas)

    per_layer_results = {}
    per_layer_y_preds = []
    final_regressors = []
    pcas = []
    scalars = []

    nlayers = activations[run_names[0]].shape[0]

    for layer in tqdm(range(nlayers), desc="Ridge LORO"):
        if n_pcs is None:
            base_components = 0
        elif n_pcs == 0:
            layer_all = np.concatenate([activations[r][layer] for r in run_names], axis=0)
            base_components = determine_npcas(layer_all)
        else:
            base_components = n_pcs

        alpha_fold_preds = {alpha: [] for alpha in ridge_alphas}
        alpha_fold_targets = {alpha: [] for alpha in ridge_alphas}

        for test_run in run_names:
            train_runs = [r for r in run_names if r != test_run]

            X_train_raw = np.concatenate([activations[r][layer] for r in train_runs], axis=0)
            y_train = np.concatenate([y_by_run[r] for r in train_runs], axis=0)
            X_test_raw = activations[test_run][layer]
            y_test = y_by_run[test_run]

            if standardize_acts:
                fold_scaler = StandardScaler()
                X_train = fold_scaler.fit_transform(X_train_raw)
                X_test = fold_scaler.transform(X_test_raw)
            else:
                X_train = X_train_raw
                X_test = X_test_raw

            n_components = base_components
            if n_components > 0:
                n_components = min(n_components, X_train.shape[0], X_train.shape[1])
                fold_pca = PCA(n_components=n_components)
                X_train = fold_pca.fit_transform(X_train)
                X_test = fold_pca.transform(X_test)

            for alpha in ridge_alphas:
                reg = Ridge(alpha=alpha)
                reg.fit(X_train, y_train)
                pred = reg.predict(X_test)
                alpha_fold_preds[alpha].append(pred)
                alpha_fold_targets[alpha].append(y_test)

        best_r = -1.0
        best_alpha = None
        best_mse = None
        best_preds = None

        for alpha in ridge_alphas:
            curr_pred = np.concatenate(alpha_fold_preds[alpha], axis=0)
            curr_true = np.concatenate(alpha_fold_targets[alpha], axis=0)
            current_r = pearsonr(curr_pred.flatten(), curr_true.flatten())[0]

            if current_r > best_r:
                best_r = current_r
                best_alpha = alpha
                best_mse = np.sum((curr_pred - curr_true) ** 2)
                best_preds = curr_pred

        print(f"Layer {layer} - Best Alpha: {best_alpha:.2f} (r={best_r:.4f})")

        X_full = np.concatenate([activations[r][layer] for r in run_names], axis=0)
        y_full = np.concatenate([y_by_run[r] for r in run_names], axis=0)

        if standardize_acts:
            acts_scaler = StandardScaler()
            X_full = acts_scaler.fit_transform(X_full)
        else:
            acts_scaler = None

        n_components = base_components
        if n_components > 0:
            n_components = min(n_components, X_full.shape[0], X_full.shape[1])
            pca = PCA(n_components=n_components)
            X_full = pca.fit_transform(X_full)
        else:
            pca = None

        final_regressor = Ridge(alpha=best_alpha)
        final_regressor.fit(X_full, y_full)

        results = {
            'mse': float(best_mse),
            'r': float(best_r),
            'best_alpha': float(best_alpha),
            'nsamples': int(y_full.shape[0]),
            'cv_scheme': 'leave_one_run_out',
            'n_heldout_runs': int(len(run_names)),
            'npcas': int(n_components)
        }
        per_layer_results[layer] = results
        per_layer_y_preds.append(best_preds.tolist())
        final_regressors.append(final_regressor)
        pcas.append(pca)
        scalars.append(acts_scaler)

    return per_layer_results, per_layer_y_preds, final_regressors, pcas, scalars, y_concat, y_scaler


def pls_decode_leave_one_run_out(activations, betas, avg_vertices, standardize_acts, standardize_betas, n_components_list=[10, 20, 30, 40, 50], n_pcs=None):
    run_names = sorted(activations.keys())
    if len(run_names) < 2:
        raise ValueError("leave_one_run_out requires at least 2 runs/tasks.")

    y_by_run, y_concat, y_scaler = _prepare_runwise_targets(betas, avg_vertices, standardize_betas)

    per_layer_results = {}
    per_layer_y_preds = []
    final_regressors = []
    scalars = []

    nlayers = activations[run_names[0]].shape[0]

    for layer in tqdm(range(nlayers), desc="PLS LORO"):
        comp_fold_preds = {int(n): [] for n in n_components_list}
        comp_fold_targets = {int(n): [] for n in n_components_list}

        for test_run in run_names:
            train_runs = [r for r in run_names if r != test_run]

            X_train_raw = np.concatenate([activations[r][layer] for r in train_runs], axis=0)
            y_train = np.concatenate([y_by_run[r] for r in train_runs], axis=0)
            X_test_raw = activations[test_run][layer]
            y_test = y_by_run[test_run]

            if standardize_acts:
                fold_scaler = StandardScaler()
                X_train = fold_scaler.fit_transform(X_train_raw)
                X_test = fold_scaler.transform(X_test_raw)
            else:
                X_train = X_train_raw
                X_test = X_test_raw

            for n_comps in n_components_list:
                n_comps_int = int(n_comps)
                valid_n = min(n_comps_int, X_train.shape[0], X_train.shape[1])
                if valid_n < 1:
                    continue

                reg = PLSRegression(n_components=valid_n, scale=False)
                reg.fit(X_train, y_train)

                pred = reg.predict(X_test)
                if pred.ndim > 1:
                    pred = pred.flatten()

                comp_fold_preds[n_comps_int].append(pred)
                comp_fold_targets[n_comps_int].append(y_test)

        best_r = -1.0
        best_n = int(n_components_list[0])
        best_mse = None
        best_preds = None

        for n_comps in n_components_list:
            n_comps_int = int(n_comps)
            if len(comp_fold_preds[n_comps_int]) == 0:
                continue

            curr_pred = np.concatenate(comp_fold_preds[n_comps_int], axis=0)
            curr_true = np.concatenate(comp_fold_targets[n_comps_int], axis=0)
            current_r = pearsonr(curr_pred.flatten(), curr_true.flatten())[0]

            if current_r > best_r:
                best_r = current_r
                best_n = n_comps_int
                best_mse = np.sum((curr_pred - curr_true) ** 2)
                best_preds = curr_pred

        print(f"Layer {layer} - Best n_comps: {best_n} (r={best_r:.4f})")

        X_full = np.concatenate([activations[r][layer] for r in run_names], axis=0)
        y_full = np.concatenate([y_by_run[r] for r in run_names], axis=0)

        if standardize_acts:
            acts_scaler = StandardScaler()
            X_full = acts_scaler.fit_transform(X_full)
        else:
            acts_scaler = None

        valid_best_n = min(best_n, X_full.shape[0], X_full.shape[1])
        final_regressor = PLSRegression(n_components=valid_best_n, scale=False)
        final_regressor.fit(X_full, y_full)

        results = {
            'mse': float(best_mse),
            'r': float(best_r),
            'best_n_components': int(best_n),
            'nsamples': int(y_full.shape[0]),
            'cv_scheme': 'leave_one_run_out',
            'n_heldout_runs': int(len(run_names))
        }
        per_layer_results[layer] = results
        per_layer_y_preds.append(best_preds.tolist())
        final_regressors.append(final_regressor)
        scalars.append(acts_scaler)

    return per_layer_results, per_layer_y_preds, final_regressors, scalars, y_concat, y_scaler

def infer(activations, regressor, pca, scalar, standardize_acts):
    """
    Perform inference using the trained regression model

    Parameters:
        activations (tensor): Activations tensor
        regressor (RidgeRegression): Trained Ridge regression model
        pca (PCA): Trained PCA model
        scalar (StandardScaler): Trained StandardScaler for activations
        standardize_acts (bool): Whether to standardize activations

    Returns:
        y_preds (list): Predicted betas
    """

    X = activations

    if standardize_acts:
        X = scalar.transform(X)

    if pca is not None:
        X = pca.transform(X)
    
    y_pred = regressor.predict(X)
    y_pred = np.expand_dims(y_pred, axis=1) 

    return y_pred
    