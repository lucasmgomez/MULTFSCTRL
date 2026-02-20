import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from tqdm import tqdm

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
        
def pca_ridge_infer(activations, regressor, pca, scalar, standardize_acts):
    """
    Perform inference using trained PCA-Ridge Regression models

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
    
def pls_decode(activations, betas, avg_vertices, standardize_acts, standardize_betas, n_components_list=[10, 20, 30, 40, 50], n_pcs=None):
    """
    Perform Grid Search using K-Fold CV to select the best n_components for PLS.
    
    Parameters:
        n_components_list (list): List of component counts to test (e.g. [10, 20, 30]). 
                                  Note: PLS usually needs fewer components than PCA.
        n_pcs: Ignored (kept for compatibility with old function calls).
    """

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