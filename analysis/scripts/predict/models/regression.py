import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

def pca_ridge_decode(activations, betas, avg_vertices, standardize_acts, standardize_betas, ridge_alpha, n_pcs=64):
    """
    Perform decoding using PCA-Ridge Regression

    Parameters:
        activations (tensor): Activations tensor
        betas (tensor): Betas tensor
        avg_vertices (bool): Whether to average betas across vertices
        standardize_acts (bool): Whether to standardize activations
        standardize_betas (bool): Whether to standardize betas
        ridge_alpha (float): Ridge regression alpha parameter
        n_pcs (int or None): Number of principal components to use (if None, determine automatically)
    """

    if avg_vertices:
        y = np.mean(betas, axis=1) # mean of vertices
    else:
        y = betas

    if standardize_betas:
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    per_layer_results = {}
    per_layer_y_preds = []

    scalars = []
    regressors = []
    pcas = []
    for layer in tqdm(range(activations.shape[0])):

        if n_pcs is None:
            n_components = 0
        elif n_pcs == 0:
            n_components = determine_npcas(activations[layer])
        elif n_pcs > 0:
            n_components = n_pcs

        X = activations[layer]
        if standardize_acts:
            acts_scaler = StandardScaler()
            X = acts_scaler.fit_transform(X)

        kf = KFold(n_splits=10, shuffle=True, random_state=5112000)

        y_pred = np.zeros(y.shape)


        regressor = Ridge(alpha=ridge_alpha)

        if n_components > 0:
            pca = PCA(n_components=n_components)
            X = pca.fit_transform(X)

        # Cross-validation loop
        for xidx, yidx in kf.split(X):
            X_train, X_test = X[xidx], X[yidx]
            y_train, _ = y[xidx], y[yidx]

            regressor.fit(X_train, y_train)

            y_pred[yidx] = regressor.predict(X_test)

        y_true = np.expand_dims(y, axis=1)
        y_pred = np.expand_dims(y_pred, axis=1) 
        mse = np.sum((y_pred - y_true)**2)
        r = pearsonr(y_pred, y_true)[0][0] 
        results = {'mse': float(mse), 'r': float(r), 'nsamples': int(y_true.shape[0]), 'kf_nsplits': int(kf.get_n_splits()), 'npcas': int(n_components)} 
        per_layer_results[layer] = results
        print('r:', r)

        per_layer_y_preds.append(y_pred.tolist())

        # Full training on all data for final checkpoint
        regressor.fit(X, y)
        regressors.append(regressor)
        pcas.append(pca)
        scalars.append(acts_scaler)

    return per_layer_results, per_layer_y_preds, regressors, pcas, scalars, y
        
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
    