import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
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
        standardize_acts (bool): Whether to standardize activations
        standardize_betas (bool): Whether to standardize betas
        alpha (float): Ridge regression alpha parameter
    """

    per_layer_results = {}
    per_layer_y_preds = []
    regressors = []
    for layer in tqdm(range(activations.shape[0])):

        # Determine the number of principal components to use
        if n_pcs is None:
            n_components = 0
        elif n_pcs == 0:
            n_components = determine_npcas(activations[layer])
        elif n_pcs > 0:
            n_components = n_pcs

        # Flatten the activations and class vectors
        X = activations[layer]

        if avg_vertices:
            y = np.mean(betas, axis=1) # mean of vertices
        else:
            y = betas

        # Standardize activations and betas if required
        if standardize_acts:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        if standardize_betas:
            y = (y - y.mean()) / y.std()

        # Initialize K-Fold Cross-Validation
        kf = KFold(n_splits=10, shuffle=True, random_state=5112000)

        # Arrays to store predictions
        y_pred = np.zeros(y.shape)

        # Train a Ridge Regression model using the principal components
        regressor = Ridge(alpha=ridge_alpha)

        if n_components > 0:
            pca = PCA(n_components=n_components)
            X = pca.fit_transform(X)

        # Cross-validation loop
        for xidx, yidx in kf.split(X):
            # Split into training and testing sets
            X_train, X_test = X[xidx], X[yidx]
            y_train, _ = y[xidx], y[yidx]

            regressor.fit(X_train, y_train)

            # Predict on the test set
            y_pred[yidx] = regressor.predict(X_test)

        y = y
        y = np.expand_dims(y, axis=1) # with mean
        y_pred = np.expand_dims(y_pred, axis=1) # with mean
        mse = np.sum((y_pred - y)**2)
        r = pearsonr(y_pred, y)[0][0] # with mean
        results = {'mse': float(mse), 'r': float(r), 'nsamples': int(y.shape[0]), 'kf_nsplits': int(kf.get_n_splits()), 'npcas': int(n_components)} 
        per_layer_results[layer] = results
        print('r:', r)

        per_layer_y_preds.append(y_pred.tolist())

        # Full training on all data for final checkpoint
        regressor.fit(X, y)
        regressors.append(regressor)

    return per_layer_results, per_layer_y_preds, regressors
        
def pca_ridge_infer(activations, regressor, avg_vertices, standardize_acts, ridge_alpha, n_pcs=64)
    """
    Perform inference using trained PCA-Ridge Regression models

    Parameters:
        activations (tensor): Activations tensor
        regressor (RidgeRegression): Trained Ridge regression model
    """

    y_preds = []
    
    # Determine the number of principal components to use
    if n_pcs is None:
        n_components = 0
    elif n_pcs == 0:
        n_components = determine_npcas(activations[layer])
    elif n_pcs > 0:
        n_components = n_pcs

    X = activations

    # Standardize activations if required
    if standardize_acts:
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    if n_components > 0:
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
    
    y_pred = regressor.predict(X)

    y_pred = np.expand_dims(y_pred, axis=1) # with mean
    y_preds.append(y_pred.tolist())

    # Stack predictions from all layers
    y_preds = np.concatenate(y_preds, axis=1)

    return y_preds
    