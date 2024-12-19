import numpy as np

def normalizefea(X):
    """
    L2 normalize
    """
    feanorm = np.maximum(1e-14,np.sum(X**2,axis=1))
    X_out = X/(feanorm[:,None]**0.5)
    return X_out