import numpy as np

def fit_scaler(X, method):
    """
    Calculate and return scaling parameters for the data based on the chosen method.
    Supports 'z_score' (standardization) and 'minmax' (min-max normalization).
    """
    if method not in ["z_score", "minmax"]:
        raise ValueError("Method must be either 'z_score' or 'minmax'.")
    
    # Ensure X is a numpy array
    X = np.array(X)
    
    # Compute parameters
    scaler_params = {
        "method": method,
        "mean": np.mean(X, axis=0).tolist(),
        "std": np.std(X, axis=0).tolist(),
        "min": np.min(X, axis=0).tolist(),
        "max": np.max(X, axis=0).tolist()
    }
    
    return scaler_params

def transform_data(X, scaler_params):
    """
    Apply scaling transformation to data using pre-calculated scaling parameters.
    """
    method = scaler_params["method"]
    X = np.array(X)  # Ensure X is a numpy array
    
    # Z-score (standardization) transformation
    if method == "z_score":
        mean = np.array(scaler_params["mean"])
        std = np.array(scaler_params["std"])
        return (X - mean) / (std + 1e-15)  # Added small epsilon to avoid division by zero
    
    # Min-Max scaling transformation
    elif method == "minmax":
        min_vals = np.array(scaler_params["min"])
        max_vals = np.array(scaler_params["max"])
        return (X - min_vals) / ((max_vals - min_vals) + 1e-15)  # Added small epsilon
    
    # Invalid method (shouldn't reach this due to prior check in fit_scaler)
    else:
        raise ValueError("Invalid scaling method. Must be 'z_score' or 'minmax'.")

def fit_transform_data(X, method):
    """
    Fit the scaler and transform the data.
    """
    scaler_params = fit_scaler(X, method)
    transformed_data = transform_data(X, scaler_params)
    return transformed_data, scaler_params
