import numpy as np

def fit_scaler(X, method="z_score"):
    """
    Calculate scaling parameters for the data.
    """
    if method not in ["z_score", "minmax"]:
        raise ValueError("Method must be either 'z_score' or 'minmax'.")
    
    # Convert input to numpy array if it's a pandas DataFrame/Series
    X = np.array(X)
    
    # Calculate parameters and convert to native Python lists for JSON serialization
    params = {
        "method": method,
        "mean": np.mean(X, axis=0).tolist(),
        "scale": np.std(X, axis=0).tolist(),
        "min": np.min(X, axis=0).tolist(),
        "max": np.max(X, axis=0).tolist()
    }
    
    return params

def transform_data(X, params):
    """
    Transform data using pre-calculated scaling parameters.
    """
    method = params["method"]
    
    # Convert parameters back to numpy arrays for calculations
    if method == "z_score":
        if "mean" not in params or "scale" not in params:
            raise RuntimeError("Missing required scaling parameters for z-score normalization")
        mean = np.array(params["mean"])
        scale = np.array(params["scale"])
        return (X - mean) / (scale + 1e-15)
        
    elif method == "minmax":
        if "min" not in params or "max" not in params:
            raise RuntimeError("Missing required scaling parameters for min-max normalization")
        min_vals = np.array(params["min"])
        max_vals = np.array(params["max"])
        return (X - min_vals) / ((max_vals - min_vals) + 1e-15)
    
    else:
        raise ValueError("Method must be either 'z_score' or 'minmax'.")

def fit_transform_data(X, method="z_score"):
    """
    Fit the scaler and transform the data in one step.
    """
    params = fit_scaler(X, method)
    transformed_data = transform_data(X, params)
    return transformed_data, params