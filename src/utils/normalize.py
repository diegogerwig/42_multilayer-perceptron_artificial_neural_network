# import numpy as np

# def z_score_scale(X):
#     """
#     Standardize features by removing the mean and scaling to unit variance (z-score normalization)
#     """
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     return (X - mean) / (std + 1e-15), mean, std

# def minmax_scale(X):
#     """
#     Scale features to a fixed range [0, 1] using min-max scaling
#     """
#     min_vals = np.min(X, axis=0)
#     max_vals = np.max(X, axis=0)
#     return (X - min_vals) / ((max_vals - min_vals) + 1e-15), min_vals, max_vals

# def scale_data(X, method="z_score"):
#     """
#     Scale data using the specified method and return scaled data along with scaling parameters
    
#     Parameters:
#         X : array-like
#             Data to scale
#         method : str, optional (default="z_score")
#             Scaling method: "z_score" or "minmax"
            
#     Returns:
#         scaled_X : array-like
#             Scaled data
#         params : tuple
#             Scaling parameters (mean, std) for z_score or (min, max) for minmax
#     """
#     if method == "z_score":
#         scaled_X, param1, param2 = z_score_scale(X)
#     elif method == "minmax":
#         scaled_X, param1, param2 = minmax_scale(X)
#     else:
#         raise ValueError("Method must be either 'z_score' or 'minmax'")
        
#     return scaled_X, (param1, param2)

# def transform_data(X, method="z_score", params=None):
#     """
#     Transform data using pre-computed scaling parameters
    
#     Parameters:
#         X : array-like
#             Data to transform
#         method : str, optional (default="z_score")
#             Scaling method: "z_score" or "minmax"
#         params : tuple
#             Scaling parameters (mean, std) for z_score or (min, max) for minmax
            
#     Returns:
#         transformed_X : array-like
#             Transformed data
#     """
#     if params is None:
#         raise ValueError("Scaling parameters must be provided")
        
#     param1, param2 = params
    
#     if method == "z_score":
#         return (X - param1) / (param2 + 1e-15)
#     elif method == "minmax":
#         return (X - param1) / ((param2 - param1) + 1e-15)
#     else:
#         raise ValueError("Method must be either 'z_score' or 'minmax'")



import numpy as np

class Scaler:
    def __init__(self, method="z_score"):
        self.method = method
        self.mean = None
        self.scale = None
        self.min = None
        self.max = None
    
    def fit(self, X):
            self.mean = np.mean(X, axis=0)
            self.scale = np.std(X, axis=0)
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)

    def transform(self, X):
        if self.method == "z_score":
            if self.mean is None or self.scale is None:
                raise RuntimeError("You must fit the scaler before transforming data")
            return (X - self.mean) / (self.scale + 1e-15)
        elif self.method == 'minmax':
            if self.min is None or self.max is None:
                raise RuntimeError("You must fit the scaler before transforming data.")
            return (X - self.min) / ((self.max - self.min) + 1e-15)
        else:
            raise ValueError("Method must be either 'z_score' or 'minmax'.")
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

