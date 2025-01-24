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

