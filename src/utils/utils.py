import pandas as pd
import numpy as np

YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
MAGENTA = "\033[1;34m"
END = "\033[0m"

def get_accuracy(y_pred, y_true):
    """
    Calculate accuracy for softmax outputs and one-hot encoded labels.
    
    Args:
        y_pred: model predictions (softmax probabilities), shape (n_samples, n_classes)
        y_true: true labels (one-hot encoded), shape (n_samples, n_classes)
        
    Returns:
        float: accuracy score
    """
    # For binary predictions with softmax
    predictions = (y_pred[:, 1] > 0.5).astype(int)  # Take probability of class 1
    # Calculate accuracy
    return (predictions == y_true).mean()

def load(path: str, header: str = "header") -> pd.DataFrame:
    HANDLED_ERRORS = (FileNotFoundError, PermissionError,
                      ValueError, IsADirectoryError)
    try:
        df = pd.read_csv(path) if header is not None \
                               else pd.read_csv(path, header=None)
        print(f"{GREEN}Loading dataset of dimensions {YELLOW}{df.shape}{END}")
        return df
    except HANDLED_ERRORS as error:
        print(f"{YELLOW}{__name__}: {type(error).__name__}: {error}{END}")
        return exit(1)
