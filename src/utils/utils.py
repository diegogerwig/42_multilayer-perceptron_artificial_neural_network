import pandas as pd
import numpy as np
from colorama import Fore, Style

def get_accuracy(y_pred, y_true):
    """
    Calculate accuracy for softmax outputs and one-hot encoded labels.
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
        print(f"{Fore.GREEN}Loading dataset of dimensions {Fore.YELLOW}{df.shape}{Style.RESET_ALL}")
        return df
    except HANDLED_ERRORS as error:
        print(f"{Fore.YELLOW}{__name__}: {type(error).__name__}: {error}{Style.RESET_ALL}")
        return exit(1)