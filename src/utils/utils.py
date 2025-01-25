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
