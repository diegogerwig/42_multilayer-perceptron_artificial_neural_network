#!/usr/bin/env python3
import numpy as np

def binary_cross_entropy(y_true, y_pred, parameters=None, lambda_reg=0.01):
    """
    Calculate binary cross entropy loss with optional L2 regularization
    
    Parameters:
    y_true: numpy array of true labels (shape: (m, 1))
    y_pred: numpy array of predicted probabilities (shape: (m, 1))
    parameters: dictionary containing the weights for L2 regularization (optional)
    lambda_reg: regularization parameter (default: 0.01)
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Binary cross entropy formula: -1/m * Σ(y*log(p) + (1-y)*log(1-p))
    m = y_true.shape[0]
    bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # L2 regularization (if parameters provided)
    if parameters is not None:
        l2_reg = 0
        weight_keys = [key for key in parameters.keys() if key.startswith('W')]
        for key in weight_keys:
            l2_reg += np.sum(np.square(parameters[key]))
        l2_reg = (lambda_reg / (2 * m)) * l2_reg
        return bce_loss + l2_reg
    
    return bce_loss

def compute_accuracy(predictions, Y):
    """
    Calculate accuracy between predictions and true labels
    """
    if len(Y.shape) > 1:
        Y = np.argmax(Y, axis=1)
    return np.mean(predictions == Y)

def evaluate_predictions(predictions, Y):
    """
    Calculate and display comprehensive evaluation metrics
    """
    Y_true = np.argmax(Y, axis=1) if len(Y.shape) > 1 else Y
    
    true_positives = np.sum((predictions == 1) & (Y_true == 1))
    true_negatives = np.sum((predictions == 0) & (Y_true == 0))
    false_positives = np.sum((predictions == 1) & (Y_true == 0))
    false_negatives = np.sum((predictions == 0) & (Y_true == 1))
    
    accuracy = (true_positives + true_negatives) / len(Y)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {
            'true_positives': int(true_positives),
            'true_negatives': int(true_negatives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        }
    }
    
    return metrics