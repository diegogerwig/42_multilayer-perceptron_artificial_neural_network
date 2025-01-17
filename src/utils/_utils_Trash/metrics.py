#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Compute binary cross entropy with numerical stability
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def evaluate_predictions(y_pred, y_true):
    """
    Evaluate predictions with multiple metrics
    Returns dict with accuracy, precision, recall, F1, and AUC
    """
    y_true_class = np.argmax(y_true, axis=1)
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = np.mean(y_pred == y_true_class)
    
    # Precision, Recall, F1
    metrics['precision'] = precision_score(y_true_class, y_pred)
    metrics['recall'] = recall_score(y_true_class, y_pred)
    metrics['f1'] = f1_score(y_true_class, y_pred)
    
    # AUC
    try:
        metrics['auc'] = roc_auc_score(y_true_class, y_pred)
    except:
        metrics['auc'] = 0.5  # Default for failed calculation
    
    # Confusion matrix components
    metrics['confusion_matrix'] = {
        'true_positives': np.sum((y_pred == 1) & (y_true_class == 1)),
        'false_positives': np.sum((y_pred == 1) & (y_true_class == 0)),
        'true_negatives': np.sum((y_pred == 0) & (y_true_class == 0)),
        'false_negatives': np.sum((y_pred == 0) & (y_true_class == 1))
    }
    
    return metrics

def calculate_class_weights(y):
    """
    Calculate balanced class weights for imbalanced datasets
    """
    classes = np.unique(np.argmax(y, axis=1))
    weights = {}
    n_samples = len(y)
    
    for c in classes:
        c_count = np.sum(np.argmax(y, axis=1) == c)
        weights[c] = n_samples / (len(classes) * c_count)
    
    return weights