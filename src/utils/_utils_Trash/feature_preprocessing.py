#!/usr/bin/env python3
import numpy as np
from scipy import stats
import json

def remove_outliers(X, Y, threshold=3):
    """Remove outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(X))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], Y[mask]

def handle_skewness(X):
    """Apply log1p transformation to handle skewed features"""
    # Use log1p which handles zeros better than plain log
    return np.log1p(X - X.min(axis=0) + 1)

def detect_constant_features(X, threshold=0.01):
    """Detect and remove near-constant features"""
    std = np.std(X, axis=0)
    mask = std > threshold
    return X[:, mask], mask

def detect_correlated_features(X, threshold=0.95):
    """Detect and remove highly correlated features"""
    corr_matrix = np.corrcoef(X.T)
    # Create a mask for features to keep
    mask = np.ones(X.shape[1], dtype=bool)
    
    for i in range(corr_matrix.shape[0]):
        if mask[i]:  # Only check if feature hasn't been masked
            for j in range(i + 1, corr_matrix.shape[1]):
                if mask[j] and abs(corr_matrix[i, j]) > threshold:
                    # Keep feature with higher variance
                    if np.var(X[:, i]) < np.var(X[:, j]):
                        mask[i] = False
                        break
                    else:
                        mask[j] = False
    
    return X[:, mask], mask

def preprocess_features(X, Y, config=None):
    """
    Enhanced preprocessing pipeline for features
    
    Parameters:
        X: np.ndarray - Input features
        Y: np.ndarray - Target variables
        config: dict - Preprocessing configuration
    
    Returns:
        X_processed: np.ndarray - Processed features
        Y_processed: np.ndarray - Processed targets
        preprocessing_info: dict - Information about preprocessing steps
    """
    if config is None:
        config = {
            'remove_outliers': True,
            'handle_skewness': True,
            'remove_constant': True,
            'remove_correlated': True,
            'outlier_threshold': 3,
            'correlation_threshold': 0.95,
            'constant_threshold': 0.01
        }
    
    preprocessing_info = {
        'original_shape': X.shape,
        'steps': [],
        'feature_masks': {}
    }
    
    # 1. Remove outliers
    if config.get('remove_outliers', True):
        X_processed, Y_processed = remove_outliers(
            X, Y, 
            threshold=config.get('outlier_threshold', 3)
        )
        preprocessing_info['steps'].append({
            'step': 'outlier_removal',
            'samples_before': X.shape[0],
            'samples_after': X_processed.shape[0]
        })
        X, Y = X_processed, Y_processed
    
    # 2. Handle skewness
    if config.get('handle_skewness', True):
        X = handle_skewness(X)
        preprocessing_info['steps'].append({
            'step': 'skewness_handling'
        })
    
    # 3. Remove constant features
    if config.get('remove_constant', True):
        X, constant_mask = detect_constant_features(
            X, 
            threshold=config.get('constant_threshold', 0.01)
        )
        preprocessing_info['feature_masks']['constant'] = constant_mask.tolist()
        preprocessing_info['steps'].append({
            'step': 'constant_feature_removal',
            'features_before': len(constant_mask),
            'features_after': X.shape[1]
        })
    
    # 4. Remove highly correlated features
    if config.get('remove_correlated', True):
        X, correlation_mask = detect_correlated_features(
            X, 
            threshold=config.get('correlation_threshold', 0.95)
        )
        preprocessing_info['feature_masks']['correlation'] = correlation_mask.tolist()
        preprocessing_info['steps'].append({
            'step': 'correlation_feature_removal',
            'features_before': len(correlation_mask),
            'features_after': X.shape[1]
        })
    
    # 5. Standardize features
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X = (X - mean) / std
    
    preprocessing_info['normalization'] = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    
    # Save preprocessing info
    with open('./models/preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=4)
    
    return X, Y, preprocessing_info

def transform_new_data(X, preprocessing_info):
    """Transform new data using saved preprocessing parameters"""
    # Apply feature masks
    if 'constant' in preprocessing_info['feature_masks']:
        X = X[:, preprocessing_info['feature_masks']['constant']]
    
    if 'correlation' in preprocessing_info['feature_masks']:
        X = X[:, preprocessing_info['feature_masks']['correlation']]
    
    # Apply skewness handling if it was used
    if any(step['step'] == 'skewness_handling' for step in preprocessing_info['steps']):
        X = handle_skewness(X)
    
    # Apply normalization
    mean = np.array(preprocessing_info['normalization']['mean'])
    std = np.array(preprocessing_info['normalization']['std'])
    X = (X - mean) / std
    
    return X