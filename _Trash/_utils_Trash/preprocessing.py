#!/usr/bin/env python3
import numpy as np
from scipy import stats
import warnings

def preprocess_breast_cancer_data(X, y=None):
    """
    Enhanced preprocessing optimized for breast cancer data
    """
    def robust_scale(data):
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        scale = np.where(iqr > 0, iqr, 1.0)
        median = np.median(data, axis=0)
        return (data - median) / scale

    X_processed = X.copy()
    original_shape = X_processed.shape

    # 1. Robust scaling (less sensitive to outliers)
    X_processed = robust_scale(X_processed)

    # 2. Log transform highly skewed features
    skewness = stats.skew(X_processed, axis=0)
    for i in range(X_processed.shape[1]):
        if abs(skewness[i]) > 1.0:
            # Add small constant to ensure positivity
            min_val = np.min(X_processed[:, i])
            if min_val < 0:
                X_processed[:, i] = X_processed[:, i] - min_val + 1e-6
            X_processed[:, i] = np.log1p(X_processed[:, i])

    # 3. Feature aggregation
    # Create interaction features for radius, texture, and area
    radius_idx = [i for i, col in enumerate(range(X.shape[1])) if 'radius' in str(col)]
    texture_idx = [i for i, col in enumerate(range(X.shape[1])) if 'texture' in str(col)]
    area_idx = [i for i, col in enumerate(range(X.shape[1])) if 'area' in str(col)]

    if radius_idx and area_idx:
        radius_area = X_processed[:, radius_idx[0]] * X_processed[:, area_idx[0]]
        X_processed = np.column_stack([X_processed, radius_area])

    # 4. Remove highly correlated features
    corr_matrix = np.corrcoef(X_processed.T)
    mask = np.ones(X_processed.shape[1], dtype=bool)
    variances = np.var(X_processed, axis=0)
    
    for i in range(len(corr_matrix)-1):
        for j in range(i+1, len(corr_matrix)):
            if mask[i] and mask[j] and abs(corr_matrix[i,j]) > 0.90:  # Reduced threshold
                if variances[i] < variances[j]:
                    mask[i] = False
                else:
                    mask[j] = False
    
    X_processed = X_processed[:, mask]

    # 5. Final standardization
    mean = np.mean(X_processed, axis=0)
    std = np.std(X_processed, axis=0) + 1e-8
    X_processed = (X_processed - mean) / std

    # 6. Remove outliers using robust statistics
    z_scores = np.abs((X_processed - np.median(X_processed, axis=0)) / 
                     (np.percentile(X_processed, 75, axis=0) - 
                      np.percentile(X_processed, 25, axis=0)))
    outlier_mask = np.all(z_scores < 3, axis=1)
    
    X_processed = X_processed[outlier_mask]
    if y is not None:
        y = y[outlier_mask]

    preprocessing_info = {
        'shapes': {
            'original': original_shape,
            'processed': X_processed.shape,
        },
        'feature_mask': mask.tolist(),
        'scaling': {
            'mean': mean.tolist(),
            'std': std.tolist()
        },
        'extra_features': {
            'radius_idx': radius_idx,
            'area_idx': area_idx,
            'texture_idx': texture_idx
        }
    }

    if y is not None:
        return X_processed, y, preprocessing_info
    return X_processed, preprocessing_info

def transform_new_data(X, preprocessing_info):
    """Transform new data using preprocessing parameters"""
    X_processed = X.copy()

    # 1. Robust scaling
    X_processed = (X_processed - np.median(X_processed, axis=0)) / (
        np.percentile(X_processed, 75, axis=0) - 
        np.percentile(X_processed, 25, axis=0)
    )

    # 2. Add interaction features if present in training
    if ('extra_features' in preprocessing_info and 
        'radius_idx' in preprocessing_info['extra_features'] and 
        'area_idx' in preprocessing_info['extra_features']):
        
        radius_idx = preprocessing_info['extra_features']['radius_idx']
        area_idx = preprocessing_info['extra_features']['area_idx']
        
        if radius_idx and area_idx:
            radius_area = X_processed[:, radius_idx[0]] * X_processed[:, area_idx[0]]
            X_processed = np.column_stack([X_processed, radius_area])

    # 3. Apply feature mask
    mask = np.array(preprocessing_info['feature_mask'])
    X_processed = X_processed[:, mask]

    # 4. Apply final scaling
    mean = np.array(preprocessing_info['scaling']['mean'])
    std = np.array(preprocessing_info['scaling']['std'])
    X_processed = (X_processed - mean) / std

    return X_processed