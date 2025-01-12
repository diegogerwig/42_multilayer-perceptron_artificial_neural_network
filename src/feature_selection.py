#!/usr/bin/env python3
import numpy as np
import pandas as pd
import json
import os

def read_column_names(filename='./data/data_columns_names.txt'):
    """
    Read and clean feature names from file
    """
    try:
        names = []
        with open(filename, 'r') as f:
            content = f.read()
            content = content[content.find('[')+1:content.rfind(']')]
            names = [name.strip().strip('"').strip() for name in content.split(',')]
            names = [name for name in names if name]  # Remove empty strings
            
            print("\nFull column list:")
            for i, name in enumerate(names):
                print(f"{i}: {name}")
            
            # Return only feature names (skip ID and Diagnosis)
            feature_names = names[2:]  # Skip first two columns
            print("\nFeature names:")
            for i, name in enumerate(feature_names):
                print(f"{i}: {name}")
            
            return feature_names
            
    except Exception as e:
        print(f"\nError reading column names from {filename}: {e}")
        return None

def select_features_train(X_train, diagnosis, X_test, min_correlation=0.70):
    """
    Select features based on correlation with diagnosis.
    diagnosis should be 1 for Malignant (M) and 0 for Benign (B).
    """
    print(f"Data shape: {X_train.shape}")
    
    # Get feature names
    feature_names = read_column_names()
    if not feature_names:
        raise ValueError("Could not read feature names")
    
    if len(feature_names) != X_train.shape[1]:
        print(f"\nWarning: Number of feature names ({len(feature_names)}) doesn't match data dimensions ({X_train.shape[1]})")
        print("\nAvailable feature names:")
        for i, name in enumerate(feature_names):
            print(f"{i}: {name}")
        raise ValueError("Feature names and data dimensions mismatch")
    
    # Calculate correlations
    df = pd.DataFrame(X_train)
    df['diagnosis'] = diagnosis
    
    # Calculate correlations with diagnosis
    correlations = abs(df.corr()['diagnosis']).drop('diagnosis')
    selected_features = correlations[correlations >= min_correlation].sort_values(ascending=False)
    selected_indices = np.array([int(i) for i in selected_features.index])
    
    print("\nSelected features -> correlation with DIAGNOSIS >= 0.70:")
    print("{:<3} {:<25} {:<15}".format("ID", "Feature Name", "Correlation"))
    print("-" * 45)

    # Print features with correlations in tabulated format
    for i, (idx, corr) in enumerate(selected_features.items(), 1):
        feat_name = feature_names[int(idx)]
        print("{:<3} {:<25} {:.3f}".format(i, feat_name, corr))
    
    # Save selected feature information
    os.makedirs('./models', exist_ok=True)
    save_data = {
        'selected_indices': selected_indices.tolist(),
        'feature_names': [feature_names[int(i)] for i in selected_indices]
    }
    
    with open('./models/selected_features.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved selected feature indices to './models/selected_features.json'")
    
    return X_train[:, selected_indices], X_test[:, selected_indices]

def select_features_predict(X, feature_indices_file='./models/selected_features.json'):
    """
    Select features for prediction using saved indices
    """
    try:
        with open(feature_indices_file, 'r') as f:
            data = json.load(f)
            selected_indices = np.array(data['selected_indices'], dtype=int)
            
        print(f"\nLoading {len(selected_indices)} selected features:")
        if 'feature_names' in data:
            for i, name in enumerate(data['feature_names'], 1):
                print(f"{i}. {name}")
                
        return X[:, selected_indices]
    except Exception as e:
        raise Exception(f"Error loading selected features: {str(e)}")