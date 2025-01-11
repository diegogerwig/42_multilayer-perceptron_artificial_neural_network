#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import json
import os

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def predict(X, parameters):
    """
    Forward propagation for prediction.
    """
    cache = {'A0': X}
    L = len([key for key in parameters.keys() if key.startswith('W')])

    for l in range(1, L+1):
        A_prev = cache[f'A{l-1}']
        Z = np.dot(A_prev, parameters[f'W{l}'].T) + parameters[f'b{l}'].T
        
        if l < L:  # Hidden layers with ReLU
            A = np.maximum(0, Z)  # ReLU activation
        else:  # Output layer with softmax
            A = softmax(Z)
        
        cache[f'A{l}'] = A
    
    predictions = np.argmax(cache[f'A{L}'], axis=1)
    probabilities = cache[f'A{L}']
    return predictions, probabilities

def categorical_cross_entropy(y_true, y_pred):
    """
    Calculate categorical cross entropy loss.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def select_features(X, feature_indices_file='./models/selected_features.npy'):
    """
    Select the same features that were used during training.
    """
    if not os.path.exists(feature_indices_file):
        raise FileNotFoundError(f"Selected features file not found at {feature_indices_file}")
    
    selected_indices = np.load(feature_indices_file)
    print(f"Loading {len(selected_indices)} selected features from training")
    return X[:, selected_indices]

def evaluate_predictions(predictions, Y):
    """
    Evaluate model predictions with various metrics.
    """
    Y_true = np.argmax(Y, axis=1)
    
    true_positives = np.sum((predictions == 1) & (Y_true == 1))
    true_negatives = np.sum((predictions == 0) & (Y_true == 0))
    false_positives = np.sum((predictions == 1) & (Y_true == 0))
    false_negatives = np.sum((predictions == 0) & (Y_true == 1))
    
    accuracy = (true_positives + true_negatives) / len(Y)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

def main():
    parser = argparse.ArgumentParser(
        description='Make predictions using the trained neural network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset', 
        required=True,
        help='Path to the dataset CSV file'
    )
    parser.add_argument(
        '--model_params', 
        default='./models/model_params.npy',
        help='Path to the model parameters file'
    )
    args = parser.parse_args()
    
    try:
        # Load model parameters
        if not os.path.exists(args.model_params):
            raise FileNotFoundError(f"Model parameters file not found at {args.model_params}")
        
        parameters = np.load(args.model_params, allow_pickle=True).item()
        print(f"\nLoaded model parameters from {args.model_params}")
        
        # Print model architecture
        print("\nModel architecture:")
        for key in sorted(parameters.keys()):
            if key.startswith('W'):
                print(f"  {key} shape: {parameters[key].shape}")
        
        # Load and prepare data
        print(f"\nLoading dataset from {args.dataset}")
        data = pd.read_csv(args.dataset, header=None)
        X = data.iloc[:, 2:].values  # Skip ID and Diagnosis columns
        print(f"Original input shape: {X.shape}")
        
        # Select features
        X = select_features(X)
        print(f"Input shape after feature selection: {X.shape}")
        
        # Convert labels to one-hot encoding
        Y = np.zeros((len(data), 2))
        Y[data.iloc[:, 1] == 'M', 1] = 1  # Malignant
        Y[data.iloc[:, 1] == 'B', 0] = 1  # Benign
        
        # Standardize features
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        
        # Verify input dimensions match model expectations
        input_dim = parameters['W1'].shape[1]
        if X.shape[1] != input_dim:
            raise ValueError(f"Model expects {input_dim} features but got {X.shape[1]} features.\n"
                           f"This usually means the feature selection during training selected a "
                           f"different number of features than what's saved in selected_features.npy")
        
        # Make predictions
        print("\nMaking predictions...")
        predictions, probabilities = predict(X, parameters)
        
        # Calculate loss
        loss = categorical_cross_entropy(Y, probabilities)
        print(f'Categorical Cross-Entropy Loss: {loss:.4f}')
        
        # Evaluate predictions
        evaluate_predictions(predictions, Y)
        
        # Print detailed predictions for first few samples
        print("\nDetailed predictions for first 5 samples:")
        print("ID | True Label | Predicted | Confidence")
        print("-" * 45)
        for i in range(min(5, len(predictions))):
            true_label = "M" if Y[i, 1] == 1 else "B"
            pred_label = "M" if predictions[i] == 1 else "B"
            confidence = probabilities[i, predictions[i]]
            print(f"{data.iloc[i, 0]:<3} | {true_label:^10} | {pred_label:^9} | {confidence:^10.4f}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()