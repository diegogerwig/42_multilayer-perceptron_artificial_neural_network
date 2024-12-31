#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(X, parameters):
    cache = {'A0': X.T}
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        cache[f'Z{l}'] = np.dot(parameters[f'W{l}'], cache[f'A{l-1}']) + parameters[f'b{l}']
        cache[f'A{l}'] = sigmoid(cache[f'Z{l}'])
    
    return cache[f'A{L}'].T > 0.5, cache[f'A{L}'].T

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def evaluate_predictions(predictions, Y):
    true_positives = np.sum((predictions == 1) & (Y == 1))
    true_negatives = np.sum((predictions == 0) & (Y == 0))
    false_positives = np.sum((predictions == 1) & (Y == 0))
    false_negatives = np.sum((predictions == 0) & (Y == 1))
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model_params', default='model_params.npy')
    args = parser.parse_args()
    
    # Load model and normalization parameters
    parameters = np.load(args.model_params, allow_pickle=True).item()
    norm_params = np.load('model_norm.npz')
    
    # Load and prepare data
    data = pd.read_csv(args.dataset, header=None)
    X = data.iloc[:, 2:].values  # Skip id and diagnosis
    Y = (data.iloc[:, 1] == 'M').astype(int).values.reshape(-1, 1)
    
    # Standardize using saved parameters
    X = (X - norm_params['mean']) / norm_params['std']
    
    # Make predictions
    predictions, probabilities = predict(X, parameters)
    
    # Calculate loss
    loss = binary_cross_entropy(Y, probabilities)
    print(f'\nTest Data Shape: {X.shape}')
    print(f'Binary Cross-Entropy Loss: {loss:.4f}')
    
    # Evaluate predictions
    evaluate_predictions(predictions, Y)

if __name__ == "__main__":
    main()