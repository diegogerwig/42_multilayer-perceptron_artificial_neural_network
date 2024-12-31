#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def predict(X, parameters):
    cache = {'A0': X.T}
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        cache[f'Z{l}'] = np.dot(parameters[f'W{l}'], cache[f'A{l-1}']) + parameters[f'b{l}']
        cache[f'A{l}'] = sigmoid(cache[f'Z{l}'])
    
    return cache[f'A{L}'].T > 0.5, cache[f'A{L}'].T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model_params', default='model_params.npy')
    parser.add_argument('--model_topology', default='model_topology.json')
    args = parser.parse_args()
    
    # Load model
    parameters = np.load(args.model_params, allow_pickle=True).item()
    with open(args.model_topology) as f:
        topology = json.load(f)
    
    # Load data
    data = pd.read_csv(args.dataset)
    X = data.drop('diagnosis', axis=1).values
    Y = data['diagnosis'].values.reshape(-1, 1)
    
    # Make predictions
    predictions, probabilities = predict(X, parameters)
    
    # Calculate metrics
    accuracy = np.mean(predictions == Y)
    loss = binary_cross_entropy(Y, probabilities)
    
    print(f'Test Data Shape: {X.shape}')
    print(f'Loss: {loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()