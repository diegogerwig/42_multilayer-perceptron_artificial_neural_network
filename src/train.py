#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def init_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return parameters

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Prevent division by zero
    return (X - mean) / std, mean, std

def forward_propagation(X, parameters):
    cache = {'A0': X.T}
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        cache[f'Z{l}'] = np.dot(parameters[f'W{l}'], cache[f'A{l-1}']) + parameters[f'b{l}']
        cache[f'A{l}'] = sigmoid(cache[f'Z{l}'])
    
    return cache

def backward_propagation(parameters, cache, X, Y):
    grads = {}
    L = len(parameters) // 2
    m = X.shape[0]
    
    dAL = -(np.divide(Y.T, cache[f'A{L}']) - np.divide(1 - Y.T, 1 - cache[f'A{L}']))
    
    for l in reversed(range(1, L + 1)):
        dZ = dAL * sigmoid_derivative(cache[f'A{l}'])
        grads[f'dW{l}'] = 1/m * np.dot(dZ, cache[f'A{l-1}'].T)
        grads[f'db{l}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dAL = np.dot(parameters[f'W{l}'].T, dZ)
    
    return grads

def train_model(train_data, text_data, layers, epochs, learning_rate, batch_size):
    # Prepare data
    X_train = train_data.iloc[:, 2:].values  # Skip id and diagnosis
    Y_train = (train_data.iloc[:, 1] == 'M').astype(int).values.reshape(-1, 1)
    X_text = text_data.iloc[:, 2:].values
    Y_text = (text_data.iloc[:, 1] == 'M').astype(int).values.reshape(-1, 1)
    
    # Standardize
    X_train, mean, std = standardize(X_train)
    X_text = (X_text - mean) / std
    
    # Save normalization params
    os.makedirs('./models', exist_ok=True)
    np.savez('./models/model_norm.npz', mean=mean, std=std)
    
    print(f'x_train shape : {X_train.shape}')
    print(f'x_text shape : {X_text.shape}')
    
    # Initialize network
    layer_dims = [X_train.shape[1]] + layers + [1]
    parameters = init_parameters(layer_dims)
    
    # Save network topology
    topology = {
        'layers': layer_dims,
        'activation': 'sigmoid'
    }
    with open('./models/model_topology.json', 'w') as f:
        json.dump(topology, f)
    
    # Training loop
    for epoch in range(epochs):
        # Mini-batch training
        indices = np.random.permutation(X_train.shape[0])
        
        for i in range(0, X_train.shape[0], batch_size):
            batch_indices = indices[i:min(i + batch_size, X_train.shape[0])]
            batch_X = X_train[batch_indices]
            batch_Y = Y_train[batch_indices]
            
            cache = forward_propagation(batch_X, parameters)
            grads = backward_propagation(parameters, cache, batch_X, batch_Y)
            
            # Update parameters
            for l in range(1, len(layer_dims)):
                parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
                parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
        
        # Calculate metrics
        train_cache = forward_propagation(X_train, parameters)
        train_loss = binary_cross_entropy(Y_train.T, train_cache[f'A{len(layer_dims)-1}'].T)
        
        text_cache = forward_propagation(X_text, parameters)
        text_loss = binary_cross_entropy(Y_text.T, text_cache[f'A{len(layer_dims)-1}'].T)
        
        print(f'epoch {epoch+1:02d}/{epochs} - loss: {train_loss:.4f} - text_loss: {text_loss:.4f}')
    
    # Save model parameters
    np.save('./models/model_params.npy', parameters)
    print("> saving model './models/model_params.npy' to disk...")
    
    return parameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--text_data', required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[24, 24, 24])
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    train_data = pd.read_csv(args.train_data, header=None)
    text_data = pd.read_csv(args.text_data, header=None)
    
    train_model(train_data, text_data, args.layers, args.epochs, args.learning_rate, args.batch_size)

if __name__ == "__main__":
    main()