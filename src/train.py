#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import json
import os
from plot import plot_learning_curves

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def binary_cross_entropy(y_true, y_pred, parameters, lambda_reg=0.01):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    m = y_true.shape[1]
    
    # Add L2 regularization
    l2_reg = 0
    for l in range(len(parameters) // 2):
        l2_reg += np.sum(np.square(parameters[f'W{l+1}']))
    l2_reg = (lambda_reg / (2 * m)) * l2_reg
    
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) + l2_reg

def init_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        scale = np.sqrt(2./layer_dims[l-1])
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * scale
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return parameters

def forward_propagation(X, parameters, training=True, dropout_rate=0.3):
    cache = {'A0': X.T}
    L = len(parameters) // 2
    dropout_mask = {}
    
    for l in range(1, L+1):
        Z = np.dot(parameters[f'W{l}'], cache[f'A{l-1}']) + parameters[f'b{l}']
        if l == L:
            cache[f'A{l}'] = sigmoid(Z)
        else:
            A = relu(Z)
            if training:
                dropout_mask[f'D{l}'] = np.random.rand(*A.shape) > dropout_rate
                A *= dropout_mask[f'D{l}'] / (1 - dropout_rate)
            cache[f'A{l}'] = A
            
    cache['dropout_mask'] = dropout_mask
    return cache

def backward_propagation(parameters, cache, X, Y, lambda_reg=0.01):
    grads = {}
    L = len(parameters) // 2
    m = X.shape[0]
    
    dA = -(np.divide(Y.T, cache[f'A{L}']) - np.divide(1 - Y.T, 1 - cache[f'A{L}']))
    
    for l in reversed(range(1, L + 1)):
        if l == L:
            dZ = dA * cache[f'A{L}'] * (1 - cache[f'A{L}'])
        else:
            dZ = dA * relu_derivative(cache[f'A{l}'])
            dZ *= cache['dropout_mask'][f'D{l}'] / (1 - 0.3)
        
        dW = 1/m * np.dot(dZ, cache[f'A{l-1}'].T) + (lambda_reg/m) * parameters[f'W{l}']
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        
        grads[f'dW{l}'] = dW
        grads[f'db{l}'] = db
        
        if l > 1:
            dA = np.dot(parameters[f'W{l}'].T, dZ)
    
    return grads

def select_features(X_train, Y_train, X_text, num_features=10):
    df = pd.DataFrame(X_train)
    df['target'] = Y_train.ravel()
    correlations = abs(df.corr()['target']).sort_values(ascending=False)
    selected_features = correlations[1:num_features+1].index.astype(int).values
    
    print("Selected features:", selected_features)
    return X_train[:, selected_features], X_text[:, selected_features]

def compute_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def train_model(train_data, text_data, layers, epochs, learning_rate, batch_size):
    X_train = train_data.iloc[:, 2:].values
    Y_train = (train_data.iloc[:, 1] == 'M').astype(int).values.reshape(-1, 1)
    X_text = text_data.iloc[:, 2:].values
    Y_text = (text_data.iloc[:, 1] == 'M').astype(int).values.reshape(-1, 1)
    
    X_train, X_text = select_features(X_train, Y_train, X_text)
    
    X_train = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)
    X_text = (X_text - np.mean(X_text, axis=0)) / (np.std(X_text, axis=0) + 1e-8)
    
    print(f'x_train shape : {X_train.shape}')
    print(f'x_text shape : {X_text.shape}')
    
    layer_dims = [X_train.shape[1]] + layers + [1]
    parameters = init_parameters(layer_dims)
    
    os.makedirs('./models', exist_ok=True)
    with open('./models/model_topology.json', 'w') as f:
        json.dump({'layers': layer_dims, 'activation': 'relu'}, f)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 50
    min_delta = 0.001
    patience_counter = 0
    
    initial_lr = learning_rate
    decay_rate = 0.01
    
    for epoch in range(epochs):
        learning_rate = initial_lr / (1 + decay_rate * epoch)
        indices = np.random.permutation(X_train.shape[0])
        
        for i in range(0, X_train.shape[0], batch_size):
            batch_indices = indices[i:min(i + batch_size, X_train.shape[0])]
            batch_X = X_train[batch_indices]
            batch_Y = Y_train[batch_indices]
            
            cache = forward_propagation(batch_X, parameters, training=True)
            grads = backward_propagation(parameters, cache, batch_X, batch_Y)
            
            for key in parameters:
                parameters[key] -= learning_rate * grads[f'd{key}']
        
        train_cache = forward_propagation(X_train, parameters, training=False)
        train_predictions = train_cache[f'A{len(layer_dims)-1}'].T > 0.5
        train_loss = binary_cross_entropy(Y_train.T, train_cache[f'A{len(layer_dims)-1}'].T, parameters)
        train_acc = compute_accuracy(train_predictions, Y_train)
        
        text_cache = forward_propagation(X_text, parameters, training=False)
        text_predictions = text_cache[f'A{len(layer_dims)-1}'].T > 0.5
        text_loss = binary_cross_entropy(Y_text.T, text_cache[f'A{len(layer_dims)-1}'].T, parameters)
        text_acc = compute_accuracy(text_predictions, Y_text)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(text_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(text_acc)
        
        print(f'epoch {epoch+1:02d}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} '
              f'- val_loss: {text_loss:.4f} - val_acc: {text_acc:.4f}')
        
        if text_loss < (best_val_loss - min_delta):
            best_val_loss = text_loss
            patience_counter = 0
            np.save('./models/model_params.npy', parameters)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
    
    print("> saving model './models/model_params.npy' to disk...")
    network_info = {
        'layers': layer_dims,
        'lr': learning_rate,
        'batch_size': batch_size
    }
    plot_learning_curves(history, network_info)
    return parameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--text_data', required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[32, 16, 8])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    train_data = pd.read_csv(args.train_data, header=None)
    text_data = pd.read_csv(args.text_data, header=None)
    
    train_model(train_data, text_data, args.layers, args.epochs, args.learning_rate, args.batch_size)

if __name__ == "__main__":
    main()