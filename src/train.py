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

def batch_normalize(Z, gamma, beta, epsilon=1e-8):
    mean = np.mean(Z, axis=1, keepdims=True)
    variance = np.var(Z, axis=1, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta

def binary_cross_entropy(y_true, y_pred, parameters, lambda_reg=0.01):
    """
    Calculate binary cross entropy loss with L2 regularization.
    
    Args:
        y_true: true labels
        y_pred: predicted probabilities
        parameters: network parameters dictionary
        lambda_reg: L2 regularization strength
    
    Returns:
        total loss (cross entropy + L2 regularization)
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    m = y_true.shape[0]
    
    # Calculate cross entropy
    ce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Add L2 regularization only for weight matrices
    l2_reg = 0
    weight_keys = [key for key in parameters.keys() if key.startswith('W')]
    for key in weight_keys:
        l2_reg += np.sum(np.square(parameters[key]))
    l2_reg = (lambda_reg / (2 * m)) * l2_reg
    
    return ce_loss + l2_reg

def init_parameters(layer_dims):
    """
    Initialize network parameters.
    
    Args:
        layer_dims: list containing the dimensions of each layer
    
    Returns:
        parameters: dictionary containing initialized parameters
    """
    parameters = {}
    L = len(layer_dims)  # Total number of layers
    
    for l in range(1, L):
        # He initialization for weights
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
        
        # Batch normalization parameters for hidden layers only (not input or output layer)
        if l < L-1:  # Only for hidden layers
            parameters[f'gamma{l}'] = np.ones((layer_dims[l], 1))
            parameters[f'beta{l}'] = np.zeros((layer_dims[l], 1))
    
    return parameters

def forward_propagation(X, parameters, training=True, dropout_rate=0.5):
    """
    Forward propagation step.
    
    Args:
        X: input data of shape (n_samples, n_features)
        parameters: dictionary containing network parameters
        training: boolean indicating if in training mode
        dropout_rate: dropout probability
    
    Returns:
        cache: dictionary containing activations and dropout masks
    """
    cache = {'A0': X}
    L = len([key for key in parameters.keys() if key.startswith('W')])  # Number of layers
    dropout_mask = {}
    
    for l in range(1, L+1):
        A_prev = cache[f'A{l-1}']
        
        # Linear forward
        Z = np.dot(A_prev, parameters[f'W{l}'].T) + parameters[f'b{l}'].T
        
        # Apply batch normalization and ReLU for hidden layers
        if l < L:  # For hidden layers
            Z = batch_normalize(Z.T, parameters[f'gamma{l}'], parameters[f'beta{l}'])
            Z = Z.T
            A = relu(Z)
            if training:
                dropout_mask[f'D{l}'] = np.random.rand(*A.shape) > dropout_rate
                A *= dropout_mask[f'D{l}'] / (1 - dropout_rate)
        else:  # For output layer
            A = sigmoid(Z)
        
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
    
    cache['dropout_mask'] = dropout_mask
    return cache

def batch_normalize_backward(dZ, Z, gamma, beta, epsilon=1e-8):
    """
    Backward pass for batch normalization.
    """
    N = Z.shape[1]
    
    # Step 1: Calculate mean
    mu = np.mean(Z, axis=1, keepdims=True)
    
    # Step 2: Subtract mean
    Z_centered = Z - mu
    
    # Step 3: Calculate variance and std
    var = np.var(Z, axis=1, keepdims=True)
    std = np.sqrt(var + epsilon)
    Z_norm = Z_centered / std
    
    # Calculate gradients
    dgamma = np.sum(dZ * Z_norm, axis=1, keepdims=True)
    dbeta = np.sum(dZ, axis=1, keepdims=True)
    
    # Calculate gradient with respect to Z
    dZ_norm = dZ * gamma
    dvar = np.sum(dZ_norm * Z_centered * -0.5 * (var + epsilon)**(-1.5), axis=1, keepdims=True)
    dmu = np.sum(dZ_norm * -1/std, axis=1, keepdims=True) + dvar * np.mean(-2 * Z_centered, axis=1, keepdims=True)
    dZ = dZ_norm/std + 2*dvar*Z_centered/N + dmu/N
    
    return dZ, dgamma, dbeta

def backward_propagation(parameters, cache, X, Y, lambda_reg=0.05, dropout_rate=0.5):
    """
    Backward propagation step.
    """
    grads = {}
    L = len([key for key in parameters.keys() if key.startswith('W')])
    m = X.shape[0]
    
    # Initialize gradient at output layer
    dA = -(np.divide(Y, cache[f'A{L}']) - np.divide(1 - Y, 1 - cache[f'A{L}']))
    
    for l in reversed(range(1, L + 1)):
        A = cache[f'A{l}']
        A_prev = cache[f'A{l-1}']
        
        if l == L:  # Output layer
            dZ = dA * A * (1 - A)  # Gradient for sigmoid
        else:  # Hidden layers
            dZ = dA * relu_derivative(cache[f'Z{l}'])
            # Apply batch normalization backward pass
            if f'gamma{l}' in parameters:
                dZ, dgamma, dbeta = batch_normalize_backward(
                    dZ.T, 
                    cache[f'Z{l}'].T,
                    parameters[f'gamma{l}'],
                    parameters[f'beta{l}']
                )
                dZ = dZ.T
                grads[f'dgamma{l}'] = dgamma
                grads[f'dbeta{l}'] = dbeta
            
            # Apply dropout
            if f'D{l}' in cache['dropout_mask']:
                dZ *= cache['dropout_mask'][f'D{l}'] / (1 - dropout_rate)
        
        # Calculate gradients for weights and biases
        dW = 1/m * np.dot(dZ.T, A_prev) + (lambda_reg/m) * parameters[f'W{l}']
        db = 1/m * np.sum(dZ, axis=0, keepdims=True).T
        
        grads[f'dW{l}'] = dW
        grads[f'db{l}'] = db
        
        if l > 1:  # Compute dA for next layer
            dA = np.dot(dZ, parameters[f'W{l}'])
    
    return grads

def train_model(train_data, text_data, layers, epochs, learning_rate, batch_size, dropout_rate=0.5):
    X_train = train_data.iloc[:, 2:].values
    Y_train = (train_data.iloc[:, 1] == 'M').astype(int).values.reshape(-1, 1)
    X_text = text_data.iloc[:, 2:].values
    Y_text = (text_data.iloc[:, 1] == 'M').astype(int).values.reshape(-1, 1)
    
    X_train, X_text = select_features(X_train, Y_train, X_text)
    
    # Standardize features
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
            
            cache = forward_propagation(batch_X, parameters, training=True, dropout_rate=dropout_rate)
            grads = back

def read_column_names(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def select_features(X_train, Y_train, X_text, min_correlation=0.70, column_names_file='./data/data_columns_names.txt'):
   df = pd.DataFrame(X_train[:, 2:])  # Skip ID and Diagnosis
   df['target'] = Y_train.ravel()
   
   correlations = abs(df.corr()['target']).drop('target')
   selected_features = correlations[correlations >= min_correlation].sort_values(ascending=False)
   selected_indices = selected_features.index.astype(int).values + 2  # Adjust indices
   
   try:
       column_names = read_column_names(column_names_file)
       print("\nSelected features with correlation >= 0.70:")
       for idx, feat in enumerate(selected_indices):
           print(f"{idx+1}. Column {feat}: {column_names[feat]} (correlation: {selected_features.iloc[idx]:.3f})")
   except:
       print("\nSelected features:", selected_indices)
   
   return X_train[:, selected_indices], X_text[:, selected_indices]


def compute_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def train_model(train_data, text_data, layers, epochs, learning_rate, batch_size, dropout_rate=0.5):
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
            
            cache = forward_propagation(batch_X, parameters, training=True, dropout_rate=dropout_rate)
            grads = backward_propagation(parameters, cache, batch_X, batch_Y, dropout_rate=dropout_rate)
            
            for key in parameters:
                parameters[key] -= learning_rate * grads[f'd{key}']
        
        # Evaluate on train set
        train_cache = forward_propagation(X_train, parameters, training=False, dropout_rate=dropout_rate)
        train_predictions = train_cache[f'A{len(layer_dims)-1}'] > 0.5
        train_loss = binary_cross_entropy(Y_train, train_cache[f'A{len(layer_dims)-1}'], parameters)
        train_acc = compute_accuracy(train_predictions, Y_train)
        
        # Evaluate on test set
        text_cache = forward_propagation(X_text, parameters, training=False, dropout_rate=dropout_rate)
        text_predictions = text_cache[f'A{len(layer_dims)-1}'] > 0.5
        text_loss = binary_cross_entropy(Y_text, text_cache[f'A{len(layer_dims)-1}'], parameters)
        text_acc = compute_accuracy(text_predictions, Y_text)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(text_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(text_acc)
        
        print(f'epoch {epoch+1:02d}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} '
              f'- val_loss: {text_loss:.4f} - val_acc: {text_acc:.4f}')
        
        # Early stopping check
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
        'batch_size': batch_size,
        'dropout_rate': dropout_rate
    }
    plot_learning_curves(history, network_info)
    return parameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--text_data', required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[64, 32, 16])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    args = parser.parse_args()
    
    train_data = pd.read_csv(args.train_data, header=None)
    text_data = pd.read_csv(args.text_data, header=None)
    
    train_model(
        train_data, 
        text_data, 
        args.layers, 
        args.epochs, 
        args.learning_rate, 
        args.batch_size,
        args.dropout_rate
    )

if __name__ == "__main__":
    main()