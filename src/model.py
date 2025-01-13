#!/usr/bin/env python3
import numpy as np
from activation_functions import relu, relu_derivative, softmax

def batch_normalize(Z, gamma, beta, epsilon=1e-8):
    """
    Apply batch normalization
    """
    mean = np.mean(Z, axis=1, keepdims=True)
    variance = np.var(Z, axis=1, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta

def batch_normalize_backward(dZ, Z, gamma, beta, epsilon=1e-8):
    """
    Backward pass for batch normalization
    """
    N = Z.shape[1]
    mu = np.mean(Z, axis=1, keepdims=True)
    var = np.var(Z, axis=1, keepdims=True)
    Z_centered = Z - mu
    std = np.sqrt(var + epsilon)
    Z_norm = Z_centered / std
    
    dgamma = np.sum(dZ * Z_norm, axis=1, keepdims=True)
    dbeta = np.sum(dZ, axis=1, keepdims=True)
    
    dZ_norm = dZ * gamma
    dvar = np.sum(dZ_norm * Z_centered * -0.5 * (var + epsilon)**(-1.5), axis=1, keepdims=True)
    dmu = np.sum(dZ_norm * -1/std, axis=1, keepdims=True) + dvar * np.mean(-2 * Z_centered, axis=1, keepdims=True)
    dZ = dZ_norm/std + 2*dvar*Z_centered/N + dmu/N
    
    return dZ, dgamma, dbeta

def init_parameters(layer_dims):
    """
    Initialize network parameters using He initialization
    """
    parameters = {}
    L = len(layer_dims)
    
    # Initialize weights and biases for each layer (except input layer)
    #   Values for weights: He initialization 
    #   Values for biases: 0
    # Initialize gamma and beta for batch normalization layers
    #   Values for gamma (scale): 1
    #   Values for beta (shift): 0

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
        
        if l < L-1:  # Batch norm params only for hidden layers
            parameters[f'gamma{l}'] = np.ones((layer_dims[l], 1))
            parameters[f'beta{l}'] = np.zeros((layer_dims[l], 1))
    
    return parameters

def forward_propagation(X, parameters, training=True, dropout_rate=0.5):
    """
    Forward propagation with ReLU for hidden layers and softmax for output layer
    """
    cache = {'A0': X}
    L = len([key for key in parameters.keys() if key.startswith('W')])
    dropout_mask = {}
    
    for l in range(1, L+1):
        A_prev = cache[f'A{l-1}']
        Z = np.dot(A_prev, parameters[f'W{l}'].T) + parameters[f'b{l}'].T
        
        if l < L:  # Hidden layers - use ReLU
            Z = batch_normalize(Z.T, parameters[f'gamma{l}'], parameters[f'beta{l}'])
            Z = Z.T
            A = relu(Z)  # ReLU activation for hidden layers
            if training:
                dropout_mask[f'D{l}'] = np.random.rand(*A.shape) > dropout_rate
                A *= dropout_mask[f'D{l}'] / (1 - dropout_rate)
        else:  # Output layer - use softmax
            A = softmax(Z)  # Softmax activation for output layer
        
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
    
    cache['dropout_mask'] = dropout_mask
    return cache

def backward_propagation(parameters, cache, X, Y, lambda_reg=0.05, dropout_rate=0.5):
    """
    Backward propagation with proper derivatives for ReLU and softmax
    """
    grads = {}
    L = len([key for key in parameters.keys() if key.startswith('W')])
    m = X.shape[0]
    
    # For output layer (softmax + cross-entropy derivative)
    dZ_L = cache[f'A{L}'] - Y
    
    for l in reversed(range(1, L + 1)):
        A = cache[f'A{l}']
        A_prev = cache[f'A{l-1}']
        
        if l == L:
            dZ = dZ_L  # Use softmax derivative for output layer
        else:
            # Use ReLU derivative for hidden layers
            dZ = dA * relu_derivative(cache[f'Z{l}'])
            
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
            
            if f'D{l}' in cache['dropout_mask']:
                dZ *= cache['dropout_mask'][f'D{l}'] / (1 - dropout_rate)
        
        dW = 1/m * np.dot(dZ.T, A_prev) + (lambda_reg/m) * parameters[f'W{l}']
        db = 1/m * np.sum(dZ, axis=0, keepdims=True).T
        
        grads[f'dW{l}'] = dW
        grads[f'db{l}'] = db
        
        if l > 1:
            dA = np.dot(dZ, parameters[f'W{l}'])
    
    return grads