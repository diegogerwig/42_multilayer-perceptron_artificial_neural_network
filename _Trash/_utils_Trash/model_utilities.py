#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def gelu(x):
    """Gaussian Error Linear Unit activation"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def init_parameters(layer_dims):
    """
    Initialize parameters using He initialization
    """
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        # He initialization
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
        
    return parameters

def forward_propagation(X, parameters, training=True, dropout_rate=0.0, use_gelu=False):
    """
    Forward propagation with dropout and GELU option
    """
    cache = {}
    A = X.T  # Transpose for matrix multiplication
    L = len([k for k in parameters.keys() if k.startswith('W')]) // 2 + 1
    
    for l in range(1, L):
        A_prev = A
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        
        # Linear forward
        Z = np.dot(W, A_prev) + b
        
        # Activation
        if l == L-1:
            # Softmax for output layer
            A = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = A / np.sum(A, axis=0, keepdims=True)
        else:
            # GELU or ReLU for hidden layers
            if use_gelu:
                A = gelu(Z)
            else:
                A = np.maximum(0, Z)  # ReLU
            
            # Apply dropout during training
            if training and dropout_rate > 0:
                D = np.random.rand(*A.shape) > dropout_rate
                A = np.multiply(A, D)
                A = A / (1 - dropout_rate)  # Scale
                cache[f'D{l}'] = D
        
        cache[f'A{l}'] = A
        cache[f'Z{l}'] = Z
    
    return cache

def backward_propagation(parameters, cache, X, Y, lambda_reg=0.01, dropout_rate=0.0, use_gelu=False):
    """
    Backward propagation with L2 regularization and dropout
    """
    grads = {}
    m = X.shape[0]
    L = len([k for k in parameters.keys() if k.startswith('W')]) // 2 + 1
    
    # Initialize gradients for output layer
    AL = cache[f'A{L-1}']
    dZ = AL - Y.T
    
    # Backward pass
    for l in reversed(range(1, L)):
        W = parameters[f'W{l}']
        
        if l > 1:
            A_prev = cache[f'A{l-1}']
        else:
            A_prev = X.T
            
        # Compute gradients
        dW = (1/m) * np.dot(dZ, A_prev.T) + (lambda_reg/m) * W
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        if l > 1:
            dA_prev = np.dot(W.T, dZ)
            
            # Apply dropout mask if used
            if dropout_rate > 0:
                dA_prev = np.multiply(dA_prev, cache[f'D{l-1}'])
                dA_prev = dA_prev / (1 - dropout_rate)
            
            # Derivative of activation function
            Z = cache[f'Z{l-1}']
            if use_gelu:
                # Approximate GELU derivative
                dZ = dA_prev * (0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (Z + 0.044715 * Z**3))) + 
                               Z * np.exp(-(Z**2)/2) / np.sqrt(2*np.pi))
            else:
                dZ = dA_prev * (Z > 0)  # ReLU derivative
        
        grads[f'dW{l}'] = dW
        grads[f'db{l}'] = db
    
    return grads

def label_smoothing(y, alpha=0.1):
    """
    Apply label smoothing to target values
    alpha: smoothing factor (0-1)
    """
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    n_classes = y.shape[1]
    smooth_y = y * (1 - alpha) + (alpha / n_classes)
    return smooth_y
