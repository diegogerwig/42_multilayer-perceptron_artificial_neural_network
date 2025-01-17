#!/usr/bin/env python3
import numpy as np
from utils.activation_functions import relu, relu_derivative, softmax, gelu, gelu_derivative

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

def layer_normalize(Z, gamma, beta, epsilon=1e-8):
    """
    Apply layer normalization
    """
    mean = np.mean(Z, axis=1, keepdims=True)
    variance = np.var(Z, axis=1, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta

def layer_normalize_backward(dZ, Z, gamma, beta, epsilon=1e-8):
    """
    Backward pass for layer normalization
    """
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
    dZ = dZ_norm/std + 2*dvar*Z_centered/Z.shape[1] + dmu/Z.shape[1]
    
    return dZ, dgamma, dbeta

def init_parameters(layer_dims):
    """
    Initialize network parameters with improved initialization
    """
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        # Inicialización mejorada usando Kaiming/He con un factor de escala
        scale = np.sqrt(1.5)  # Factor de escala para aumentar la varianza inicial
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * scale * np.sqrt(2./layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
        
        if l < L-1:  # Solo para capas ocultas
            # Parámetros de normalización
            parameters[f'gamma{l}'] = np.ones((layer_dims[l], 1)) * 1.0  # Inicialización con 1
            parameters[f'beta{l}'] = np.zeros((layer_dims[l], 1))
            
            # Layer normalization con inicialización mejorada
            parameters[f'gamma_ln{l}'] = np.ones((layer_dims[l], 1)) * 1.0
            parameters[f'beta_ln{l}'] = np.zeros((layer_dims[l], 1))
            
            # Skip connections con inicialización escalada
            if layer_dims[l] == layer_dims[l-1]:
                parameters[f'skip_scale{l}'] = np.ones((1,)) * 0.1  # Inicialización más conservadora
    
    return parameters
    
def backward_propagation(parameters, cache, X, Y, lambda_reg=0.05, dropout_rate=0.5, use_gelu=False):
    """
    Backward propagation with improvements
    """
    grads = {}
    L = len([key for key in parameters.keys() if key.startswith('W')])
    m = X.shape[0]
    
    # Initialize gradient for output layer
    dA_L = cache[f'A{L}'] - Y  # Output layer gradient (cross-entropy with softmax)
    
    for l in reversed(range(1, L + 1)):
        A_prev = cache[f'A{l-1}']
        
        if l == L:
            # Output layer gradient
            dZ = dA_L
            dW = 1/m * np.dot(dZ.T, A_prev) + (lambda_reg/m) * parameters[f'W{l}']
            db = 1/m * np.sum(dZ, axis=0, keepdims=True).T
        else:
            # Hidden layer gradient
            # First compute dZ from dA
            if use_gelu:
                dZ = dA * gelu_derivative(cache[f'Z{l}'].T)
            else:
                dZ = dA * relu_derivative(cache[f'Z{l}'].T)
            
            # Apply dropout if used during training
            if f'D{l}' in cache['dropout_mask']:
                dZ *= cache['dropout_mask'][f'D{l}'].T / (1 - dropout_rate)
            
            # Compute gradients for normalization if present
            if f'gamma{l}' in parameters:
                dZ_norm = dZ.T
                dZ_norm, dgamma, dbeta = batch_normalize_backward(
                    dZ_norm,
                    cache[f'Z{l}'],
                    parameters[f'gamma{l}'],
                    parameters[f'beta{l}']
                )
                dZ = dZ_norm.T
                grads[f'dgamma{l}'] = dgamma
                grads[f'dbeta{l}'] = dbeta
            
            # Compute weight and bias gradients
            dW = 1/m * np.dot(dZ.T, A_prev) + (lambda_reg/m) * parameters[f'W{l}']
            db = 1/m * np.sum(dZ, axis=0, keepdims=True).T
            
            # Handle residual connections if present
            if f'skip_scale{l}' in parameters and dZ.shape[1] == A_prev.shape[1]:
                grads[f'dskip_scale{l}'] = np.sum(dZ * A_prev) / m
        
        # Store gradients
        grads[f'dW{l}'] = dW
        grads[f'db{l}'] = db
        
        # Compute dA for next layer
        if l > 1:
            dA = np.dot(dZ, parameters[f'W{l}'])
            # Add residual gradient if present
            if f'skip_scale{l-1}' in parameters:
                dA += dZ * parameters[f'skip_scale{l-1}']
    
    return grads

def forward_propagation(X, parameters, training=True, dropout_rate=0.5, use_gelu=False):
    """
    Forward propagation with improvements
    """
    cache = {'A0': X}
    L = len([key for key in parameters.keys() if key.startswith('W')])
    dropout_mask = {}
    
    for l in range(1, L+1):
        A_prev = cache[f'A{l-1}']
        Z = np.dot(parameters[f'W{l}'], A_prev.T) + parameters[f'b{l}']
        
        if l < L:  # Hidden layers
            # Batch normalization
            if f'gamma{l}' in parameters:
                Z = batch_normalize(Z, parameters[f'gamma{l}'], parameters[f'beta{l}'])
            
            # Activation
            if use_gelu:
                A = gelu(Z)
            else:
                A = relu(Z)
            
            # Dropout
            if training:
                dropout_mask[f'D{l}'] = np.random.rand(*A.shape) > dropout_rate
                A *= dropout_mask[f'D{l}'] / (1 - dropout_rate)
            
            # Residual connection
            if f'skip_scale{l}' in parameters and A.shape == A_prev.T.shape:
                A = A + parameters[f'skip_scale{l}'] * A_prev.T
            
            A = A.T
        else:  # Output layer
            A = softmax(Z.T)
        
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
    
    cache['dropout_mask'] = dropout_mask
    return cache

def label_smoothing(Y, alpha=0.1):
    """
    Apply label smoothing to target values
    """
    classes = Y.shape[1]
    Y_smooth = Y * (1 - alpha) + alpha / classes
    return Y_smooth