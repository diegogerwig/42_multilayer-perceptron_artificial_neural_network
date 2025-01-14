#!/usr/bin/env python3
import numpy as np

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU activation function
    """
    return (x > 0).astype(float)

def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation function
    Alternative to ReLU that's smoother
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    """
    Derivative of GELU activation function
    """
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))) + \
           0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))**2) * \
           np.sqrt(2 / np.pi) * (1 + 0.134145 * x**2)

def softmax(Z):
    """
    Compute softmax activation function
    Z: input array
    returns: softmax probabilities
    """
    # Subtract max for numerical stability
    exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)