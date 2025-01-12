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

def softmax(Z):
    """
    Compute softmax activation function
    Z: input array
    returns: softmax probabilities
    """
    # Subtract max for numerical stability
    exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)