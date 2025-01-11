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

def softmax(x):
    """
    Softmax activation function for output layer
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)