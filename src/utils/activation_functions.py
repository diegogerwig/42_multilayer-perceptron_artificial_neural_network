import numpy as np

def sigmoid(Z):
    """
    Compute the sigmoid activation function.
    """
    Z = np.clip(Z, -500, 500)  # Clip values to avoid overflow                      
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    """
    Compute the derivative of the sigmoid activation function.
    """
    return A * (1 - A)

def relu(z):
    """
    Compute the ReLU (Rectified Linear Unit) activation function.
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    Compute the derivative of the ReLU activation function.
    """
    return (z > 0) * 1

def softmax(Z):
    """
    Compute the softmax activation function.
        """
    assert len(Z.shape) == 2, "Input must be 2-dimensional with shape (batch_size, n_classes)"
    Z_max = np.max(Z, axis=1, keepdims=1)
    e_x = np.exp(Z - Z_max)  # Subtract max for numerical stability
    div = np.sum(e_x, axis=1, keepdims=1)
    return e_x / div