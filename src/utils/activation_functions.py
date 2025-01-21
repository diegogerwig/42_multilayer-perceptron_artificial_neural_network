import numpy as np

def sigmoid(Z):
    """
    Compute the sigmoid activation function.
    
    Args:
        Z (numpy.ndarray): Input array
        
    Returns:
        numpy.ndarray: Sigmoid activation applied element-wise
    """
    Z = np.clip(Z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    """
    Compute the derivative of the sigmoid activation function.
    
    Args:
        A (numpy.ndarray): Output of sigmoid function
        
    Returns:
        numpy.ndarray: Derivative of sigmoid
    """
    return A * (1 - A)

def relu(z):
    """
    Compute the ReLU (Rectified Linear Unit) activation function.
    
    Args:
        z (numpy.ndarray): Input array
        
    Returns:
        numpy.ndarray: ReLU activation applied element-wise
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    Compute the derivative of the ReLU activation function.
    
    Args:
        z (numpy.ndarray): Input array
        
    Returns:
        numpy.ndarray: Derivative of ReLU
    """
    return (z > 0) * 1

def leaky_relu(Z, alpha=0.1):
    """
    Compute the Leaky ReLU activation function.
    
    Args:
        Z (numpy.ndarray): Input array
        alpha (float): Slope for negative values. Defaults to 0.1
        
    Returns:
        numpy.ndarray: Leaky ReLU activation applied element-wise
    """
    Z = np.clip(Z, -500, 500)  # Prevent overflow
    return np.where(Z > 0, Z, alpha * Z)

def leaky_relu_derivative(Z, alpha=0.1):
    """
    Compute the derivative of the Leaky ReLU activation function.
    
    Args:
        Z (numpy.ndarray): Input array
        alpha (float): Slope for negative values. Defaults to 0.1
        
    Returns:
        numpy.ndarray: Derivative of Leaky ReLU
    """
    return np.where(Z > 0, 1, alpha)

def tanh(Z):
    """
    Compute the hyperbolic tangent activation function.
    
    Args:
        Z (numpy.ndarray): Input array
        
    Returns:
        numpy.ndarray: Tanh activation applied element-wise
    """
    return np.tanh(Z)

def tanh_derivative(A):
    """
    Compute the derivative of the hyperbolic tangent activation function.
    
    Args:
        A (numpy.ndarray): Output of tanh function
        
    Returns:
        numpy.ndarray: Derivative of tanh
    """
    return 1 - A ** 2

def softmax(Z):
    """
    Compute the softmax activation function.
    
    Args:
        Z (numpy.ndarray): Input array with shape (batch_size, n_classes)
        
    Returns:
        numpy.ndarray: Softmax probabilities with same shape as input
    """
    assert len(Z.shape) == 2, "Input must be 2-dimensional with shape (batch_size, n_classes)"
    Z_max = np.max(Z, axis=1, keepdims=1)
    e_x = np.exp(Z - Z_max)  # Subtract max for numerical stability
    div = np.sum(e_x, axis=1, keepdims=1)
    return e_x / div