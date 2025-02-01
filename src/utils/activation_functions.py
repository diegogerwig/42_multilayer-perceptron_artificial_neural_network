import numpy as np

def sigmoid(input_values):
    """
    Compute the sigmoid activation function, which maps any input to a value between 0 and 1.
    Commonly used in binary classification problems.
    """
    # Clip values to prevent overflow in exp() calculation
    # Numbers outside [-500, 500] would cause numerical instability
    safe_values = np.clip(input_values, -500, 500)
    
    # Calculate sigmoid: 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-safe_values))

def sigmoid_derivative(activated_values):
    """
    Compute the derivative of the sigmoid activation function.
    Used during backpropagation for gradient calculations.
    """
    # The input should already be the output of sigmoid function
    # Derivative can be computed as: f(x) * (1 - f(x))
    return activated_values * (1 - activated_values)

def relu(input_values):
    """
    Compute the ReLU (Rectified Linear Unit) activation function.
    Returns the input if positive, 0 if negative.
    Most commonly used activation function in modern neural networks.
    """
    # Return the input if positive, 0 if negative
    # Helps prevent vanishing gradient problem and allows for sparse activation
    return np.maximum(0, input_values)

def relu_derivative(input_values):
    """
    Compute the derivative of the ReLU activation function.
    Used during backpropagation for gradient calculations.
    """
    # Return 1 for inputs > 0, and 0 for inputs <= 0
    # Note: The derivative at exactly x=0 is technically undefined,
    # but we conventionally choose it to be 0
    return (input_values > 0).astype(int)

def softmax(input_values):
    """
    Compute the softmax activation function, which converts a vector of raw numbers
    into probabilities that sum to 1.
    """
    # Check that input has the correct shape (batch_size, n_classes)
    if len(input_values.shape) != 2:
        raise ValueError("Input must be 2D with shape (batch_size, n_classes)")
    
    # This is done to prevent numerical overflow when calculating exponentials
    max_values = np.max(input_values, axis=1, keepdims=True)
    
    # This ensures the largest possible exponent will be e^0 = 1
    shifted_values = input_values - max_values
    
    # This converts all numbers to positive values and enhances differences
    exponentials = np.exp(shifted_values)
    
    # This will be our denominator for normalization
    sum_exponentials = np.sum(exponentials, axis=1, keepdims=True)
    
    # This normalizes the values into probabilities that sum to 1
    probabilities = exponentials / sum_exponentials
    
    return probabilities