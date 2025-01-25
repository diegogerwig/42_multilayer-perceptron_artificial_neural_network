import numpy as np

def gradient_descent(W, b, dW, db, learning_rate=0.01, state=None):
    """
    Gradient Descent optimizer function.
    
    Args:
        W: List of weight matrices
        b: List of bias vectors
        dW: List of weight gradients
        db: List of bias gradients
        learning_rate: Learning rate for optimization
        state: Not used in Gradient Descent, kept for consistent interface
        
    Returns:
        tuple: (updated weights, updated biases, state)
    """
    for i in range(len(W)):
        W[i] -= learning_rate * dW[i]
        b[i] -= learning_rate * db[i]
    
    return W, b, None

def sgd(W, b, dW, db, learning_rate=0.01, state=None):
    """
    SGD optimizer function (Stochastic Gradient Descent).
    Similar to Gradient Descent, but only updates weights and biases once per batch.
        
    Args:
        W: List of weight matrices
        b: List of bias vectors
        dW: List of weight gradients
        db: List of bias gradients
        learning_rate: Learning rate for optimization
        state: Not used in SGD, kept for consistent interface
        
    Returns:
        tuple: (updated weights, updated biases, state)
    """
    for i in range(len(W)):
        W[i] -= learning_rate * dW[i]
        b[i] -= learning_rate * db[i]
    
    return W, b, None

def momentum(W, b, dW, db, learning_rate=0.01, momentum_coef=0.9, state=None):
    """
    Momentum optimizer function (Stochastic Gradient Descent with momentum).
    Uses an exponentially decaying average of past gradients to update weights and biases.
    
    Args:
        W: List of weight matrices
        b: List of bias vectors
        dW: List of weight gradients
        db: List of bias gradients
        learning_rate: Learning rate for optimization
        momentum_coef: Momentum coefficient
        state: Dictionary containing velocity states for weights and biases
        
    Returns:
        tuple: (updated weights, updated biases, updated state)
    """
    # Initialize state if None
    if state is None:
        state = {
            'vW': [np.zeros_like(w) for w in W],
            'vb': [np.zeros_like(bias) for bias in b]
        }
    
    vW = state['vW']
    vb = state['vb']
    
    for i in range(len(W)):
        vW[i] = momentum_coef * vW[i] + learning_rate * dW[i]
        vb[i] = momentum_coef * vb[i] + learning_rate * db[i]
        W[i] -= vW[i]
        b[i] -= vb[i]
    
    state = {'vW': vW, 'vb': vb}
    return W, b, state