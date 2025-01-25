import numpy as np

def binary_cross_entropy(y_true, y_pred):
    """
    Calculates binary cross-entropy loss.

    The binary cross-entropy (BCE) function measures the performance of a classification model where the output is a probability between 0 and 1. It calculates the average difference between predicted probabilities and actual binary labels (0 or 1), where:

        - A loss value closer to 0 indicates better predictions
        - Higher loss values indicate worse predictions

    It's commonly used in machine learning, particularly for binary classification problems like spam detection, image classification, or medical diagnosis.
    """
    # Small constant to avoid log(0)
    epsilon = 1e-15
    num_samples = y_true.shape[0]
    
    # If input is softmax output (m,2), use positive class probability
    if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
        y_pred = y_pred[:, 1]
    
    # Ensure correct data shape (m,1)
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    
    # Clip values between epsilon and 1-epsilon to avoid log(0) and log(1) respectively
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate BCE: -1/N * Σ(y*log(p) + (1-y)*log(1-p))
    loss = -1/num_samples * np.sum(
        y_true * np.log(y_pred) + 
        (1 - y_true) * np.log(1 - y_pred)
    )
    
    return loss