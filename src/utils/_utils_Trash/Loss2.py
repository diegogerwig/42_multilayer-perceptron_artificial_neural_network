import numpy as np

class Loss:
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """
        Binary cross-entropy loss that works with softmax output.
        Args:
            y_true: Binary labels (0 or 1)
            y_pred: Softmax output (probability distribution over classes)
        """
        epsilon = 1e-15
        m = y_true.shape[0]
        
        # If input is softmax output (shape m,2), use probability of positive class
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        
        # Reshape predictions and labels to match
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        
        # Clip values to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate binary cross-entropy
        loss = - 1 / m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    @staticmethod
    def sparse_categorical_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(np.log(y_pred[np.arange(m), y_true])) / m

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred)) / m
