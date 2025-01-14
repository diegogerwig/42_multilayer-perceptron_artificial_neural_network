#!/usr/bin/env python3
import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep

    def update(self, parameters, grads):
        """
        Update parameters using Adam optimization
        """
        if not self.m:  # Initialize if first update
            for key in parameters:
                self.m[key] = np.zeros_like(parameters[key])
                self.v[key] = np.zeros_like(parameters[key])
        
        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        
        for key in parameters:
            if f'd{key}' in grads:
                # Update biased first moment estimate
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[f'd{key}']
                # Update biased second raw moment estimate
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grads[f'd{key}'])
                
                # Update parameters
                parameters[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.epsilon)
        
        return parameters