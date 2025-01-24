#!/usr/bin/env python3
import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0.0, amsgrad=True):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.v = {}  # Second moment estimates
        self.s = {}  # First moment estimates
        self.v_max = {}  # Maximum second moments for AMSGrad
        self.t = 0   # Time step
        
    def initialize(self, parameters):
        """Initialize moment estimates"""
        for key in parameters.keys():
            if key.startswith('W') or key.startswith('b'):
                self.v[key] = np.zeros_like(parameters[key])
                self.s[key] = np.zeros_like(parameters[key])
                if self.amsgrad:
                    self.v_max[key] = np.zeros_like(parameters[key])
    
    def update(self, parameters, grads):
        """Update parameters using improved Adam optimization"""
        if not self.v or not self.s:
            self.initialize(parameters)
        
        self.t += 1
        parameters_updated = {}
        
        # Bias correction terms
        v_correction = 1 - (self.beta1 ** self.t)
        s_correction = 1 - (self.beta2 ** self.t)
        
        # Learning rate schedule with warmup
        if self.t < 1000:  # Warmup period
            current_lr = self.learning_rate * (self.t / 1000)
        else:
            current_lr = self.learning_rate
        
        for key in parameters.keys():
            if key.startswith('W') or key.startswith('b'):
                grad_key = 'd' + key
                if grad_key not in grads:
                    continue
                
                grad = grads[grad_key]
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * parameters[key]
                
                # Update biased first moment estimate
                self.s[key] = self.beta1 * self.s[key] + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.power(grad, 2)
                
                if self.amsgrad:
                    # Update maximum of v
                    self.v_max[key] = np.maximum(self.v_max[key], self.v[key])
                    v_corrected = self.v_max[key] / s_correction
                else:
                    v_corrected = self.v[key] / s_correction
                
                # Compute bias-corrected first moment estimate
                s_corrected = self.s[key] / v_correction
                
                # Update parameters with gradient clipping
                grad_norm = np.linalg.norm(s_corrected)
                if grad_norm > 1.0:
                    s_corrected = s_corrected / grad_norm
                
                parameters_updated[key] = (parameters[key] - 
                                         current_lr * s_corrected / 
                                         (np.sqrt(v_corrected) + self.epsilon))
            else:
                parameters_updated[key] = parameters[key]
        
        return parameters_updated