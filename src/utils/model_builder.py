#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class ModelBuilder:
    @staticmethod
    def build_model(input_shape, params):
        """
        Build and compile the neural network model
        """
        model = Sequential([
            # Input layer
            Input(shape=(input_shape,)),
            
            # First hidden layer
            Dense(
                params['hidden_layers'][0],
                activation='gelu' if params['use_gelu'] else 'relu',
                kernel_regularizer=l2(params['lambda_reg'])
            ),
            Dropout(params['dropout_rate']),
            
            # Additional hidden layers
            *[
                layer for units in params['hidden_layers'][1:]
                for layer in [
                    Dense(
                        units,
                        activation='gelu' if params['use_gelu'] else 'relu',
                        kernel_regularizer=l2(params['lambda_reg'])
                    ),
                    Dropout(params['dropout_rate'])
                ]
            ],
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model

    @staticmethod
    def create_lr_schedule(initial_lr, warmup_epochs, cycle_length, min_lr_factor):
        """
        Create a learning rate schedule with warmup and cosine decay
        """
        def lr_schedule(epoch):
            # Warmup period
            if epoch < warmup_epochs:
                return float(initial_lr * (epoch + 1) / warmup_epochs)
                
            # Cosine decay with restarts
            epoch_in_cycle = (epoch - warmup_epochs) % cycle_length
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_in_cycle / cycle_length))
            
            min_lr = initial_lr * min_lr_factor
            return float(min_lr + (initial_lr - min_lr) * cosine_decay)
            
        return lr_schedule