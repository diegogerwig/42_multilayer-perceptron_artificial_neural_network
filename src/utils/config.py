#!/usr/bin/env python3

# Grid search parameter configuration
PARAM_GRID = {
    'hidden_layers': [
        [16, 8],           # Basic architecture
        [32, 16],          # Wider architecture
        [64, 32],          # Large architecture
        [16, 12, 8],       # Basic deep architecture
        [32, 16, 8],       # Medium deep architecture
        [64, 32, 16]       # Large deep architecture
    ],
    'learning_rate': [
        0.001,            # Standard learning rate
        0.0003,           # Conservative learning rate
        0.0001            # Very conservative learning rate
    ],
    'dropout_rate': [
        0.1,              # Mild dropout
        0.15,             # Moderate dropout
        0.2               # Aggressive dropout
    ],
    'batch_size': [
        32,               # Small batch
        64,               # Medium batch
        128               # Large batch
    ],
    'epochs': [200],      # Fixed epochs, will use early stopping
    'lambda_reg': [
        0.0001,           # Very mild L2 regularization
        0.001,            # Mild L2 regularization
        0.01              # Moderate L2 regularization
    ],
    'use_gelu': [True]    # Use only GELU for simplicity
}

# Training configuration
TRAINING_CONFIG = {
    'early_stopping': {
        'patience': 20,
        'min_delta': 0.001
    },
    'learning_rate_schedule': {
        'warmup_epochs': 5,
        'cycle_length': 10,
        'min_lr_factor': 0.1
    },
    'label_smoothing': 0.05,
    'cross_validation': {
        'n_splits': 5,
        'shuffle': True,
        'random_state': 42
    }
}

# Model configuration
MODEL_CONFIG = {
    'optimizer': 'adam',
    'metrics': ['accuracy', 'AUC'],
    'monitor_metric': 'val_loss',
    'monitor_mode': 'min'
}

# Data configuration
DATA_CONFIG = {
    'validation_split': 0.2,
    'random_state': 42,
    'feature_selection': {
        'min_correlation': 0.7
    }
}

# Logging configuration
LOG_CONFIG = {
    'log_dir': './logs',
    'model_dir': './models',
    'results_dir': './results'
}