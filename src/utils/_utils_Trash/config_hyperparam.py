#!/usr/bin/env python3

# Optimized grid search parameters
PARAM_GRID = {
    'hidden_layers': [
        [128, 64, 32],         # Deep and wide
        [256, 128, 64, 32],    # Very deep and wide
        [64, 64, 32, 32]       # Balanced depth
    ],
    'learning_rate': [
        0.0005,               # Very conservative
        0.0001                # Ultra conservative
    ],
    'dropout_rate': [
        0.3,                  # Stronger dropout
        0.4                   # Very strong dropout
    ],
    'batch_size': [16, 32],   # Smaller batches
    'lambda_reg': [
        0.0005,              # Light regularization
        0.0001               # Very light regularization
    ],
    'use_gelu': [True]       # Always use GELU
}

# Training configuration
TRAINING_CONFIG = {
    'early_stopping': {
        'patience': 10,       # More aggressive stopping
        'min_delta': 0.0005   # Smaller improvements required
    },
    'learning_rate_schedule': {
        'warmup_epochs': 10,
        'cycle_length': 15,
        'min_lr_factor': 0.05
    },
    'label_smoothing': 0.2,  # Increased smoothing
    'cross_validation': {
        'n_splits': 5,
        'shuffle': True,
        'random_state': 42
    }
}

# Model configuration
MODEL_CONFIG = {
    'optimizer': 'adam',
    'metrics': [
        'accuracy',
        'precision',
        'recall',
        'AUC'
    ],
    'monitor_metric': 'val_loss',
    'monitor_mode': 'min',
    'layer_init': 'he_uniform'
}

# Data configuration
DATA_CONFIG = {
    'validation_split': 0.2,
    'random_state': 42,
    'preprocessing': {
        'outlier_threshold': 3,
        'correlation_threshold': 0.90,
        'skew_threshold': 1.0,
        'min_variance': 1e-5
    }
}