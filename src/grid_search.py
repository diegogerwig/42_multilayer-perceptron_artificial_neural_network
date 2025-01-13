#!/usr/bin/env python3

import os
import sys
import warnings
import logging
from contextlib import contextmanager
import argparse
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from itertools import product
import absl.logging

# Disable all warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').disabled = True
absl.logging.set_verbosity(absl.logging.ERROR)

# Configure TensorFlow environment
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',           # Disable all TF logging
    'CUDA_VISIBLE_DEVICES': '-1',          # Force CPU
    'TF_ENABLE_ONEDNN_OPTS': '0',          # Disable oneDNN
    'TF_SILENCE_CUDA_WARNINGS': '1',       # Silence CUDA
    'XLA_FLAGS': '--xla_gpu_cuda_data_dir=""',  # Disable XLA CUDA
    'TF_XLA_FLAGS': '--tf_xla_enable_xla_devices=false',  # Disable XLA devices
    'AUTOGRAPH_VERBOSITY': '0',            # Disable autograph
    'KMP_WARNINGS': 'off',                 # Disable KMP
    'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',
    'TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE': 'False'
})

# Additional TF configurations
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

# Disable TF2 behavior warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

@contextmanager
def suppress_stdout_stderr():
    """Context manager to redirect stdout and stderr to devnull"""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    devnull = open(os.devnull, 'w')
    try:
        sys.stdout = sys.stderr = devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def create_model(input_shape, hidden_layers, learning_rate, dropout_rate):
    """Creates and compiles the neural network model"""
    model = Sequential()
    
    # First hidden layer
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(dropout_rate))
    
    # Additional hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Output layer - single neuron with sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"

def print_progress(current, total, params, mean_loss=None, start_time=None):
    """Prints progress bar and current iteration details"""
    width = 80
    progress = current / total
    filled = int(width * progress)
    bar = Colors.BLUE + '=' * filled + Colors.YELLOW + '-' * (width - filled) + Colors.ENDC
    
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    eta_str = ""
    if start_time and current > 1:
        elapsed = (datetime.now() - start_time).total_seconds()
        iterations_left = total - current
        time_per_iteration = elapsed / (current - 1)
        eta_seconds = int(iterations_left * time_per_iteration)
        eta_str = f"ETA: {Colors.YELLOW}{format_time(eta_seconds)}{Colors.ENDC} | "
    
    if mean_loss is not None:
        loss_color = Colors.GREEN if mean_loss < 0.3 else Colors.YELLOW if mean_loss < 0.5 else Colors.RED
        print(f"\r[{Colors.BLUE}{timestamp}{Colors.ENDC}] [{bar}] {current}/{total} | "
              f"{eta_str}"
              f"Loss: {loss_color}{mean_loss:.4f}{Colors.ENDC} | "
              f"lr={params['learning_rate']}, layers={params['hidden_layers']}", end='')
    else:
        print(f"\r[{Colors.BLUE}{timestamp}{Colors.ENDC}] [{bar}] {current}/{total} | "
              f"{eta_str}"
              f"Evaluating: lr={params['learning_rate']}, layers={params['hidden_layers']}", end='')

def custom_grid_search(X, y, param_grid, n_splits=3):
    """Performs hyperparameter search with cross-validation"""
    # Ensure y is in the correct shape for binary classification
    y = y.reshape(-1, 1) if len(y.shape) == 1 else y
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    total_combinations = len(param_combinations)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_loss = np.inf
    best_params = None
    results = []
    
    print(f"\n{Colors.HEADER}Evaluating {Colors.BOLD}{total_combinations}{Colors.ENDC}"
          f"{Colors.HEADER} parameter combinations...{Colors.ENDC}")
    print(f"Using {Colors.BOLD}{n_splits}-fold{Colors.ENDC} cross validation\n")
    
    search_start_time = datetime.now()
    for idx, params in enumerate(param_combinations, 1):
        print_progress(idx, total_combinations, params, start_time=search_start_time)
        fold_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = create_model(
                input_shape=X.shape[1],
                hidden_layers=params['hidden_layers'],
                learning_rate=params['learning_rate'],
                dropout_rate=params['dropout_rate']
            )
            
            history = model.fit(
                X_train, y_train,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            val_loss = history.history['val_loss'][-1]
            fold_losses.append(val_loss)
        
        mean_loss = np.mean(fold_losses)
        std_loss = np.std(fold_losses)
        
        print_progress(idx, total_combinations, params, mean_loss, search_start_time)
        print()
        
        results.append({
            'params': params,
            'mean_loss': mean_loss,
            'std_loss': std_loss
        })
        
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = params
            print(f"\n{Colors.GREEN}[NEW BEST RESULT] Loss: {best_loss:.4f}{Colors.ENDC}")
    
    return best_params, best_loss, results

def configure_tensorflow():
    """Configure TensorFlow to use memory growth and CPU only"""
    # Disable GPU
    try:
        tf.config.set_visible_devices([], 'GPU')
    except:
        pass
    
    # Allow memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass

def main():
    # Configure TensorFlow before any model operations
    configure_tensorflow()
    
    parser = argparse.ArgumentParser(
        description='Neural Network Training with Feature Selection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--train_data', required=True, help='Path to training data CSV file')
    
    args = parser.parse_args()
    
    print(f"\n{Colors.HEADER}Loading data...{Colors.ENDC}")
    
    # Load data
    data = pd.read_csv(args.train_data, header=None)
    features = data.iloc[:, 2:].values
    diagnosis = data.iloc[:, 1].values
    
    # Convert labels
    y = np.where(diagnosis == 'M', 1, 0)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    # Select features using correlation
    from feature_selection import select_features_train
    # Create a dummy test set since we only need training features
    X_selected, _ = select_features_train(X, y, X.copy(), min_correlation=0.70)
    X = X_selected  # Use only the selected features
    
    print(f"{Colors.BLUE}Processed dataset: {Colors.BOLD}{X.shape[0]}{Colors.ENDC}{Colors.BLUE} samples, "
          f"{Colors.BOLD}{X.shape[1]}{Colors.ENDC}{Colors.BLUE} features{Colors.ENDC}")
    
    # Create models directory
    os.makedirs('./models', exist_ok=True)
    
    # Parameter grid
    param_grid = {
        'hidden_layers': [
            [12, 6],
            [16, 8],
            [24, 12],
            [32, 16],
            [16, 12, 8],
            [24, 16, 8],
        ],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'dropout_rate': [0.1, 0.2, 0.3],
        'batch_size': [16, 32, 64],
        'epochs': [200]
    }
    
    # Run grid search
    start_time = datetime.now()
    best_params, best_loss, results = custom_grid_search(X, y, param_grid)
    end_time = datetime.now()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save grid search results
    results_df = pd.DataFrame([{
        'hidden_layers': str(r['params']['hidden_layers']),
        'learning_rate': r['params']['learning_rate'],
        'dropout_rate': r['params']['dropout_rate'],
        'batch_size': r['params']['batch_size'],
        'epochs': r['params']['epochs'],
        'mean_loss': r['mean_loss'],
        'std_loss': r['std_loss']
    } for r in results])
    
    results_path = f'./models/grid_search_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    
    # Save best parameters
    best_params_df = pd.DataFrame([{
        'timestamp': timestamp,
        'loss': best_loss,
        'hidden_layers': str(best_params['hidden_layers']),
        'learning_rate': best_params['learning_rate'],
        'dropout_rate': best_params['dropout_rate'],
        'batch_size': best_params['batch_size'],
        'epochs': best_params['epochs'],
        'n_selected_features': X.shape[1]
    }])
    
    best_params_path = './models/best_params.csv'
    if os.path.exists(best_params_path):
        best_params_history = pd.read_csv(best_params_path)
        best_params_df = pd.concat([best_params_history, best_params_df], ignore_index=True)
    
    best_params_df.to_csv(best_params_path, index=False)
    
    # Print final results
    print(f"\n{Colors.HEADER}" + "="*50 + Colors.ENDC)
    print(f"{Colors.BOLD}FINAL RESULTS{Colors.ENDC}")
    print(f"{Colors.HEADER}" + "="*50 + Colors.ENDC)
    print(f"\n{Colors.BLUE}Best parameters found:{Colors.ENDC}")
    for param, value in best_params.items():
        print(f"{Colors.YELLOW}{param}: {Colors.ENDC}{value}")
    print(f"\n{Colors.GREEN}Best loss: {best_loss:.4f}{Colors.ENDC}")
    print(f"\n{Colors.BLUE}Total execution time: {Colors.BOLD}{end_time - start_time}{Colors.ENDC}")
    
    print(f"\n{Colors.GREEN}Results saved to {results_path}{Colors.ENDC}")
    print(f"{Colors.GREEN}Best parameters history updated in {best_params_path}{Colors.ENDC}")

if __name__ == "__main__":
    main()