#!/usr/bin/env python3

# Custom context manager to suppress stdout and stderr
import os
import sys
import warnings
import logging
from contextlib import contextmanager
import io

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    # Save the original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

# Environment variables for TensorFlow and CUDA
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',           # FATAL
    'CUDA_VISIBLE_DEVICES': '-1',          # Disable GPU
    'TF_ENABLE_ONEDNN_OPTS': '0',          # Disable oneDNN
    'PYTHONWARNINGS': 'ignore',            # Disable Python warnings
    'TF_SILENCE_CUDA_WARNINGS': '1',       # Silence CUDA warnings
    'XLA_FLAGS': '--xla_gpu_cuda_data_dir=""',  # Disable XLA CUDA
    'TF_XLA_FLAGS': '--tf_xla_enable_xla_devices=false',  # Disable XLA
    'AUTOGRAPH_VERBOSITY': '0',            # Disable autograph warnings
    'KMP_WARNINGS': 'off',                 # Disable KMP warnings
    'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',  # Use pure Python implementation
    'TF_RESTART_TASK_ON_CUDA_MALLOC_FAILURE': 'false'  # Disable CUDA malloc retry
})

# Silence everything during imports
with suppress_stdout_stderr():
    import argparse
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from datetime import datetime
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam
    from itertools import product

# Additional TF configurations after import
tf.get_logger().setLevel('ERROR')

# Redirect stderr to devnull before importing tensorflow
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Environment variables for TensorFlow and CUDA
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',           # FATAL
    'CUDA_VISIBLE_DEVICES': '-1',          # Disable GPU
    'TF_ENABLE_ONEDNN_OPTS': '0',          # Disable oneDNN
    'PYTHONWARNINGS': 'ignore',            # Disable Python warnings
    'TF_SILENCE_CUDA_WARNINGS': '1',       # Silence CUDA warnings
    'XLA_FLAGS': '--xla_gpu_cuda_data_dir=""',  # Disable XLA CUDA
    'TF_XLA_FLAGS': '--tf_xla_enable_xla_devices=false',  # Disable XLA
    'AUTOGRAPH_VERBOSITY': '0',            # Disable autograph warnings
    'KMP_WARNINGS': 'off',                 # Disable KMP warnings
    'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',  # Use pure Python implementation
    'TF_RESTART_TASK_ON_CUDA_MALLOC_FAILURE': 'false'  # Disable CUDA malloc retry
})

# Configure logging
logging.getLogger('tensorflow').disabled = True
warnings.filterwarnings('ignore')

# Silence everything during imports
with suppress_stdout_stderr():
    import argparse
    import numpy as np
    import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from itertools import product

# Restore stderr
sys.stderr = stderr

# Additional TF configurations
tf.get_logger().setLevel('ERROR')

def create_model(input_shape, hidden_layers, learning_rate, dropout_rate):
    model = Sequential()
    
    # First hidden layer
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(dropout_rate))
    
    # Additional hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Output layer with 2 neurons for binary classification
    model.add(Dense(2, activation='softmax'))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_data(path):
    """Loads data from a CSV file."""
    data = pd.read_csv(path, header=None)
    features = data.iloc[:, 2:].values
    diagnosis = data.iloc[:, 1].values
    return features, diagnosis

def preprocess_data(features, labels):
    """Preprocesses data: scaling and label conversion."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    numeric_labels = np.where(labels == 'M', 1, 0)
    return scaled_features, numeric_labels

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def format_time(seconds):
    """Convert seconds to human readable string"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"

def print_progress(current, total, params, mean_score=None, start_time=None):
    """Prints a progress bar and current iteration details with ETA"""
    width = 30
    progress = current / total
    filled = int(width * progress)
    bar = Colors.BLUE + '=' * filled + Colors.YELLOW + '-' * (width - filled) + Colors.ENDC
    
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    # Calculate ETA
    eta_str = ""
    if start_time and current > 1:  # Need at least one iteration to estimate time
        elapsed = (datetime.now() - start_time).total_seconds()
        iterations_left = total - current
        time_per_iteration = elapsed / (current - 1)  # -1 because we want completed iterations
        eta_seconds = int(iterations_left * time_per_iteration)
        eta_str = f"ETA: {Colors.YELLOW}{format_time(eta_seconds)}{Colors.ENDC} | "
    
    if mean_score is not None:
        score_color = Colors.GREEN if mean_score > 0.85 else Colors.YELLOW if mean_score > 0.75 else Colors.RED
        print(f"\r[{Colors.BLUE}{timestamp}{Colors.ENDC}] [{bar}] {current}/{total} | "
              f"{eta_str}"
              f"Accuracy: {score_color}{mean_score:.4f}{Colors.ENDC} | "
              f"lr={params['learning_rate']}, layers={params['hidden_layers']}", end='')
    else:
        print(f"\r[{Colors.BLUE}{timestamp}{Colors.ENDC}] [{bar}] {current}/{total} | "
              f"{eta_str}"
              f"Evaluating: lr={params['learning_rate']}, layers={params['hidden_layers']}", end='')

def custom_grid_search(X, y, param_grid, n_splits=3):
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    total_combinations = len(param_combinations)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_score = -np.inf
    best_params = None
    results = []
    
    print(f"\n{Colors.HEADER}Starting evaluation of {Colors.BOLD}{total_combinations}{Colors.ENDC}{Colors.HEADER} parameter combinations...{Colors.ENDC}")
    print(f"Using {Colors.BOLD}{n_splits}-fold{Colors.ENDC} cross validation\n")
    
    search_start_time = datetime.now()
    for idx, params in enumerate(param_combinations, 1):
        print_progress(idx, total_combinations, params, start_time=search_start_time)
        fold_scores = []
        
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
            
            val_score = history.history['val_accuracy'][-1]
            fold_scores.append(val_score)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print_progress(idx, total_combinations, params, mean_score, search_start_time)
        print()
        
        results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score
        })
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            print(f"\n{Colors.GREEN}[NEW BEST RESULT] Accuracy: {best_score:.4f}{Colors.ENDC}")
    
    return best_params, best_score, results

def main():
    parser = argparse.ArgumentParser(
        description='Custom Grid Search with Cross-Validation for Neural Network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--train_data',
        required=True,
        help='Path to training data CSV file'
    )
    
    args = parser.parse_args()
    
    print(f"\n{Colors.HEADER}Loading data...{Colors.ENDC}")
    features, diagnosis = load_data(args.train_data)
    X, y = preprocess_data(features, diagnosis)
    
    print(f"{Colors.BLUE}Dataset loaded: {Colors.BOLD}{X.shape[0]}{Colors.ENDC}{Colors.BLUE} samples, "
          f"{Colors.BOLD}{X.shape[1]}{Colors.ENDC}{Colors.BLUE} features{Colors.ENDC}")
    
    # Ensure models directory exists
    os.makedirs('./models', exist_ok=True)
    
    # Grid design following the rule of thumb: decreasing neuron count between layers
    param_grid = {
        'hidden_layers': [
            [16, 8],          # Simple architecture
            [24, 12],         # Slightly larger
            [32, 16],         # Moderate size
            [16, 12, 8],      # Deeper, narrow
            [24, 16, 8],      # Deeper, moderate
            [32, 24, 16]      # Deeper, wider
        ],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'dropout_rate': [0.1, 0.2, 0.3],  # Lower dropout rates for smaller networks
        'batch_size': [16, 32, 64],
        'epochs': [50]
    }
    
    start_time = datetime.now()
    best_params, best_score, results = custom_grid_search(X, y, param_grid)
    end_time = datetime.now()
    
    print(f"\n{Colors.HEADER}" + "="*50 + Colors.ENDC)
    print(f"{Colors.BOLD}FINAL RESULTS{Colors.ENDC}")
    print(f"{Colors.HEADER}" + "="*50 + f"{Colors.ENDC}")
    print(f"\n{Colors.BLUE}Best parameters found:{Colors.ENDC}")
    for param, value in best_params.items():
        print(f"{Colors.YELLOW}{param}: {Colors.ENDC}{value}")
    print(f"\n{Colors.GREEN}Best accuracy: {best_score:.4f}{Colors.ENDC}")
    print(f"\n{Colors.BLUE}Total execution time: {Colors.BOLD}{end_time - start_time}{Colors.ENDC}")
    
    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'./models/grid_search_results_{timestamp}.csv'
    
    # Convert results for saving
    results_df = pd.DataFrame([{
        'hidden_layers': str(r['params']['hidden_layers']),
        'learning_rate': r['params']['learning_rate'],
        'dropout_rate': r['params']['dropout_rate'],
        'batch_size': r['params']['batch_size'],
        'epochs': r['params']['epochs'],
        'mean_score': r['mean_score'],
        'std_score': r['std_score']
    } for r in results])
    
    results_df.to_csv(results_path, index=False)
    
    # Save best model parameters separately
    best_params_df = pd.DataFrame([{
        'timestamp': timestamp,
        'accuracy': best_score,
        'hidden_layers': str(best_params['hidden_layers']),
        'learning_rate': best_params['learning_rate'],
        'dropout_rate': best_params['dropout_rate'],
        'batch_size': best_params['batch_size'],
        'epochs': best_params['epochs']
    }])
    
    best_params_path = './models/best_params.csv'
    
    # Append or create best parameters history
    if os.path.exists(best_params_path):
        best_params_history = pd.read_csv(best_params_path)
        best_params_df = pd.concat([best_params_history, best_params_df], ignore_index=True)
    
    best_params_df.to_csv(best_params_path, index=False)
    print(f"\n{Colors.GREEN}Results saved to {results_path}{Colors.ENDC}")
    print(f"{Colors.GREEN}Best parameters history updated in {best_params_path}{Colors.ENDC}")

if __name__ == "__main__":
    main()