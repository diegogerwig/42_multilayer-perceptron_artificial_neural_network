#!/usr/bin/env python3

import os
import sys
import time
import warnings
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from utils.feature_selection import select_features_train
import absl.logging
import json

# Import configurations
from utils.config import PARAM_GRID, TRAINING_CONFIG, MODEL_CONFIG, DATA_CONFIG, LOG_CONFIG

# Suppress all warnings and logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
absl.logging.set_verbosity(absl.logging.ERROR)

# Configure TensorFlow environment
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',           # Disable all TF logging
    'CUDA_VISIBLE_DEVICES': '-1',          # Force CPU
    'TF_ENABLE_ONEDNN_OPTS': '0',          # Disable oneDNN
    'TF_SILENCE_CUDA_WARNINGS': '1',       # Silence CUDA
    'AUTOGRAPH_VERBOSITY': '0',            # Disable autograph
    'KMP_WARNINGS': 'off',                 # Disable KMP
    'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',
    'TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE': 'False'
})

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class ModelBuilder:
    @staticmethod
    def build_model(input_shape, params):
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(
                params['hidden_layers'][0],
                activation='gelu' if params['use_gelu'] else 'relu',
                kernel_regularizer=l2(params['lambda_reg'])
            ),
            Dropout(params['dropout_rate']),
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
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    @staticmethod
    def create_lr_schedule(initial_lr):
        def lr_schedule(epoch):
            return float(initial_lr * (1 / (1 + epoch * 0.01)))
        return lr_schedule

class GridSearch:
    def __init__(self, model_builder, param_grid, n_splits=5):
        self.model_builder = model_builder
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.start_time = None
        self.last_update = 0
        
    def format_time(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m"
        else:
            return f"{int(minutes)}m"
            
    def print_progress(self, current, total, params, loss=None, is_best=False):
        width = 70
        progress = current / total
        filled = int(width * progress)
        bar = Colors.BLUE + '=' * filled + Colors.YELLOW + '-' * (width - filled) + Colors.ENDC
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Calculate ETA
        if self.start_time is None:
            self.start_time = time.time()
        
        eta_str = ""
        if current > 1:
            elapsed = time.time() - self.start_time
            iterations_left = total - current
            time_per_iteration = elapsed / (current - 1)
            eta_seconds = int(iterations_left * time_per_iteration)
            eta_str = f"ETA: {Colors.YELLOW}{self.format_time(eta_seconds)}{Colors.ENDC} | "
        
        loss_color = Colors.GREEN if loss and loss < 0.3 else Colors.YELLOW if loss and loss < 0.5 else Colors.RED
        
        if loss is not None:
            msg = (f"[{Colors.BLUE}{timestamp}{Colors.ENDC}] [{bar}] {current}/{total} | "
                  f"{eta_str}"
                  f"Loss: {loss_color}{loss:.4f}{Colors.ENDC} | "
                  f"lr={params['learning_rate']}, layers={params['hidden_layers']}")
        else:
            msg = (f"[{Colors.BLUE}{timestamp}{Colors.ENDC}] [{bar}] {current}/{total} | "
                  f"{eta_str}"
                  f"Evaluating: lr={params['learning_rate']}, layers={params['hidden_layers']}")
            
        print(f"\r{msg}", end="")
        if is_best:
            print(f"\n{Colors.GREEN}[NEW BEST RESULT] Loss: {loss:.4f}{Colors.ENDC}")
        
    def run(self, X, y):
        param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                            for v in product(*self.param_grid.values())]
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        best_score = float('inf')
        best_params = None
        results = []
        
        for idx, params in enumerate(param_combinations, 1):
            self.print_progress(idx, len(param_combinations), params)
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Build and train model
                model = self.model_builder.build_model(X.shape[1], params)
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=TRAINING_CONFIG['early_stopping']['patience'],
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.LearningRateScheduler(
                        self.model_builder.create_lr_schedule(params['learning_rate'])
                    )
                ]
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                )
                
                best_val_score = min(history.history['val_loss'])
                fold_scores.append(best_val_score)
            
            mean_score = np.mean(fold_scores)
            is_best = mean_score < best_score
            
            self.print_progress(idx, len(param_combinations), params, mean_score, is_best)
            print()
            
            if is_best:
                best_score = mean_score
                best_params = params.copy()
            
            results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': np.std(fold_scores)
            })
            
        return best_params, best_score, results

def main():
    parser = argparse.ArgumentParser(
        description='Neural Network Hyperparameter Grid Search',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data',
        required=True,
        help='Path to training data CSV file'
    )
    
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}🔍 Grid search{Colors.ENDC}")
    
    try:
        print(f"\n{Colors.BLUE}Loading data...{Colors.ENDC}")
        data = pd.read_csv(args.data, header=None)
        features = data.iloc[:, 2:].values
        diagnosis = data.iloc[:, 1].values
        
        # Convert labels
        y = np.where(diagnosis == 'M', 1, 0).reshape(-1, 1)
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        
        # Feature selection
        print(f"{Colors.BLUE}Selecting features...{Colors.ENDC}")
        X_selected, selected_features = select_features_train(X, y, X.copy())
        
        # Save scaler and selected features
        os.makedirs(LOG_CONFIG['model_dir'], exist_ok=True)
        pd.to_pickle(scaler, os.path.join(LOG_CONFIG['model_dir'], 'scaler.pkl'))
        
        # Save selected features indices
        selected_features_path = os.path.join(LOG_CONFIG['model_dir'], 'selected_features.json')
        with open(selected_features_path, 'w') as f:
            json.dump({
                'selected_indices': selected_features.tolist() if isinstance(selected_features, np.ndarray) else selected_features,
                'n_features': len(selected_features)
            }, f, indent=4)
            
        # Try both approaches: full dataset and selected features
        print(f"\n{Colors.HEADER}Comparing full dataset vs selected features:{Colors.ENDC}")
        print(f"{Colors.BLUE}Full dataset:{Colors.ENDC} {Colors.BOLD}{X.shape[0]}{Colors.ENDC} samples, "
              f"{Colors.BOLD}{X.shape[1]}{Colors.ENDC} features")
        print(f"{Colors.BLUE}Selected features:{Colors.ENDC} {Colors.BOLD}{X_selected.shape[0]}{Colors.ENDC} samples, "
              f"{Colors.BOLD}{X_selected.shape[1]}{Colors.ENDC} features")
        
        # Create smaller parameter grid for comparison
        comparison_grid = {
            'hidden_layers': [[32, 16]],  # Use one architecture
            'learning_rate': [0.001],     # Use one learning rate
            'dropout_rate': [0.15],       # Use one dropout rate
            'batch_size': [64],           # Use one batch size
            'epochs': [200],
            'lambda_reg': [0.001],
            'use_gelu': [True]
        }
        
        print(f"\n{Colors.HEADER}Running comparison...{Colors.ENDC}")
        model_builder = ModelBuilder()
        
        # Test with full dataset
        print(f"\n{Colors.BLUE}Testing full dataset:{Colors.ENDC}")
        grid_search_full = GridSearch(
            model_builder, 
            comparison_grid, 
            n_splits=TRAINING_CONFIG['cross_validation']['n_splits']
        )
        _, full_score, _ = grid_search_full.run(X, y)
        
        # Test with selected features
        print(f"\n{Colors.BLUE}Testing selected features:{Colors.ENDC}")
        grid_search_selected = GridSearch(
            model_builder, 
            comparison_grid, 
            n_splits=TRAINING_CONFIG['cross_validation']['n_splits']
        )
        _, selected_score, _ = grid_search_selected.run(X_selected, y)
        
        # Compare results
        print(f"\n{Colors.HEADER}Comparison Results:{Colors.ENDC}")
        print(f"{Colors.BLUE}Full dataset loss:{Colors.ENDC} {Colors.BOLD}{full_score:.4f}{Colors.ENDC}")
        print(f"{Colors.BLUE}Selected features loss:{Colors.ENDC} {Colors.BOLD}{selected_score:.4f}{Colors.ENDC}")
        
        # Use the better approach for the full grid search
        if selected_score < full_score:
            print(f"\n{Colors.GREEN}Selected features perform better. Using selected features for grid search.{Colors.ENDC}")
            X_final = X_selected
            is_using_selected = True
        else:
            print(f"\n{Colors.GREEN}Full dataset performs better. Using full dataset for grid search.{Colors.ENDC}")
            X_final = X
            is_using_selected = False
            
        print(f"\n{Colors.HEADER}Starting full grid search with best approach...{Colors.ENDC}")
        
        # Calculate total combinations
        n_combinations = sum(1 for _ in product(*PARAM_GRID.values()))
        print(f"Evaluating {Colors.BOLD}{n_combinations}{Colors.ENDC} parameter combinations...")
        print(f"Using {Colors.BOLD}{TRAINING_CONFIG['cross_validation']['n_splits']}-fold{Colors.ENDC} cross validation\n")
        
        # Run grid search with best approach
        model_builder = ModelBuilder()
        grid_search = GridSearch(
            model_builder, 
            PARAM_GRID, 
            n_splits=TRAINING_CONFIG['cross_validation']['n_splits']
        )
        best_params, best_score, results = grid_search.run(X_final, y)
        
        # Save results
        os.makedirs(LOG_CONFIG['results_dir'], exist_ok=True)
        results_df = pd.DataFrame([{
            **r['params'],
            'mean_score': r['mean_score'],
            'std_score': r['std_score']
        } for r in results])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(LOG_CONFIG['results_dir'], f'grid_search_results_{timestamp}.csv')
        results_df.to_csv(results_path, index=False)
        
        # Save best parameters
        best_params_path = os.path.join(LOG_CONFIG['model_dir'], 'best_params.json')
        best_params_data = {
            'timestamp': timestamp,
            'best_score': float(best_score),
            'best_params': best_params,
            'feature_selection': {
                'used_selected_features': is_using_selected,
                'full_dataset_score': float(full_score),
                'selected_features_score': float(selected_score),
                'n_original_features': X.shape[1],
                'n_selected_features': X_selected.shape[1],
                'selected_features_indices': selected_features.tolist() if isinstance(selected_features, np.ndarray) else selected_features
            },
            'dataset_info': {
                'n_samples': X_final.shape[0],
                'n_features': X_final.shape[1],
            },
            'training_config': TRAINING_CONFIG
        }
        
        with open(best_params_path, 'w') as f:
            json.dump(best_params_data, f, indent=4)
        
        # Print final results
        print(f"\n{Colors.HEADER}Grid Search completed!{Colors.ENDC}")
        print(f"\n{Colors.BLUE}Best parameters found:{Colors.ENDC}")
        for param, value in best_params.items():
            print(f"- {param}: {value}")
        print(f"{Colors.GREEN}Best validation score: {best_score:.4f}{Colors.ENDC}")
        print(f"\n{Colors.GREEN}Results saved to: {results_path}{Colors.ENDC}")
        print(f"{Colors.GREEN}Best parameters saved to: {best_params_path}{Colors.ENDC}")
        
    except Exception as e:
        print(f"\n{Colors.RED}An error occurred: {str(e)}{Colors.ENDC}")
        raise

if __name__ == "__main__":
    main()