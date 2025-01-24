#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from itertools import product
import json
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import logging
import time

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
class GridSearch:
    def __init__(self, model_builder, param_grid, n_splits=3):
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
        width = 50  # Reduced from 70 for cleaner output
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
                  f"Loss: {loss_color}{loss:.4f}{Colors.ENDC}")
        else:
            msg = (f"[{Colors.BLUE}{timestamp}{Colors.ENDC}] [{bar}] {current}/{total} | "
                  f"{eta_str}"
                  f"Testing configuration {current}")
            
        print(f"\r{msg}", end="")
        if is_best:
            print(f"\n{Colors.GREEN}[NEW BEST] Loss: {loss:.4f}{Colors.ENDC}")
        
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
            
            # Use only 3 folds instead of all for faster evaluation
            for fold, (train_idx, val_idx) in enumerate(list(cv.split(X, y))[:3], 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Build and train model
                model = self.model_builder.build_model(X.shape[1], params)
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,  # Reduced patience
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.LearningRateScheduler(
                        self.model_builder.create_lr_schedule(params['learning_rate'])
                    )
                ]
                
                # Faster training with reduced epochs
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,  # Reduced from 200
                    batch_size=params['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                )
                
                best_val_score = min(history.history['val_loss'])
                fold_scores.append(best_val_score)
                
                # Early stopping for the whole fold if first results are poor
                if fold == 1 and best_val_score > 0.5:
                    break
            
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
            
            # Early stopping for the whole grid search if we find a very good model
            if best_score < 0.25:
                print(f"\n{Colors.GREEN}Found excellent model early. Stopping search.{Colors.ENDC}")
                break
            
        return best_params, best_score, results