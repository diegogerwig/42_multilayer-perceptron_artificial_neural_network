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

class GridSearch:
    def __init__(self, model_builder, param_grid, training_config, log_config):
        self.model_builder = model_builder
        self.param_grid = param_grid
        self.training_config = training_config
        self.log_config = log_config
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.log_config['log_dir'], exist_ok=True)
        os.makedirs(self.log_config['model_dir'], exist_ok=True)
        os.makedirs(self.log_config['results_dir'], exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(self.log_config['log_dir'], 
                    f'grid_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
                ),
                logging.StreamHandler()
            ]
        )
    
    def _get_callbacks(self, params):
        """Create callbacks for model training"""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.training_config['early_stopping']['patience'],
            min_delta=self.training_config['early_stopping']['min_delta'],
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate schedule
        lr_schedule = self.model_builder.create_lr_schedule(
            initial_lr=params['learning_rate'],
            **self.training_config['learning_rate_schedule']
        )
        callbacks.append(LearningRateScheduler(lr_schedule))
        
        return callbacks
    
    def run(self, X, y):
        """
        Run grid search with cross-validation
        
        Args:
            X: Input features
            y: Target variables
        """
        param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                            for v in product(*self.param_grid.values())]
        cv = StratifiedKFold(**self.training_config['cross_validation'])
        
        results = []
        best_score = float('inf')
        best_params = None
        
        logging.info(f"Starting grid search with {len(param_combinations)} combinations")
        
        for idx, params in enumerate(param_combinations, 1):
            logging.info(f"\nEvaluating combination {idx}/{len(param_combinations)}")
            logging.info(f"Parameters: {params}")
            
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Build and train model
                model = self.model_builder.build_model(X.shape[1], params)
                callbacks = self._get_callbacks(params)
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Get best validation score
                best_val_score = min(history.history['val_loss'])
                fold_scores.append(best_val_score)
                
                logging.info(f"Fold {fold} validation loss: {best_val_score:.4f}")
            
            # Calculate mean and std of scores
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            logging.info(f"Mean validation loss: {mean_score:.4f} (±{std_score:.4f})")
            
            # Save results
            results.append({
                'params': params,
                'mean_score': float(mean_score),
                'std_score': float(std_score)
            })
            
            # Update best parameters if needed
            if mean_score < best_score:
                best_score = mean_score
                best_params = params
                logging.info(f"New best score: {best_score:.4f}")
        
        # Save results
        self._save_results(results, best_params, best_score)
        
        return best_params, best_score, results
    
    def _save_results(self, results, best_params, best_score):
        """Save grid search results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all results
        results_df = pd.DataFrame([{
            **r['params'],
            'mean_score': r['mean_score'],
            'std_score': r['std_score']
        } for r in results])
        
        results_path = os.path.join(
            self.log_config['results_dir'],
            f'grid_search_results_{timestamp}.csv'
        )
        results_df.to_csv(results_path, index=False)
        
        # Save best parameters
        best_params_path = os.path.join(
            self.log_config['results_dir'],
            'best_params.json'
        )
        
        with open(best_params_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'best_score': float(best_score),
                'best_params': best_params
            }, f, indent=4)
        
        logging.info(f"Results saved to {results_path}")
        logging.info(f"Best parameters saved to {best_params_path}")