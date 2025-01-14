#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import json
import os
import pickle
from utils.model import (
    init_parameters, forward_propagation, backward_propagation,
    label_smoothing
)
from utils.optimizer import AdamOptimizer
from utils.metrics import binary_cross_entropy
from utils.feature_selection import select_features_train
from utils.plot import plot_learning_curves

def compute_cost_improved(Y, A, parameters, lambda_reg=0.01):
    """
    Compute cost with improvements:
    - Label smoothing
    - Adaptive L2 regularization
    """
    m = Y.shape[0]
    Y_smooth = label_smoothing(Y)
    
    # Cross entropy with label smoothing
    cross_entropy = -np.sum(Y_smooth * np.log(A + 1e-8)) / m
    
    # Adaptive L2 regularization based on layer size
    l2_reg = 0
    for l in range(1, len([k for k in parameters.keys() if k.startswith('W')]) + 1):
        layer_size = parameters[f'W{l}'].shape[0]
        adaptive_lambda = lambda_reg * np.sqrt(layer_size) / 100
        l2_reg += adaptive_lambda * np.sum(np.square(parameters[f'W{l}']))
    
    return cross_entropy + l2_reg

def train_model(features, Y_train, val_features=None, Y_val=None, network_config=None):
    """
    Train the neural network model with all improvements
    """
    print("\nSelecting features...")
    X_train, selected_features = select_features_train(features, Y_train[:, 1].reshape(-1, 1), features)
    
    # Get selected features
    with open('./models/selected_features.json', 'r') as f:
        data = json.load(f)
        if isinstance(data, dict):
            feature_indices = data.get('selected_indices', [])
        else:
            feature_indices = data
    
    print(f"Using {len(feature_indices)} selected features")
    
    # Apply feature selection to validation data if provided
    has_validation = val_features is not None and Y_val is not None
    if has_validation:
        X_val = val_features[:, feature_indices]
    
    # Standardize features
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - X_mean) / X_std
    if has_validation:
        X_val = (X_val - X_mean) / X_std
    
    # Save normalization parameters
    normalization_params = {'mean': X_mean.tolist(), 'std': X_std.tolist()}
    with open('./models/normalization_params.json', 'w') as f:
        json.dump(normalization_params, f)
    
    print(f'Training data shape after feature selection:   {X_train.shape}')
    if has_validation:
        print(f'Validation data shape after feature selection: {X_val.shape}')
    
    # Initialize model
    layer_dims = [X_train.shape[1]] + network_config['layers'] + [2]
    parameters = init_parameters(layer_dims)
    
    # Initialize Adam optimizer
    optimizer = AdamOptimizer(
        learning_rate=network_config['learning_rate'],
        beta1=0.9,
        beta2=0.999
    )
    
    # Save model topology
    os.makedirs('./models', exist_ok=True)
    with open('./models/model_topology.json', 'w') as f:
        json.dump({
            'layers': layer_dims,
            'activation': 'softmax',
            'use_gelu': network_config.get('use_gelu', False)
        }, f)
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'learning_rates': []
    }
    if has_validation:
        history.update({
            'val_loss': [],
            'val_acc': []
        })
    
    # Training configuration
    best_loss = float('inf')
    patience = 50
    min_delta = 0.001
    patience_counter = 0
    
    # Learning rate schedule configuration
    warmup_epochs = 5
    cycle_length = 10
    initial_lr = network_config['learning_rate']
    min_lr = initial_lr * 0.1
    use_gelu = network_config.get('use_gelu', False)
    
    print("\nTraining configuration:")
    print(f"- Learning rate:    {initial_lr}")
    print(f"- Batch size:       {network_config['batch_size']}")
    print(f"- Dropout rate:     {network_config['dropout_rate']}")
    print(f"- Use GELU:         {use_gelu}")
    print(f"- Warmup epochs:    {warmup_epochs}")
    
    print("\nTraining started...")
    for epoch in range(network_config['epochs']):
        # Learning rate schedule
        if epoch < warmup_epochs:
            # Warmup period - linear increase
            current_lr = initial_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing with warm restarts
            cycle_progress = (epoch - warmup_epochs) % cycle_length
            current_lr = min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * cycle_progress / cycle_length)) / 2
        
        optimizer.learning_rate = current_lr
        history['learning_rates'].append(current_lr)
        
        # Shuffle training data
        indices = np.random.permutation(X_train.shape[0])
        
        # Mini-batch training
        for i in range(0, X_train.shape[0], network_config['batch_size']):
            batch_indices = indices[i:min(i + network_config['batch_size'], X_train.shape[0])]
            batch_X = X_train[batch_indices]
            batch_Y = Y_train[batch_indices]
            
            # Forward propagation
            cache = forward_propagation(
                batch_X, 
                parameters, 
                training=True, 
                dropout_rate=network_config['dropout_rate'],
                use_gelu=use_gelu
            )
            
            # Backward propagation
            grads = backward_propagation(
                parameters, 
                cache, 
                batch_X, 
                batch_Y,
                lambda_reg=network_config.get('lambda_reg', 0.01),
                dropout_rate=network_config['dropout_rate'],
                use_gelu=use_gelu
            )
            
            # Update parameters using Adam
            parameters = optimizer.update(parameters, grads)
        
        # Calculate training metrics
        cache = forward_propagation(X_train, parameters, training=False, use_gelu=use_gelu)
        train_loss = compute_cost_improved(
            Y_train, 
            cache[f'A{len(layer_dims)-1}'], 
            parameters,
            lambda_reg=network_config.get('lambda_reg', 0.01)
        )
        train_predictions = np.argmax(cache[f'A{len(layer_dims)-1}'], axis=1)
        train_acc = np.mean(train_predictions == Y_train.argmax(axis=1))
        
        # Save training metrics
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        
        # Calculate validation metrics if validation data exists
        if has_validation:
            val_cache = forward_propagation(X_val, parameters, training=False, use_gelu=use_gelu)
            val_loss = compute_cost_improved(
                Y_val, 
                val_cache[f'A{len(layer_dims)-1}'], 
                parameters,
                lambda_reg=network_config.get('lambda_reg', 0.01)
            )
            val_predictions = np.argmax(val_cache[f'A{len(layer_dims)-1}'], axis=1)
            val_acc = np.mean(val_predictions == Y_val.argmax(axis=1))
            
            history['val_loss'].append(float(val_loss))
            history['val_acc'].append(float(val_acc))
            
            if epoch % 10 == 0:
                print(f'epoch {epoch+1:04d}/{network_config["epochs"]} - '
                      f'lr: {current_lr:.6f} - '
                      f'loss: {train_loss:.4f} - acc: {train_acc:.4f} - '
                      f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
            
            # Early stopping check with validation loss
            if val_loss < (best_loss - min_delta):
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                with open('./models/model_params.pkl', 'wb') as f:
                    pickle.dump(parameters, f)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation loss: {best_loss:.4f}")
                    break
        else:
            # Early stopping check with training loss
            if epoch % 10 == 0:
                print(f'epoch {epoch+1:04d}/{network_config["epochs"]} - '
                      f'lr: {current_lr:.6f} - '
                      f'loss: {train_loss:.4f} - acc: {train_acc:.4f}')
            
            if train_loss < (best_loss - min_delta):
                best_loss = train_loss
                patience_counter = 0
                # Save best model
                with open('./models/model_params.pkl', 'wb') as f:
                    pickle.dump(parameters, f)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best training loss: {best_loss:.4f}")
                    break
    
    # Save final history
    network_info = {
        'layers': layer_dims,
        'lr': current_lr,
        'batch_size': network_config['batch_size'],
        'dropout_rate': network_config['dropout_rate'],
        'use_gelu': use_gelu,
        'lambda_reg': network_config.get('lambda_reg', 0.01)
    }
    
    with open('./models/training_history.json', 'w') as f:
        json.dump({
            'history': history,
            'network_info': network_info
        }, f)
    
    print("\n> Training completed")
    print("> Model saved to './models/model_params.pkl'")
    print("> Model topology saved to './models/model_topology.json'")
    print("> Feature normalization parameters saved to './models/normalization_params.json'")
    
    # Generate plot only if validation data exists
    if has_validation:
        print("\nGenerating learning curves plot...")
        plot_learning_curves(history, network_info)

def main():
    parser = argparse.ArgumentParser(
        description='Neural Network for Binary Classification of Breast Cancer Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--train_data',
        required=True,
        help='Path to training data CSV file'
    )

    # Optional arguments with defaults
    parser.add_argument(
        '--val_data',
        help='Path to validation data CSV file'
    )
    parser.add_argument(
        '--layer',
        type=int,
        nargs='+',
        default=[16, 8],  
        help='Hidden layer dimensions. Example: --layer 128 64 32'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2000,  # Más épocas para mejor convergencia
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0003,  # Learning rate más bajo para mejor estabilidad
        help='Initial learning rate for Adam optimizer'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,  # Batch size más grande para mejor generalización
        help='Number of samples per gradient update'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.15,  # Dropout más suave
        help='Dropout rate for regularization (between 0 and 1)'
    )
    parser.add_argument(
        '--lambda_reg',
        type=float,
        default=0.0001,  # Regularización L2 más suave
        help='L2 regularization parameter'
    )
    parser.add_argument(
        '--use_gelu',
        action='store_true',
        help='Use GELU activation instead of ReLU'
    )
    parser.add_argument(
        '--loss',
        default='binaryCrossentropy',
        help='Loss function to use'
    )
    
    args = parser.parse_args()
    
    print("\nTraining configuration:")
    print(f"- Training data:    {args.train_data}")
    print(f"- Validation data:  {args.val_data}")
    print(f"- Hidden layers:    {args.layer}")
    print(f"- Epochs:           {args.epochs}")
    print(f"- Learning rate:    {args.learning_rate}")
    print(f"- Batch size:       {args.batch_size}")
    print(f"- Dropout rate:     {args.dropout_rate}")
    print(f"- Lambda reg:       {args.lambda_reg}")
    print(f"- Use GELU:         {args.use_gelu}")
    print(f"- Loss function:    {args.loss}")
    
    # Load training data
    train_data = pd.read_csv(args.train_data, header=None)
    features = train_data.iloc[:, 2:].values  # Skip ID and Diagnosis
    diagnosis = train_data.iloc[:, 1].values  # Get Diagnosis column
    print(f"\nOriginal training feature shape:   {features.shape}")
    
    # Load validation data if provided
    val_features = None
    Y_val = None
    if args.val_data:
        val_data = pd.read_csv(args.val_data, header=None)
        val_features = val_data.iloc[:, 2:].values  # Skip ID and Diagnosis
        val_diagnosis = val_data.iloc[:, 1].values  # Get Diagnosis column
        print(f"Original validation feature shape: {val_features.shape}")
        
        # Convert validation targets to one-hot encoding
        Y_val = np.zeros((len(val_data), 2))
        Y_val[val_diagnosis == 'M', 1] = 1  # Malignant
        Y_val[val_diagnosis == 'B', 0] = 1  # Benign
    
    # Convert training targets to one-hot encoding
    Y_train = np.zeros((len(train_data), 2))
    Y_train[diagnosis == 'M', 1] = 1  # Malignant
    Y_train[diagnosis == 'B', 0] = 1  # Benign
    
    # Create network configuration dictionary
    network_config = {
        'layers': args.layer,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'dropout_rate': args.dropout_rate,
        'lambda_reg': args.lambda_reg,
        'use_gelu': args.use_gelu,
        'beta1': 0.9,    # Default Adam optimizer parameters
        'beta2': 0.999,
        'epsilon': 1e-8,
        'warmup_epochs': 10,  # Warmup period
        'min_lr': args.learning_rate * 0.1,
        'cycle_length': 20,   # Cosine annealing cycle length
        'label_smoothing': 0.05  # Label smoothing parameter
    }
    
    train_model(features, Y_train, val_features, Y_val, network_config)

if __name__ == "__main__":
    main()