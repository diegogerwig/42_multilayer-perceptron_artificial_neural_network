#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import json
import os
import pickle
from model import init_parameters, forward_propagation, backward_propagation
from metrics import categorical_cross_entropy
from feature_selection import select_features_train
from plot import plot_learning_curves

def train_model(features, Y_train, val_features, Y_val, network_config):
    """
    Train the neural network model using both training and validation data.
    """
    print("\nSelecting features based on correlation with target...")
    X_train, selected_features = select_features_train(features, Y_train[:, 1].reshape(-1, 1), features)
    
    # Cargar los índices seleccionados
    with open('./models/selected_features.json', 'r') as f:
        data = json.load(f)
        if isinstance(data, dict):
            feature_indices = data.get('selected_indices', [])
        else:
            feature_indices = data
    
    print(f"Using {len(feature_indices)} selected features")
    
    # Seleccionar las mismas características para validación
    X_val = val_features[:, feature_indices]
    
    # Standardize features
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    
    # Save normalization parameters
    normalization_params = {'mean': X_mean.tolist(), 'std': X_std.tolist()}
    with open('./models/normalization_params.json', 'w') as f:
        json.dump(normalization_params, f)
    
    print(f'Training data shape after feature selection: {X_train.shape}')
    print(f'Validation data shape after feature selection: {X_val.shape}')
    
    # Initialize model
    layer_dims = [X_train.shape[1]] + network_config['layers'] + [2]
    parameters = init_parameters(layer_dims)
    
    # Save model topology
    os.makedirs('./models', exist_ok=True)
    with open('./models/model_topology.json', 'w') as f:
        json.dump({'layers': layer_dims, 'activation': 'softmax'}, f)
    
    # Initialize history with validation metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 50
    min_delta = 0.001
    patience_counter = 0
    initial_lr = network_config['learning_rate']
    decay_rate = 0.01
    
    print("\nTraining started...")
    for epoch in range(network_config['epochs']):
        learning_rate = initial_lr / (1 + decay_rate * epoch)
        indices = np.random.permutation(X_train.shape[0])
        
        # Mini-batch training
        for i in range(0, X_train.shape[0], network_config['batch_size']):
            batch_indices = indices[i:min(i + network_config['batch_size'], X_train.shape[0])]
            batch_X = X_train[batch_indices]
            batch_Y = Y_train[batch_indices]
            
            cache = forward_propagation(batch_X, parameters, training=True, dropout_rate=network_config['dropout_rate'])
            grads = backward_propagation(parameters, cache, batch_X, batch_Y, dropout_rate=network_config['dropout_rate'])
            
            for key in parameters:
                if f'd{key}' in grads:
                    parameters[key] -= learning_rate * grads[f'd{key}']
        
        # Calculate training metrics
        cache = forward_propagation(X_train, parameters, training=False)
        train_loss = categorical_cross_entropy(Y_train, cache[f'A{len(layer_dims)-1}'])
        train_predictions = np.argmax(cache[f'A{len(layer_dims)-1}'], axis=1)
        train_acc = np.mean(train_predictions == Y_train.argmax(axis=1))
        
        # Calculate validation metrics
        val_cache = forward_propagation(X_val, parameters, training=False)
        val_loss = categorical_cross_entropy(Y_val, val_cache[f'A{len(layer_dims)-1}'])
        val_predictions = np.argmax(val_cache[f'A{len(layer_dims)-1}'], axis=1)
        val_acc = np.mean(val_predictions == Y_val.argmax(axis=1))
        
        # Save metrics
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        
        if epoch % 10 == 0:
            print(f'epoch {epoch+1:04d}/{network_config["epochs"]} - '
                  f'loss: {train_loss:.4f} - acc: {train_acc:.4f} - '
                  f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
        
        # Early stopping check with validation loss
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            with open('./models/model_params.pkl', 'wb') as f:
                pickle.dump(parameters, f)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
    
    # Save final history for plotting
    network_info = {
        'layers': layer_dims,
        'lr': learning_rate,
        'batch_size': network_config['batch_size'],
        'dropout_rate': network_config['dropout_rate']
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
    
    # Generate and show plot
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
    parser.add_argument(
        '--val_data',
        required=True,
        help='Path to validation data CSV file'
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=[64, 32, 16],
        help='Hidden layer dimensions. Example: --layers 64 32 16'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate for gradient descent'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Number of samples per gradient update'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.5,
        choices=range(0, 100),
        metavar="[0-1]",
        help='Dropout rate for regularization (between 0 and 1)'
    )
    
    args = parser.parse_args()
    
    print("\nTraining configuration:")
    print(f"- Training data:    {args.train_data}")
    print(f"- Validation data:  {args.val_data}")
    print(f"- Hidden layers:    {args.layers}")
    print(f"- Epochs:           {args.epochs}")
    print(f"- Learning rate:    {args.learning_rate}")
    print(f"- Batch size:       {args.batch_size}")
    print(f"- Dropout rate:     {args.dropout_rate}")
    
    # Load training data
    train_data = pd.read_csv(args.train_data, header=None)
    features = train_data.iloc[:, 2:].values  # Skip ID and Diagnosis
    diagnosis = train_data.iloc[:, 1].values  # Get Diagnosis column
    print(f"\nOriginal training feature shape: {features.shape}")
    
    # Load validation data
    val_data = pd.read_csv(args.val_data, header=None)
    val_features = val_data.iloc[:, 2:].values  # Skip ID and Diagnosis
    val_diagnosis = val_data.iloc[:, 1].values  # Get Diagnosis column
    print(f"Original validation feature shape: {val_features.shape}")
    
    # Convert targets to one-hot encoding
    Y_train = np.zeros((len(train_data), 2))
    Y_train[diagnosis == 'M', 1] = 1  # Malignant
    Y_train[diagnosis == 'B', 0] = 1  # Benign
    
    Y_val = np.zeros((len(val_data), 2))
    Y_val[val_diagnosis == 'M', 1] = 1  # Malignant
    Y_val[val_diagnosis == 'B', 0] = 1  # Benign
    
    # Create network configuration dictionary
    network_config = {
        'layers': args.layers,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'dropout_rate': args.dropout_rate
    }
    
    train_model(features, Y_train, val_features, Y_val, network_config)

if __name__ == "__main__":
    main()