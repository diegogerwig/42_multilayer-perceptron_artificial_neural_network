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

def train_model(features, Y_train, network_config):
    """
    Train the neural network model.
    
    Parameters:
        features: numpy array of features (without ID and Diagnosis)
        Y_train: numpy array of one-hot encoded target values
        network_config: dictionary containing network configuration
            - layers: list of hidden layer dimensions
            - epochs: number of training epochs
            - learning_rate: initial learning rate
            - batch_size: size of mini-batches
            - dropout_rate: dropout probability
    """
    # Select features based on correlation
    print("\nSelecting features based on correlation with target...")
    X_train, _ = select_features_train(features, Y_train[:, 1].reshape(-1, 1), features)
    
    # Standardize features
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - X_mean) / X_std
    
    # Save normalization parameters
    normalization_params = {'mean': X_mean.tolist(), 'std': X_std.tolist()}
    with open('./models/normalization_params.json', 'w') as f:
        json.dump(normalization_params, f)
    
    print(f'Training data shape after feature selection: {X_train.shape}')
    
    # Initialize model
    layer_dims = [X_train.shape[1]] + network_config['layers'] + [2]  # 2 output neurons for softmax
    parameters = init_parameters(layer_dims)
    
    # Save model topology
    os.makedirs('./models', exist_ok=True)
    with open('./models/model_topology.json', 'w') as f:
        json.dump({'layers': layer_dims, 'activation': 'softmax'}, f)
    
    # Save training history and plot
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_loss = float('inf')
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
            
            # Forward and backward passes
            cache = forward_propagation(batch_X, parameters, training=True, dropout_rate=network_config['dropout_rate'])
            grads = backward_propagation(parameters, cache, batch_X, batch_Y, dropout_rate=network_config['dropout_rate'])
            
            # Update parameters
            for key in parameters:
                if f'd{key}' in grads:
                    parameters[key] -= learning_rate * grads[f'd{key}']
        
        # Calculate loss
        cache = forward_propagation(X_train, parameters, training=False)
        train_loss = categorical_cross_entropy(Y_train, cache[f'A{len(layer_dims)-1}'], parameters)
        train_predictions = np.argmax(cache[f'A{len(layer_dims)-1}'], axis=1)
        train_acc = np.mean(train_predictions == Y_train.argmax(axis=1))
        
        # Save metrics for plotting
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        if epoch % 10 == 0:
            print(f'epoch {epoch+1:04d}/{network_config["epochs"]} - loss: {train_loss:.4f} - acc: {train_acc:.4f}')
        
        # Early stopping check
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
    
    # Save model and plot learning curves
    network_info = {
        'layers': layer_dims,
        'lr': learning_rate,
        'batch_size': network_config['batch_size'],
        'dropout_rate': network_config['dropout_rate']
    }
    
    # Save history for later plotting
    with open('./models/training_history.json', 'w') as f:
        json.dump({
            'history': {k: [float(v) for v in vals] for k, vals in history.items()},
            'network_info': network_info
        }, f)
    
    # Import and use plot function
    from plot import plot_learning_curves
    plot_learning_curves(history, network_info)
    
    print("\n> Training completed")
    print("> Model saved to './models/model_params.pkl'")
    print("> Model topology saved to './models/model_topology.json'")
    print("> Feature normalization parameters saved to './models/normalization_params.json'")
    print("> Training history saved to './models/training_history.json'")
    print("> Learning curves saved to './plots/learning_curves.png'")

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
    print(f"- Training data:  {args.train_data}")
    print(f"- Hidden layers:  {args.layers}")
    print(f"- Epochs:         {args.epochs}")
    print(f"- Learning rate:  {args.learning_rate}")
    print(f"- Batch size:     {args.batch_size}")
    print(f"- Dropout rate:   {args.dropout_rate}")
    
    # Load data and extract features/target
    train_data = pd.read_csv(args.train_data, header=None)
    
    # Extract features and target
    features = train_data.iloc[:, 2:].values  # Skip ID and Diagnosis
    diagnosis = train_data.iloc[:, 1].values  # Get Diagnosis column
    
    print(f"\nOriginal feature shape: {features.shape}")
    
    # Convert target to one-hot encoding
    Y_train = np.zeros((len(train_data), 2))
    Y_train[diagnosis == 'M', 1] = 1  # Malignant
    Y_train[diagnosis == 'B', 0] = 1  # Benign
    
    # Create network configuration dictionary
    network_config = {
        'layers': args.layers,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'dropout_rate': args.dropout_rate
    }
    
    train_model(features, Y_train, network_config)

if __name__ == "__main__":
    main()