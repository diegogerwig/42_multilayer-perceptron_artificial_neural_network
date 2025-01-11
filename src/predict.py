#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import json
import os
import pickle
from model import forward_propagation
from metrics import evaluate_predictions, categorical_cross_entropy
from feature_selection import select_features_predict
from plot import plot_learning_curves

def load_model(model_path='./models/model_params.pkl'):
    """
    Load model parameters and configuration
    """
    print("\nLoading model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model parameters file not found at {model_path}")
    
    # Load model parameters
    with open(model_path, 'rb') as f:
        parameters = pickle.load(f)
    
    # Load model topology
    with open('./models/model_topology.json', 'r') as f:
        topology = json.load(f)
    
    # Load normalization parameters
    with open('./models/normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    
    print("Model loaded successfully")
    return parameters, topology, norm_params

def load_plot_data():
    """
    Load training history and network information for plotting
    """
    try:
        with open('./models/training_history.json', 'r') as f:
            data = json.load(f)
            history = data['history']
            network_info = data['network_info']
        return history, network_info
    except Exception as e:
        raise Exception(f"Could not load training history: {str(e)}")

def predict(X, parameters):
    """
    Make predictions using trained model
    """
    cache = forward_propagation(X, parameters, training=False)
    L = len([key for key in parameters.keys() if key.startswith('W')])
    predictions = np.argmax(cache[f'A{L}'], axis=1)
    probabilities = cache[f'A{L}']
    return predictions, probabilities

def main():
    parser = argparse.ArgumentParser(
        description='Make predictions using the trained neural network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--test_data', 
        required=True,
        help='Path to test data CSV file'
    )
    parser.add_argument(
        '--model_params', 
        default='./models/model_params.pkl',
        help='Path to the model parameters file'
    )
    args = parser.parse_args()
    
    try:
        # Load model and its configuration
        parameters, topology, norm_params = load_model(args.model_params)
        
        # Load and prepare data
        print(f"\nProcessing dataset: {args.dataset}")
        data = pd.read_csv(args.dataset, header=None)
        X = data.iloc[:, 2:].values  # Skip ID and Diagnosis columns

        # Convert labels to one-hot encoding
        Y = np.zeros((len(data), 2))
        Y[data.iloc[:, 1] == 'M', 1] = 1  # Malignant
        Y[data.iloc[:, 1] == 'B', 0] = 1  # Benign
        
        # Select features
        X = select_features_predict(X)
        print(f"Input shape after feature selection: {X.shape}")
        
        # Normalize features
        X = (X - np.array(norm_params['mean'])) / np.array(norm_params['std'])
        
        # Make predictions
        print("\nMaking predictions...")
        predictions, probabilities = predict(X, parameters)
        
        # Calculate loss and metrics
        loss = categorical_cross_entropy(Y, probabilities)
        print(f'\nCategorical Cross-Entropy Loss: {loss:.4f}')
        
        # Evaluate predictions
        metrics = evaluate_predictions(predictions, Y)
        
        # Update and show plot
        try:
            # Load training history
            history, network_info = load_plot_data()
            
            # Update history with test metrics
            history['test_loss'] = loss
            history['test_acc'] = metrics['accuracy']
            
            # Create and show plot
            plot_learning_curves(history, network_info)
            print("\nLearning curves plot has been updated with test results")
            
        except Exception as e:
            print(f"\nWarning: Could not create learning curves plot: {str(e)}")
        
        # Print detailed predictions
        print("\nDetailed predictions for first 5 samples:")
        print("ID | True Label | Predicted | Confidence")
        print("-" * 45)
        for i in range(min(5, len(predictions))):
            true_label = "M" if Y[i, 1] == 1 else "B"
            pred_label = "M" if predictions[i] == 1 else "B"
            confidence = probabilities[i, predictions[i]]
            print(f"{data.iloc[i, 0]:<3} | {true_label:^10} | {pred_label:^9} | {confidence:^10.4f}")

    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()