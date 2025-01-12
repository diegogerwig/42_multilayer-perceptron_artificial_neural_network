#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import json
import os
import pickle
from model import forward_propagation
from metrics import evaluate_predictions, binary_cross_entropy
from feature_selection import select_features_predict
from plot import plot_learning_curves

def load_model(model_path='./models/model_params.pkl'):
    """Load model parameters and configuration."""
    print("\nLoading model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model parameters file not found at {model_path}")
    
    try:
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
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def predict(X, parameters):
    """Make predictions using trained model."""
    try:
        cache = forward_propagation(X, parameters, training=False)
        L = len([key for key in parameters.keys() if key.startswith('W')])
        predictions = np.argmax(cache[f'A{L}'], axis=1)
        probabilities = cache[f'A{L}']
        return predictions, probabilities
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

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
        # Check if test data exists
        if not os.path.exists(args.test_data):
            raise FileNotFoundError(f"Test data file not found: {args.test_data}")
            
        # Load model and its configuration
        parameters, topology, norm_params = load_model(args.model_params)
        
        # Load and prepare data
        print(f"\nProcessing dataset: {args.test_data}")
        test_df = pd.read_csv(args.test_data, header=None)
        X = test_df.iloc[:, 2:].values  # Skip ID and Diagnosis columns

        # Convert labels to one-hot encoding
        Y = np.zeros((len(test_df), 2))
        Y[test_df.iloc[:, 1] == 'M', 1] = 1  # Malignant
        Y[test_df.iloc[:, 1] == 'B', 0] = 1  # Benign
        
        # Select features
        print("\nSelecting features...")
        X = select_features_predict(X)
        print(f"Input shape after feature selection: {X.shape}")
        
        # Normalize features
        print("Normalizing features...")
        X = (X - np.array(norm_params['mean'])) / np.array(norm_params['std'])
        
        # Make predictions
        print("\nMaking predictions...")
        predictions, probabilities = predict(X, parameters)
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = evaluate_predictions(predictions, Y)
        test_loss = float(binary_cross_entropy(Y, probabilities))
        
        # Print metrics
        print("\nTest Results:")
        print(f"LOSS:      {test_loss:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"Confusion Matrix:")
        print(f"TRUE POS: {metrics['confusion_matrix']['true_positives']}\t| FALSE POS: {metrics['confusion_matrix']['false_positives']}")
        print(f"FALSE NEG: {metrics['confusion_matrix']['false_negatives']}\t| TRUE NEG: {metrics['confusion_matrix']['true_negatives']}")
        
        # # Print detailed predictions
        # print("\nSample Predictions (first 5):")
        # print("ID | True Label | Predicted | Confidence")
        # print("-" * 45)
        # for i in range(min(5, len(predictions))):
        #     true_label = "M" if Y[i, 1] == 1 else "B"
        #     pred_label = "M" if predictions[i] == 1 else "B"
        #     confidence = probabilities[i, predictions[i]]
        #     sample_id = test_df.iloc[i, 0]
        #     print(f"{sample_id:<3} | {true_label:^10} | {pred_label:^9} | {confidence:^10.4f}")
        
        print("\nPrediction completed successfully.")

    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()