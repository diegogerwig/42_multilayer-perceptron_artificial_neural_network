#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import json
import os
import pickle
from utils.model import forward_propagation
from utils.metrics import evaluate_predictions, binary_cross_entropy
from utils.preprocessing import transform_new_data

def load_model(model_path='./models/model_params.pkl'):
    """Load model parameters and configuration."""
    print("\nLoading model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model parameters file not found at {model_path}")
    
    try:
        # Load model parameters
        with open(model_path, 'rb') as f:
            parameters = pickle.load(f)
        
        # Load model topology and preprocessing info
        with open('./models/model_topology.json', 'r') as f:
            topology = json.load(f)
            use_gelu = topology.get('use_gelu', False)
        
        # Load preprocessing information
        with open('./models/preprocessing_info.json', 'r') as f:
            preprocessing_info = json.load(f)
        
        print("Model loaded successfully")
        return parameters, topology, preprocessing_info, use_gelu
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def predict(X, parameters, use_gelu=False):
    """Make predictions using trained model."""
    try:
        # Forward pass
        cache = forward_propagation(X, parameters, training=False, use_gelu=use_gelu)
        L = len([key for key in parameters.keys() if key.startswith('W')])
        
        # Get predictions and probabilities
        probabilities = cache[f'A{L}']
        predictions = np.argmax(probabilities, axis=1)
        
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
        parameters, topology, preprocessing_info, use_gelu = load_model(args.model_params)
        
        # Load and prepare data
        print(f"\nProcessing dataset: {args.test_data}")
        test_df = pd.read_csv(args.test_data, header=None)
        X = test_df.iloc[:, 2:].values  # Skip ID and Diagnosis columns

        # Convert labels to one-hot encoding
        Y = np.zeros((len(test_df), 2))
        Y[test_df.iloc[:, 1] == 'M', 1] = 1  # Malignant
        Y[test_df.iloc[:, 1] == 'B', 0] = 1  # Benign
        
        print(f"Original input shape: {X.shape}")
        
        # Apply preprocessing
        print("\nApplying preprocessing...")
        X = transform_new_data(X, preprocessing_info)
        print(f"Input shape after preprocessing: {X.shape}")
        
        # Make predictions
        print("\nMaking predictions...")
        predictions, probabilities = predict(X, parameters, use_gelu)
        
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
        print(f"AUC:       {metrics['auc']:.4f}")
        print("\nConfusion Matrix:")
        print(f"TRUE POS:  {metrics['confusion_matrix']['true_positives']}\t| FALSE POS: {metrics['confusion_matrix']['false_positives']}")
        print(f"FALSE NEG: {metrics['confusion_matrix']['false_negatives']}\t| TRUE NEG: {metrics['confusion_matrix']['true_negatives']}")
        
        # Save predictions to file
        results_df = test_df.copy()
        results_df['Predicted'] = ['M' if p == 1 else 'B' for p in predictions]
        results_df['M_Probability'] = probabilities[:, 1]
        
        output_file = os.path.join(os.path.dirname(args.test_data), 'predictions.csv')
        results_df.to_csv(output_file, index=False, header=False)
        print(f"\nPredictions saved to: {output_file}")
        
        print("\nPrediction completed successfully.")

    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()