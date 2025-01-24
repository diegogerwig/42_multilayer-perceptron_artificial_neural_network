#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import pickle
import json
import os
from utils.normalize import transform_data
from utils.mlp_functions import forward_propagation, ACTIVATIONS_FUNCTIONS, LOSS_FUNCTIONS
from colorama import init, Fore, Style

init()  # Initialize colorama for colored text output

def load_model(filepath='trained_model'):
    """Load model weights and configuration"""
    print(f"{Fore.YELLOW}📂 Loading Model:{Style.RESET_ALL}")
    
    # Load model and weights from pickle file
    pickle_path = f'./models/{filepath}.pkl'
    if not os.path.exists(pickle_path):
        print(f"\n{Fore.RED}❗ Error: Model not found!{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   The model file should be at: {Fore.BLUE}{pickle_path}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   Please train the model first or check the file path.{Style.RESET_ALL}\n")
        exit(1)
    
    with open(pickle_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"{Fore.WHITE}   - Model loaded from:  {Fore.BLUE}{pickle_path}{Style.RESET_ALL}")
    
    # Load scaler parameters
    scaler_path = './models/scaler_params.json'
    if not os.path.exists(scaler_path):
        print(f"\n{Fore.RED}❗ Error: Scaler parameters not found!{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   The scaler file should be at: {Fore.BLUE}{scaler_path}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   Please train the model first or check the file path.{Style.RESET_ALL}\n")
        exit(1)
    
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    
    print(f"{Fore.WHITE}   - Scaler loaded from: {Fore.BLUE}{scaler_path}{Style.RESET_ALL}")
    
    return model_data, scaler_params

def evaluate_model(y_true, y_pred, probas):
    """Calculate and return model performance metrics"""
    # Convert predictions to binary format if needed
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Calculate metrics
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate AUC-ROC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, probas[:, 1])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': {
            'true_positives': int(TP),
            'true_negatives': int(TN),
            'false_positives': int(FP),
            'false_negatives': int(FN)
        }
    }

def predict_data(args):
    """Make predictions using the trained model"""
    try:
        # Load model and scaler
        model_data, scaler_params = load_model(args.model)
        
        # Load test data
        data_test = pd.read_csv(args.test_data, header=None)
        X_test = data_test.iloc[:, 2:]  # Features
        y_test = data_test.iloc[:, 1].map({'B': 0, 'M': 1}).values  # Labels
        
        print(f"\n{Fore.YELLOW}📊 Data Information:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Test set shape:     {Fore.BLUE}{X_test.shape}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Number of features: {Fore.BLUE}{X_test.shape[1]}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Class distribution: {Fore.BLUE}B: {(y_test == 0).sum()}, M: {(y_test == 1).sum()}{Style.RESET_ALL}")
        
        # Scale features using saved scaler parameters
        X_test = transform_data(X_test, scaler_params)
        
        # Get model configuration
        W = model_data['W']
        b = model_data['b']
        config = model_data['model_data']
        
        print(f"\n{Fore.YELLOW}🔍 Model Configuration:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Hidden layers:     {Fore.BLUE}{config['hidden_layer_sizes']}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Output size:       {Fore.BLUE}{config['output_layer_size']}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Activation:        {Fore.BLUE}{config['activation']}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Output activation: {Fore.BLUE}{config['output_activation']}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Loss function:     {Fore.BLUE}{config['loss']}{Style.RESET_ALL}")
        
        # Get activation functions from dictionary
        activation_func, _ = ACTIVATIONS_FUNCTIONS[config['activation']] 
        output_activation = ACTIVATIONS_FUNCTIONS["softmax"][0] 

        # Make predictions
        print(f"\n{Fore.YELLOW}🎯 Making Predictions...{Style.RESET_ALL}")
        probabilities, _ = forward_propagation(X_test, W, b, activation_func, output_activation)
        predictions = np.argmax(probabilities, axis=1)
        
        # Calculate metrics
        metrics = evaluate_model(y_test, predictions, probabilities)

        # Calculate loss
        test_loss = LOSS_FUNCTIONS[args.loss](y_test, probabilities)

        # Print results
        print(f"\n{Fore.YELLOW}📈 Test Results:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - LOSS:      {Fore.BLUE}{test_loss:.4f}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Accuracy:  {Fore.BLUE}{metrics['accuracy']:.4f}{Style.RESET_ALL}")
        # print(f"{Fore.WHITE}   - Precision: {Fore.BLUE}{metrics['precision']:.4f}{Style.RESET_ALL}")
        # print(f"{Fore.WHITE}   - Recall:    {Fore.BLUE}{metrics['recall']:.4f}{Style.RESET_ALL}")
        # print(f"{Fore.WHITE}   - F1 Score:  {Fore.BLUE}{metrics['f1']:.4f}{Style.RESET_ALL}")
        # print(f"{Fore.WHITE}   - AUC:       {Fore.BLUE}{metrics['auc']:.4f}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}📊 Confusion Matrix:{Style.RESET_ALL}")
        cm = metrics['confusion_matrix']
        print(f"{Fore.WHITE}   TRUE POS:  {Fore.BLUE}{cm['true_positives']:3d}{Fore.WHITE} | FALSE POS: {Fore.BLUE}{cm['false_positives']:3d}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   FALSE NEG: {Fore.BLUE}{cm['false_negatives']:3d}{Fore.WHITE} | TRUE NEG:  {Fore.BLUE}{cm['true_negatives']:3d}{Style.RESET_ALL}")
        
        # Save predictions
        results_df = data_test.copy()
        results_df['Predicted'] = ['M' if p == 1 else 'B' for p in predictions]
        results_df['M_Probability'] = probabilities[:, 1]
        
        output_file = os.path.join(os.path.dirname(args.test_data), 'predictions.csv')
        results_df.to_csv(output_file, index=False, header=False)
        print(f"\n{Fore.YELLOW}💾 Results:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Predictions saved to: {Fore.BLUE}{output_file}{Style.RESET_ALL}")
        
    except Exception as error:
        print(f"\n{Fore.RED}❌ Error: {type(error).__name__}: {error}{Style.RESET_ALL}")
        import traceback
        print(traceback.format_exc())
        exit(1)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"""
{Fore.YELLOW}🔮 Neural Network Prediction Tool{Style.RESET_ALL}
{Fore.WHITE}Make predictions using a trained multilayer perceptron neural network.

{Fore.YELLOW}📋 Usage Example:{Style.RESET_ALL}
{Fore.BLUE}  python predict.py --test_data ./data/test.csv --model trained_model{Style.RESET_ALL}
"""
    )

    # Data arguments
    data_args = parser.add_argument_group(f'{Fore.YELLOW}📁 Data Arguments{Style.RESET_ALL}')
    data_args.add_argument(
        '--test_data',
        default='./data/processed/data_test.csv',
        help=f'{Fore.WHITE}Path to the test data CSV file{Style.RESET_ALL}'
    )
    data_args.add_argument(
        '--model',
        default='trained_model',
        help=f'{Fore.WHITE}Name of the model to use (default: trained_model){Style.RESET_ALL}'
    )
    data_args.add_argument(
        '--loss', 
        default='binaryCrossentropy',
        choices=['binaryCrossentropy', 'sparseCategoricalCrossentropy', 'categoricalCrossentropy'],
        help=f'{Fore.WHITE}Loss function (default: binaryCrossentropy){Style.RESET_ALL}'
    )

    args = parser.parse_args()
    
    # Print help reminder and configuration
    print(f'{Fore.YELLOW}💡 Quick Help:{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   Use {Fore.GREEN}--help{Fore.WHITE} or {Fore.GREEN}-h{Fore.WHITE} for detailed usage information\n')

    print(f'{Fore.YELLOW}🔧 Configuration:{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Test data: {Fore.BLUE}{args.test_data}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Model:     {Fore.BLUE}{args.model}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Loss:      {Fore.BLUE}{args.loss}{Style.RESET_ALL}\n')

    predict_data(args)

if __name__ == "__main__":
    main()