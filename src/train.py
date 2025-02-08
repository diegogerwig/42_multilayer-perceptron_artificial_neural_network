import pandas as pd
import numpy as np
import argparse
import pickle
import json
import os
import sys
from utils.normalize import fit_transform_data, transform_data
from utils.plot import plot_learning_curves, plot_model_analysis
from utils.mlp_functions import create_model_config, fit_network
from colorama import init, Fore, Style

init(autoreset=True)  # Initialize colorama for colored text output. Autoreset colors after each print.

def load_data(data_path):
    """Load and prepare dataset"""
    data = pd.read_csv(data_path)
    X = data.iloc[:, 2:]
    y = data.iloc[:, 1].map({'B': 0, 'M': 1}).values
    return X, y

def save_model(config, W, b, filepath='model', skip_input=False):
    """Save model weights and configuration"""
    # Create models directory if it doesn't exist
    os.makedirs('./models', exist_ok=True)
    
    # Prepare model data
    model_data = {
        'hidden_layer_sizes': config['hidden_layer_sizes'],
        'output_layer_size': config['output_layer_size'],
        'activation': config['activation_name'],
        'output_activation': config['output_activation_name'],
        'loss': config['loss_name'],
    }
    
    # Save weights and model data using pickle
    pickle_data = {
        'model_data': model_data,
        'W': W,
        'b': b
    }
    pickle_path = f'./models/{filepath}.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(pickle_data, f)
    
    # Save model topology configuration in JSON
    json_topology_path = f'./models/{filepath}_topology.json'
    with open(json_topology_path, 'w') as f:
        json.dump(model_data, f, indent=4)
    
    # Save weights and biases in JSON format
    weights_biases_data = {
        'weights': [w.tolist() for w in W],
        'biases': [b.tolist() for b in b]
    }
    json_weights_path = f'./models/{filepath}_weights.json'
    with open(json_weights_path, 'w') as f:
        json.dump(weights_biases_data, f, indent=4)
    
    print(f"\n{Fore.YELLOW}💾 Saving model:")
    print(f"{Fore.WHITE}   - Model file PICKLE:  {Fore.BLUE}{pickle_path}")
    print(f"{Fore.WHITE}   - Topology JSON:      {Fore.BLUE}{json_topology_path}")
    print(f"{Fore.WHITE}   - Weights/Bias JSON:  {Fore.BLUE}{json_weights_path}")
    
    # Print additional info about the model structure
    print(f"\n{Fore.YELLOW}📐 Model Architecture:")
    print(f"{Fore.WHITE}   - INPUT  layer size:        {Fore.BLUE}{W[0].shape[0]}")
    for i, w in enumerate(W[:-1]):  # Iterate through hidden layers
        print(f"{Fore.WHITE}   - HIDDEN layer {i+1} size:      {Fore.BLUE}{w.shape[1]}")
    print(f"{Fore.WHITE}   - OUTPUT layer size:        {Fore.BLUE}{W[-1].shape[1]}")

    plot_model_analysis(W, b, skip_input)

def validate_layer_sizes(value):
    """
    Validate layer sizes to ensure they are positive integers.
    """
    try:
        layers = [int(x) for x in value]
        
        # Check for non-positive values
        invalid_layers = [(i+1, size) for i, size in enumerate(layers) if size <= 0]
        if invalid_layers:
            invalid_str = ", ".join([f"layer {pos}: {size}" for pos, size in invalid_layers])
            raise argparse.ArgumentTypeError(
                f"\n{Fore.RED}❌ Error: Invalid layer sizes detected!\n"
                f"{Fore.WHITE}The following layers have invalid sizes (must be > 0):\n"
                f"{Fore.YELLOW}{invalid_str}"
            )
        
        return layers
        
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"\n{Fore.RED}❌ Error: All layer sizes must be integers!\n"
            f"{Fore.WHITE}Received invalid value: {Fore.YELLOW}{value}"
        )

def train_val_split(X_train, y_train, val_path, args):
    """Handle validation data preparation"""
    if not os.path.exists(val_path):
        print(f"❗ Warning: Using 12,5% of training data as validation set.")
        train_indices = range(len(X_train))
        val_size = int(0.125 * len(train_indices))
        val_indices = train_indices[-val_size:]
        train_indices = train_indices[:-val_size]
        
        X_val = X_train.iloc[val_indices]
        y_val = y_train[val_indices]
        X_train = X_train.iloc[train_indices]
        y_train = y_train[train_indices]
    else:
        X_val, y_val = load_data(val_path)
    
    return X_train, X_val, y_train, y_val

def print_data_info(X_train, X_val, y_train):
    """Print dataset information"""
    print(f"\n📊 {Fore.YELLOW}Data Information:")
    print(f"   - Training set shape:   {Fore.BLUE}{X_train.shape}")
    print(f"   - Validation set shape: {Fore.BLUE}{X_val.shape}")
    print(f"   - Number of features:   {Fore.BLUE}{X_train.shape[1]}")
    print(f"   - Class distribution:   {Fore.BLUE}B: {(y_train == 0).sum()}, M: {(y_train == 1).sum()}")

def prepare_data_scaling(X_train, X_val, args):
    """Handle data scaling"""
    scale_method = 'z_score' if args.standardize == 'z_score' else 'minmax'
    X_train_scaled, scaler_params = fit_transform_data(X_train, method=scale_method)
    X_val_scaled = transform_data(X_val, scaler_params)
    
    os.makedirs('./models', exist_ok=True)
    scaler_params['method'] = scale_method
    with open('./models/scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=4)
    
    print(f"   - Scaling method:   {Fore.BLUE}{scale_method}")
    print(f"   - Scaler params:    {Fore.BLUE}./models/scaler_params.json")
    
    return X_train_scaled, X_val_scaled

def setup_early_stopping(args):
    """Configure early stopping"""
    if args.early_stopping.lower() == 'true':
        config = {
            'enabled': True,
            'patience': args.patience, # Number of epochs to wait before stopping
            'min_delta': 0.001         # Minimum change in loss to be considered an improvement
        }
        print(f"\n🛑 Early Stopping: {Fore.GREEN}ENABLED {Fore.WHITE}with patience: {Fore.BLUE}{args.patience}\n")
        return config
    print(f"\n🛑 Early Stopping: {Fore.RED}DISABLED\n")
    return None

def print_model_config(args):
    """Print model configuration"""
    print(f"\n🔍 {Fore.YELLOW}Model Configuration:")
    configs = {
        'Hidden layers': args.layers,
        'Hidden activation': args.activation,
        'Output activation': 'softmax',
        'Loss function': args.loss,
        'Epochs': args.epochs,
        'Batch size': args.batch_size,
        'Learning rate': args.learning_rate,
        'Optimizer': args.optimizer,
        'Weight init': args.weight_init,
        'Standardization': args.standardize,
        'Early stopping': 'Enabled' if args.early_stopping.lower() == 'true' else 'Disabled',
        'Seed': args.seed
    }
    for key, value in configs.items():
        print(f"   - {key:<22} {Fore.BLUE}{value}")

def calculate_metrics(y_true, y_pred):
    """Calculate additional performance metrics"""
    # Convert probabilities to class predictions
    y_pred_class = (y_pred >= 0.5).astype(int)
    
    # Calculate True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((y_true == 1) & (y_pred_class == 1))
    fp = np.sum((y_true == 0) & (y_pred_class == 1))
    tn = np.sum((y_true == 0) & (y_pred_class == 0))
    fn = np.sum((y_true == 1) & (y_pred_class == 0))
    
    # Calculate metrics

    # Precision: Value of correctly predicted positive observations to the total predicted positive observations. Best value at 1 and worst at 0. Shows how many of the predicted positives are actually positive.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall: Value of correctly predicted positive observations to the all observations in actual class. Best value at 1 and worst at 0. Shows how many of the actual positives are predicted positive.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score: Harmonic mean of precision and recall. Best value at 1 and worst at 0. Represents a balance between precision and recall.
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn
    }

def print_training_results(history, y_val, val_predictions):
    """Print training metrics and additional performance metrics"""
    print(f"\n📈 {Fore.YELLOW}Training Results:")
    metrics = {
        'Training   LOSS': history['train_losses'][-1],
        'Validation LOSS': history['val_losses'][-1],
        'Training   ACCURACY': history['train_accuracies'][-1],
        'Validation ACCURACY': history['val_accuracies'][-1]
    }
    for key, value in metrics.items():
        print(f"   - {key:<22} {Fore.BLUE}{value:.4f}")
    
    print(f"\n🎯 {Fore.YELLOW}Additional Performance Metrics:")
    additional_metrics = calculate_metrics(y_val, val_predictions)
    
    # Print classification metrics
    print(f"\n   📊 {Fore.YELLOW}Classification Metrics:")
    for key in ['Precision', 'Recall', 'F1 Score']:
        print(f"   - {key:<22} {Fore.BLUE}{additional_metrics[key]:.4f}")
    
    # Print confusion matrix
    print(f"\n   {Fore.YELLOW}📊 Confusion Matrix:")
    print(f"   {Fore.WHITE}   TRUE POS:  {Fore.BLUE}{additional_metrics['True Positives']:3d}{Fore.WHITE} | FALSE NEG: {Fore.BLUE}{additional_metrics['False Negatives']:3d}")
    print(f"   {Fore.WHITE}   FALSE POS: {Fore.BLUE}{additional_metrics['False Positives']:3d}{Fore.WHITE} | TRUE NEG:  {Fore.BLUE}{additional_metrics['True Negatives']:3d}")

def train_model(args):
    """Train the neural network model"""
    try:
        train_path = os.path.join(args.data_dir, 'data_training.csv')
        val_path = os.path.join(args.data_dir, 'data_validation.csv')
        
        if not os.path.exists(train_path):
            print(f"\n{Fore.RED}❗ Error: Training data file not found!")
            print(f"{Fore.WHITE}   The file should be at: {Fore.BLUE}{train_path}")
            print(f"{Fore.WHITE}   Please check the file path and try again.\n")
            return

        # Load and prepare data
        X_train, y_train = load_data(train_path)
        X_train, X_val, y_train, y_val = train_val_split(X_train, y_train, val_path, args)
        
        print_data_info(X_train, X_val, y_train)
        
        print(f"\n🔄 {Fore.YELLOW}Training Phase:")

        # Prepare data for training
        X_train_scaled, X_val_scaled = prepare_data_scaling(X_train, X_val, args)
        
        # Setup early stopping
        early_stopping_config = setup_early_stopping(args)
        
        # Create model configuration
        config = create_model_config(
            hidden_layer_sizes=args.layers,
            output_layer_size=2,
            activation=args.activation,
            output_activation="softmax",
            loss=args.loss,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weight_initializer=args.weight_init,
            seed=args.seed,
            optimizer=args.optimizer
        )
        
        # Train the model
        W, b, history, val_predictions = fit_network(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            config,
            early_stopping_config
        )
        
        save_model(config, W, b, 'trained_model', getattr(args, 'skip_input', False))
        
        print_model_config(args)
        print_training_results(history, y_val, val_predictions)
        
        plot_learning_curves(
            history['train_losses'],
            history['val_losses'],
            history['train_accuracies'],
            history['val_accuracies'],
            getattr(args, 'skip_input', False)
        )
                           
    except Exception as error:
        print(f"❌ Error: {type(error).__name__}: {error}")
        import traceback
        print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"""
{Fore.YELLOW}🧠 Neural Network Training Tool
{Fore.WHITE}   Trains a multilayer perceptron neural network for classification using softmax output.

{Fore.YELLOW}📋 Usage Example:
{Fore.BLUE}   python train.py --layers 16 8 4 --epochs 1000 --learning_rate 0.0005{Fore.RESET}
"""
    )

    # Data arguments
    data_args = parser.add_argument_group(f'{Fore.YELLOW}📁 Data Arguments{Fore.RESET}')
    data_args.add_argument(
                        '--data_dir',
                        default='./data/processed',
                        help=f'{Fore.WHITE}Directory containing the processed datasets')

    # Model architecture parameters
    architecture = parser.add_argument_group(f'{Fore.YELLOW}🏗️  Architecture Parameters{Fore.RESET}')
    architecture.add_argument('--layers', 
                            nargs='+', 
                            type=int, 
                            default=[40, 30, 10, 5],
                            help=f'{Fore.WHITE}Hidden layer sizes (default: 40 30 10 5)')
    architecture.add_argument('--activation', 
                            default='relu',
                            choices=['relu', 'sigmoid'],
                            help=f'{Fore.WHITE}Hidden layer activation function (default: relu)')

    # Training parameters
    train_params = parser.add_argument_group(f'{Fore.YELLOW}📈 Training Parameters{Fore.RESET}')
    train_params.add_argument('--epochs', 
                            type=int, 
                            default=500,
                            help=f'{Fore.WHITE}Number of training epochs (default: 500)')
    train_params.add_argument('--batch_size', 
                            type=int, 
                            default=64,
                            help=f'{Fore.WHITE}Mini-batch size (default: 64)')
    train_params.add_argument('--learning_rate', 
                            type=float, 
                            default=0.0006,
                            help=f'{Fore.WHITE}Learning rate (default: 0.0006)')
    train_params.add_argument('--loss', 
                            default='binaryCrossentropy',
                            choices=['binaryCrossentropy'],
                            help=f'{Fore.WHITE}Loss function (always binaryCrossentropy)')

    # Optimization parameters
    optimization = parser.add_argument_group(f'{Fore.YELLOW}🔬 Optimization Parameters{Fore.RESET}')
    optimization.add_argument('--optimizer', 
                            default='gradient_descent',
                            choices=['gradient_descent','sgd', 'momentum'],
                            help=f'{Fore.WHITE}Optimizer (default: gradient_descent)')
    optimization.add_argument('--weight_init',
                            default='HeUniform',
                            choices=['Random', 'HeNormal', 'HeUniform', 'GlorotNormal', 'GlorotUniform'],
                            help=f'{Fore.WHITE}Weight initialization method (default: HeUniform)')
    optimization.add_argument('--standardize', 
                            default='z_score',
                            choices=['z_score', 'minmax'],
                            help=f'{Fore.WHITE}Data standardization method (default: z_score)')
    optimization.add_argument('--early_stopping',
                            type=str,
                            default='false',
                            choices=['true', 'false'],
                            help=f'{Fore.WHITE}Enable or disable early stopping (default: false)')
    optimization.add_argument('--patience', 
                            type=int, 
                            default=5,
                            help=f'{Fore.WHITE}Early stopping patience (default: 5)')
    optimization.add_argument('--seed', 
                            type=int, 
                            default=None,
                            help=f'{Fore.WHITE}Random seed for reproducibility (default: NONE)')

    # Parse arguments and train the model
    other = parser.add_argument_group(f'{Fore.YELLOW}🔧 Other Parameters{Fore.RESET}')
    other.add_argument('--skip-input',
                       action='store_true',
                       help='Skip input prompts for plots')

    args = parser.parse_args()

    # Validate layer sizes after parsing
    try:
        args.layers = validate_layer_sizes(args.layers)
    except argparse.ArgumentTypeError as e:
        print(e)
        sys.exit(1)
   
    # Print help reminder and configuration
    print(f'{Fore.YELLOW}💡 Quick Help:')
    print(f'{Fore.WHITE}   Use {Fore.GREEN}--help{Fore.WHITE} or {Fore.GREEN}-h{Fore.WHITE} for detailed usage information\n')

    print(f'{Fore.YELLOW}🔧 Configuration:')
    print(f'{Fore.WHITE}   - Data directory:     {Fore.BLUE}{args.data_dir}')
    print(f'{Fore.WHITE}   - Hidden layers:      {Fore.BLUE}{args.layers}')
    print(f'{Fore.WHITE}   - Activation:         {Fore.BLUE}{args.activation}')
    print(f'{Fore.WHITE}   - Output activation:  {Fore.BLUE}softmax')  # Always softmax for classification
    print(f'{Fore.WHITE}   - Output size:        {Fore.BLUE}2')        # Softmax requires 2 output classes
    print(f'{Fore.WHITE}   - Loss function:      {Fore.BLUE}{args.loss}')
    print(f'{Fore.WHITE}   - Epochs:             {Fore.BLUE}{args.epochs}')
    print(f'{Fore.WHITE}   - Batch size:         {Fore.BLUE}{args.batch_size}')
    print(f'{Fore.WHITE}   - Learning rate:      {Fore.BLUE}{args.learning_rate}')
    print(f'{Fore.WHITE}   - Optimizer:          {Fore.BLUE}{args.optimizer}')
    print(f'{Fore.WHITE}   - Weight init:        {Fore.BLUE}{args.weight_init}')
    print(f'{Fore.WHITE}   - Standardization:    {Fore.BLUE}{args.standardize}')
    print(f'{Fore.WHITE}   - Early stopping:     {Fore.BLUE}{"Enabled" if args.early_stopping.lower() == "true" else "Disabled"}')
    print(f'{Fore.WHITE}   - Random seed:        {Fore.BLUE}{args.seed}\n')

    train_model(args)

if __name__ == "__main__":
    main()