import pandas as pd
import numpy as np
import argparse
import pickle
import json
import os
import sys
from utils.normalize import fit_transform_data, transform_data
from utils.plot import plot_learning_curves
from utils.mlp_functions import create_model_config, fit_network
from colorama import init, Fore, Style

init(autoreset=True)  # Initialize colorama for colored text output. Autoreset colors after each print.

def load_data(data_path):
    """Load and prepare dataset"""
    data = pd.read_csv(data_path)
    X = data.iloc[:, 2:]
    y = data.iloc[:, 1].map({'B': 0, 'M': 1}).values
    return X, y

def save_model(config, W, b, filepath='model'):
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
    
    # Save model configuration in JSON (excluding weights)
    json_path = f'./models/{filepath}.json'
    with open(json_path, 'w') as f:
        json.dump(model_data, f, indent=4)
    
    print(f"\n{Fore.YELLOW}💾 Saving model:")
    print(f"{Fore.WHITE}   - Pickle file: {Fore.BLUE}{pickle_path}")
    print(f"{Fore.WHITE}   - JSON config: {Fore.BLUE}{json_path}")

def train_test_split(X_train, y_train, val_path, args):
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
            'patience': args.patience,
            'min_delta': 0.001
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
        'Activation': args.activation,
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

def print_training_results(history):
    """Print training metrics"""
    print(f"\n📈 {Fore.YELLOW}Training Results:")
    metrics = {
        'Training LOSS': history['train_losses'][-1],
        'Validation LOSS': history['val_losses'][-1],
        'Training ACCURACY': history['train_accuracies'][-1],
        'Validation ACCURACY': history['val_accuracies'][-1]
    }
    for key, value in metrics.items():
        print(f"   - {key:<22} {Fore.BLUE}{value:.4f}")

def train_model(args):
    """Train the neural network model"""
    try:
        train_path = os.path.join(args.data_dir, 'data_training.csv')
        val_path = os.path.join(args.data_dir, 'data_validation.csv')
        
        if not os.path.exists(train_path):
            print(f"❌ Error: Training data file not found at {train_path}")
            return

        # Load and prepare data
        X_train, y_train = load_data(train_path)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, val_path, args)
        
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
        W, b, history = fit_network(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            config,
            early_stopping_config
        )
        
        save_model(config, W, b, 'trained_model')
        
        print_model_config(args)
        print_training_results(history)
        
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
{Fore.WHITE}Trains a multilayer perceptron neural network for classification using softmax output.

{Fore.YELLOW}📋 Usage Example:
{Fore.BLUE}  python train.py --layers 16 8 4 --epochs 100 --learning_rate 0.0005
"""
    )

    # Data arguments
    data_args = parser.add_argument_group(f'{Fore.YELLOW}📁 Data Arguments')
    data_args.add_argument(
                        '--data_dir',
                        default='./data/processed',
                        help=f'{Fore.WHITE}Directory containing the processed datasets'
    )

    # Model architecture parameters
    architecture = parser.add_argument_group(f'{Fore.YELLOW}🏗️  Architecture Parameters')
    architecture.add_argument('--layers', 
                            nargs='+', 
                            type=int, 
                            default=[24, 16],
                            help=f'{Fore.WHITE}Hidden layer sizes (default: 16 8 4)')
    architecture.add_argument('--activation', 
                            default='relu',
                            choices=['relu', 'sigmoid'],
                            help=f'{Fore.WHITE}Hidden layer activation function (default: relu)')

    # Training parameters
    train_params = parser.add_argument_group(f'{Fore.YELLOW}📈 Training Parameters')
    train_params.add_argument('--epochs', 
                            type=int, 
                            default=1000,
                            help=f'{Fore.WHITE}Number of training epochs (default: 100)')
    train_params.add_argument('--batch_size', 
                            type=int, 
                            default=64,
                            help=f'{Fore.WHITE}Mini-batch size (default: 32)')
    train_params.add_argument('--learning_rate', 
                            type=float, 
                            default=0.0006,
                            help=f'{Fore.WHITE}Learning rate (default: 0.1)')
    train_params.add_argument('--loss', 
                            default='binaryCrossentropy',
                            choices=['binaryCrossentropy'],
                            help=f'{Fore.WHITE}Loss function (always binaryCrossentropy)')

    # Optimization parameters
    optimization = parser.add_argument_group(f'{Fore.YELLOW}🔬 Optimization Parameters')
    optimization.add_argument('--optimizer', 
                            default='gradient_descent',
                            choices=['gradient_descent','sgd', 'momentum'],
                            help=f'{Fore.WHITE}Optimizer (default: gradient_descent)')
    optimization.add_argument('--weight_init', 
                            default='HeUniform',  
                            choices=['HeNormal', 'HeUniform', 'GlorotNormal', 'GlorotUniform'],  
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
    parser.add_argument('--skip-input',
                       action='store_true',
                       help='Skip input prompts for plots')

    args = parser.parse_args()
   
    # Print help reminder and configuration
    print(f'{Fore.YELLOW}💡 Quick Help:')
    print(f'{Fore.WHITE}   Use {Fore.GREEN}--help{Fore.WHITE} or {Fore.GREEN}-h{Fore.WHITE} for detailed usage information\n')

    print(f'{Fore.YELLOW}🔧 Configuration:')
    print(f'{Fore.WHITE}   - Data directory:     {Fore.BLUE}{args.data_dir}')
    print(f'{Fore.WHITE}   - Hidden layers:      {Fore.BLUE}{args.layers}')
    print(f'{Fore.WHITE}   - Activation:         {Fore.BLUE}{args.activation}')
    print(f'{Fore.WHITE}   - Output activation:  {Fore.BLUE}softmax')  # Always softmax
    print(f'{Fore.WHITE}   - Output size:        {Fore.BLUE}2')        # Always 2
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