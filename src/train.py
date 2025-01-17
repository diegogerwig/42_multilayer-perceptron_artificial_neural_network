#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import pickle
import json
import os
from utils.Mlp import MLP
from utils.Scaler import Scaler
from utils.display import plot_learning_curves
from utils.EarlyStopping import EarlyStopping
from colorama import init, Fore, Style

init()  # Initialize colorama for colored text output

def save_model(model, W, b, filepath='trained_model'):
    """Save model using both pickle and JSON formats
    
    Args:
        model: The trained MLP model
        W: List of weight matrices
        b: List of bias vectors
        filepath: Base name for saved files (without extension)
    """
    os.makedirs('./models', exist_ok=True)
    
    # Prepare model architecture and parameters
    model_config = {
        'hidden_layer_sizes': model.hidden_layer_sizes,
        'output_layer_size': model.output_layer_size,
        'activation': model.activation_name,
        'output_activation': model.output_activation_name,
        'loss': model.loss_name,
        'learning_rate': model.learning_rate,
        'batch_size': model.batch_size,
        'solver': model.solver.name,
        'weight_initializer': model.weight_initializer
    }
    
    # Save complete model with weights using pickle
    pickle_data = {
        'config': model_config,
        'weights': W,
        'biases': b,
        'train_losses': model.train_losses,
        'val_losses': model.val_losses,
        'train_accuracies': model.train_accuracies,
        'val_accuracies': model.val_accuracies
    }
    
    pickle_path = f'./models/{filepath}.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(pickle_data, f)
        
    # Save model configuration as JSON (human-readable)
    json_path = f'./models/{filepath}.json'
    with open(json_path, 'w') as f:
        json.dump(model_config, f, indent=4)
        
    print(f"\n{Fore.YELLOW}💾 Model saved:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}   - Complete model: {Fore.BLUE}{pickle_path}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}   - Configuration: {Fore.BLUE}{json_path}{Style.RESET_ALL}")

def load_model(filepath='trained_model'):
    """Load a saved model
    
    Args:
        filepath: Base name of the saved model files (without extension)
    
    Returns:
        model_config: Dictionary containing model configuration
        W: List of weight matrices
        b: List of bias vectors
    """
    pickle_path = f'./models/{filepath}.pkl'
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"\n{Fore.GREEN}✓ Model loaded successfully from {Fore.BLUE}{pickle_path}{Style.RESET_ALL}")
        return data['config'], data['weights'], data['biases']
    
    except FileNotFoundError:
        print(f"{Fore.RED}❌ Error: Model file not found at {pickle_path}{Style.RESET_ALL}")
        exit(1)
    except Exception as e:
        print(f"{Fore.RED}❌ Error loading model: {str(e)}{Style.RESET_ALL}")
        exit(1)

def train_model(args):
    # Load and preprocess data
    data_train = pd.read_csv(os.path.join(args.data_dir, 'data_training.csv'), header=None)
    data_val = pd.read_csv(os.path.join(args.data_dir, 'data_validation.csv'), header=None)

    # Split features and target and convert to numpy arrays
    X_train = data_train.iloc[:, 2:].to_numpy().astype(float)
    y_train = data_train.iloc[:, 1].map({'B': 0, 'M': 1}).to_numpy().reshape(-1, 1)  # Reshape to column vector
    
    X_val = data_val.iloc[:, 2:].to_numpy().astype(float)
    y_val = data_val.iloc[:, 1].map({'B': 0, 'M': 1}).to_numpy().reshape(-1, 1)  # Reshape to column vector

    print(f"\n{Fore.YELLOW}📊 Data Information:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}   - Training set shape:   {Fore.BLUE}{X_train.shape}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}   - Validation set shape: {Fore.BLUE}{X_val.shape}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}   - Number of features:   {Fore.BLUE}{X_train.shape[1]}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}   - Class distribution:   {Fore.BLUE}B: {(y_train == 0).sum()}, M: {(y_train == 1).sum()}{Style.RESET_ALL}")

    print(f"\n{Fore.YELLOW}🔄 Training Phase:{Style.RESET_ALL}")

    # Scale data
    scaler = Scaler(method=args.standardize)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save scaler parameters
    os.makedirs('./models', exist_ok=True)
    scaler_params = {
        'method': args.standardize,
        'mean': scaler.mean,
        'scale': scaler.scale,
        'min': scaler.min,
        'max': scaler.max
    }
    np.save('./models/scaler_params.npy', scaler_params)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # Create and train model con arquitectura ajustada
    model = MLP(hidden_layer_sizes=[30, 15, 8],   # Hidden layers
                output_layer_size=1,              # Binary classification
                activation=args.activation,
                output_activation=args.output_activation,
                epochs=args.epochs,
                loss=args.loss,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                random_seed=args.seed,
                weight_initializer=args.weight_init,
                solver=args.solver)
    
    try:
        best_W, best_b = model.fit(X_train, y_train, X_val, y_val, early_stopping)
        save_model(model, best_W, best_b, 'trained_model')
    except Exception as error:
        print(f"{Fore.RED}❌ Error: {type(error).__name__}: {error}{Style.RESET_ALL}")
        import traceback
        print(traceback.format_exc())
        exit(1)

    print(model)
    plot_learning_curves(model.train_losses, model.val_losses, 
                        model.train_accuracies, model.val_accuracies)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"""
{Fore.YELLOW}🧠 Neural Network Training Tool{Style.RESET_ALL}
{Fore.WHITE}Trains a multilayer perceptron neural network with specified parameters.

{Fore.YELLOW}📋 Usage Example:{Style.RESET_ALL}
{Fore.BLUE}  python train.py --layers 16 8 4 --epochs 100{Style.RESET_ALL}
"""
    )

    # Data arguments
    data_args = parser.add_argument_group(f'{Fore.YELLOW}📁 Data Arguments{Style.RESET_ALL}')
    data_args.add_argument(
        '--data_dir',
        default='./data/processed',
        help=f'{Fore.WHITE}Directory containing the processed datasets{Style.RESET_ALL}'
    )

    # Model architecture parameters
    architecture = parser.add_argument_group(f'{Fore.YELLOW}🏗️  Architecture Parameters{Style.RESET_ALL}')
    architecture.add_argument('--layers', nargs='+', type=int, default=[16, 8, 4],
                            help=f'{Fore.WHITE}Hidden layer sizes (default: 16 8 4){Style.RESET_ALL}')
    architecture.add_argument('--output_size', type=int, default=1,
                            help=f'{Fore.WHITE}Output layer size (default: 1){Style.RESET_ALL}')
    architecture.add_argument('--activation', default='relu',
                            help=f'{Fore.WHITE}Hidden layer activation function (default: relu){Style.RESET_ALL}')
    architecture.add_argument('--output_activation', default='sigmoid',
                            help=f'{Fore.WHITE}Output layer activation function (default: sigmoid){Style.RESET_ALL}')

    # Training parameters
    train_params = parser.add_argument_group(f'{Fore.YELLOW}📈 Training Parameters{Style.RESET_ALL}')
    train_params.add_argument('--epochs', type=int, default=100,
                         help=f'{Fore.WHITE}Number of training epochs (default: 100){Style.RESET_ALL}')
    train_params.add_argument('--batch_size', type=int, default=32,
                         help=f'{Fore.WHITE}Mini-batch size (default: 32){Style.RESET_ALL}')
    train_params.add_argument('--learning_rate', type=float, default=0.1,
                         help=f'{Fore.WHITE}Learning rate (default: 0.1){Style.RESET_ALL}')
    train_params.add_argument('--loss', 
                         default='binaryCrossentropy',
                         choices=['sparseCategoricalCrossentropy', 'binaryCrossentropy', 'categoricalCrossentropy'],
                         help=f'{Fore.WHITE}Loss function (default: binaryCrossentropy){Style.RESET_ALL}')

    # Optimization parameters
    optimization = parser.add_argument_group(f'{Fore.YELLOW}⚙️  Optimization Parameters{Style.RESET_ALL}')
    optimization.add_argument('--solver', 
                            default='sgd',
                            choices=['sgd', 'momentum'],
                            help=f'{Fore.WHITE}Optimizer (default: sgd){Style.RESET_ALL}')
    optimization.add_argument('--weight_init', 
                            default='HeUniform',  # Corregido a HeUniform
                            choices=['HeNormal', 'HeUniform', 'GlorotNormal', 'GlorotUniform'],  # Opciones correctas
                            help=f'{Fore.WHITE}Weight initialization method (default: HeUniform){Style.RESET_ALL}')
    optimization.add_argument('--standardize', 
                            default='z_score',
                            choices=['z_score', 'minmax'],
                            help=f'{Fore.WHITE}Data standardization method (default: z_score){Style.RESET_ALL}')
    optimization.add_argument('--patience', 
                            type=int, 
                            default=10,
                            help=f'{Fore.WHITE}Early stopping patience (default: 10){Style.RESET_ALL}')
    optimization.add_argument('--seed', 
                            type=int, 
                            default=42,
                            help=f'{Fore.WHITE}Random seed for reproducibility (default: 42){Style.RESET_ALL}')

    args = parser.parse_args()

    # Print help reminder and configuration
    print(f'{Fore.YELLOW}💡 Quick Help:{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   Use {Fore.GREEN}--help{Fore.WHITE} or {Fore.GREEN}-h{Fore.WHITE} for detailed usage information\n')

    print(f'{Fore.YELLOW}🔧 Configuration:{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Data directory:     {Fore.BLUE}{args.data_dir}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Hidden layers:      {Fore.BLUE}{args.layers}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Activation:         {Fore.BLUE}{args.activation}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Output activation:  {Fore.BLUE}{args.output_activation}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Epochs:             {Fore.BLUE}{args.epochs}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Batch size:         {Fore.BLUE}{args.batch_size}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Learning rate:      {Fore.BLUE}{args.learning_rate}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Loss function:      {Fore.BLUE}{args.loss}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Optimizer:          {Fore.BLUE}{args.solver}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Weight init:        {Fore.BLUE}{args.weight_init}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Standardization:    {Fore.BLUE}{args.standardize}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Early stop:         {Fore.BLUE}{args.patience}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Random seed:        {Fore.BLUE}{args.seed}{Style.RESET_ALL}\n')

    train_model(args)

if __name__ == "__main__":
    main()