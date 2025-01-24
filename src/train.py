#!/usr/bin/env python3
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

init()  # Initialize colorama for colored text output


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
    
    print(f"\n{Fore.YELLOW}💾 Saving model:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}   - Pickle file: {Fore.BLUE}{pickle_path}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}   - JSON config: {Fore.BLUE}{json_path}{Style.RESET_ALL}")

def train_model(args):
    """Train the neural network model"""
    try:
        # Load training data
        train_path = os.path.join(args.data_dir, 'data_training.csv')
        val_path = os.path.join(args.data_dir, 'data_validation.csv')
        
        if not os.path.exists(train_path):
            print(f"{Fore.RED}❌ Error: Training data file not found at {train_path}{Style.RESET_ALL}")
            return
            
        data_train = pd.read_csv(train_path, header=None)
        X_train = data_train.iloc[:, 2:]  
        y_train = data_train.iloc[:, 1].map({'B': 0, 'M': 1}).values

        # Handle validation data
        if not os.path.exists(val_path):
            print(f"{Fore.YELLOW}❗ Warning: Validation data file not found at {val_path}. Using 15% of training data as validation set.{Style.RESET_ALL}")
            # Split training data for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=args.seed)
        else:
            data_val = pd.read_csv(val_path, header=None)
            X_val = data_val.iloc[:, 2:]
            y_val = data_val.iloc[:, 1].map({'B': 0, 'M': 1}).values

        print(f"\n{Fore.YELLOW}📊 Data Information:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Training set shape:   {Fore.BLUE}{X_train.shape}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Validation set shape: {Fore.BLUE}{X_val.shape}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Number of features:   {Fore.BLUE}{X_train.shape[1]}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Class distribution:   {Fore.BLUE}B: {(y_train == 0).sum()}, M: {(y_train == 1).sum()}{Style.RESET_ALL}")

        # Rest of the function remains the same
        print(f"\n{Fore.YELLOW}🔄 Training Phase:{Style.RESET_ALL}")

        scale_method = 'z_score' if args.standardize == 'z_score' else 'minmax'
        X_train, scaler_params = fit_transform_data(X_train, method=scale_method)
        X_val = transform_data(X_val, scaler_params)

        os.makedirs('./models', exist_ok=True)
        scaler_params['method'] = scale_method 
        scaler_params_path = './models/scaler_params.json'
        with open(scaler_params_path, 'w') as f:
            json.dump(scaler_params, f, indent=4)

        print(f"{Fore.WHITE}   - Scaler params:    {Fore.BLUE}{scaler_params_path}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Scaling method:   {Fore.BLUE}{scale_method}{Style.RESET_ALL}")

        early_stopping_config = None
        if args.early_stopping.lower() == 'true':
            print(f"\n{Fore.YELLOW}🛑 Early Stopping:{Style.RESET_ALL}")
            early_stopping_config = {
                'enabled': True,
                'patience': args.patience,
                'min_delta': 0.001
            }
            print(f"{Fore.WHITE}   - Enabled with patience: {Fore.BLUE}{args.patience}\n{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}🛑 Early Stopping: {Fore.BLUE}Disabled\n{Style.RESET_ALL}")

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
            random_seed=args.seed,
            solver=args.solver
        )
        
        W, b, history = fit_network(
            X_train, y_train, 
            X_val, y_val, 
            config,
            early_stopping_config
        )
        
        save_model(config, W, b, 'trained_model')
        
        print(f"\n{Fore.YELLOW}🔍 Model Configuration:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Hidden layers:      {Fore.BLUE}{args.layers}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Activation:         {Fore.BLUE}{args.activation}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Output activation:  {Fore.BLUE}softmax{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Loss function:      {Fore.BLUE}{args.loss}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Epochs:             {Fore.BLUE}{args.epochs}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Batch size:         {Fore.BLUE}{args.batch_size}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Learning rate:      {Fore.BLUE}{args.learning_rate}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Optimizer:          {Fore.BLUE}{args.solver}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Weight init:        {Fore.BLUE}{args.weight_init}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Standardization:    {Fore.BLUE}{args.standardize}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Early stopping:     {Fore.BLUE}{'Enabled' if args.early_stopping.lower() == 'true' else 'Disabled'}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Random seed:        {Fore.BLUE}{args.seed}{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}📈 Training Results:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Training LOSS:       {Fore.BLUE}{history['train_losses'][-1]:.4f}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Validation LOSS:     {Fore.BLUE}{history['val_losses'][-1]:.4f}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Training ACCURACY:   {Fore.BLUE}{history['train_accuracies'][-1]:.4f}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}   - Validation ACCURACY: {Fore.BLUE}{history['val_accuracies'][-1]:.4f}{Style.RESET_ALL}")

        plot_learning_curves(
            history['train_losses'], 
            history['val_losses'],
            history['train_accuracies'], 
            history['val_accuracies'],
            getattr(args, 'skip_input', False)
        )
                           
    except Exception as error:
        print(f"{Fore.RED}❌ Error: {type(error).__name__}: {error}{Style.RESET_ALL}")
        import traceback
        print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"""
{Fore.YELLOW}🧠 Neural Network Training Tool{Style.RESET_ALL}
{Fore.WHITE}Trains a multilayer perceptron neural network for classification using softmax output.

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
    architecture.add_argument('--layers', nargs='+', type=int, default=[12, 8],
                            help=f'{Fore.WHITE}Hidden layer sizes (default: 16 8 4){Style.RESET_ALL}')
    architecture.add_argument('--activation', default='relu',
                            choices=['relu', 'sigmoid'],
                            help=f'{Fore.WHITE}Hidden layer activation function (default: relu){Style.RESET_ALL}')

    # Training parameters
    train_params = parser.add_argument_group(f'{Fore.YELLOW}📈 Training Parameters{Style.RESET_ALL}')
    train_params.add_argument('--epochs', type=int, default=1000,
                         help=f'{Fore.WHITE}Number of training epochs (default: 100){Style.RESET_ALL}')
    train_params.add_argument('--batch_size', type=int, default=32,
                         help=f'{Fore.WHITE}Mini-batch size (default: 32){Style.RESET_ALL}')
    train_params.add_argument('--learning_rate', type=float, default=0.0005,
                         help=f'{Fore.WHITE}Learning rate (default: 0.1){Style.RESET_ALL}')
    train_params.add_argument('--loss', 
                         default='binaryCrossentropy',
                         choices=['binaryCrossentropy'],
                         help=f'{Fore.WHITE}Loss function (always binaryCrossentropy){Style.RESET_ALL}')

    # Optimization parameters
    optimization = parser.add_argument_group(f'{Fore.YELLOW}⚙️  Optimization Parameters{Style.RESET_ALL}')
    optimization.add_argument('--solver', 
                            default='sgd',
                            choices=['sgd', 'momentum'],
                            help=f'{Fore.WHITE}Optimizer (default: sgd){Style.RESET_ALL}')
    optimization.add_argument('--weight_init', 
                            default='HeUniform',  
                            choices=['HeNormal', 'HeUniform', 'GlorotNormal', 'GlorotUniform'],  
                            help=f'{Fore.WHITE}Weight initialization method (default: HeUniform){Style.RESET_ALL}')
    optimization.add_argument('--standardize', 
                            default='z_score',
                            choices=['z_score', 'minmax'],
                            help=f'{Fore.WHITE}Data standardization method (default: z_score){Style.RESET_ALL}')
    optimization.add_argument('--early_stopping',
                            type=str,
                            default='false',
                            choices=['true', 'false'],
                            help=f'{Fore.WHITE}Enable or disable early stopping (default: false){Style.RESET_ALL}')
    optimization.add_argument('--patience', 
                            type=int, 
                            default=4,
                            help=f'{Fore.WHITE}Early stopping patience (default: 10){Style.RESET_ALL}')
    optimization.add_argument('--seed', 
                            type=int, 
                            default=None,
                            help=f'{Fore.WHITE}Random seed for reproducibility (default: NONE){Style.RESET_ALL}')

    # Parse arguments and train the model
    parser.add_argument('--skip-input',
                       action='store_true',
                       help='Skip input prompts for plots')

    args = parser.parse_args()
   
    # Print help reminder and configuration
    print(f'{Fore.YELLOW}💡 Quick Help:{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   Use {Fore.GREEN}--help{Fore.WHITE} or {Fore.GREEN}-h{Fore.WHITE} for detailed usage information\n')

    print(f'{Fore.YELLOW}🔧 Configuration:{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Data directory:     {Fore.BLUE}{args.data_dir}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Hidden layers:      {Fore.BLUE}{args.layers}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Activation:         {Fore.BLUE}{args.activation}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Output activation:  {Fore.BLUE}softmax{Style.RESET_ALL}')  # Always softmax
    print(f'{Fore.WHITE}   - Output size:        {Fore.BLUE}2{Style.RESET_ALL}')        # Always 2
    print(f'{Fore.WHITE}   - Loss function:      {Fore.BLUE}{args.loss}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Epochs:             {Fore.BLUE}{args.epochs}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Batch size:         {Fore.BLUE}{args.batch_size}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Learning rate:      {Fore.BLUE}{args.learning_rate}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Optimizer:          {Fore.BLUE}{args.solver}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Weight init:        {Fore.BLUE}{args.weight_init}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Standardization:    {Fore.BLUE}{args.standardize}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Early stopping:     {Fore.BLUE}{"Enabled" if args.early_stopping.lower() == "true" else "Disabled"}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Random seed:        {Fore.BLUE}{args.seed}{Style.RESET_ALL}\n')

    train_model(args)

if __name__ == "__main__":
    main()