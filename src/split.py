#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os
from colorama import init, Fore, Style

init()  # Initialize colorama for colored text output

def split_data(data_path, output_dir, test_size, val_size, random_seed):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = pd.read_csv(data_path, header=None)
    np.random.seed(random_seed)
    
    # First split: training+validation vs test
    indices = np.random.permutation(len(data))
    split_point = int(len(data) * (1 - test_size))
    trainval_idx, test_idx = indices[:split_point], indices[split_point:]
    
    # Split training data into train and validation
    trainval_data = data.iloc[trainval_idx]
    test_data = data.iloc[test_idx]
    
    # Calculate validation size relative to training+validation data
    val_samples = int(len(trainval_data) * val_size)
    train_indices = np.random.permutation(len(trainval_data))
    train_idx, val_idx = train_indices[val_samples:], train_indices[:val_samples]
    
    # Create final datasets
    train_data = trainval_data.iloc[train_idx]
    val_data = trainval_data.iloc[val_idx]
    
    # Save datasets
    train_data.to_csv(os.path.join(output_dir, 'data_training.csv'), index=False, header=False)
    val_data.to_csv(os.path.join(output_dir, 'data_validation.csv'), index=False, header=False)
    test_data.to_csv(os.path.join(output_dir, 'data_test.csv'), index=False, header=False)
    
    # Print dataset shapes with colors
    print(f'{Fore.YELLOW}📂 Dataset split results:{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - ORIGINAL    set shape: {Fore.BLUE}{data.shape}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - TRAIN       set shape: {Fore.BLUE}{train_data.shape}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - VALIDATION  set shape: {Fore.BLUE}{val_data.shape}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - TEST        set shape: {Fore.BLUE}{test_data.shape}{Style.RESET_ALL}')

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"""
{Fore.YELLOW}🔍 Dataset Splitter Tool{Style.RESET_ALL}
{Fore.WHITE}Splits a dataset into training, validation, and test sets with specified proportions.

{Fore.YELLOW}📋 Usage Example:{Style.RESET_ALL}
{Fore.BLUE}  python split_data.py --dataset ./data/raw/data.csv --test_size 0.20{Style.RESET_ALL}
"""
    )
    
    # Required arguments group
    required = parser.add_argument_group(f'{Fore.YELLOW}📁 Data Arguments{Style.RESET_ALL}')
    required.add_argument(
        '--dataset', 
        default='./data/raw/data.csv',
        help=f'{Fore.WHITE}Input dataset path (CSV file){Style.RESET_ALL}'
    )
    required.add_argument(
        '--output',
        default='./data/processed',
        help=f'{Fore.WHITE}Output directory for split datasets{Style.RESET_ALL}'
    )

    # Split parameters group
    split_params = parser.add_argument_group(f'{Fore.YELLOW}📊 Split Parameters{Style.RESET_ALL}')
    split_params.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help=f'{Fore.WHITE}Test set size (default: 0.2){Style.RESET_ALL}'
    )
    split_params.add_argument(
        '--val_size',
        type=float,
        default=0.1,
        help=f'{Fore.WHITE}Validation set size (default: 0.1){Style.RESET_ALL}'
    )
    
    # Other parameters group
    other = parser.add_argument_group(f'{Fore.YELLOW}⚙️  Other Parameters{Style.RESET_ALL}')
    other.add_argument(
        '--seed',
        type=int,
        default=None,
        help=f'{Fore.WHITE}Random seed for reproducibility (default: NONE){Style.RESET_ALL}'
    )

    args = parser.parse_args()

    args.train_size = round(1 - args.test_size - args.val_size, 2)

    print(f'{Fore.YELLOW}💡 Quick Help:{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   Use {Fore.GREEN}--help{Fore.WHITE} or {Fore.GREEN}-h{Fore.WHITE} for detailed usage information')
    print(f'{Fore.WHITE}   Available arguments: {Fore.BLUE}--dataset, --output, --train_size, --test_size, --val_size, --seed{Style.RESET_ALL}\n')

    print(f'{Fore.YELLOW}🔨 Splitting dataset...{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Dataset:         {Fore.BLUE}{args.dataset}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Output dir:      {Fore.BLUE}{args.output}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Train size:      {Fore.BLUE}{args.train_size:.2f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Test size:       {Fore.BLUE}{args.test_size:.2f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Validation size: {Fore.BLUE}{args.val_size:.2f}{Style.RESET_ALL}')
    print(f'{Fore.WHITE}   - Random seed:     {Fore.BLUE}{args.seed}{Style.RESET_ALL}\n')

    if not 0 < args.train_size + args.test_size + args.val_size <= 1:
        print(f'{Fore.RED}❌ Error: Sum of train_size + test_size + val_size must be between 0 and 1{Style.RESET_ALL}')
        return

    split_data(args.dataset, args.output, args.test_size, args.val_size, args.seed)

if __name__ == "__main__":
    main()