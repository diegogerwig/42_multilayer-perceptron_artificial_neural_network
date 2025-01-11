#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os

def split_data(data_path, output_dir, test_size=0.2, val_size=0.1, random_seed=42):
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
    
    print(f'   - ORIGINAL    set shape: {data.shape}')
    print(f'   - TRAIN       set shape: {train_data.shape}')
    print(f'   - VALIDATION  set shape: {val_data.shape}')
    print(f'   - TEST        set shape: {test_data.shape}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Input dataset path')
    parser.add_argument('--output', default='./data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size (from training data)')
    args = parser.parse_args()
    
    split_data(args.dataset, args.output, args.test_size, args.val_size, args.seed)

if __name__ == "__main__":
    main()