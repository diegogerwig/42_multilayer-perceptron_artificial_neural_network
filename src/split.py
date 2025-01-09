#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os

def split_data(data_path, output_dir, test_size=0.2, random_seed=42):
    os.makedirs(output_dir, exist_ok=True)
    
    data = pd.read_csv(data_path, header=None)
    np.random.seed(random_seed)
    
    indices = np.random.permutation(len(data))
    split_point = int(len(data) * (1 - test_size))
    train_idx, test_idx = indices[:split_point], indices[split_point:]
    
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    
    train_data.to_csv(os.path.join(output_dir, 'data_training.csv'), index=False, header=False)
    test_data.to_csv(os.path.join(output_dir, 'data_test.csv'), index=False, header=False)
    
    print(f'   - ORIGINAL set shape: {data.shape}')
    print(f'   - TRAIN    set shape: {train_data.shape}')
    print(f'   - TEST     set shape: {test_data.shape}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Input dataset path')
    parser.add_argument('--output', default='./data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_size', type=float, default=0.2, help='Validation set size')
    args = parser.parse_args()
    
    split_data(args.dataset, args.output, args.test_size, args.seed)

if __name__ == "__main__":
    main()