import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def train_test_split(X, y, test_size=0.2, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    indices = np.random.permutation(X.shape[0])
    split_point = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split_point], indices[split_point:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def init_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return parameters

def forward_propagation(X, parameters):
    cache = {'A0': X.T}
    L = len(parameters) // 2
    
    for l in range(1, L):
        cache[f'Z{l}'] = np.dot(parameters[f'W{l}'], cache[f'A{l-1}']) + parameters[f'b{l}']
        cache[f'A{l}'] = sigmoid(cache[f'Z{l}'])
    
    # Output layer with softmax
    cache[f'Z{L}'] = np.dot(parameters[f'W{L}'], cache[f'A{L-1}']) + parameters[f'b{L}']
    cache[f'A{L}'] = sigmoid(cache[f'Z{L}'])  # Using sigmoid for binary classification
    
    return cache

def backward_propagation(parameters, cache, X, Y):
    grads = {}
    L = len(parameters) // 2
    m = X.shape[0]
    
    dAL = -(np.divide(Y.T, cache[f'A{L}']) - np.divide(1 - Y.T, 1 - cache[f'A{L}']))
    
    for l in reversed(range(1, L + 1)):
        dZ = dAL * sigmoid_derivative(cache[f'A{l}'])
        grads[f'dW{l}'] = 1/m * np.dot(dZ, cache[f'A{l-1}'].T)
        grads[f'db{l}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dAL = np.dot(parameters[f'W{l}'].T, dZ)
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    return parameters

def compute_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def train_network(X_train, Y_train, X_val, Y_val, layer_dims, epochs=100, learning_rate=0.01, batch_size=32):
    parameters = init_parameters(layer_dims)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Mini-batch training
        indices = np.random.permutation(X_train.shape[0])
        
        for i in range(0, X_train.shape[0], batch_size):
            batch_indices = indices[i:min(i + batch_size, X_train.shape[0])]
            batch_X = X_train[batch_indices]
            batch_Y = Y_train[batch_indices]
            
            cache = forward_propagation(batch_X, parameters)
            grads = backward_propagation(parameters, cache, batch_X, batch_Y)
            parameters = update_parameters(parameters, grads, learning_rate)
        
        # Calculate metrics
        train_cache = forward_propagation(X_train, parameters)
        train_preds = predict(X_train, parameters)
        train_loss = binary_cross_entropy(Y_train.T, train_cache[f'A{len(layer_dims)-1}'].T)
        train_acc = compute_accuracy(train_preds, Y_train)
        
        val_cache = forward_propagation(X_val, parameters)
        val_preds = predict(X_val, parameters)
        val_loss = binary_cross_entropy(Y_val.T, val_cache[f'A{len(layer_dims)-1}'].T)
        val_acc = compute_accuracy(val_preds, Y_val)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
    
    return parameters, history

def plot_metrics(history):
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='training loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='training accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    Y = (data['diagnosis'] == 'M').astype(int)
    X = data.drop(['diagnosis', 'id'], axis=1).values
    
    X = standardize(X)
    
    return X, Y.values.reshape(-1, 1)

def predict(X, parameters):
    cache = forward_propagation(X, parameters)
    return cache[f'A{len(parameters)//2}'].T > 0.5

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Configuration')
    parser.add_argument('--layers', nargs='+', type=int, default=[24, 24, 24],
                      help='Hidden layer sizes')
    parser.add_argument('--epochs', type=int, default=70,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--loss', type=str, default='binaryCrossentropy',
                      choices=['binaryCrossentropy'],
                      help='Loss function')
    return parser.parse_args()

def save_model(parameters, filename='model.npy'):
    np.save(filename, parameters)
    print(f"> saving model '{filename}' to disk...")

def load_model(filename='model.npy'):
    return np.load(filename, allow_pickle=True).item()

def main():
    args = parse_arguments()
    
    # Load and preprocess data
    X, Y = preprocess_data('data.csv')
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_seed=42)
    
    # Print shapes
    print(f'x_train shape : {X_train.shape}')
    print(f'x_valid shape : {X_val.shape}')
    
    # Define network architecture
    input_size = X_train.shape[1]
    layer_dims = [input_size] + args.layers + [1]
    
    # Train model
    parameters, history = train_network(
        X_train, Y_train, X_val, Y_val,
        layer_dims=layer_dims,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    # Save model
    save_model(parameters)
    
    # Plot metrics
    plot_metrics(history)

if __name__ == "__main__":
    main()