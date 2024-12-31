import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def init_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return parameters

def forward_propagation(X, parameters):
    cache = {'A0': X.T}
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        cache[f'Z{l}'] = np.dot(parameters[f'W{l}'], cache[f'A{l-1}']) + parameters[f'b{l}']
        cache[f'A{l}'] = sigmoid(cache[f'Z{l}'])
    
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

def train_model(X_train, Y_train, X_val, Y_val, layer_dims, epochs=100, learning_rate=0.01, batch_size=32):
    parameters = init_parameters(layer_dims)
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            batch_X = X_train[i:min(i + batch_size, X_train.shape[0])]
            batch_Y = Y_train[i:min(i + batch_size, Y_train.shape[0])]
            
            cache = forward_propagation(batch_X, parameters)
            grads = backward_propagation(parameters, cache, batch_X, batch_Y)
            parameters = update_parameters(parameters, grads, learning_rate)
        
        # Calculate metrics
        train_cache = forward_propagation(X_train, parameters)
        train_loss = binary_cross_entropy(Y_train.T, train_cache[f'A{len(layer_dims)-1}'].T)
        
        val_cache = forward_propagation(X_val, parameters)
        val_loss = binary_cross_entropy(Y_val.T, val_cache[f'A{len(layer_dims)-1}'].T)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')
    
    return parameters, history

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='training loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    Y = (data['diagnosis'] == 'M').astype(int)
    X = data.drop(['diagnosis', 'id'], axis=1)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, Y.values.reshape(-1, 1)

def predict(X, parameters):
    cache = forward_propagation(X, parameters)
    return cache[f'A{len(parameters)//2}'].T > 0.5

def main():
    # Load and preprocess data
    X, Y = preprocess_data('data.csv')
    
    # Split data
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Define network architecture
    layer_dims = [30, 24, 24, 24, 1]  # input, hidden layers, output
    
    # Train model
    parameters, history = train_model(X_train, Y_train, X_val, Y_val, 
                                    layer_dims, epochs=70, learning_rate=0.01)
    
    # Plot results
    plot_history(history)
    
    # Get predictions
    train_predictions = predict(X_train, parameters)
    val_predictions = predict(X_val, parameters)
    
    # Calculate accuracy
    train_accuracy = np.mean(train_predictions == Y_train)
    val_accuracy = np.mean(val_predictions == Y_val)
    
    print(f'\nTraining accuracy: {train_accuracy:.4f}')
    print(f'Validation accuracy: {val_accuracy:.4f}')

if __name__ == "__main__":
    main()