import numpy as np
from utils.weight_init import random, he_normal, he_uniform, glorot_normal, glorot_uniform
from utils.optimizer import gradient_descent, sgd, momentum
from utils.utils import get_accuracy
from utils.early_stopping import check_early_stopping
from utils.activation_functions import (
    sigmoid, sigmoid_derivative,
    relu, relu_derivative,
    softmax
)
from utils.loss_functions import binary_cross_entropy
from colorama import init, Fore, Style

init(autoreset=True)

ACTIVATIONS_FUNCTIONS: dict[str, tuple] = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "softmax": (softmax, None)
}

LOSS_FUNCTIONS: dict[str, callable] = {
    "binaryCrossentropy": binary_cross_entropy,
}


def create_model_config(
    hidden_layer_sizes,
    output_layer_size,
    activation,
    output_activation,
    loss,
    learning_rate,
    epochs,
    batch_size,
    weight_initializer,
    seed,
    optimizer
):
    """Create a configuration dictionary for the neural network."""
    # Validate and set optimizer
    if optimizer not in ["gradient_descent", "sgd", "momentum"]:
        print(f"{Fore.YELLOW}Error: Unknown optimizer '{optimizer}'."
              f'Available choices: ["gradient_descent", "sgd", "momentum"]')
        exit(1)
    
    optimizer_config = {
        'name': optimizer,
        'learning_rate': learning_rate,
        'momentum': 0.9 if optimizer == "momentum" else None
    }
    
    # Validate and set activation function
    if activation not in ACTIVATIONS_FUNCTIONS:
        print(f"{Fore.YELLOW}Error: Unknown activation function '{activation}'.\n"
              f"Available choices: {list(ACTIVATIONS_FUNCTIONS.keys())}")
        exit(1)
    
    activation_func, activation_derivative = ACTIVATIONS_FUNCTIONS[activation]
    
    # Validate loss function
    if loss not in LOSS_FUNCTIONS:
        print(f"{Fore.YELLOW}Error: Unknown loss function '{loss}'.\n"
              f"Available choices: {list(LOSS_FUNCTIONS.keys())}")
        exit(1)
    
    return {
        'hidden_layer_sizes': hidden_layer_sizes,
        'output_layer_size': output_layer_size,
        'activation': activation_func,
        'activation_derivative': activation_derivative,
        'activation_name': activation,
        'output_activation': softmax,
        'output_activation_name': "softmax",
        'loss': LOSS_FUNCTIONS[loss],
        'loss_name': loss,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'weight_initializer': weight_initializer,
        'seed': seed,
        'optimizer': optimizer_config
    }

def feed_forward_propagation(X, W, b, activation_func, output_activation):
    """Feed forward pass through the network."""
    # Initialize list to store activations for backpropagation
    A = []
    a = X
    
    # Process hidden layers with specified activation
    for i in range(len(W) - 1):
        z = np.dot(a, W[i]) + b[i]  # Linear transformation
        a = activation_func(z)      # Activation function
        A.append(a)                 # Save activation for backpropagation
    
    # Output layer always uses softmax and outputs shape (m, 2)
    z = np.dot(a, W[-1]) + b[-1]    # Linear transformation
    if z.shape[1] != 2:
        raise ValueError(f"Expected output to have 2 columns, got shape {z.shape}")
    output = output_activation(z)   # Softmax activation
    return output, A

def back_propagation(X, y, output, A, W, activation_derivative):
    """Backward propagation to compute gradients."""
    # Initialize gradients
    m = X.shape[0]
    dW, db = [], []  # Gradients for weights and biases
    
    # Convert outputs and labels to appropriate shapes
    output = np.array(output)  # Shape (m, 2) from softmax
    y = np.array(y).reshape(-1, 1)  # Shape (m, 1)
    
    # Initial gradient for softmax with binary cross-entropy
    dz_full = np.zeros_like(output)
    dz_full[:, 1] = output[:, 1] - y.ravel()  # Gradient for positive class. ravel() is a NumPy function that flattens a multi-dimensional array into a 1D array
    dz_full[:, 0] = -dz_full[:, 1]            # Gradient for negative class
    
    for i in reversed(range(len(W))):
        a_prev = A[i - 1] if i > 0 else X
        
        # Compute gradients with respect to weights and biases
        dW_i = np.dot(a_prev.T, dz_full)
        db_i = np.sum(dz_full, axis=0, keepdims=True)
        
        # Clip gradients to prevent explosion
        clip_value = 1.0
        dW_i = np.clip(dW_i, -clip_value, clip_value)
        db_i = np.clip(db_i, -clip_value, clip_value)
        
        # Save gradients
        dW.insert(0, dW_i)
        db.insert(0, db_i)
        
        # Propagate the gradient to previous layer
        if i > 0:
            da = np.dot(dz_full, W[i].T)
            dz_full = da * activation_derivative(A[i - 1])
            dz_full = np.clip(dz_full, -clip_value, clip_value)
    
    return dW, db

def init_network(layer_sizes, weight_initializer, random_seed=None):
    """Initialize network weights and biases."""
    # Initialize lists to store weights and biases
    W = []
    b = []
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]
        
        biases = np.zeros((1, output_size))  # Initialize biases to zeros
        
        # Weight initialization
        match weight_initializer:
            case "Random":
                weight = random(input_size, output_size)
            case "HeNormal":
                weight = he_normal(input_size, output_size)
            case "HeUniform":
                weight = he_uniform(input_size, output_size)
            case "GlorotNormal":
                weight = glorot_normal(input_size, output_size)
            case "GlorotUniform":
                weight = glorot_uniform(input_size, output_size)
            case _:
                print(f"{Fore.YELLOW}Error while initializing weights")
                exit(1)

        print(f"{Fore.GREEN}Layer {i+1} - Weight shape: {weight.shape} - Bias shape: {biases.shape}")
        print(f"   Weight: {weight}")
        print(f"   Bias: {biases}")
        
        W.append(weight)
        b.append(biases)
    
    return W, b

def fit_network(X_train, y_train, X_val, y_val, config, early_stopping_config):
    """Train the neural network model."""
    # Convert inputs to numpy arrays and ensure correct shapes
    X_train = np.array(X_train)             # Shape (m, n) where m is the number of samples and n is the number of features
    y_train = np.ravel(np.array(y_train))   # Shape (m,)   Ravel() is a NumPy function that flattens a multi-dimensional array into a 1D array
    
    # Validate validation data
    if X_val is not None and y_val is not None:
        X_val = np.array(X_val)
        y_val = np.ravel(np.array(y_val))
    
    # Initialize network
    input_layer_size = X_train.shape[1]  # Number of features
    layer_sizes = [input_layer_size] + config['hidden_layer_sizes'] + [config['output_layer_size']]  # Full network architecture
    W, b = init_network(layer_sizes, config['weight_initializer'], config['seed'])  # Initialize weights and biases
    
    # Initialize metrics tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    # Initialize best weights and biases for early stopping
    best_W, best_b = None, None
    early_stopping_state = None
    final_val_predictions = None
    
    # Initialize early stopping if configured
    if early_stopping_config and early_stopping_config.get('enabled', False):
        early_stopping_state = {
            'best_loss': None,
            'counter': 0,
            'early_stop': False
        }
    
    # Initialize optimizer state
    optimizer_state = None

    # Training loop
    for epoch in range(config['epochs']):
        # Mini-batch training
        for i in range(0, len(X_train), config['batch_size']):
            batch_X = X_train[i:i+config['batch_size']]
            batch_y = y_train[i:i+config['batch_size']]
            
            output, A = feed_forward_propagation(batch_X, W, b, config['activation'], config['output_activation'])
            dW, db = back_propagation(batch_X, batch_y, output, A, W, config['activation_derivative'])
            
            # Update weights and biases
            match config['optimizer']['name']:
                case 'momentum':
                    W, b, optimizer_state = momentum(W, b, dW, db, config['optimizer']['learning_rate'], 
                                                config['optimizer']['momentum'], optimizer_state)
                case 'sgd':
                    W, b, optimizer_state = sgd(W, b, dW, db, config['optimizer']['learning_rate'], optimizer_state)
                case _:  # default: gradient descent
                    W, b, optimizer_state = gradient_descent(W, b, dW, db, config['optimizer']['learning_rate'], optimizer_state)
        
        # Compute metrics
        train_output, _ = feed_forward_propagation(X_train, W, b, config['activation'], config['output_activation'])
        val_output, _ = feed_forward_propagation(X_val, W, b, config['activation'], config['output_activation'])
        
        # Store validation predictions for the current epoch
        final_val_predictions = val_output[:, 1]  # Store probabilities for positive class
        
        try:
            train_loss = config['loss'](y_train, train_output)
            val_loss = config['loss'](y_val, val_output)
        except:
            raise ValueError("Invalid loss function for this training model")
        
        train_accuracy = get_accuracy(train_output, y_train)
        val_accuracy = get_accuracy(val_output, y_val)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        epoch_width = len(str(config['epochs']))
        print(f"{Fore.MAGENTA}epoch {Fore.CYAN}{epoch+1:>{epoch_width}}/{config['epochs']}"
              f"{Fore.GREEN} - {Fore.MAGENTA}train_loss: {Fore.CYAN}{train_loss:.4f}"
              f"{Fore.GREEN} - {Fore.MAGENTA}val_loss: {Fore.CYAN}{val_loss:.4f}"
              f"{Fore.GREEN} - {Fore.MAGENTA}train_acc: {Fore.CYAN}{train_accuracy:.4f}"
              f"{Fore.GREEN} - {Fore.MAGENTA}val_acc: {Fore.CYAN}{val_accuracy:.4f}")
        
        # Check early stopping if enabled
        if early_stopping_state is not None:
            should_stop, early_stopping_state = check_early_stopping(
                val_loss,
                early_stopping_state,
                patience=early_stopping_config.get('patience'),
                min_delta=early_stopping_config.get('min_delta')
            )
            
            # Save best weights when improvement is seen
            if early_stopping_state['counter'] == 0:
                best_W = [w.copy() for w in W]
                best_b = [b.copy() for b in b]
                # Store validation predictions for best model
                final_val_predictions = val_output[:, 1]
            
            if should_stop:
                print(f"{Fore.GREEN}Early stopping triggered at epoch {epoch + 1}")
                # Restore best weights
                W, b = best_W, best_b
                break
    
    # If early stopping wasn't used, make sure we have the final predictions
    if final_val_predictions is None:
        val_output, _ = feed_forward_propagation(X_val, W, b, config['activation'], config['output_activation'])
        final_val_predictions = val_output[:, 1]
    
    return W, b, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }, final_val_predictions