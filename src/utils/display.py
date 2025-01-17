import matplotlib.pyplot as plt
import numpy as np
import os
from utils.Scaler import Scaler
from colorama import init, Fore, Style

init()  # Initialize colorama for colored text output

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs('./models/plots', exist_ok=True)
    
    # Save the figure
    plot_path = './models/plots/learning_curves.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"{Fore.YELLOW}📊 Learning curves saved:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}   - Plot saved to: {Fore.BLUE}{plot_path}{Style.RESET_ALL}")
