import matplotlib.pyplot as plt
import numpy as np
# from utils.Scaler import Scaler
from utils.normalize import Scaler  
import os

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    # Set dark theme style
    plt.style.use('dark_background')
    
    # Create plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)
    
    # Custom dark theme colors
    background_color = '#1f1f1f'
    text_color = '#e0e0e0'
    grid_color = '#404040'
    
    # Create figure with adjusted size
    fig = plt.figure(figsize=(12, 4))  # Reduced figure size for display
    fig.patch.set_facecolor(background_color)
    
    # Create subplots with specific size ratios
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Common style settings
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(background_color)
        ax.tick_params(colors=text_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.spines['top'].set_color(grid_color) 
        ax.spines['right'].set_color(grid_color)
        ax.spines['left'].set_color(grid_color)
        ax.grid(True, linestyle='--', color=grid_color, alpha=0.3)
    
    # Plot 1: Loss Curves
    ax1.plot(train_losses, label='Training Loss', color='#00ff00', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='#ff3366', linewidth=2)
    ax1.set_xlabel('Epochs', color=text_color, fontsize=10)
    ax1.set_ylabel('Loss', color=text_color, fontsize=10)
    ax1.set_title('Learning Curves - Loss', color=text_color, fontsize=11, pad=10)
    ax1.legend(facecolor=background_color, edgecolor=grid_color, fontsize=9)
    
    # Plot 2: Accuracy Curves
    ax2.plot(train_accuracies, label='Training Accuracy', color='#00ff00', linewidth=2)
    ax2.plot(val_accuracies, label='Validation Accuracy', color='#ff3366', linewidth=2)
    ax2.set_xlabel('Epochs', color=text_color, fontsize=10)
    ax2.set_ylabel('Accuracy', color=text_color, fontsize=10)
    ax2.set_title('Learning Curves - Accuracy', color=text_color, fontsize=11, pad=10)
    ax2.legend(facecolor=background_color, edgecolor=grid_color, fontsize=9)
    
    # Plot 3: Loss Landscape
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    contour = ax3.contourf(X, Y, Z, levels=100, cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax3, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.label.set_color(text_color)
    cbar.ax.tick_params(colors=text_color)
    cbar.outline.set_edgecolor(grid_color)
    cbar.set_label("Cost Function", color=text_color, fontsize=9)
    
    # Scale losses for visualization
    scaler = Scaler(method="minmax")
    norm_losses = scaler.fit_transform(np.array(train_losses).reshape(-1, 1)).flatten()

    # train_losses_array = np.array(train_losses).reshape(-1, 1)
    # norm_losses, _ = scale_data(train_losses_array, method="minmax")
    # norm_losses = norm_losses.flatten()
    
    # Create spiral path
    t = np.linspace(0, 4*np.pi, len(train_losses))
    radius = 3 * (1 - np.exp(-norm_losses*2))
    path_x = radius * np.cos(t)
    path_y = radius * np.sin(t)
    
    # Plot path
    ax3.plot(path_x, path_y, color='#ffffff', alpha=0.5, linewidth=1.5)
    ax3.plot(path_x, path_y, 'x', color='#ffff00', markersize=3, alpha=0.7)
    ax3.plot(path_x[0], path_y[0], 'o', color='#00ff00', markersize=6, label='Start')
    ax3.plot(path_x[-1], path_y[-1], 'o', color='#ff3366', markersize=6, label='End')
    
    ax3.set_xlabel("X₁", color=text_color, fontsize=10)
    ax3.set_ylabel("X₂", color=text_color, fontsize=10)
    ax3.set_title("Loss Landscape & Gradient Path", color=text_color, fontsize=11, pad=10)
    ax3.legend(facecolor=background_color, edgecolor=grid_color, fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save high quality version
    save_path = './plots/learning_curves.png'
    # Save with larger size for better quality when needed
    fig.set_size_inches(20, 6)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=background_color)
    print(f"Plot saved to: {save_path}")
    
    # Reset to smaller size for display
    fig.set_size_inches(24, 8)
    plt.show()