import matplotlib.pyplot as plt
import os
import webbrowser
import platform
import subprocess
from pathlib import Path

def plot_learning_curves(history, network_info):
    """
    Plot and save learning curves with enhanced visualization.
    Shows complete training and test curves.
    """
    # Define custom colors
    background_color = '#2F3035'  # Dark gray background
    grid_color = '#3D3F44'        # Slightly lighter gray for grid
    text_color = '#E0E0E0'        # Light gray for text
    train_color = '#00A6FB'       # Bright blue for training
    val_color = '#FF6B6B'        # Coral red for test
    
    # Create figure with custom background
    fig = plt.figure(figsize=(15, 7), facecolor=background_color)
    
    # Main title with custom styling
    fig.suptitle("Neural Network Training Results", 
                 y=0.98,
                 fontsize=16, 
                 fontweight='bold',
                 color=text_color)
    
    # Architecture subtitle
    arch_str = ' → '.join(str(x) for x in network_info['layers'])
    plt.figtext(0.5, 0.91, 
                f"Architecture: [{arch_str}] | Learning Rate: {network_info['lr']:.4f} | Batch Size: {network_info['batch_size']}", 
                ha='center', 
                fontsize=10,
                color=text_color)

    # Loss subplot
    ax1 = plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot both loss curves
    ax1.plot(epochs, history['train_loss'], color=train_color, 
             label=f'Training Loss: {history["train_loss"][-1]:.4f}', linewidth=2, linestyle='-')
    ax1.plot(epochs, history['val_loss'], color=val_color, 
             label=f'Validation Loss: {history["val_loss"][-1]:.4f}', linewidth=2, linestyle='-')
    
    # Configure subplot
    ax1.set_title('Loss Curves', pad=20, color=text_color)
    ax1.set_xlabel('Epoch', color=text_color)
    ax1.set_ylabel('Loss', color=text_color)
    ax1.legend(loc='upper right', facecolor=grid_color, edgecolor=grid_color)
    ax1.grid(True, linestyle='--', alpha=0.3, color=grid_color)
    ax1.set_facecolor(background_color)
    
    # Style ticks
    ax1.tick_params(colors=text_color)
    for spine in ax1.spines.values():
        spine.set_color(grid_color)
    
    # Accuracy subplot
    ax2 = plt.subplot(1, 2, 2)
    
    # Plot both accuracy curves
    ax2.plot(epochs, history['train_acc'], color=train_color, 
             label=f'Training Acc: {history["train_acc"][-1]:.4f}', linewidth=2, linestyle='-')
    ax2.plot(epochs, history['val_acc'], color=val_color, 
             label=f'Validation Acc: {history["val_acc"][-1]:.4f}', linewidth=2, linestyle='-')
    
    # Configure subplot
    ax2.set_title('Accuracy Curves', pad=20, color=text_color)
    ax2.set_xlabel('Epoch', color=text_color)
    ax2.set_ylabel('Accuracy', color=text_color)
    ax2.legend(loc='lower right', facecolor=grid_color, edgecolor=grid_color)
    ax2.grid(True, linestyle='--', alpha=0.3, color=grid_color)
    ax2.set_facecolor(background_color)
    
    # Style ticks
    ax2.tick_params(colors=text_color)
    for spine in ax2.spines.values():
        spine.set_color(grid_color)
    
    # Fix axis ticks
    ax2.set_xlim([1, len(epochs)])
    ax2.set_ylim([0, 1.1])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Style legend text color
    for ax in [ax1, ax2]:
        legend = ax.get_legend()
        for text in legend.get_texts():
            text.set_color(text_color)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    
    # Save plot
    plot_path = './plots/learning_curves.png'
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                facecolor=background_color)
    plt.close()
    
    # Try to open the plot
    try:
        if platform.system().lower() == 'linux':
            subprocess.run(['xdg-open', plot_path], check=True)
        else:
            webbrowser.open('file://' + os.path.abspath(plot_path))
        print(f"\nPlot saved and opened: {plot_path}")
    except Exception as e:
        print(f"\nPlot saved as: {plot_path}")
        print(f"Note: Could not open viewer automatically ({str(e)})")