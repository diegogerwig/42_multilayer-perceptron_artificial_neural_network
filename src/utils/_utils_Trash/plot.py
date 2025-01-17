#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import subprocess

def open_file(filepath):
    """Open a file using system's default application"""
    try:
        abs_path = os.path.abspath(filepath)
        subprocess.run(f"xdg-open '{abs_path}'", shell=True)
    except Exception as e:
        print(f"Error opening file: {str(e)}")
        print(f"You can find the file at: {abs_path}")

def plot_learning_curves(history, network_info=None):
    """Plot the learning curves with clean design"""
    # Set style and colors
    plt.style.use('dark_background')
    
    BACKGROUND_COLOR = '#1E1E1E'
    GRID_COLOR = '#333333'
    TEXT_COLOR = '#FFFFFF'
    TRAIN_COLOR = '#00C853'
    VAL_COLOR = '#FF4081'
    
    # Create plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)
    
    # Create figure
    fig = plt.figure(figsize=(15, 7))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    if network_info:
        # Create concise network info string
        arch_text = (
            f"Architecture: {' → '.join(map(str, network_info['layers']))}  |  "
            f"LR: {network_info['lr']:.6f}  |  "
            f"Batch: {network_info['batch_size']}  |  "
            f"Dropout: {network_info['dropout_rate']}"
        )
        
        # Add title with architecture info below
        plt.suptitle("Neural Network Training Results", 
                    fontsize=16, 
                    fontweight='bold', 
                    color=TEXT_COLOR,
                    y=0.95)
        
        plt.figtext(0.5, 0.88, arch_text,
                   ha='center',
                   fontsize=10,
                   color='#CCCCCC',
                #    bbox=dict(facecolor='#2A2A2A',
                #            edgecolor='#404040',
                #            alpha=0.5,
                #            pad=5,
                #            boxstyle='round')
                           )
    
    # Create subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    # Plot training loss
    ax1.set_facecolor(BACKGROUND_COLOR)
    ax1.plot(history['train_loss'], label='Training Loss',
             color=TRAIN_COLOR, linewidth=2, alpha=0.9)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss',
                color=VAL_COLOR, linewidth=2, alpha=0.9)
    ax1.set_title('Model Loss', fontsize=12, pad=15, color=TEXT_COLOR)
    ax1.set_xlabel('Epoch', fontsize=10, color=TEXT_COLOR)
    ax1.set_ylabel('Loss', fontsize=10, color=TEXT_COLOR)
    ax1.grid(True, linestyle='--', alpha=0.2, color=GRID_COLOR)
    ax1.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR)
    
    # Plot training accuracy
    ax2.set_facecolor(BACKGROUND_COLOR)
    ax2.plot(history['train_acc'], label='Training Accuracy',
             color=TRAIN_COLOR, linewidth=2, alpha=0.9)
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Validation Accuracy',
                color=VAL_COLOR, linewidth=2, alpha=0.9)
    ax2.set_title('Model Accuracy', fontsize=12, pad=15, color=TEXT_COLOR)
    ax2.set_xlabel('Epoch', fontsize=10, color=TEXT_COLOR)
    ax2.set_ylabel('Accuracy', fontsize=10, color=TEXT_COLOR)
    ax2.grid(True, linestyle='--', alpha=0.2, color=GRID_COLOR)
    ax2.legend(facecolor=BACKGROUND_COLOR, edgecolor=GRID_COLOR)
    
    # Style adjustments for both plots
    for ax in [ax1, ax2]:
        ax.spines['bottom'].set_color(GRID_COLOR)
        ax.spines['top'].set_color(GRID_COLOR)
        ax.spines['left'].set_color(GRID_COLOR)
        ax.spines['right'].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    # plt.show()

    plot_path = './plots/learning_curves.png'
    plt.savefig(plot_path,
                dpi=300,
                bbox_inches='tight',
                facecolor=BACKGROUND_COLOR,
                edgecolor='none')
    print(f"Plot saved as: {plot_path}")
    
    # Try to open the plot
    open_file(plot_path)
    
    # Close the figure to free memory
    plt.close(fig)