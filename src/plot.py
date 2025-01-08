import matplotlib.pyplot as plt
import os
import webbrowser
import platform
import subprocess
from pathlib import Path

def plot_learning_curves(history, network_info):
    """
    Plot and save learning curves with improved visualization.
    """
    # Set style
    plt.style.use('seaborn-darkgrid')
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle("Neural Network Training Results", 
                 y=0.98,
                 fontsize=16, 
                 fontweight='bold')
    
    # Add subtitle with network information
    arch_str = ' → '.join(str(x) for x in network_info['layers'])
    plt.figtext(0.5, 0.91, 
                f"Architecture: [{arch_str}] | Learning Rate: {network_info['lr']:.4f} | Batch Size: {network_info['batch_size']}", 
                ha='center', 
                fontsize=10)

    # Loss subplot
    ax1 = plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax1.set_title('Loss Curves', pad=20)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Fix axis ticks
    ax1.set_xlim([1, len(epochs)])
    ax1.set_ylim([0, max(max(history['train_loss']), max(history['val_loss'])) * 1.1])
    
    # Accuracy subplot
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Accuracy Curves', pad=20)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Fix axis ticks
    ax2.set_xlim([1, len(epochs)])
    ax2.set_ylim([0, 1.1])  # Accuracy is between 0 and 1
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    
    # Save plot
    plot_path = './plots/learning_curves.png'
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Get absolute path
    abs_path = os.path.abspath(plot_path)
    
    def is_wsl():
        """Check if running under Windows Subsystem for Linux"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False

    # Try to open the plot
    try:
        if platform.system().lower() == 'linux':
            if is_wsl():
                try:
                    windows_path = subprocess.check_output(['wslpath', '-w', abs_path]).decode().strip()
                    subprocess.run(['cmd.exe', '/c', 'start', windows_path], 
                                check=True,
                                stderr=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL)
                except:
                    subprocess.run(['xdg-open', abs_path],
                                check=True,
                                stderr=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL)
            else:
                subprocess.run(['xdg-open', abs_path],
                            check=True,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL)
        else:
            webbrowser.open('file://' + abs_path)
    except Exception as e:
        print(f"\nPlot saved as: {abs_path}")
        print(f"Note: Could not open viewer automatically ({str(e)})")