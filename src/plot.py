import matplotlib.pyplot as plt
import os
import webbrowser
import platform
import subprocess
from pathlib import Path

def plot_learning_curves(history, network_info):
    """
    Plot and save learning curves, then open them in the appropriate viewer.
    """
    plt.figure(figsize=(15, 6))

    plt.suptitle("Neural Network Training Results", 
                y=1.05,
                fontsize=16,
                fontweight='bold')
    
    plt.title(f"Architecture: {network_info['layers']} | Learning Rate: {network_info['lr']:.4f} | " +
             f"Batch Size: {network_info['batch_size']}", 
             fontsize=10,
             pad=20)  
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='training loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='training accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Ensure plots directory exists and save figure
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

    # Try to open the plot with appropriate viewer
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