import matplotlib.pyplot as plt
import os
import webbrowser

def plot_learning_curves(history, network_info):
    plt.figure(figsize=(15, 6))

    plt.suptitle(f"Neural Network Training Results\n" + 
                f"Architecture: {network_info['layers']} | Learning Rate: {network_info['lr']:.4f} | " +
                f"Batch Size: {network_info['batch_size']}", y=1.02)
    
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
    plot_path = './plots/learning_curves.png'
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    abs_path = os.path.abspath(plot_path)
    webbrowser.open('file://' + abs_path)