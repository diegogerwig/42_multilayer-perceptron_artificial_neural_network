import matplotlib.pyplot as plt
import numpy as np
import os
import threading
from utils.normalize import fit_transform_data

import matplotlib.pyplot as plt
import numpy as np
import os
import threading
from sklearn.metrics import roc_curve
import seaborn as sns

def wait_for_input(skip_input=False):
    input("\nPress Enter to continue...")
    plt.close('all')

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, skip_input=False):
    """
    Plot learning curves with interactive continuation options.
    """
    # Set dark theme style
    plt.style.use('dark_background')
    
    # Create plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)
    
    # Custom dark theme colors
    background_color = '#1f1f1f'
    text_color = '#e0e0e0'
    grid_color = '#404040'
    
    # Create figure with adjusted size
    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor(background_color)

    # Add main title with custom styling
    fig.suptitle('ARTIFICIAL NEURAL NETWORK', 
                fontsize=16,
                fontweight='bold',
                color=text_color,
                y=0.95)  

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
    
    contour = ax3.contourf(X, Y, Z, levels=100, cmap='plasma')
    cbar = fig.colorbar(contour, ax=ax3, fraction=0.02, pad=0.04)
    cbar.ax.yaxis.label.set_color(text_color)
    cbar.ax.tick_params(colors=text_color)
    cbar.outline.set_edgecolor(grid_color)
    cbar.set_label("Cost Function", color=text_color, fontsize=9)
    
    # Scale losses for visualization
    train_losses_array = np.array(train_losses).reshape(-1, 1)
    norm_losses, _ = fit_transform_data(train_losses_array, method="minmax")
    norm_losses = norm_losses.flatten()
    
    # Create spiral path
    t = np.linspace(0, 4*np.pi, len(train_losses))
    radius = 3 * (1 - np.exp(-norm_losses*2))
    path_x = radius * np.cos(t)
    path_y = radius * np.sin(t)
    
    # Plot path
    ax3.plot(path_x, path_y, color='#ffffff', alpha=0.5, linewidth=1.5)
    ax3.plot(path_x, path_y, 'x', color='#ffff00', markersize=3, alpha=0.7)
    ax3.plot(path_x[0], path_y[0], 'o', color='#00ff00', markersize=10, label='Start')
    ax3.plot(path_x[-1], path_y[-1], 'o', color='#ff3366', markersize=10, label='End')
    
    ax3.set_xlabel("X₁", color=text_color, fontsize=10)
    ax3.set_ylabel("X₂", color=text_color, fontsize=10)
    ax3.set_title("Loss Landscape & Gradient Path", color=text_color, fontsize=11, pad=10)
    ax3.legend(facecolor=background_color, edgecolor=grid_color, fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save high quality version
    save_path = './plots/learning_curves.png'
    fig.set_size_inches(15, 5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=background_color)
    print(f"\n📷 Plot saved to: {save_path}")

    if not skip_input:
        print("\n❗ You can either:")
        print("   - Close the plot window")
        print("   - Press Enter to continue")
        
        # Create a thread to wait for Enter key
        input_thread = threading.Thread(target=wait_for_input)
        input_thread.daemon = True
        input_thread.start()
        
        # Show plot only if not skipping input
        plt.show()
    
    plt.close('all')  # Close all figures to prevent memory leaks

def plot_prediction_results(metrics, probas=None, y_true=None, skip_input=False):
    """
    Plot prediction results with interactive continuation options.
    """
    plt.style.use('dark_background')
    
    # Style settings
    background_color = '#1f1f1f'
    text_color = '#e0e0e0'
    grid_color = '#404040'
    
    # Create plots directory
    os.makedirs('./plots', exist_ok=True)
    
    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor(background_color)
    
    fig.suptitle('MODEL PREDICTION RESULTS', 
                fontsize=16,
                fontweight='bold',
                color=text_color,
                y=0.95)
    
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Common style
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(background_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.grid(True, linestyle='--', color=grid_color, alpha=0.3)
    
    # Plot 1: Metrics Bar Plot
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    values = [metrics[m] for m in metrics_to_plot]
    
    bars = ax1.bar(metrics_to_plot, values, color=['#00ff00', '#00bbff', '#ff3366', '#ffff00', '#ff00ff'])
    ax1.set_ylim(0, 1)
    ax1.set_title('Performance Metrics', color=text_color, fontsize=11, pad=10)
    ax1.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', color=text_color)
    
    # Plot 2: ROC Curve
    if probas is not None and y_true is not None:
        fpr, tpr, _ = roc_curve(y_true, probas[:, 1])
        ax2.plot(fpr, tpr, color='#00ff00', linewidth=2, label=f'ROC (AUC = {metrics["auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], color='#ff3366', linestyle='--', label='Random')
        ax2.set_xlabel('False Positive Rate', color=text_color)
        ax2.set_ylabel('True Positive Rate', color=text_color)
        ax2.set_title('ROC Curve', color=text_color, fontsize=11, pad=10)
        ax2.legend(facecolor=background_color, edgecolor=grid_color)
    
    # Plot 3: Confusion Matrix
    cm = np.array([
        [metrics['confusion_matrix']['true_negatives'], metrics['confusion_matrix']['false_positives']],
        [metrics['confusion_matrix']['false_negatives'], metrics['confusion_matrix']['true_positives']]
    ])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', ax=ax3, 
                cbar_kws={'label': 'Count'})
    ax3.set_xlabel('Predicted', color=text_color)
    ax3.set_ylabel('Actual', color=text_color)
    ax3.set_title('Confusion Matrix', color=text_color, fontsize=11, pad=10)
    
    plt.tight_layout()
    
    # Save plot
    save_path = './plots/prediction_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=background_color)
    print(f"\n📷 Plot saved to: {save_path}")

    if not skip_input:
        print("\n❗ You can either:")
        print("   - Close the plot window")
        print("   - Press Enter to continue")
        
        # Create thread for input
        input_thread = threading.Thread(target=wait_for_input)
        input_thread.daemon = True
        input_thread.start()
        
        # Show plot if not skipping
        plt.show()
    
    plt.close('all')