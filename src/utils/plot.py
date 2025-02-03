import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
import threading
from utils.normalize import fit_transform_data
from sklearn.metrics import roc_curve
import seaborn as sns
from colorama import init, Fore, Style

init(autoreset=True)  # Initialize colorama for colored text output. Autoreset colors after each print.


def wait_for_input(skip_input=False):
    input("\nPress Enter to continue... \n")
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
    fig = plt.figure(figsize=(32, 8))
    fig.patch.set_facecolor(background_color)

    # Add main title with custom styling
    fig.suptitle('ARTIFICIAL NEURAL NETWORK', 
                fontsize=20,
                fontweight='bold',
                color=text_color,
                y=0.98)  

    # Create subplots with specific size ratios
    gs = fig.add_gridspec(1, 4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3], projection='3d')
    
    # Common style settings
    for ax in [ax1, ax2, ax3, ax4]:
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
    ax3.set_title("2D Loss Landscape & Gradient Path", color=text_color, fontsize=11, pad=10)
    ax3.legend(facecolor=background_color, edgecolor=grid_color, fontsize=9)
    
    # Plot 4: Loss Landscape 3D
    surface = ax4.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
    z_path = path_x**2 + path_y**2
    ax4.plot3D(path_x, path_y, z_path, color='#ffffff', alpha=0.5, linewidth=1.5)
    ax4.scatter3D(path_x[0], path_y[0], z_path[0], color='#00ff00', s=100, label='Start')
    ax4.scatter3D(path_x[-1], path_y[-1], z_path[-1], color='#ff3366', s=100, label='End')

    ax4.set_xlabel("X₁", color=text_color, fontsize=10)
    ax4.set_ylabel("X₂", color=text_color, fontsize=10)
    ax4.set_zlabel("Cost", color=text_color, fontsize=10)
    ax4.set_title("3D Loss Landscape & Gradient Path", color=text_color, fontsize=11, pad=10)
    ax4.legend(facecolor=background_color, edgecolor=grid_color, fontsize=9)

    # Adjust layout
    plt.tight_layout()
    
    # Save high quality version
    save_path = './plots/learning_curves.png'
    # fig.set_size_inches(32, 8)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=background_color)
    print(f"\n📷 Plot saved to: {Fore.BLUE}{save_path}")

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
    
    fig = plt.figure(figsize=(24, 8))
    fig.patch.set_facecolor(background_color)
    
    fig.suptitle('MODEL PREDICTION RESULTS', 
                fontsize=20,
                fontweight='bold',
                color=text_color,
                y=0.98)
    
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
        [metrics['confusion_matrix']['true_positives'], metrics['confusion_matrix']['false_negatives']],
        [metrics['confusion_matrix']['false_positives'], metrics['confusion_matrix']['true_negatives']]
    ])

    # Create custom colormap
    colors = ['#8B0000', '#2e5c35']
    custom_cmap = ListedColormap(colors, N=2)

    # Plot heatmap with custom colors
    sns.heatmap(cm, 
        annot=True, 
        fmt='d',
        cmap=custom_cmap,
        ax=ax3,
        cbar=False,
        annot_kws={
            'size': 40,
            'weight': 'bold',
            'color': 'white'
        },
        square=True
    )

    # Add category labels in correct positions
    labels = ['TRUE\nPOSITIVE', 'FALSE\nNEGATIVE', 
            'FALSE\nPOSITIVE', 'TRUE\nNEGATIVE']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (i, j), label in zip(positions, labels):
        ax3.text(j + 0.5, i + 0.75, label,
            ha='center', 
            va='center',
            color='white', 
            fontsize=20
        )

    # Customize axes
    ax3.set_xlabel('Predicted', color=text_color, fontsize=12, labelpad=10)
    ax3.set_ylabel('Verified', color=text_color, fontsize=12, labelpad=10)
    ax3.set_title('Confusion Matrix', color=text_color, fontsize=14, pad=20)

    # Remove ticks
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.tight_layout()
        
    # Save plot
    save_path = './plots/prediction_results.png'
    # fig.set_size_inches(24, 8)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=background_color)
    print(f"\n📷 Plot saved to: {Fore.BLUE}{save_path}")

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

def plot_model_analysis(W, b, skip_input=False):
    """Plot model architecture analysis with interactive visualization"""
    plt.style.use('dark_background')
    
    # Create plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)
    
    # Custom colors
    background_color = '#1f1f1f'
    text_color = '#e0e0e0'
    grid_color = '#404040'
    
    # Create figure
    fig = plt.figure(figsize=(32, 8))
    fig.patch.set_facecolor(background_color)

    # Add main title
    fig.suptitle('MODEL ARCHITECTURE ANALYSIS', 
                fontsize=20,
                fontweight='bold',
                color=text_color,
                y=0.98)

    # Create subplots
    gs = fig.add_gridspec(1, 4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Common style settings
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(background_color)
        ax.tick_params(colors=text_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.spines['top'].set_color(grid_color)
        ax.spines['right'].set_color(grid_color)
        ax.spines['left'].set_color(grid_color)
        ax.grid(True, linestyle='--', color=grid_color, alpha=0.3)
    
    # Plot 1: Network Architecture as connected layers
    layer_sizes = [w.shape[0] for w in W] + [W[-1].shape[1]]
    max_size = max(layer_sizes)
    
    # Create visualization of connected layers
    for i, size in enumerate(layer_sizes):
        x = [i] * size
        y = np.linspace(-size/2, size/2, size)
        # Draw nodes
        ax1.scatter([i]*len(y), y, color='#00ff00', s=100, alpha=0.6)
        # If not the last layer, draw connections
        if i < len(layer_sizes)-1:
            next_size = layer_sizes[i+1]
            next_y = np.linspace(-next_size/2, next_size/2, next_size)
            for current_y in y:
                for ny in next_y:
                    ax1.plot([i, i+1], [current_y, ny], color='#404040', alpha=0.1)
    
    ax1.set_title('Network Architecture', color=text_color, pad=20)
    ax1.set_xticks(range(len(layer_sizes)))
    ax1.set_xticklabels([f'Layer {i}\n({size} neurons)' for i, size in enumerate(layer_sizes)])
    
    # Plot 2: Weight Distributions
    for i, w in enumerate(W):
        sns.kdeplot(data=w.flatten(), ax=ax2, label=f'Layer {i+1}', alpha=0.6)
    ax2.set_xlabel('Weight Value', color=text_color)
    ax2.set_ylabel('Density', color=text_color)
    ax2.set_title('Weight Distributions', color=text_color, pad=20)
    ax2.legend(facecolor=background_color, edgecolor=grid_color)
    
    # Plot 3: Weight Magnitudes Boxplot
    box_data = [w.flatten() for w in W]
    ax3.boxplot(box_data, patch_artist=True,
                boxprops=dict(facecolor='#00ff00', color=text_color, alpha=0.6),
                medianprops=dict(color='#ff3366'),
                whiskerprops=dict(color=text_color),
                capprops=dict(color=text_color),
                flierprops=dict(color='#00ff00', markerfacecolor='#00ff00', alpha=0.6))
    
    ax3.set_xticklabels([f'Layer {i+1}' for i in range(len(W))])
    ax3.set_title('Weight Ranges by Layer', color=text_color, pad=20)
    ax3.set_ylabel('Weight Value', color=text_color)
    
    # Plot 4: Parameter Counts
    params_per_layer = [w.size + b.size for w, b in zip(W, b)]
    ax4.bar(range(len(W)), params_per_layer, color='#00ff00', alpha=0.6)
    ax4.set_title('Parameters per Layer', color=text_color, pad=20)
    ax4.set_xlabel('Layer', color=text_color)
    ax4.set_ylabel('Number of Parameters', color=text_color)
    ax4.set_xticks(range(len(W)))
    ax4.set_xticklabels([f'Layer {i+1}' for i in range(len(W))])
    
    # Add values above bars
    for i, v in enumerate(params_per_layer):
        ax4.text(i, v, str(v), ha='center', va='bottom', color=text_color)
    
    plt.tight_layout()
    
    # Save plot
    save_path = './plots/model_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=background_color)
    print(f"\n📷 Plot saved to: {Fore.BLUE}{save_path}")

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