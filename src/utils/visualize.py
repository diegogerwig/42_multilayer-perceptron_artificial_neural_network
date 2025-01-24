# src/visualize.py
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from colorama import Fore, Style

def set_dark_style():
    plt.style.use('dark_background')
    return {
        'bg_color': '#1a1a1a',
        'node_color': '#00a8e8',
        'text_color': '#ffffff',
        'pos_weight': '#00ff00',
        'neg_weight': '#ff4444'
    }

def visualize_model(model_path="./models/trained_model.pkl", show_plot=True):
    print(f"\n{Fore.YELLOW}📊 Visualizing Neural Network{Style.RESET_ALL}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    colors = set_dark_style()
    os.makedirs('./plots', exist_ok=True)
    
    # 1. Architecture Visualization
    visualize_architecture(model, colors, show_plot)
    
    # 2. Weights Distribution
    visualize_weights_distribution(model, colors, show_plot)
    
    # 3. Layer Activation Pattern
    visualize_layer_pattern(model, colors, show_plot)

def visualize_architecture(model, colors, show_plot):
    W = model['W']
    layer_sizes = [W[0].shape[0]] + [w.shape[1] for w in W]
    
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(colors['bg_color'])
    ax = plt.gca()
    ax.set_facecolor(colors['bg_color'])
    
    left, right = 0.1, 0.9
    bottom, top = 0.1, 0.9
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(layer_sizes)-2)] + ['Output']
    
    for n, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        layer_top = v_spacing*(size - 1)/2. + (top + bottom)/2.
        plt.text(n*h_spacing + left, top + 0.1, name, ha='center', color=colors['text_color'])
        
        for m in range(size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                              color=colors['node_color'], fill=False, linewidth=2)
            ax.add_artist(circle)
    
    for n, (size_a, size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(size_b - 1)/2. + (top + bottom)/2.
        weights = W[n]
        max_weight = np.abs(weights).max()
        
        for i in range(size_a):
            for j in range(size_b):
                weight = weights[i, j]
                color = colors['neg_weight'] if weight < 0 else colors['pos_weight']
                alpha = min(0.9, abs(weight) / max_weight)
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                [layer_top_a - i*v_spacing, layer_top_b - j*v_spacing],
                                c=color, alpha=alpha, linewidth=1.5)
                ax.add_artist(line)
    
    plt.axis('off')
    plt.title('Neural Network Architecture', pad=20, size=14, color=colors['text_color'])
    plt.savefig('./plots/network_architecture.png', dpi=300, bbox_inches='tight', facecolor=colors['bg_color'])
    print(f"{Fore.WHITE}   - Architecture saved: {Fore.BLUE}./plots/network_architecture.png{Style.RESET_ALL}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_weights_distribution(model, colors, show_plot):
    plt.figure(figsize=(10, 6))
    plt.gcf().patch.set_facecolor(colors['bg_color'])
    plt.gca().set_facecolor(colors['bg_color'])
    
    for i, w in enumerate(model['W']):
        plt.hist(w.flatten(), bins=50, alpha=0.5, label=f'Layer {i+1}')
    
    plt.title('Weights Distribution by Layer', color=colors['text_color'])
    plt.xlabel('Weight Value', color=colors['text_color'])
    plt.ylabel('Frequency', color=colors['text_color'])
    plt.legend(facecolor=colors['bg_color'], labelcolor=colors['text_color'])
    plt.tick_params(colors=colors['text_color'])
    
    plt.savefig('./plots/weights_distribution.png', dpi=300, bbox_inches='tight', facecolor=colors['bg_color'])
    print(f"{Fore.WHITE}   - Weights distribution saved: {Fore.BLUE}./plots/weights_distribution.png{Style.RESET_ALL}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_layer_pattern(model, colors, show_plot):
    plt.figure(figsize=(12, 4))
    plt.gcf().patch.set_facecolor(colors['bg_color'])
    plt.gca().set_facecolor(colors['bg_color'])
    
    for i, w in enumerate(model['W']):
        plt.subplot(1, len(model['W']), i+1)
        plt.imshow(np.abs(w), cmap='viridis')
        plt.title(f'Layer {i+1} Pattern', color=colors['text_color'])
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('./plots/layer_patterns.png', dpi=300, bbox_inches='tight', facecolor=colors['bg_color'])
    print(f"{Fore.WHITE}   - Layer patterns saved: {Fore.BLUE}./plots/layer_patterns.png{Style.RESET_ALL}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    visualize_model(show_plot=True)