import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
import os
from colorama import init, Fore, Style

init()

def set_dark_style():
   plt.style.use('dark_background')
   return {
       'bg_color': '#1a1a1a',
       'node_color': '#00a8e8',
       'text_color': '#ffffff',
       'pos_weight': '#00ff00',
       'neg_weight': '#ff4444'
   }

def zoom_factory(event, ax, base_scale=1.2):
   def zoom(event):
       cur_xlim = ax.get_xlim()
       cur_ylim = ax.get_ylim()
       xdata = event.xdata
       ydata = event.ydata
       
       if event.button == 'up':
           scale_factor = 1/base_scale
       elif event.button == 'down':
           scale_factor = base_scale
       else:
           scale_factor = 1

       new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
       new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

       relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
       rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

       ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * relx])
       ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * rely])
       plt.draw()
   return zoom

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
   
   feature_names = []
   output_names = ['Bening', 'Malign']
   layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(layer_sizes)-2)] + ['Output']
   
   layer_tops = []
   node_positions = {}

   for n, (size, name) in enumerate(zip(layer_sizes, layer_names)):
       layer_top = v_spacing*(size - 1)/2. + (top + bottom)/2.
       layer_tops.append(layer_top)
       node_positions[n] = []
       
       plt.text(n*h_spacing + left, top + 0.1, name, ha='center', color=colors['text_color'],
               fontsize=12, fontweight='bold')
       
       for m in range(size):
           x, y = n*h_spacing + left, layer_top - m*v_spacing
           node_positions[n].append((x, y))
           circle = plt.Circle((x, y), v_spacing/4., color=colors['node_color'], 
                             fill=False, linewidth=2, zorder=4)
           ax.add_artist(circle)
           
           if n == 0 and m < len(feature_names):
               plt.text(x - v_spacing/4., y, feature_names[m], color=colors['text_color'],
                       ha='right', va='center', fontsize=8)
           elif n == len(layer_sizes)-1 and m < len(output_names):
               plt.text(x + v_spacing/4., y, output_names[m], color=colors['text_color'],
                       ha='left', va='center', fontsize=8)
           else:
               plt.text(x - v_spacing/4., y, f'N{m}', color=colors['text_color'],
                       ha='right', va='center', fontsize=8)

   weights_text = []
   for n, (size_a, size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
       weights = W[n]
       max_weight = np.abs(weights).max()
       
       for i in range(size_a):
           for j in range(size_b):
               start = node_positions[n][i]
               end = node_positions[n+1][j]
               weight = weights[i, j]
               
               color = colors['neg_weight'] if weight < 0 else colors['pos_weight']
               alpha = min(0.9, abs(weight) / max_weight)
               width = 1.5 * (abs(weight) / max_weight + 0.5)
               
               line = plt.Line2D([start[0], end[0]], [start[1], end[1]],
                               color=color, alpha=alpha, linewidth=width,
                               picker=5, zorder=2)
               ax.add_artist(line)
               
               mid_x = (start[0] + end[0]) / 2
               mid_y = (start[1] + end[1]) / 2
               bbox_props = dict(boxstyle="round,pad=0.3", fc='black', ec=color, alpha=0.7)
               
               txt = ax.text(mid_x, mid_y, f'{weight:.2f}', 
                           color=colors['text_color'],
                           visible=False, ha='center', va='center',
                           bbox=bbox_props, zorder=3)
               weights_text.append((line, txt))

   def on_pick(event):
       for line, txt in weights_text:
           txt.set_visible(line == event.artist)
       fig.canvas.draw()

   fig.canvas.mpl_connect('pick_event', on_pick)
   fig.canvas.mpl_connect('scroll_event', lambda event: zoom_factory(event, ax)(event))
   
   ax.set_xlim(left - 0.1, right + 0.1)
   ax.set_ylim(bottom - 0.1, top + 0.1)
   
   plt.axis('off')
   plt.title('Neural Network Architecture\nClick connections to see weights • Use scroll to zoom', 
             pad=20, size=14, color=colors['text_color'], fontweight='bold')
   
   plt.tight_layout()
   plt.savefig('./plots/network_architecture.png', dpi=300, bbox_inches='tight', 
               facecolor=colors['bg_color'])
   print(f"{Fore.WHITE}   - Architecture saved: {Fore.BLUE}./plots/network_architecture.png{Style.RESET_ALL}")
   
   if show_plot:
       plt.show()
   else:
       plt.close()

def visualize_model(model_path="./models/trained_model.pkl", show_plot=True):
   print(f"\n{Fore.YELLOW}📊 Visualizing Neural Network{Style.RESET_ALL}")
   
   with open(model_path, 'rb') as f:
       model = pickle.load(f)
   
   colors = set_dark_style()
   os.makedirs('./plots', exist_ok=True)
   visualize_architecture(model, colors, show_plot)

if __name__ == "__main__":
   visualize_model(show_plot=True)