import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd

# Load model
with open("./models/trained_model.pkl", "rb") as file:
    model_data = pickle.load(file)

# Get number of features from weights shape
n_features = model_data['W'][0].shape[0]
X_sample = np.random.rand(1, n_features)

# Get layer outputs
def get_layer_outputs(W, b, X):
    layer_outputs = []
    X_current = X
    for i in range(len(W)):
        X_current = np.tanh(np.dot(X_current, W[i]) + b[i])
        layer_outputs.append(X_current)
    return layer_outputs

activations = get_layer_outputs(model_data['W'], model_data['b'], X_sample)

# Plot activations
fig, axes = plt.subplots(1, len(activations), figsize=(15, 5))
if len(activations) == 1:
    axes = [axes]
    
for i, activation in enumerate(activations):
    ax = axes[i]
    sns.heatmap(activation, cmap="viridis", cbar=True, ax=ax)
    ax.set_title(f"Layer {i+1}")

plt.tight_layout()
plt.savefig('./plots/layer_activations.png')
plt.close()

# Global Activation Heatmap
df_activations = pd.DataFrame(np.concatenate(activations, axis=1))
plt.figure(figsize=(12, 6))
sns.heatmap(df_activations, cmap="coolwarm", linewidths=0.5)
plt.xlabel("Neurons")
plt.ylabel("Example")
plt.title("Neuron Activations")
plt.savefig('./plots/global_activations.png')
plt.close()