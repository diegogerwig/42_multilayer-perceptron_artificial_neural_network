# 42_multilayer-perceptron_artificial_neural_network


## Documentation

### *Feedforward*:

Input data flows through the network in one direction (forward).

Each layer processes data sequentially:
1. **Weighted Sum**: Calculate the weighted sum of inputs and add the bias.
2. **Activation Function**: Apply an activation function (e.g., sigmoid) to introduce non-linearity.
3. **Output**: Pass the result to the next layer.

There are no loops or cycles; data moves strictly in the direction of input → hidden → output layers.

### *Backpropagation*:

An algorithm used to calculate gradients for network training.

**Steps**:
1. **Output Error**: Compute the error at the output layer (e.g., difference between predicted and actual values).
2. **Backward Propagation**:
   - Propagate the error backward through the network using the chain rule.
   - Calculate local gradients for each layer.
3. **Gradients**:
   - Compute partial derivatives of the error with respect to weights and biases.

**Key formula**:
$$
{\text{Error Gradient} = \text{Local Gradient} \times \text{Downstream Gradient}}
$$

### *Gradient Descent*:

An optimization algorithm to minimize the network’s error or loss.

**Steps**:
1. **Calculate Error Gradients**: Use backpropagation to compute gradients.
2. **Update Parameters**: Adjust weights and biases in the direction opposite to the gradient.
3. **Learning Rate**: Controls the step size for updates.

**Variants**:
- **Mini-batch Gradient Descent**: Updates weights using a small random subset of the data instead of the entire dataset.

### *Interplay Between the Three Processes*:
1. Perform feedforward to generate predictions.
2. Use backpropagation to compute gradients based on the predictions.
3. Apply gradient descent to update the weights and biases.
4. Repeat until the model converges (i.e., achieves minimal error).

### *Overfitting*:

Overfitting occurs when a model learns noise or random fluctuations in the training data instead of the underlying patterns, leading to poor generalization on unseen data.

**Methods to Avoid Overfitting**:
- **Regularization**: Add a penalty term to the loss function (e.g., L1 or L2 regularization).
- **Dropout**: Randomly disable neurons during training to prevent reliance on specific features.
- **Data Augmentation**: Increase dataset size by transforming the data (e.g., rotations, scaling).
- **Early Stopping**: Halt training when performance on a validation set starts to deteriorate.
- **Cross-Validation**: Use techniques like k-fold cross-validation to ensure robustness.

### *Activation Functions*:

Activation functions introduce non-linearities into the model, enabling it to learn complex patterns.

**Common Types**:
1. **Sigmoid**:
   - Squashes inputs into a range between 0 and 1.
   - Commonly used in binary classification.
   - Formula:
$$
{\sigma(x) = \frac{1}{1 + e^{-x}}}
$$

2. **Tanh**:
   - Squashes inputs into a range between -1 and 1.
   - Centered around zero, often preferred over sigmoid.
   - Formula:
$$
{\text{Tanh}(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}}
$$

3. **ReLU** (Rectified Linear Unit):
   - Outputs positive signals and zero for negative inputs.
   - Monotonic derivative; computationally efficient.
   - Formula:
$$
{\displaystyle ReLU = {\begin{aligned}&{\begin{cases}0&{\text{if }}x\leq 0\\x&{\text{if }}x>0\end{cases}}=&\max\{0,x\}\end{aligned}}}
$$

4. **Leaky ReLU**:
   - Similar to ReLU but allows a small gradient for negative inputs.
   - Formula:
$$
{\text{Leaky ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}}
$$
where $\alpha$ is a small constant (e.g., 0.01).

5. **Softmax**:
   - Converts outputs into probabilities.
   - Commonly used in multi-class classification.
   - Formula:
$$
{\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}}
$$

![Activation Functions](./images/activation_functions.png)

[Learn More: Activation Functions](https://medium.com/@cmukesh8688/activation-functions-sigmoid-tanh-relu-leaky-relu-softmax-50d3778dcea5)

### *Binary cross-entropy error function*

Binary Cross-Entropy (BCE) is a loss function commonly used in binary classification problems, where the task is to classify data into one of two classes (e.g., 0 or 1).

Key Points:
- Penalty for Wrong Predictions: The function heavily penalizes predictions that are confident but incorrect.
- Logarithmic Nature: The use of the logarithm emphasizes small differences when the prediction is close to the true label and larger differences when predictions are far off.
- Interpretation: BCE is minimized when the predicted probabilities align closely with the true labels, making it an effective metric for binary classification.