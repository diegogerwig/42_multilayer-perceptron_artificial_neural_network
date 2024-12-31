# 42_multilayer-perceptron_artificial_neural_network



## Documentation

#### *Feedforward*:

Input data flows through network in one direction (forward)

Each layer processes data sequentially:
- Calculate weighted sum of inputs + bias
- Apply activation function (like sigmoid)
- Pass result to next layer

No loops/cycles - data moves input→hidden→output

#### *Backpropagation*:

Algorithm to calculate gradients for network training

Works backwards from output to input:
- Calculate error at output layer
- Propagate error backwards through network using chain rule
- Compute partial derivatives for weights/biases

Key formula: Error gradient = Local gradient × Downstream gradient

#### *Gradient Descent*:

Optimization algorithm to minimize error/loss

Steps:
- Calculate error gradient using backpropagation
- Update weights/biases in opposite direction of gradient
- Learning rate controls step size

Mini-batch variation: Updates based on subset of data at a time

**The three work together cyclically**:
1. Feedforward to get predictions
2. Backpropagation to compute gradients
3. Gradient descent to update parameters
4. Repeat until convergence