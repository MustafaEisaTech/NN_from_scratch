# Simple Neural Network Implementation

## Overview
This project implements a basic neural network from scratch using NumPy. The network is designed to classify data points based on two input features using a simple feed-forward architecture with backpropagation for training.

## Architecture
The neural network consists of:
- Input layer: 2 neurons (for 2 features)
- Hidden layer: 2 neurons with sigmoid activation
- Output layer: 1 neuron with sigmoid activation

## Mathematical Foundation

### Activation Function
The network uses the sigmoid activation function:

$\sigma(x) = \frac{1}{1 + e^{-x}}$

With its derivative:

$\sigma'(x) = \sigma(x)(1 - \sigma(x))$

### Loss Function
Mean Squared Error (MSE) is used as the loss function:

$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

### Training
The network employs Stochastic Gradient Descent (SGD) with backpropagation:
- Learning rate: 0.1
- Epochs: 1000
- Weight updates follow the formula: $w = w - \alpha \frac{\partial L}{\partial w}$
where $\alpha$ is the learning rate and $\frac{\partial L}{\partial w}$ is the partial derivative of the loss with respect to each weight.

## Code Structure

### Key Components:
1. `sigmoid()`: Implements the sigmoid activation function
2. `derive_sigmoid()`: Computes the derivative of sigmoid
3. `MSE()`: Calculates the Mean Squared Error
4. `NeuralNet` class: Main implementation with:
   - Random weight initialization
   - Feedforward propagation
   - Backpropagation training

## Usage Example
```python
# Create and train the network
network = NeuralNet()
network.train(data, all_y_trues)

# Make predictions
emily = np.array([-7, -3])
frank = np.array([20, 2])
print("Emily: %.3f" % network.feedforward(emily))
print("Frank: %.3f" % network.feedforward(frank))
```

## Dependencies
- NumPy

