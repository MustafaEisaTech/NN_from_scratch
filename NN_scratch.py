import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def MSE(y_i, yhat_i):
    return ((y_i - yhat_i) ** 2).mean()
'''
    y_i = np.array([1, 0, 0, 1])
        yhat_i = np.array([0, 0, 0, 0])
'''

def derive_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


'''
weights = np.array([0, 1])
bias = 4

n = Neuron(weights, bias)


x = np.array([2, 3])

print(n.feedforward(x))
'''

'''class NeuralNet:
    def __init__(self):
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        output_h1 = self.h1.feedforward(x)
        output_h2 = self.h2.feedforward(x)
        output_o1 = self.o1.feedforward(np.array([output_h1, output_h2]))
        return output_o1
    
bias = 0
weights = np.array([0, 1])

NN = NeuralNet()
x = np.array([2, 3])

print(NN.feedforward(x))
'''
class NeuralNet:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    def train(self, data, all_y_true):

        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_true):
                 sum_h1 = self.w1 * x[0] + self.w2 *x[1] + self.b1
                 h1 = sigmoid(sum_h1)

                 sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                 h2 = sigmoid(sum_h2)

                 sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                 o1 = sigmoid(sum_o1)
                 y_pred = o1
                 # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                 d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron o1
                 d_ypred_d_w5 = h1 * derive_sigmoid(sum_o1)
                 d_ypred_d_w6 = h2 * derive_sigmoid(sum_o1)
                 d_ypred_d_b3 = derive_sigmoid(sum_o1)

                 d_ypred_d_h1 = self.w5 * derive_sigmoid(sum_o1)
                 d_ypred_d_h2 = self.w6 * derive_sigmoid(sum_o1)

            # Neuron h1
                 d_h1_d_w1 = x[0] * derive_sigmoid(sum_h1)
                 d_h1_d_w2 = x[1] * derive_sigmoid(sum_h1)
                 d_h1_d_b1 = derive_sigmoid(sum_h1)

                    # Neuron h2
                 d_h2_d_w3 = x[0] * derive_sigmoid(sum_h2)
                 d_h2_d_w4 = x[1] * derive_sigmoid(sum_h2)
                 d_h2_d_b2 = derive_sigmoid(sum_h2)

                    # --- Update weights and biases
                    # Neuron h1
                 self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                 self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                 self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                    # Neuron h2
                 self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                 self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                 self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                    # Neuron o1
                 self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                 self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                 self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Calculate total loss at the end of each epoch
        if epoch % 10 == 0:
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = MSE(all_y_trues, y_preds)
            print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = NeuralNet()
network.train(data, all_y_trues)
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) 
print("Frank: %.3f" % network.feedforward(frank)) 
        