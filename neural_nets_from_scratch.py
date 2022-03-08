import numpy as np


def Linear(x, dir='forward'):
    if dir == "backward":
        return 1
    else:
        return x
class Layer:
    # Takes number of inputs from previous layer and number of neurons for this layer
    def __init__(self, n_input, n_neurons, activation=Linear()):
        self.weights = np.random.randn(n_input+1, n_neurons)
        self.X = 0
        self.activation = activation

    def forward(self, x):
        self.X = np.insert(x, 0, 1)
        return np.dot(x, self.weights)

    def backward(self, previous_layer_derivative, learning_rate=10e-3):
        self.weights += np.dot(np.dot(previous_layer_derivative, self.activation(self.X, 'backward')), self.X)
        return np.dot(np.dot(previous_layer_derivative, self.activation(self.X, 'backward')), self.weights)


class Network:
    # Takes list of layers
    def __init__(self, layers=[]):
        self.layers = layers
