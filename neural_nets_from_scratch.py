import numpy as np

class Layer:
    # Takes number of inputs from previous layer and number of neurons for this layer
    def __init__(self, n_input, n_neurons):
        self.weights = np.random.randn(n_input+1, n_neurons)
        self.X = 0

    def forward(self, x):
        self.X = np.insert(x, 0, 1)
        return np.dot(x, self.weights)

    def backward(self, previous_layer_derivative, learning_rate=10e-3):
        self.weights += np.dot(previous_layer_derivative, self.X)
        return np.dot(previous_layer_derivative, self.weights)


class Network:
    # Takes list of layers
    def __init__(self, layers=[]):
        self.layers = layers
