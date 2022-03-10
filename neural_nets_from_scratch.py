import numpy as np
import matplotlib.pyplot as plt

class Layer:
    # Takes number of inputs from previous layer and number of neurons for this layer
    def __init__(self, n_input, n_neurons, activation='linear'):
        self.weights = np.random.randn(n_input+1, n_neurons)
        self.X = 0
        self.activation = activation

    def forward(self, x):
        #x=examples, inputs    w=number of inputs, neurons
        self.X = np.concatenate([np.ones((x.shape[0],1)), x], axis=1)
        if self.activation == 'linear':
            return np.dot(self.X, self.weights)

    def backward(self, previous_layer_derivative, learning_rate=10e-3):
        if self.activation == 'linear':
            self.weights += np.dot(self.X.T, previous_layer_derivative)
            return np.dot(previous_layer_derivative, self.weights[1:, :])


class Network:
    # Takes list of layers
    def __init__(self, layers=[], loss='mean_squared_error'):
        self.layers = layers
        self.loss = []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, previous_layer_derivative):
        for layer in self.layers:
            previous_layer_derivative = layer.backward(previous_layer_derivative)

    def train(self, inputs, y, epochs=1):
        for i in range(epochs):
            self.loss.append(np.sum(np.square(y - self.forward(inputs))))
            self.backward(2*(y - self.forward(inputs)))

    def graph_loss(self): 
        plt.plot(range(len(self.loss)), self.loss)
        plt.show()


x = np.array([[1], [2], [3], [4], [5]])
y = np.array([[2], [4], [6], [8], [10]])

m = Network([Layer(1, 1)])
m.train(x, y, 10)
m.graph_loss()