import re
import numpy as np
import matplotlib.pyplot as plt

class Linear:
    def __init__(self):
        self.X = 0

    def __str__(self):
        return 'no'
    
    def forward(self, X):
        self.X = X
        return self.X
    
    def backward(self):
        return 1

class Sigmoid:
    def __init__(self):
        self.X = 0
    
    def __str__(self):
        return "sigmoid"

    def forward(self, X):
        self.X = X
        return 1/(1+np.exp(-self.X))
    
    def backward(self):
        return (1/(1+np.exp(-self.X)))*(1-(1/(1+np.exp(-(self.X)))))

class Relu:
    def __init__(self):
        self.X = 0

    def forward(self, X):
        self.X = X
        return np.where(X > 0, X, 0)

    def backward(self):
        return np.where(self.X > 0, 1, 0)

class Sine:
    def __init__(self):
        self.X = 0

    def forward(self, X):
        self.X = X
        return np.sin(X)

    def backward(self):
        return np.cos(self.X)

class Layer:
    def __init__(self, n_input, n_neurons, activation=Linear()):
        self.weights = np.random.randn(n_input+1, n_neurons)
        self.X = 0
        self.activation = activation
        self.name = ''

    def __str__(self):
        return f"{self.name} has {self.weights.shape[0]} inputs and {self.weights.shape[1]} outputs which totals {self.weights.size} weights and applies {self.activation} activation"

    def forward(self, x):
        self.X = np.concatenate([np.ones((x.shape[0],1)), x], axis=1)
        return self.activation.forward(np.dot(self.X, self.weights))

    def backward(self, previous_layer_derivative, learning_rate):
        self.weights += np.dot(self.X.T, previous_layer_derivative*self.activation.backward())/self.X.shape[0]*learning_rate
        return np.dot(previous_layer_derivative*self.activation.backward(), self.weights[1:, :].T)/self.X.shape[0]
            

class Network:
    def __init__(self, layers=[], loss='mean_squared_error'):
        self.layers = layers
        self.loss = []
        for epoch, layer in enumerate(self.layers):
            layer.name = "Layer_" + str(epoch)
    
    def __str__(self):
        string = ""
        for layer in self.layers:
            string += f"{layer} \n"
        return string

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, previous_layer_derivative, learning_rate):
        for layer in reversed(self.layers):
            previous_layer_derivative = layer.backward(previous_layer_derivative, learning_rate)

    def train(self, inputs, y, epochs=1, learning_rate=1e-2, batch_size=None):
        lr = learning_rate
        batch_size = batch_size if type(batch_size) is int and batch_size > 0 else inputs.shape[0]
        for epoch in range(1, epochs+1):
            if not type(lr) is int and not type(lr) is float:
                lr = learning_rate(epoch)

            loss_ = 0

            for i in range(int(inputs.shape[0]/batch_size)):
                loss_ += np.sum(np.square(y[i*batch_size:(i+1)*batch_size] - self.forward(inputs[i*batch_size:(i+1)*batch_size])))
                self.backward(2*(y[i*batch_size:(i+1)*batch_size] - self.forward(inputs[i*batch_size:(i+1)*batch_size])), learning_rate=lr)
            if inputs.shape[0]%batch_size != 0:
                loss_ += np.sum(np.square(y[-(inputs.shape[0]%batch_size):] - self.forward(inputs[-(inputs.shape[0]%batch_size):])))
                self.backward(2*(y[-(inputs.shape[0]%batch_size):] - self.forward(inputs[-(inputs.shape[0]%batch_size):])), learning_rate=lr)

            self.loss.append(loss_/inputs.shape[0])

    def predict(self, X): 
        return self.forward(X)

