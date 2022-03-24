import numpy as np
from activation_functions import *
from loss_functions import *
from optimizers import *
from regularizers import *

class Layer:
    def __init__(self, n_input, n_neurons, activation=Linear(), optimizer=None, regularizer=None, *args):
        self.weights = np.random.uniform(-1, 1, (n_input+1, n_neurons))
        self.X = 0
        self.activation = activation
        self.name = ''
        self.optimizer = optimizer
        self.regularizer = regularizer

    def __str__(self):
        return f"{self.name} has {self.weights.shape[0]} inputs and {self.weights.shape[1]} neurons which totals {self.weights.size} weights and applies {self.activation} activation"

    def forward(self, x, **kwargs):
        self.X = np.concatenate([np.ones((x.shape[0],1)), x], axis=1)
        return self.activation.forward(np.dot(self.X, self.weights))

    def backward(self, previous_layer_derivative, learning_rate):
        if self.regularizer is None:
            self.weights += self.optimizer(np.dot(self.X.T, previous_layer_derivative*self.activation.backward()))*learning_rate/self.X.shape[0]
        else:
            self.weights += self.optimizer(np.dot(self.X.T, previous_layer_derivative*self.activation.backward()))*learning_rate/self.X.shape[0] - self.regularizer(self.weights)

        return np.dot(previous_layer_derivative*self.activation.backward(), self.weights[1:, :].T)
            



class Dropout:
    def __init__(self, drop_rate=0.2):
        self.drop_rate = drop_rate
        self.name = ''
        self.optimizer = np.NaN

    def __str__(self):
        return f"Dropout with rate {self.drop_rate}"

    def forward(self, X, **kwargs):
        for karg in kwargs:
            if karg == 'training':
                if kwargs[karg] == True:
                    return X * np.where(np.random.random(X.shape) > self.drop_rate, 1, 0)
        return X

    def backward(self, previous_layer_derivative, learning_rate):
        return previous_layer_derivative



class Network:
    def __init__(self, layers=[], loss=MSE(), optimizer=SGD(), callbacks=None):
        self.layers = layers
        self.loss = []
        self.loss_func = loss
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.training = False
        for i, layer in enumerate(self.layers):
            if layer.name == "":
                layer.name = "Layer_" + str(i)
            if layer.optimizer is None:
                layer.optimizer = optimizer.new()
    
    def __str__(self):
        string = f"Model with {len(self.layers)} layers and {self.loss_func} loss function and {self.optimizer} optimizer\n"
        for layer in self.layers:
            string += f"{layer} \n"
        return string

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x, training=self.training)
        return x
    
    def backward(self, previous_layer_derivative, learning_rate):
        for layer in reversed(self.layers):
            previous_layer_derivative = layer.backward(previous_layer_derivative, learning_rate)

    def train(self, inputs, y, epochs=1, learning_rate=1e-2, batch_size=None):
        self.training = True
        lr = learning_rate
        batch_size = batch_size if type(batch_size) is int and batch_size > 0 else inputs.shape[0]
        for epoch in range(1, epochs+1):
            if self.training == True:
                if not type(lr) is int and not type(lr) is float:
                    lr = learning_rate(epoch)

                p = np.random.permutation(inputs.shape[0])
                inputs, y = inputs[p], y[p]
                loss_ = 0

                for i in range(int(inputs.shape[0]/batch_size)):
                    y_hat = self.forward(inputs[i*batch_size:(i+1)*batch_size])
                    loss_ += np.sum(self.loss_func.forward(y=y[i*batch_size:(i+1)*batch_size], y_hat=y_hat))
                    self.backward(self.loss_func.backward(y=y[i*batch_size:(i+1)*batch_size], y_hat=y_hat), learning_rate=lr)
                if inputs.shape[0]%batch_size != 0:
                    y_hat = self.forward(inputs[-(inputs.shape[0]%batch_size):])
                    loss_ += np.sum(self.loss_func.forward(y=y[-(inputs.shape[0]%batch_size):], y_hat=y_hat))
                    self.backward(self.loss_func.backward(y=y[-(inputs.shape[0]%batch_size):], y_hat=y_hat), learning_rate=lr)

                self.loss.append(loss_/inputs.shape[0])


                if self.callbacks is not None:
                    if self.callbacks is type(list):
                        for callback in self.callbacks: 
                            callback(self)
                    else:
                        self.callbacks(self)
            else:
                break
        self.training = False

    def predict(self, X): 
        return self.forward(X)




class Early_Stopping:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.epochs_without_improvement = 0
        self.lowest_loss = np.inf

    def __call__(self, Net):
        if self.lowest_loss - Net.loss[len(Net.loss)-1] > self.min_delta:
            self.lowest_loss = Net.loss[len(Net.loss)-1]
            self.epochs_without_improvement = 0
        
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement == self.patience:
            Net.training = False