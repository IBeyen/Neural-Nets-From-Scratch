import numpy as np



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

    def __str__(self):
        return "relu"

    def forward(self, X):
        self.X = X
        return np.where(X > 0, X, 0)

    def backward(self):
        return np.where(self.X > 0, 1, 0)



class Sine:
    def __init__(self):
        self.X = 0

    def __str__(self):
        return "sine"

    def forward(self, X):
        self.X = X
        return np.sin(X)

    def backward(self):
        return np.cos(self.X)
