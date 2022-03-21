import numpy as np



class SGD:
    def __init__(self):
        pass
    
    def __str__(self):
        return "stochastic gradient descent"

    def new(self):
        return SGD()

    def __call__(self, dW):
        return dW




class Momentum:
    def __init__(self, Beta=0.9):
        self.V = 0
        self.b = Beta

    def __str__(self):
        return "momentum"

    def new(self):
        return Momentum(self.b)

    def __call__(self, dW):
        self.V = self.b * self.V + (1 - self.b) * dW
        return self.V




class RMSProp:
    def __init__(self, Beta=0.999, Epsilon=1e-8):
        self.S = 0
        self.b = Beta
        self.E = Epsilon
    
    def __str__(self):
        return "root mean squared propagation"

    def new(self):
        return RMSProp(self.b, self.E)

    def __call__(self, dW):
        self.S = self.b * self.S + (1 - self.b) * np.square(dW)
        return dW/np.sqrt(self.S + self.E)
    



class Adam:
    def __init__(self, Beta1=0.9, Beta2=0.999, Epsilon=1e-8):
        self.V = 0
        self.S = 0
        self.b1 = Beta1
        self.b2 = Beta2
        self.E = Epsilon

    def __str__(self):
        return "adaptive moment estimation"

    def new(self):
        return Adam(self.b1, self.b2, self.E)

    def __call__(self, dW):
        self.V = self.b1 * self.V + (1 - self.b1) * dW
        self.S = self.b2 * self.S + (1 - self.b2) * np.square(dW)
        return self.V/np.sqrt(self.S + self.E)