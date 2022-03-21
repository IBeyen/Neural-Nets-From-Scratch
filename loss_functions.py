import numpy as np



class MSE:
    def __init__(self):
        pass

    def __str__(self):
        return "mean squared error"

    def forward(self, y, y_hat):
        return np.square(y - y_hat)

    def backward(self, y, y_hat):
        return 2*(y - y_hat)




class MAE:
    def __init__(self):
        pass

    def __str__(self):
        return "mean absolute error"

    def forward(self, y, y_hat):
        return np.absolute(y - y_hat)

    def backward(self, y, y_hat):
        return np.where(y_hat >= y, 1, -1)