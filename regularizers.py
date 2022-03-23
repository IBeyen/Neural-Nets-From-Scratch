import numpy as np

class L1:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, W):
        return np.where(W >= 0, 1, -1) * self.alpha