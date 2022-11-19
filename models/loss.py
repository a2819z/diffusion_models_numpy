import numpy as np


def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

class MSE:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def __call__(self, y, t):
        self.t = t
        self.y = y
        self.loss = sum_squared_error(self.y, self.t) / self.t.shape[0]
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx =  2 * (self.y - self.t) / batch_size
        return dx