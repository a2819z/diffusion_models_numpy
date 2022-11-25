import numpy as np


def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2, axis=-1)

class MSE:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        self.scaling_factor = None

    def __call__(self, y, t, scaling_factor=None):
        self.t = t
        self.y = y
        self.scaling_factor = 1 if scaling_factor is None else scaling_factor
        self.loss = sum_squared_error(self.y, self.t) * self.scaling_factor
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx =  2 * np.reshape(self.scaling_factor, (-1, 1)) * (self.y - self.t) / batch_size
        return dx