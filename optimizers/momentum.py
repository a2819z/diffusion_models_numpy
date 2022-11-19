import numpy as np


class Momentum:
    def __init__(self, lr=1e-2, momentum=.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}

            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum * self.v - self.lr * grads[key]
            params[key] += self.v[key]