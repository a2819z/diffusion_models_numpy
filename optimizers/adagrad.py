import numpy as np


class AdaGrad:
    def __init__(self, lr=1e-2, eps=1e-7):
        self.lr = lr
        self.h = None
        self.eps = eps

    def update(self, params: dict, grads: dict):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.eps)