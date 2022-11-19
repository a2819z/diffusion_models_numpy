import sys
thismodule = sys.modules[__name__]

import numpy as np


def build_activ(activ, kwargs):
    if kwargs is None:
        kwargs = {}
    if activ is None:
        return ReLU()
    return getattr(thismodule, activ)(**kwargs)


class Sigmoid:
    def __init__(self):
        self.out = None

    def __call__(self, x):
        out =  1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class ReLU:
    def __init__(self):
        self.mask = None

    def __call__(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class LeakyReLU:
    def __init__(self, slope=0.1):
        self.mask = None
        self.slope = slope

    def __call__(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = out[self.mask] * self.slope
        return out

    def backward(self, dout):
        dout[self.mask] = self.slope
        dx = dout
        return dx


class SoftPlus:
    def __init__(self, beta=1., threshold=20):
        self.beta = beta
        self.threshold = threshold
        self.mask = None

    def __call__(self, x):
        #self.mask = (x <= self.threshold)
        #out = x.copy()
        out = np.log(1 + np.exp(self.beta * x)) / self.beta
        return out

    def backward(self, dout):
        #dout[self.mask] = 1
        def sigmoid(x):
            pos_mask = (x >= 0)
            neg_mask = ~pos_mask
            z = np.zeros_like(x)

            z[pos_mask] = np.exp(-x[pos_mask])
            z[neg_mask] = np.exp(x[neg_mask])
            top = np.ones_like(x)
            top[neg_mask] = z[neg_mask]
            return  top / (1. + z)
            
        dx = dout * sigmoid(dout)
        return dx
