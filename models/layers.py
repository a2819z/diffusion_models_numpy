import numpy as np


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def __call__(self, x, y):
        self.x = x
        self.y = y

        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.x
        dy = dout * self.y

        return dx, dy


class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def __call_(self, x, y):
        self.x = x
        self.y = y

        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def __call__(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Embed:
    def __init__(self, W):
        self.W = W
        self.x = None
        self.dW = None

    def __call__(self, x):
        self.x = x
        out = np.dot(x, self.W)
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        return dx

class ConditoinalAffine:
    def __init__(self, W, b, embedW):
        self.W = W
        self.b = b
        self.x = None
        self.t = None
        self.dW = None
        self.embedW = None

    def __call__(self, x, t):
        self.x = x
        self.t = t
        out = np.dot(x, self.W) + self.b
        embed = np.dot(t, self.embedW)
        return embed * out
