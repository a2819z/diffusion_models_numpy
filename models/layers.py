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

    def __call__(self, x, args):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout, return_grad=False):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        if return_grad:
            return dx, self.dW, self.db
        else:
            return dx


class Embed:
    def __init__(self, W):
        self.W = W
        self.x = None
        self.dW = None

    def __call__(self, x, args):
        self.x = x
        out = self.W[x]
        return out

    def backward(self, dout, return_grad=False):
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self.x, dout)
        dx = 1

        if return_grad:
            return dx, self.dW
        else:
            return dx

class ConditionalAffine:
    def __init__(self, W, b, cW):
        self.W = W
        self.b = b
        self.cW = cW
        self.dW = None
        self.db = None
        self.dcW = None

        self.affine = Affine(self.W, self.b)
        self.embed = Embed(self.cW)
        self.mul = MulLayer()

    def __call__(self, x, c):
        out = self.affine(x, x)
        embed = self.embed(c, c)
        out = self.mul(out, embed)

        return out

    def backward(self, dout):
        dx, dy = self.mul.backward(dout)
        _, self.dcW = self.embed.backward(dy, return_grad=True)
        dx, self.dW, self.db = self.affine.backward(dx, return_grad=True)

        return dx