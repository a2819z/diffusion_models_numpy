from collections import OrderedDict

import numpy as np

from models.base import NN
from models.layers import Affine
from models.activations import build_activ
from models.loss import MSE


# Denoising Score matching
class ScoreNet(NN):
    def __init__(self, cfg, conditional=False, weight_init_std=0.01):
        super(ScoreNet, self).__init__()
        if len(cfg.dims) == 1:
            raise ValueError(f'dims must ...')

        self.dims = cfg.dims
        self.layers = OrderedDict()

        for i in range(len(self.dims)-1):
            self.params[f'W{i}'] = weight_init_std * np.random.randn(self.dims[i], self.dims[i+1])
            self.params[f'b{i}'] = np.zeros(self.dims[i+1])

            self.layers[f'Affine{i}'] = Affine(self.params[f'W{i}'], self.params[f'b{i}'])

            if i != len(self.dims)-2:
                self.layers[f'Activ{i}'] = build_activ(cfg.activ.type, cfg.activ.args)

        self.loss_func = MSE()

    def __call__(self, x):
        for layer in self.layers.values():
            x = layer(x)

        return x

    def loss(self, x, t):
        y = self(x)
        return self.loss_func(y, t)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.loss_func.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for i in range(len(self.dims) - 1):
            grads[f'W{i}'], grads[f'b{i}'] = self.layers[f'Affine{i}'].dW, self.layers[f'Affine{i}'].db

        return grads