from collections import OrderedDict

import numpy as np

from models.base import NN, weight_initialization
from models.layers import Affine, ConditionalAffine
from models.activations import build_activ
from models.loss import MSE


# Denoising Score matching
class ScoreNet(NN):
    def __init__(self, cfg):
        super(ScoreNet, self).__init__()
        if len(cfg.dims) == 1:
            raise ValueError(f'dims must ...')

        self.dims = cfg.dims
        self.layers = OrderedDict()

        for i in range(len(self.dims)-1):
            self.params[f'W{i}'] = weight_initialization((np.zeros((self.dims[i], self.dims[i+1]))), cfg.init.type, **cfg.init.args)
            self.params[f'b{i}'] = np.zeros(self.dims[i+1])

            self.layers[f'Affine{i}'] = Affine(self.params[f'W{i}'], self.params[f'b{i}'])

            if i != len(self.dims)-2:
                self.layers[f'Activ{i}'] = build_activ(cfg.activ.type, cfg.activ.args)

        self.loss_func = MSE()

    def __call__(self, x):
        for layer in self.layers.values():
            x = layer(x, None)

        return x

    def loss(self, x, t):
        y = self(x)
        return np.mean(self.loss_func(y, t))

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


class NoiseConditionedScoreNet(NN):
    def __init__(self, cfg):
        super(NoiseConditionedScoreNet, self).__init__()

        if len(cfg.dims) == 1:
            raise ValueError(f'dims must ...')

        self.dims = cfg.dims
        self.t_steps = cfg.t_steps
        self.layers = OrderedDict()

        for i in range(len(self.dims)-2):
            self.params[f'W{i}'] = weight_initialization(np.zeros((self.dims[i], self.dims[i+1])), cfg.init.type, **cfg.init.args)
            self.params[f'b{i}'] = np.zeros(self.dims[i+1])
            self.params[f'cW{i}'] = weight_initialization(np.zeros((self.t_steps, self.dims[i+1])), cfg.init.type, **cfg.init.args)

            self.layers[f'ConditionalAffine{i}'] = ConditionalAffine(self.params[f'W{i}'], self.params[f'b{i}'], self.params[f'cW{i}'])

            self.layers[f'Activ{i}'] = build_activ(cfg.activ.type, cfg.activ.args)

        self.params[f'W{i+1}'] = weight_initialization(np.zeros((self.dims[-2], self.dims[-1])), cfg.init.type, **cfg.init.args)
        self.params[f'b{i+1}'] = np.zeros(self.dims[-1])
        self.layers[f'lastLayer']  = Affine(self.params[f'W{i+1}'], self.params[f'b{i+1}'])

        self.loss_func = MSE()

    def __call__(self, x, time_condition):
        for layer in self.layers.values():
            x = layer(x, time_condition)

        return x

    def loss(self, x: np.array, time_condition: np.array, t, sigmas: np.array =None):
        y = self(x, time_condition)
        scaling_factor = sigmas.squeeze() ** 2
        return np.mean(self.loss_func(y, t, scaling_factor=scaling_factor), axis=0)

    def gradient(self, x, time_condition, t, sigmas=None):
        # forward
        self.loss(x, time_condition, t, sigmas)

        # backward
        dout = 1
        dout = self.loss_func.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for i in range(len(self.dims) - 2):
            grads[f'W{i}'], grads[f'b{i}'] = self.layers[f'ConditionalAffine{i}'].dW, self.layers[f'ConditionalAffine{i}'].db
            grads[f'cW{i}'] = self.layers[f'ConditionalAffine{i}'].dcW

        grads[f'W{i+1}'], grads[f'b{i+1}'] = self.layers[f'lastLayer'].dW, self.layers[f'lastLayer'].db

        return grads