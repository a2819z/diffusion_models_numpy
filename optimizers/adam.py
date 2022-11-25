import numpy as np
from models.base import NN


class Adam(NN):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-7, clip_norm=None):
        super(Adam, self).__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.params = None
        self.clip_norm = clip_norm

    def update(self, params: dict, grads: dict):
        if self.params is None:
            self.params = {}
            self.params['t'] = 0
            self.params['m'], self.params['v'] = {}, {}

            for key, val in params.items():
                self.params['m'][key] = np.zeros_like(val)
                self.params['v'][key] = np.zeros_like(val)

        t = self.params['t'] + 1

        # TODO: gradient clipping
        #t = np.inf if self.clip_norm is None else self.clip_norm
        for key in params.keys():
            #if np.linalg.norm(grads[key]) > self.t:
                #grads[key] = grads[key] * self.t /np.linalg.norm(grads[key])

            m = self.params['m'][key]
            v = self.params['v'][key]

            m = self.beta1 * m + (1 - self.beta1) * grads[key]
            v = self.beta2 * v + (1 - self.beta2) * np.power(grads[key], 2)

            # Unbiasing
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Update optimizer's params
            self.params['m'][key] = m
            self.params['v'][key] = v
        self.params['t'] = t
