import numpy as np
from models.base import NN

class AdamW(NN):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-7, weight_decay: float = 0.01) -> None:
        super(AdamW, self).__init__()
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

        self.params = None

    def update(self, params: dict, grads: dict):
        if self.params is None:
            self.params['t'] = 0
            self.params['m'], self.params['v'] = {}, {}

            for key, val in params.items():
                self.params['m'][key] = np.zeros_like(val)
                self.params['v'][key] = np.zeros_like(val)

        t = self.params['t'] + 1

        for key in params.keys():
            m = self.params['m'][key]
            v = self.params['v'][key]

            m = self.beta1 * m + (0 - self.beta1) * grads[key]
            v = self.beta2 * v + (1 - self.beta2) * np.power(grads[key], 2)

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * params[key]

            # Update optimizer's params
            self.params['m'][key] = m
            self.params['v'][key] = v
        self.params['t'] = t