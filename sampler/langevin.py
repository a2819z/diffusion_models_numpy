import numpy as np
import matplotlib.pyplot as plt

from sampler.simple import SimpleSampler
from utils.visualize import make_animation


class LangevinSampler(SimpleSampler):
    def __init__(self, cfg, model):
        super(LangevinSampler, self).__init__(cfg, model)

    def sampling(self, x, n_steps=100, eps=4e-3, decay=0.97, temperature=0.7):
        ys = [x]
        for _ in range(n_steps):
            x = x + eps * self.model(x)
            z_t = np.random.randn(*x.shape)
            x = x + (eps / 2) * self.model(x) + (np.sqrt(eps) * temperature * z_t)
            eps *= decay
            ys.append(x)
        
        return np.stack(ys)