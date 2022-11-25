import numpy as np
import matplotlib.pyplot as plt

from sampler.base import BaseSampler
from utils.visualize import make_animation


class SimpleSampler(BaseSampler):
    def __init__(self, cfg, model):
        super(SimpleSampler, self).__init__(cfg, model)

    def sampling(self, x, n_steps=100, eps=1e-3):
        ys = [x]
        for _ in range(n_steps):
            x = x + eps * self.model(x)
        
        return np.stack(ys)
        
    def log_gradient(self, x):
        scores = self.model(x)
        scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
        scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)

        return scores_log1p

    def visualization(self, train_data, test_data=None, iter_idx=None, show=False):
        # 1. Gradient visualization
        xlim = (train_data[..., 0].min() * 1.3, train_data[..., 0].max() * 1.3)
        ylim = (train_data[..., 1].min() * 1.3, train_data[..., 1].max() * 1.3)
        xx = np.stack(np.meshgrid(np.linspace(*xlim, 50), np.linspace(*ylim, 50)), axis=-1).reshape(-1, 2)

        scores_log1p = self.log_gradient(xx)

        fig = plt.figure(figsize=(16,12))
        ax = fig.subplots()
        ax.scatter(*train_data.T, color='red', edgecolor='black', s=40)
        ax.quiver(*xx.T, *scores_log1p.T, width=0.002, color='black', alpha=0.5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        fig.savefig(self.cfg.work_dir / f'gradient_{iter_idx}.jpg')

        # 2. Score matching visualization
        if test_data is not None:
            # 2-1. Gradient in each noise level
            gs = np.stack([scores_log1p])

            # 2-2. Sampling
            ys = self.sampling(test_data, **self.cfg.sampler.run)

            # 2-3. Animation
            anim=make_animation(ys, gs, xx)
            if show:
                plt.show()
            anim.save(self.cfg.work_dir / f'sampling_{iter_idx}.gif', writer='imagemagick')