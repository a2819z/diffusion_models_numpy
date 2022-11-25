import numpy as np
import matplotlib.pyplot as plt

from sampler.base import BaseSampler
from utils.visualize import make_animation


class AnnealedLangevinSampler(BaseSampler):
    def __init__(self, cfg, model, sigma_begin=1.0, sigma_end=0.01, noise_step=10):
        super(AnnealedLangevinSampler, self).__init__(cfg, model)
        self.sigma_begin=sigma_begin
        self.sigma_end=sigma_end
        self.noise_step=noise_step
        self.sigmas = np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), noise_step))

    def sampling(self, x, n_steps=100, eps=5e-6):
        ys = [x]
        for noise_level in range(self.noise_step):
            step_size = eps * (self.sigmas[noise_level] / self.sigmas[-1])**2
            for _ in range(n_steps):
                z_t = np.random.randn(*x.shape)

                x = x + step_size / 2 * self.model(x, noise_level) + np.sqrt(step_size) * z_t
                ys.append(x)
        
        return np.stack(ys)
        
    def log_gradient(self, x, noise_level=None):
        if noise_level is None:
            noise_level = np.random.randint(0, len(self.sigmas), (x.shape[0],))
        scores = self.model(x, noise_level)
        scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
        scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)

        return scores_log1p

    def visualization(self, train_data, test_data=None, iter_idx=None, show=False):
        xlim = (train_data[..., 0].min() * 1.3, train_data[..., 0].max() * 1.3)
        ylim = (train_data[..., 1].min() * 1.3, train_data[..., 1].max() * 1.3)
        xx = np.stack(np.meshgrid(np.linspace(*xlim, 50), np.linspace(*ylim, 50)), axis=-1).reshape(-1, 2)

        # 1. Gradient visualization
        scores_log1p = self.log_gradient(xx)

        fig = plt.figure(figsize=(16,12))
        ax = fig.subplots()
        ax.scatter(*train_data.T, color='red', alpha=0.7, edgecolor='black', s=40)
        ax.quiver(*xx.T, *scores_log1p.T, width=0.002, color='black', alpha=0.5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        fig.savefig(self.cfg.work_dir / f'gradient_{iter_idx}.jpg')

        # 2. Score matching visualization
        if test_data is not None:
            # 2-1. Gradient in each noise level
            gs = []
            for noise_level in range(self.noise_step):
                _gs = self.log_gradient(xx, noise_level)
                gs.extend([_gs] * self.cfg.sampler.run.n_steps)
            gs = np.stack(gs)

            # 2-2. Sampling
            ys = self.sampling(test_data, **self.cfg.sampler.run)

            # 2-3. Result save
            ax.scatter(*ys[-1].T, color='blue', alpha=0.7, edgecolor='black', s=40)
            fig.savefig(self.cfg.work_dir / f'result_{iter_idx}.jpg')

            # 2-3. Animation
            anim=make_animation(ys, gs, xx)
            if show:
                plt.show()
            anim.save(self.cfg.work_dir / f'sampling_{iter_idx}.gif', writer='imagemagick')