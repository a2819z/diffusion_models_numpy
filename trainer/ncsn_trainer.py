import time

import numpy as np

from trainer.base_trainer import BaseTrainer


class NCSNTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, logger):
        super(NCSNTrainer, self).__init__(cfg, model, optimizer, logger)
    
    def perturbation(self, samples, sigmas):
        sigmas = sigmas[..., None]
        noise = np.random.normal(size=samples.shape)
        perturbed_samples = samples + noise * sigmas
        target = -(perturbed_samples - samples) / (sigmas ** 2)

        return perturbed_samples, target

    def train(self, train_data, test_data):
        elapsed = 0
        best = np.inf
        losses = []
        sigmas = np.exp(np.linspace(np.log(self.cfg.sigma_begin), np.log(self.cfg.sigma_end), self.cfg.num_t_steps))
        for i in range(self.cfg.steps):
            start_time = time.time()
            # Data preparation
            batch_idxs = np.random.choice(train_data.shape[0], size=self.cfg.batch_size, replace=False)
            labels = np.random.randint(0, len(sigmas), size=len(batch_idxs))

            perturbed_samples, target = self.perturbation(train_data[batch_idxs], sigmas[labels])

            # Backpropagation
            grad = self.model.gradient(perturbed_samples, labels, target, sigmas=sigmas[labels])

            # Update
            self.optimizer.update(self.model.params, grad)
            end_time = time.time()
            elapsed += end_time - start_time

            # Logging
            if (i+1) % self.cfg.log_freq == 0:
                loss = self.model.loss(perturbed_samples, labels, target, sigmas=sigmas[labels])
                self.logging(i, loss, elapsed)
                self.sampler.visualization(train_data, iter_idx=i+1)
                if loss < best:
                    best = loss
                    model_state_dict = self.model.state_dict()
                    optim_state_dict = self.optimizer.state_dict()
                    self.save_checkpoint(self.cfg.work_dir / 'checkpoint', i+1, model_state_dict, optim_state_dict, best=True)
                    #self.sampler.visualization(train_data, test_data=test_data, iter_idx=i+1)

            # Checkpoint save
            if (i+1) % self.cfg.save_freq == 0:
                model_state_dict = self.model.state_dict()
                optim_state_dict = self.optimizer.state_dict()
                self.save_checkpoint(self.cfg.work_dir / 'checkpoint', i+1, model_state_dict, optim_state_dict)

    def logging(self, iter_idx, loss, elapsed):
        self.logger.info(
            f'Iteration: {iter_idx+1:5d}/{self.cfg.steps} ({int((iter_idx+1)/self.cfg.steps*100)}%) \
            | Loss: {loss:6.4f}\
            | elapsed: {elapsed:6.2f}s \
            | ETA: {elapsed/(iter_idx+1)*self.cfg.steps - elapsed:6.2f}s'
        )