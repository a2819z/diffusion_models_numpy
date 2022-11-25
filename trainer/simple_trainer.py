import time

import numpy as np

from trainer.base_trainer import BaseTrainer
from sampler.builder import build_sampler


class SimpleTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, logger):
        super(SimpleTrainer, self).__init__(cfg, model, optimizer, logger)

    def perturbation(self, samples, sigma=0.01):
        noise = np.random.normal(size=samples.shape)
        perturbed_samples = samples + noise * sigma
        target = -(perturbed_samples - samples) / (sigma ** 2)

        return perturbed_samples, target
    
    def train(self, train_data, test_data):
        elapsed = 0
        best = np.inf
        losses = []
        for i in range(self.cfg.steps):
            start_time = time.time()
            # Data preparation
            batch_idxs = np.random.choice(train_data.shape[0], size=self.cfg.batch_size, replace=False)
            perturbed_samples, target = self.perturbation(train_data[batch_idxs], sigma=self.cfg.sigma)

            #print(perturbed_samples.shape, target.shape)
            # Backpropagation
            grad = self.model.gradient(perturbed_samples, target)

            # Update
            self.optimizer.update(self.model.params, grad)
            end_time = time.time()
            elapsed += end_time - start_time

            # Logging
            if (i+1) % self.cfg.log_freq == 0:
                loss = self.model.loss(perturbed_samples, target)
                self.logging(i, loss, elapsed)
                self.sampler.visualization(train_data, iter_idx=i)
                if loss < best:
                    best = loss
                    model_state_dict = self.model.state_dict()
                    optim_state_dict = self.optimizer.state_dict()
                    self.save_checkpoint(self.cfg.work_dir / 'checkpoint', i+1, model_state_dict, optim_state_dict, best=True)
                    #self.sampler.visualization(train_data, test_data=test_data, iter_idx=i, show=False)

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