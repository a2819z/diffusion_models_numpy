from argparse import ArgumentParser
import sys
from pathlib import Path
import os.path as osp
import time

import numpy as np
import matplotlib.pyplot as plt

from models.models import ScoreNet
from models import sampling
from data import sample_batch
from optimizers.builder import build_optimizer

from sconf import Config, dump_args, dump_yaml
from utils.logger import Logger
from utils.plot import plot_gradients, make_animation
from utils.utils import save_dict, load_dict


def parse_args_and_config():
    parser = ArgumentParser()
    parser.add_argument('cfg', type=str, help="Path of configure file")

    args, left_argv = parser.parse_known_args()
    cfg = Config(args.cfg, default='configs/default.yaml')
    cfg.argv_update(left_argv)

    cfg.work_dir = Path(osp.join(cfg.work_dir, cfg.exp_name))
    (cfg.work_dir / 'checkpoint').mkdir(parents=True, exist_ok=False)

    dump_yaml(cfg)

    return args, cfg


def save_checkpoint(save_dir, step, model_state_dict, optim_state_dict, **kwargs):
    states = {'step': step, 'model': model_state_dict, 'optim': optim_state_dict}
    states.update(kwargs)

    save_dict(states, save_dir / f'{step:06d}.pickle')
    save_dict(states, save_dir / 'latest.pickle')


def load_checkpoint(fname, model, optim):
    states = load_dict(fname)
    model.load_state_dict(states['model'])
    optim.load_state_dict(states['optim'])
    step = states['step']

    return model, optim, step


def perturbation(samples, sigma=0.01):
    noise = np.random.normal(size=samples.shape)
    perturbed_samples = samples + noise * sigma
    target = -noise / sigma

    return perturbed_samples, target


def train(args, cfg, model, optimizer, data, logger):
    elapsed = 0
    for i in range(cfg.steps):
        start_time = time.time()
        # Data preparation
        batch_idxs = np.random.choice(data.shape[0], size=cfg.batch_size, replace=False)
        perturbed_samples, target = perturbation(data[batch_idxs])

        #print(perturbed_samples.shape, target.shape)
        # Backpropagation
        grad = model.gradient(perturbed_samples, target)

        # Update
        optimizer.update(model.params, grad)
        end_time = time.time()
        elapsed += end_time - start_time

        # Logging
        if (i+1) % cfg.log_freq == 0:
            loss = model.loss(perturbed_samples, target)
            #print(model(perturbed_samples)[0], target[0])
            logger.info(
                f'Iteration: {i+1:5d}/{cfg.steps} ({int((i+1)/cfg.steps*100)}%) \
                  | Loss: {loss:6.4f}\
                  | elapsed: {elapsed:6.2f}s \
                  | ETA: {elapsed/(i+1)*cfg.steps - elapsed:6.2f}s'
            )
            #plot_gradients(model, data)
            #plt.show()

        # Checkpoint save
        if (i+1) % cfg.save_freq == 0:
            model_state_dict = model.state_dict()
            optim_state_dict = optimizer.state_dict()
            save_checkpoint(cfg.work_dir / 'checkpoint', i, model_state_dict, optim_state_dict)
            

    plot_gradients(model, data)
    plt.show()

    x = np.random.normal(size=(500, 2))
    samples = sampling.simple(model, x, n_steps=300)
    anim = make_animation(samples, model, data)
    anim.save(cfg.work_dir / "sample.gif", writer='imagemagick')
    plt.close()


if __name__ == "__main__":
    args, cfg = parse_args_and_config()

    logger_path = cfg.work_dir / 'log.log'
    logger = Logger.get(file_path=logger_path, level='info', colorize=True)

    args_str = dump_args(args)
    logger.info('Run Argv:\n> {}'.format(' '.join(sys.argv)))
    logger.info('Args:\n{}'.format(args_str))
    logger.info('Configs:\n{}'.format(cfg.dumps()))

    model = ScoreNet(cfg.model)
    optimizer = build_optimizer(cfg.optimizer)
    data = sample_batch(10**4, noise=1.)

    train(args, cfg, model, optimizer, data, logger)