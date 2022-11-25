from sampler.builder import build_sampler
from utils.utils import save_dict, load_dict


class BaseTrainer:
    def __init__(self, cfg, model, optimizer, logger):
        self.cfg = cfg

        self.model = model
        self.optimizer = optimizer

        self.sampler = build_sampler(self.cfg.sampler, self.cfg, self.model)

        self.logger = logger

    @NotImplementedError
    def train(self, n_steps, train_data):
        pass

    def save_checkpoint(self, save_dir, step, model_state_dict, optim_state_dict, best=False, **kwargs):
        states = {'step': step, 'model': model_state_dict, 'optim': optim_state_dict}
        states.update(kwargs)

        if best:
            save_dict(states, save_dir / 'best.pickle')
        else:
            save_dict(states, save_dir / f'{step:06d}.pickle')
            save_dict(states, save_dir / 'latest.pickle')

    def load_checkpoint(self, fname, model, optim=None):
        states = load_dict(fname)
        model.load_state_dict(states['model'])
        if optim is not None:
            optim.load_state_dict(states['optim'])
        step = states['step']

        return model, optim, step