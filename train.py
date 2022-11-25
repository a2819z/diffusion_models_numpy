from argparse import ArgumentParser
import sys
from pathlib import Path
import os.path as osp

from data import load_data
from models.builder import build_model
from trainer.builder import build_trainer
from optimizers.builder import build_optimizer

from sconf import Config, dump_args, dump_yaml
from utils.logger import Logger
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


if __name__ == "__main__":
    args, cfg = parse_args_and_config()

    logger_path = cfg.work_dir / 'log.log'
    logger = Logger.get(file_path=logger_path, level='info', colorize=True)

    args_str = dump_args(args)
    logger.info('Run Argv:\n> {}'.format(' '.join(sys.argv)))
    logger.info('Args:\n{}'.format(args_str))
    logger.info('Configs:\n{}'.format(cfg.dumps()))

    model = build_model(cfg.model)
    optimizer = build_optimizer(cfg.optimizer)
    train_data = load_data(type=cfg.data.type, size=10**4, noise=1., fname=cfg.data.path)
    test_data = load_data(type=cfg.data.type, size=500, noise=1., fname=cfg.data.path, test=True)

    kwargs = {'cfg': cfg, 'model': model, 'optimizer': optimizer, 'logger': logger}
    trainer = build_trainer(cfg.trainer, kwargs)
    trainer.train(train_data, test_data)