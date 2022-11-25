from argparse import ArgumentParser
from pathlib import Path
import os.path as osp

import numpy as np

from models.builder import build_model
from sampler.builder import build_sampler
from data import load_data

from sconf import Config
from utils.utils import load_checkpoint


def parse_args_and_config():
    parser = ArgumentParser()
    parser.add_argument('resume', type=str, help="Path of checkpoint")
    parser.add_argument('--epoch', default='latest.pickle', help="Path of checkpoint")
    

    args, left_argv = parser.parse_known_args()
    cfg = Config(osp.join(args.resume, 'config.yaml'), default='configs/default.yaml')
    cfg.argv_update(left_argv)

    cfg.work_dir = Path(cfg.work_dir)

    return args, cfg


if __name__ == "__main__":
    args, cfg = parse_args_and_config()

    model = build_model(cfg.model)
    ckpt_fname = cfg.work_dir / 'checkpoint' / args.epoch
    model, _, step = load_checkpoint(ckpt_fname, model)

    sampler = build_sampler(cfg.sampler, cfg, model)

    #data = np.random.normal(size=(500, 2))
    #data = generate_data(size=500, noise=1.)
    #data = np. random.randn(500, 2)
    train_data = load_data(type=cfg.data.type, size=10**4, noise=1., fname=cfg.data.path)
    test_data = load_data(type=cfg.data.type, size=10**4, noise=1., fname=cfg.data.path, test=True)
    
    sampler.visualization(train_data, test_data, step, show=True)