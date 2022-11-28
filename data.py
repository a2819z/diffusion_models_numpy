import os
import os.path as osp
import pickle

import numpy as np
from sklearn.datasets import make_swiss_roll, make_circles


def generate_swiss(size, noise=1., save=True, fname='swiss.pickle', test=False):
    if test:
        data = np.random.randn(size, 2)
    else:
        x, _ = make_swiss_roll(size, noise=noise)
        data = x[:, [0, 2]] / 10.

    if save:
        fname = fname.split('.')[0] + '_test.pickle' if test else fname
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

    return data


def generate_gauss(size, save=True, fname='gauss.pickle', test=False):
    if test:
        lim = (-10, 10)
        size = int(np.sqrt(size))
        data = np.stack(np.meshgrid(np.linspace(*lim, size), np.linspace(*lim, size)), axis=-1).reshape(-1, 2)
    else:
        x1 = np.random.normal(-5, 1, size=(int(size/5), 2))
        x2 = np.random.normal(5, 1, size=(int(size/5*4), 2))
        data = np.concatenate((x1, x2))

    if save:
        fname = fname.split('.')[0] + '_test.pickle' if test else fname
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

    return data


def generate_circle(size, save=True, fname='circle.pickle', test=False):
    if test:
        lim = (-2, 2)
        size = int(np.sqrt(size))
        data = np.stack(np.meshgrid(np.linspace(*lim, size), np.linspace(*lim, size)), axis=-1).reshape(-1, 2)
    else:
        data, _ = make_circles(n_samples=size, factor=0.4, noise=0.05)

    if save:
        fname = fname.split('.')[0] + '_test.pickle' if test else fname
        with open(fname, 'wb') as f:
            pickle.dump(data, f)
    
    return data



def load_data(type, size, noise=1., save=True, fname='swiss.pickle', test=False):
    fname = fname.split('.')[0] + '_test.pickle' if test is True else fname
    if osp.exists(fname):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        if type == 'swiss':
            data = generate_swiss(size, noise, save, test=test)
        elif type == 'gauss':
            data = generate_gauss(size, save, test=test)
        elif type == 'circle':
            data = generate_circle(size, save, test=test)
        else:
            raise ValueError(f'{type} is not supported...')
    
    return data