import optimizers


def build_optimizer(cfg):
    return getattr(optimizers, cfg.type)(**cfg.args)