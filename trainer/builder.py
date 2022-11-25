import trainer


def build_trainer(cfg, kwargs):
    cfg.kwargs = {} if cfg.kwargs is None else cfg.kwargs
    kwargs.update(cfg.kwargs)
    return getattr(trainer, cfg.type)(**kwargs)