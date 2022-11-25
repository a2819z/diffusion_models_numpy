import models


def build_model(cfg):
    return getattr(models, cfg.type)(cfg)