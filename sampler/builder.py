import sampler


def build_sampler(sampler_cfg, cfg, model):
    kwargs = {} if sampler_cfg.kwargs is None else sampler_cfg.kwargs
    kwargs['model'] = model
    kwargs['cfg'] = cfg
    return getattr(sampler, sampler_cfg.type)(**kwargs)