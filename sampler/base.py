class BaseSampler:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    @NotImplementedError
    def sampling(self, x, args):
        pass