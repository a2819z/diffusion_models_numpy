class NN:
    def __init__(self):
        self.params = {}

    def state_dict(self):
        return self.params

    def load_state_dict(self, state_dict: dict):
        for key in self.params.keys():
            param = state_dict.get(key, None)
            if param is None:
                raise ValueError(f'Key {key} is not exists in the self.params!')
            self.params[key] = param