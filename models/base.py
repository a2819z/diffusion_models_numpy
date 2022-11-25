import numpy as np 


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
            self.params[key][:] = param[:]


def weight_initialization(weight, method='std', **kwargs):
    w = np.random.randn(*weight.shape)

    if method == 'std':
        w *= kwargs.get('std', 0.01)
    elif method == 'xaiver':
        w *= np.sqrt(2 / np.sum(weight.shape))
    elif method == 'he':
        w *= np.sqrt(2 / weight.shape[0])
    else:
        raise ValueError(f'{method} is not supported for weight initialization!')

    return w