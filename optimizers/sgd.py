from models.base import NN


class SGD(NN):
    def __init__(self, lr=1e-3, weight_decay=0.0):
        super(SGD, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] + self.weight_decay * params[key]