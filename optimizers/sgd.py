from models.base import NN


class SGD(NN):
    def __init__(self, lr=1e-3):
        super(SGD, self).__init__()
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]