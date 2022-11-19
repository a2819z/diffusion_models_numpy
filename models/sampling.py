import numpy as np


def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)


def simple(model, x, n_steps=100, eps=1e-3):
    def step(x, i):
        x = x + eps * model(x)
        return x, x

    return scan(step, x, np.arange(n_steps))[1]

def langevin(model, x, key_seq, n_steps=100, eps=4e-3, decay=0.97, temperature=0.7):
    def step(state, i):
        x, key_seq, eps = state
        z_t = np.normal(size=x.shape)