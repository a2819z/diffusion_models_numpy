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


def langevin(model, x, n_steps=100, eps=4e-3, decay=0.97, temperature=0.7):
    def step(state, i):
        x, eps = state
        z_t = np.random.randn(*x.shape)
        x = x + (eps / 2) * model(x) + (np.sqrt(eps) * temperature * z_t)
        eps *= decay
        return (x, eps), x

    return scan(step, (x, eps), np.arange(n_steps))[1]


def annealed_langeving_dynamics(model, x, eps=1e-4, n_iter=100, n_step=10, sigma_begin=1, sigma_end=0.01):
    sigmas = np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), n_step))
    ys = []
    for noise_level in range(n_step):
        step_size = eps * (sigmas[noise_level] / sigmas[-1])**2
        for _ in range(n_iter):
            z_t = np.random.randn(*x.shape)

            x = x + step_size / 2 * model(x, noise_level) + np.sqrt(step_size) * z_t
            ys.append(x)

    return np.stack(ys)
        
