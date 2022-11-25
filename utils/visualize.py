import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_lines(model, data, xlim=(-1.5, 2.0), ylim=(-1.5, 2.0), nx=50, ny=50, plot_scatter=True, alpha=1.0):
    xx = np.stack(np.meshgrid(np.linspace(*xlim, nx), np.linspace(*ylim, ny)), axis=-1).reshape(-1, 2)
    scores = model(xx)
    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
    # Perform the plots
    
    if plot_scatter:
        plt.figure(figsize=(16,12))
        plt.scatter(data[:, 0], data[:, 1], alpha=0.3, color='red', edgecolor='black', s=40)
        plt.xlim(-1.5, 2.0)
        plt.ylim(-1.5, 2.0)
    
    quiver = plt.quiver(*xx.T, *scores_log1p.T, width=0.002, color='black', alpha=alpha)
    
    return quiver


def make_animation(Xt, Gs, xx):
    X0 = np.tile(Xt[:1], (10, 1, 1)) # For delay
    Xf = np.tile(Xt[-1:], (10, 1, 1))
    Xt = np.concatenate([X0, Xt, Xf])

    if Gs.shape[0] != 1:
        G0 = np.tile(Gs[:1], (10, 1, 1))
        Gf = np.tile(Gs[:1], (10, 1, 1))
        Gs = np.concatenate([G0, Gs, Gf])

    fig = plt.figure(figsize=(14, 10))
    ax = plt.gca()

    xlim = (xx[..., 0].min(), xx[..., 0].max())
    ylim = (xx[..., 1].min(), xx[..., 1].max())
    #xlim = (Gs[..., 0].min() * 0.7, Gs[..., 0].max() * 0.7)
    #ylim = (Gs[..., 1].min() * 0.7, Gs[..., 1].max() * 0.7)

    ax.axis("off")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    scatter = ax.scatter(Xt[0, :, 0], Xt[0, :, 1])
    quiver = ax.quiver(*xx.T, *Gs[0].T, width=0.002, color='black')

    def animate(i):
        scatter.set_offsets(Xt[i])
        if Gs.shape[0] != 1:
            quiver.set_UVC(*Gs[i].T)
        return (quiver, scatter,)

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=lambda: animate(0),
        frames=range(0, len(Xt)),
        interval=100,
        repeat_delay=1000,
        blit=True,
    )

    return anim