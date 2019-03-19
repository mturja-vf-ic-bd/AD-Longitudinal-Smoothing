import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib


def plot_matrix_all(connectomes, fname="matplt", vmin=0, vmax=0.25, savefig=True):
    fig = plt.figure(figsize=(20, 5))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 4),
                    axes_pad=0.3,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )
    t = 0
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")  # To print subscipt
    for ax in grid:
        ax.set_axis_off()
        im = ax.matshow(connectomes[t], vmin=vmin, vmax=vmax)
        ax.set_title(("t" + str(t)).translate(SUB))
        ax.set_xlabel("ROIs")
        ax.set_ylabel("ROIs")
        t = t + 1

        cbar = grid.cbar_axes[0].colorbar(im)

        cbar.ax.set_yticks(np.arange(vmin, vmax * 1.1, (vmax - vmin) / 4))
        cbar.ax.set_yticklabels(
            [str(vmin), str((vmax - vmin) / 4), str((vmax - vmin) / 2),
             str((vmax - vmin) * 3 / 4), str(vmax)])

    if savefig:
        fig.savefig(fname + ".png")
    else:
        plt.show()


# Plot spectrum of raw and smooth data
def plot_eigen_specturm(rw_spectrum, sm_spectrum, savefig=False):
    T = len(rw_spectrum)
    assert T == len(sm_spectrum), "raw and smooth list size doesn't match"
    fig, axes = plt.subplots(nrows=1, ncols=T, figsize=(5*T, T))
    t = 0
    for ax in axes.flat:
        ax.plot(rw_spectrum[t], color='r')
        ax.plot(sm_spectrum[t], color='b')
        ax.set_title("t" + str(t))
        ax.set_ylim(np.min(rw_spectrum), np.max(rw_spectrum))
        t = t + 1

    if savefig:
        matplotlib.use('Agg')
        fig.savefig("spectrum_plot.png")
    else:
        plt.show()
