import matplotlib.pylab as plt
from matplotlib import cm as cmm
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from cycler import cycler

plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"
plt.rc("text", usetex=True)
plt.rcParams["axes.linewidth"] = 2
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

line_properties = {
    "colour": ["black", "orange", "dodgerblue", "forestgreen"],
    "linewidth": [3.5, 4.5, 3.0, 2.5, 2, 6],
    "alpha": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "dashes": [[3, 0], [3, 0], [3, 0], [3, 0]],
    "ls": ["-", "-", "-", "-", "-", "-"],
}

colormap = cmm.viridis
color2 = [colormap(i) for i in np.linspace(0, 1, 5)][::-1]
color3 = [
    "black",
    "orange",
    "dodgerblue",
    "forestgreen",
    "black",
    "orange",
    "dodgerblue",
    "forestgreen",
]
color4 = ["darkred", "orange", "magenta", "c"]  # "burlywood"]
color6 = ["orange", "orange", "orange", "forestgreen", "forestgreen", "forestgreen"]
color7 = ["deepskyblue", "orange", "darkred", "forestgreen"]

dashes2 = [[3, 0], [3, 0], [3, 0], [3, 0], [3, 0], [3, 0]]  # V kick
dashes3 = [[3, 0], [3, 0], [3, 0], [3, 0], [4, 4], [4, 4], [4, 4], [4, 4]]  # SF crit
dashes4 = [[3, 0], [3, 0], [3, 0], [3, 0]]
dashes6 = [[3, 0], [2.5, 2.5], [15, 4], [3, 0], [2.5, 2.5], [15, 4]]  # NoRelMotion
dashes7 = [[3, 0], [3, 0], [3, 0], [3, 0]]

lw2 = [4.5, 4.5, 4.5, 4.5, 4.5, 1.4]
lw3 = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
lw4 = [2.5, 2.5, 2.5, 2.5]
lw6 = [3.75, 3.75, 3.75, 2.0, 2.0, 2.0]
lw7 = [3.5, 3.5, 3.5, 3.5]

LABEL_SIZE = 30
LEGEND_SIZE = 24


def plot_style(xsize, ysize):
    plt.rc("text", usetex=True)
    plt.rcParams.update({"figure.autolayout": True})
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["figure.figsize"] = xsize, ysize

    fig, ax = plt.subplots()
    x_minor_locator = AutoMinorLocator(5)
    y_minor_locator = AutoMinorLocator(5)
    plt.tick_params(which="both", width=1.7)
    plt.tick_params(which="major", length=9)
    plt.tick_params(which="minor", length=5)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    ax.tick_params(
        axis="both", which="both", pad=8, left=True, right=True, top=True, bottom=True
    )
    fig.set_tight_layout(False)

    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["lines.dashed_pattern"] = [6, 6]
    plt.rcParams["lines.dashdot_pattern"] = [3, 5, 1, 5]
    plt.rcParams["lines.dotted_pattern"] = [1, 3]
    plt.rcParams["lines.scale_dashes"] = False
    plt.rcParams["errorbar.capsize"] = 6

    return (fig, ax)


def fixlogax(ax, a="x"):
    if a == "x":
        labels = [item.get_text() for item in ax.get_xticklabels()]
        positions = ax.get_xticks()
        for i in range(len(positions)):
            labels[i] = "$10^{" + str(int(np.log10(positions[i]))) + "}$"
        if np.size(np.where(positions == 1)) > 0:
            labels[np.where(positions == 1)[0][0]] = "$1$"
        if np.size(np.where(positions == 10)) > 0:
            labels[np.where(positions == 10)[0][0]] = "$10$"
        if np.size(np.where(positions == 0.1)) > 0:
            labels[np.where(positions == 0.1)[0][0]] = "$0.1$"
        ax.set_xticklabels(labels)
    if a == "y":
        labels = [item.get_text() for item in ax.get_yticklabels()]
        positions = ax.get_yticks()
        for i in range(len(positions)):
            labels[i] = "$10^{" + str(int(np.log10(positions[i]))) + "}$"
        if np.size(np.where(positions == 1)) > 0:
            labels[np.where(positions == 1)[0][0]] = "$1$"
        if np.size(np.where(positions == 10)) > 0:
            labels[np.where(positions == 10)[0][0]] = "$10$"
        if np.size(np.where(positions == 0.1)) > 0:
            labels[np.where(positions == 0.1)[0][0]] = "$0.1$"
        ax.set_yticklabels(labels)
