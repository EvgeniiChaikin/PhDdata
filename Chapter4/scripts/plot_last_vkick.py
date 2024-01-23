import numpy as np

import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator

import h5py as h5
import sys
import json
from scipy import stats
from matplotlib import cm as cmm
from cycler import cycler
from matplotlib import ticker
from constants import *
from plot_style import *


def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def plot():
    runs = read_sim_data("main.json")

    for script_name, plots in runs.items():
        print(script_name)

        if script_name == sys.argv[0]:
            print("FOUND")

            for plot in plots:
                output = plot["output_file"]
                dict_sim = plot["data"]
                split = plot["split"]
                snapshot = plot["snapshot"]

                print(output)

                bins = 90
                edges = np.linspace(-1.5, 3.5, bins)  # log km/s
                centers = 0.5 * (edges[1:] + edges[:-1])

                fig, ax = plot_style(8, 5.5)

                colormap = cmm.viridis

                if split == 0:
                    colors = [colormap(i) for i in np.linspace(0, 1, len(dict_sim) - 3)]
                else:
                    colors = [colormap(i) for i in np.linspace(0, 1, len(dict_sim))]

                ax.set_prop_cycle(cycler("color", colors[::-1]))

                for counter, (key, value) in enumerate(dict_sim.items()):
                    f = h5.File(value + "/output_{:04d}.hdf5".format(snapshot), "r")

                    unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
                    unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
                    unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

                    vkick = (
                        f["/PartType0/LastSNIIKineticFeedbackvkick"][:]
                        * unit_length_in_cgs
                        / unit_time_in_cgs
                        / 1e5
                    )
                    mass = f["/PartType0/Masses"][:]

                    mask = np.where(vkick > 0.0)
                    values, _, _ = stats.binned_statistic(
                        np.log10(vkick[mask]), mass[mask], statistic="sum", bins=edges
                    )
                    print(np.max(values))
                    values /= np.sum(values)
                    print(np.max(values))

                    if split == 0:
                        if counter < len(dict_sim.items()) - 3:
                            (line,) = ax.plot(
                                10.0**centers,
                                values,
                                label=key.replace("_", "\_"),
                                lw=3.0,
                            )
                            ax.axvline(
                                x=np.median(vkick[mask]),
                                lw=1.5,
                                zorder=-10,
                                color=line.get_color(),
                                dashes=(6, 3),
                            )
                        elif counter == len(dict_sim.items()) - 3:
                            (ref1,) = ax.plot(
                                10.0**centers,
                                values,
                                lw=2.0,
                                color="black",
                                dashes=(8, 2, 2, 2),
                                zorder=1000,
                            )
                        elif counter == len(dict_sim.items()) - 2:
                            ref2 = ax.fill_between(
                                10.0**centers,
                                np.ones_like(values) / 1e5,
                                values,
                                color="chocolate",
                                alpha=0.6,
                            )
                        elif counter == len(dict_sim.items()) - 1:
                            ref3 = ax.fill_between(
                                10.0**centers,
                                np.ones_like(values) / 1e5,
                                values,
                                color="grey",
                                alpha=0.2,
                            )
                    elif split == 7:
                        (line,) = ax.plot(
                            10.0**centers,
                            values,
                            label=key.replace("_", "\_"),
                            color=color7[counter],
                            lw=lw7[counter],
                            dashes=tuple(d for d in dashes7[counter]),
                        )
                    else:
                        (line,) = ax.plot(
                            10.0**centers,
                            values,
                            label=key.replace("_", "\_"),
                            lw=3.0,
                        )
                        ax.axvline(
                            x=np.median(vkick[mask]),
                            lw=1.5,
                            zorder=-10,
                            color=line.get_color(),
                            dashes=(6, 3),
                        )

                    time = f["/Header"].attrs["Time"]
                    time_snp = time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6

                ax.set_xscale("log")
                ax.set_yscale("log")

                ax.set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

                locmin = ticker.LogLocator(
                    base=10.0,
                    subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    numticks=10,
                )

                ax.xaxis.set_minor_locator(locmin)
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())

                locmin = ticker.LogLocator(
                    base=10.0,
                    subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    numticks=10,
                )

                ax.yaxis.set_minor_locator(locmin)
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
                ax.yaxis.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])

                ax.set_ylim(10**-4, 10**1.9)
                ax.set_xlim(10**-1.2, 10**3.6)

                ax.xaxis.set_tick_params(labelsize=30)
                ax.yaxis.set_tick_params(labelsize=30)

                fixlogax(ax, "x")
                fixlogax(ax, "y")

                ax.set_xlabel(
                    "Actual kick velocity $\\Delta v$ [km s$^{-1}$]", fontsize=33
                )
                ax.set_ylabel("Mass fr. of particles", labelpad=0, fontsize=33)

                if split != 7:
                    leg1 = ax.legend(
                        fontsize=27.5,
                        ncol=2,
                        bbox_to_anchor=(-0.20, 1.4),
                        loc="upper left",
                        handlelength=0,
                        handletextpad=0,
                        columnspacing=0.25,
                        borderaxespad=0.20,
                        labelspacing=0.3,
                    )
                else:
                    leg1 = ax.legend(
                        fontsize=24.5,
                        ncol=1,
                        bbox_to_anchor=(-0.20, 1.45),
                        loc="upper left",
                    )

                if split == 0:
                    leg2 = plt.legend(
                        [ref1, ref2, ref3],
                        [
                            "H12\_M5\_fkin0p1\_v0050\_NoRelMotion\_MulKicks",
                            "H12\_M5\_fkin0p1\_v0050\_NoRelMotion",
                            "H12\_M5\_fkin0p1\_v0050",
                        ],
                        loc="upper left",
                        fontsize=LEGEND_SIZE * 0.78,
                    )

                    plt.gca().add_artist(leg1)

                if split != 7:
                    print("Setting legend")
                    hl_dict = {
                        handle.get_label(): handle for handle in leg1.legendHandles
                    }
                    for k in hl_dict:
                        hl_dict[k].set_color("white")

                    for counter, text in enumerate(leg1.get_texts()):
                        text.set_color(colors[::-1][counter])

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)
                plt.close()


if __name__ == "__main__":
    plot()
