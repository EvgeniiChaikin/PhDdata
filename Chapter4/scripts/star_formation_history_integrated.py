import numpy as np

import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator

import h5py as h5
import sys
import json
from scipy import stats
from matplotlib import cm as cmm
from cycler import cycler
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
                y_min, y_max = plot["ylims"]
                snapshot = plot["snapshot"]

                fig, ax = plot_style(8, 8)

                for counter, (key, value) in enumerate(dict_sim.items()):
                    f = h5.File(value + "/output_{:04d}.hdf5".format(snapshot), "r")

                    unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
                    unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
                    unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

                    time = f["/Header"].attrs["Time"]
                    time_snp_Myr = (
                        time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6
                    )

                    bins = 25
                    edges_Myr = np.linspace(0.0, 1e3, bins)
                    centers_Myr = 0.5 * (edges_Myr[1:] + edges_Myr[:-1])
                    values_Msun = np.zeros_like(centers_Myr)

                    birth_masses_Msun = (
                        f["/PartType4/InitialMasses"][:]
                        * unit_mass_in_cgs
                        / constants["SOLAR_MASS_IN_CGS"]
                    )
                    birth_times_Myr = (
                        f["/PartType4/BirthTimes"][:]
                        * unit_time_in_cgs
                        / constants["YEAR_IN_CGS"]
                        / 1e6
                    )

                    for ii, (l, r) in enumerate(zip(edges_Myr[:-1], edges_Myr[1:])):
                        mask = np.logical_and(l < birth_times_Myr, birth_times_Myr <= r)
                        print(np.sum(birth_masses_Msun[mask]))
                        values_Msun[ii] += np.sum(birth_masses_Msun[mask])

                    if split == 0:
                        lw = line_properties["linewidth"]
                        colors = line_properties["colour"]
                        dashes = line_properties["dashes"]
                    if split == 1:
                        colors = color2
                        lw = lw2
                        dashes = dashes2
                    elif split == 3:
                        colors = color3
                        lw = lw3
                        dashes = dashes3
                    elif split == 6:
                        colors = color6
                        lw = lw6
                        dashes = dashes6

                    ax.plot(
                        centers_Myr / 1e3,
                        np.cumsum(values_Msun),
                        lw=lw[counter],
                        color=colors[counter],
                        dashes=tuple(d for d in dashes[counter]),
                        zorder=4,
                        label=key.replace("_", "\_"),
                    )

                # Axes & labels
                ax.xaxis.set_tick_params(labelsize=31)
                ax.yaxis.set_tick_params(labelsize=31)
                ax.set_xlim(-0.02, 1.02)

                # Y log scale
                ax.set_yscale("log")
                plt.yticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10])
                fixlogax(ax, "y")
                ax.set_ylim(y_min, y_max)

                ax.set_xlabel("Time [Gyr]", fontsize=31)
                ax.set_ylabel("$M_*(<t)$ [M$_{\\odot}$]", fontsize=31)
                ax.legend(loc="lower right", fontsize=18, frameon=False)

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)
                plt.close()


if __name__ == "__main__":
    plot()
