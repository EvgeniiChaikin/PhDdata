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
                x_min, x_max = plot["xlims"]
                snapshot = plot["snapshot"]

                fig, ax = plot_style(8, 8)

                for counter, (key, value) in enumerate(dict_sim.items()):
                    print(counter)

                    f = h5.File(value + "/output_{:04d}.hdf5".format(snapshot), "r")

                    unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
                    unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
                    unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

                    boxsize_kpc = (
                        f["/Header"].attrs["BoxSize"]
                        * unit_length_in_cgs
                        / constants["PARSEC_IN_CGS"]
                        / 1e3
                    )
                    centre_kpc = boxsize_kpc / 2.0

                    time = f["/Header"].attrs["Time"]
                    time_snp_Myr = (
                        time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6
                    )

                    bins = 25
                    edges_kpc = np.linspace(x_min, x_max, bins)  # kpc
                    centers_kpc = 0.5 * (edges_kpc[1:] + edges_kpc[:-1])
                    areas_kpc2 = np.pi * (edges_kpc[1:] ** 2 - edges_kpc[:-1] ** 2)

                    pos_kpc = (
                        f[f"/PartType0/Coordinates"][:, :]
                        * unit_length_in_cgs
                        / constants["PARSEC_IN_CGS"]
                        / 1e3
                    )
                    mass_Msun = (
                        f[f"/PartType0/Masses"][:]
                        * unit_mass_in_cgs
                        / constants["SOLAR_MASS_IN_CGS"]
                    )

                    pos_kpc[:, 0] -= centre_kpc[0]
                    pos_kpc[:, 1] -= centre_kpc[0]
                    r_kpc = np.sqrt(pos_kpc[:, 0] ** 2.0 + pos_kpc[:, 1] ** 2.0)

                    values_Msun, _, _ = stats.binned_statistic(
                        r_kpc, mass_Msun, statistic="sum", bins=edges_kpc
                    )
                    Sigma_Msun_pc2 = values_Msun / areas_kpc2 / (1e3) ** 2.0

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
                        centers_kpc,
                        Sigma_Msun_pc2,
                        lw=lw[counter],
                        color=colors[counter],
                        dashes=tuple(d for d in dashes[counter]),
                        zorder=4,
                        label=key.replace("_", "\_"),
                    )

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_yscale("log")

                ax.xaxis.set_tick_params(labelsize=33)
                ax.yaxis.set_tick_params(labelsize=33)

                ax.set_xlabel("Radial distance $r$ [kpc]", fontsize=33)

                if split != 6 and split != 1:
                    leg1 = plt.legend(
                        loc="upper right",
                        fontsize=28,
                        frameon=False,
                        handlelength=0,
                        handletextpad=0,
                        borderaxespad=0.20,
                        labelspacing=0.3,
                    )

                    hl_dict = {
                        handle.get_label(): handle for handle in leg1.legendHandles
                    }
                    for k in hl_dict:
                        hl_dict[k].set_color("white")

                    for counter, text in enumerate(leg1.get_texts()):
                        text.set_color(line_properties["colour"][counter])

                ax.set_ylabel(
                    "$\\Sigma_{\\rm gas} \\, [\\rm M_\\odot \\, pc^{-2}]$",
                    fontsize=33,
                )
                ax.text(
                    0.04,
                    0.04,
                    "$t = {:.2f}$ Gyr".format(time_snp_Myr[0] / 1e3),
                    ha="left",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=32,
                )

                fixlogax(ax, "y")

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)
                plt.close()


if __name__ == "__main__":
    plot()
