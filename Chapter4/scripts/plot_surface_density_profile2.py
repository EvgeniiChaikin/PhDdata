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
                y_min, y_max = plot["ylims"]
                x_min, x_max = plot["xlims"]
                snapshot = plot["snapshot"]

                fig, (ax, ax2) = plt.subplots(
                    2, 1, figsize=(8.0, 10.0), sharex=True, sharey=False
                )

                for counter, (key, value) in enumerate(dict_sim.items()):
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

                    birth_times_Myr = (
                        f["/PartType4/BirthTimes"][:]
                        * unit_time_in_cgs
                        / constants["YEAR_IN_CGS"]
                        / 1e6
                    )

                    mask = birth_times_Myr > 0.0

                    bins = 25
                    edges_kpc = np.linspace(x_min, x_max, bins)  # kpc
                    centers_kpc = 0.5 * (edges_kpc[1:] + edges_kpc[:-1])
                    areas_kpc2 = np.pi * (edges_kpc[1:] ** 2 - edges_kpc[:-1] ** 2)

                    Sigma_Msun_pc2 = []

                    for part_type in [0, 4]:
                        pos_kpc = (
                            f[f"/PartType{part_type}/Coordinates"][:, :]
                            * unit_length_in_cgs
                            / constants["PARSEC_IN_CGS"]
                            / 1e3
                        )
                        mass_Msun = (
                            f[f"/PartType{part_type}/Masses"][:]
                            * unit_mass_in_cgs
                            / constants["SOLAR_MASS_IN_CGS"]
                        )

                        pos_kpc[:, 0] -= centre_kpc[0]
                        pos_kpc[:, 1] -= centre_kpc[0]

                        r_kpc = np.sqrt(pos_kpc[:, 0] ** 2.0 + pos_kpc[:, 1] ** 2.0)

                        if part_type == 0:
                            values_Msun, _, _ = stats.binned_statistic(
                                r_kpc, mass_Msun, statistic="sum", bins=edges_kpc
                            )
                        else:
                            values_Msun, _, _ = stats.binned_statistic(
                                r_kpc[mask],
                                mass_Msun[mask],
                                statistic="sum",
                                bins=edges_kpc,
                            )
                        Sigma_Msun_pc2.append(values_Msun / areas_kpc2 / (1e3) ** 2.0)

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

                    for c_part, a in enumerate([ax, ax2]):
                        a.plot(
                            centers_kpc,
                            Sigma_Msun_pc2[c_part],
                            lw=lw[counter],
                            color=colors[counter],
                            dashes=tuple(d for d in dashes[counter]),
                            zorder=4,
                            label=key.replace("_", "\_"),
                        )

                for a in [ax, ax2]:
                    a.tick_params(
                        axis="both",
                        which="both",
                        pad=8,
                        left=True,
                        right=True,
                        top=True,
                        bottom=True,
                    )
                    a.tick_params(which="both", width=1.7)
                    a.tick_params(which="major", length=9)
                    a.tick_params(which="minor", length=5)
                    x_minor_locator = AutoMinorLocator(5)
                    y_minor_locator = AutoMinorLocator(5)
                    a.xaxis.set_minor_locator(x_minor_locator)
                    a.yaxis.set_minor_locator(y_minor_locator)

                    a.set_xlim(x_min, x_max)
                    a.set_yscale("log")
                    a.xaxis.set_tick_params(labelsize=33)
                    a.yaxis.set_tick_params(labelsize=33)

                    a.set_yticks([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

                    locmin = ticker.LogLocator(
                        base=10.0,
                        subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                        numticks=10,
                    )

                    a.yaxis.set_minor_locator(locmin)
                    a.yaxis.set_minor_formatter(ticker.NullFormatter())

                ax2.set_xlabel("Radial distance $r$ [kpc]", fontsize=33)

                if split != 6:
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
                ax2.set_ylabel(
                    "$\\Sigma_{\\rm star} \\, [\\rm M_\\odot \\, pc^{-2}]$",
                    fontsize=33,
                )
                ax.text(
                    0.96,
                    0.96,
                    "$t = {:.2f}$ Gyr".format(time_snp_Myr[0] / 1e3),
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=32,
                )

                fixlogax(ax, "y")
                fixlogax(ax2, "y")

                ax.set_ylim(y_min, y_max)
                ax2.set_ylim(y_min / 30.0, y_max)

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)
                plt.close()


if __name__ == "__main__":
    plot()
