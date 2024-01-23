import numpy as np

import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator

import h5py as h5
import sys
import json
from scipy import stats
from matplotlib import cm as cmm
from matplotlib import ticker
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
                y_min, y_max = plot["ylims"]
                x_min, x_max = plot["xlims"]
                split = plot["split"]
                snapshot = plot["snapshot"]

                fig, ax = plot_style(8, 4.5)

                bins = 40
                edges_Myr = np.linspace(x_min, x_max, bins)
                centers_Myr = 0.5 * (edges_Myr[1:] + edges_Myr[:-1])
                values = np.zeros_like(centers_Myr)

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
                    time_Myr = time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6

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

                    sfr = f[f"/PartType0/StarFormationRates"][:]

                    XH = f["/PartType0/ElementMassFractions"][:, 0]
                    HI_frac = f["/PartType0/SpeciesFractions"][:, 0]
                    H2_frac = f["/PartType0/SpeciesFractions"][:, 2]

                    gas_HI_M_Msun = mass_Msun * XH * HI_frac
                    gas_H2_M_Msun = mass_Msun * XH * H2_frac * 2.0
                    gas_neutral_M_Msun = gas_H2_M_Msun + gas_HI_M_Msun

                    vkick_kms = (
                        f["/PartType0/LastSNIIKineticFeedbackvkick"][:]
                        * unit_length_in_cgs
                        / unit_time_in_cgs
                        / 1e5
                    )

                    feedback_times_Myr = (
                        f["/PartType0/LastSNIIKineticFeedbackTimes"][:]
                        * unit_time_in_cgs
                        / constants["YEAR_IN_CGS"]
                        / 1e6
                    )

                    MAX_HEIGHT_kpc = 0.25  # kpc
                    pos_kpc[:, 2] -= centre_kpc[2]

                    logic_0 = np.abs(pos_kpc[:, 2]) < MAX_HEIGHT_kpc  # all gas
                    logic_1 = np.logical_and(logic_0, sfr > 0.0)  # star forming
                    logic_2 = np.logical_and(
                        logic_0, np.logical_and(vkick_kms > 0.0, vkick_kms < 1e10)
                    )  # kicked
                    logic_2sfr = np.logical_and(
                        logic_1, np.logical_and(vkick_kms > 0.0, vkick_kms < 1e10)
                    )  # kicked and star forming

                    mask_sfr = np.where(logic_1)

                    for bin_c, TIMEWINDOW_Myr in enumerate(centers_Myr):
                        mask_bin = (
                            np.abs(feedback_times_Myr - time_Myr[0]) <= TIMEWINDOW_Myr
                        )

                        mask_kicked_all = np.where(np.logical_and(logic_2, mask_bin))
                        mask_kicked_sfr = np.where(np.logical_and(logic_2sfr, mask_bin))

                        sfr_fraction = np.sum(
                            gas_neutral_M_Msun[mask_kicked_sfr]
                        ) / np.sum(gas_neutral_M_Msun[mask_kicked_all])
                        values[bin_c] = sfr_fraction

                    if split != 3:
                        colors = line_properties["colour"]
                        lws = line_properties["linewidth"]
                        dashes = line_properties["dashes"]
                    else:
                        colors = color3
                        lws = lw3
                        dashes = dashes3

                    ax.plot(
                        centers_Myr / 1e3,  # To Gyr
                        values,
                        lw=lws[counter],
                        dashes=tuple(d for d in dashes[counter]),
                        color=colors[counter],
                        label=key.replace("_", "\_"),
                        # label="Only kicked particles in the neutral ISM"
                    )
                    if split != 3:
                        ax.axhline(
                            y=np.sum(gas_neutral_M_Msun[mask_sfr])
                            / np.sum(gas_neutral_M_Msun[logic_0]),
                            color=colors[counter],
                            lw=2,
                            dashes=(10, 3, 3, 3),
                            # label="All particles in the neutral ISM"
                        )

                ax.set_xlim(x_min / 1e3, x_max / 1e3)  # From Myr to Gyr
                ax.set_yscale("log")

                ax.xaxis.set_tick_params(labelsize=33)
                ax.yaxis.set_tick_params(labelsize=33)

                ax.set_xlabel(
                    "Time since last kick $\\Delta \\, t_{\\rm kick}$ [Gyr]",
                    fontsize=33,
                )
                ax.set_ylabel(
                    "Mass fr. of SF gas ($<\\Delta \\, t_{\\rm kick}$)",
                    fontsize=LABEL_SIZE * 0.8,
                )

                locmin = ticker.LogLocator(
                    base=10.0,
                    subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    numticks=10,
                )

                ax.yaxis.set_minor_locator(locmin)
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())

                ax.set_yticks(
                    [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
                )
                ax.set_ylim(y_min, y_max)

                if split > 0 and split != 3:
                    leg1 = plt.legend(loc="lower right", fontsize=21, frameon=False)

                ax.text(
                    0.03,
                    0.96,
                    "$t = {:.1f}$ Gyr".format(time_Myr[0] / 1e3),
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=33,
                )

                ax.text(
                    0.96,
                    0.94,
                    r"H\textsc{i} + H$_2$",
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=33,
                )

                fixlogax(ax, "y")

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)
                plt.close()


if __name__ == "__main__":
    plot()
