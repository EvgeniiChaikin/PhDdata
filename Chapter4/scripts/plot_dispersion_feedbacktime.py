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
                y_min, y_max = plot["ylims"]
                x_min, x_max = plot["xlims"]
                split = plot["split"]
                snapshot = plot["snapshot"]

                fig, ax = plot_style(8, 4.5)

                bins = 40
                edges_Myr = np.linspace(x_min, x_max, bins)  # kpc
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

                    XH = f["/PartType0/ElementMassFractions"][:, 0]
                    HI_frac = f["/PartType0/SpeciesFractions"][:, 0]
                    H2_frac = f["/PartType0/SpeciesFractions"][:, 2]

                    gas_HI_M_Msun = mass_Msun * XH * HI_frac
                    gas_H2_M_Msun = mass_Msun * XH * H2_frac * 2.0
                    gas_neutral_M_Msun = gas_H2_M_Msun + gas_HI_M_Msun

                    sigma2 = (
                        f["/PartType0/VelocityDispersions"][:]
                        * (unit_length_in_cgs / 1e5 / unit_time_in_cgs) ** 2
                    )  # [ (km/sec)**2]

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

                    pos_kpc[:, 2] -= centre_kpc[2]

                    MAX_HEIGHT_kpc = 0.25  # kpc

                    logic_1 = np.logical_and(
                        sigma2 < 1e20, np.abs(pos_kpc[:, 2]) < MAX_HEIGHT_kpc
                    )
                    logic_2 = np.logical_and(
                        logic_1, np.logical_and(vkick_kms > 0.0, vkick_kms < 1e10)
                    )

                    mask_general = np.where(logic_1)
                    mask_kicked = np.where(logic_2)

                    # mass x sigma1D**2
                    mass_sigma2 = gas_neutral_M_Msun * sigma2 / 3.0

                    sigma2_general = np.sum(mass_sigma2[mask_general]) / np.sum(
                        gas_neutral_M_Msun[mask_general]
                    )
                    sigma2_kicked = np.sum(mass_sigma2[mask_kicked]) / np.sum(
                        gas_neutral_M_Msun[mask_kicked]
                    )

                    for bin_c, TIMEWINDOW_Myr in enumerate(centers_Myr):

                        snii_times = np.where(
                            np.logical_and(
                                logic_2,  
                                np.abs(
                                    feedback_times_Myr - time_Myr[0]
                                )
                                <= TIMEWINDOW_Myr,
                            )
                        )
                        sigma2_bin = np.sum(mass_sigma2[snii_times]) / np.sum(
                            gas_neutral_M_Msun[snii_times]
                        )
                        sigma_bin = np.sqrt(sigma2_bin)

                        print(
                            "bin",
                            bin_c,
                            "Time lookback",
                            TIMEWINDOW_Myr,
                            "N gas",
                            np.size(mass_sigma2[snii_times]),
                        )
                        values[bin_c] = sigma_bin

                    if split < 1:
                        colors = line_properties["colour"]
                        lws = line_properties["linewidth"]
                    else:
                        colors = color3
                        lws = lw3

                    ax.plot(
                        centers_Myr / 1e3,  # To Gyr
                        values,
                        lw=lws[counter],
                        color=colors[counter],
                        label=key.replace("_", "\_"),
                    )
                    ax.axhline(
                        y=np.sqrt(sigma2_general),
                        color=colors[counter],
                        lw=2,
                        dashes=(10, 3, 3, 3),
                    )

                    print("Max values", np.max(values))

                ax.set_xlim(x_min / 1e3, x_max / 1e3)  # From Myr to Gyr
                ax.yaxis.set_ticks([10, 20, 30, 40, 50])
                ax.set_ylim(y_min, y_max)

                ax.xaxis.set_tick_params(labelsize=33)
                ax.yaxis.set_tick_params(labelsize=33)

                if split != 1:
                    print("Setting legend")
                    leg1 = ax.legend(
                        fontsize=26.0,
                        ncol=2,
                        bbox_to_anchor=(-0.15, 1.4), # -0.20 1.42
                        loc="upper left",
                        handlelength=0,
                        handletextpad=0.4, # 0.0
                        columnspacing=0.25,
                        borderaxespad=0.20,
                        labelspacing=0.3,
                    )
                    hl_dict = {
                        handle.get_label(): handle for handle in leg1.legendHandles
                    }
                    for k in hl_dict:
                        hl_dict[k].set_color("white")

                    for counter, text in enumerate(leg1.get_texts()):
                        print(counter, text)
                        text.set_color(colors[counter])

                ax.text(
                    0.03,
                    0.04,
                    "$t = {:.1f}$ Gyr".format(time_Myr[0] / 1e3),
                    ha="left",
                    va="bottom",
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

                ax.set_ylabel(
                    "$\\sigma_{\\rm turb} \\, [\\rm km \\, s^{-1}]$ ($<\\Delta \\, t_{\\rm kick}$)",
                    fontsize=29, labelpad=23.0
                )

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)
                plt.close()


if __name__ == "__main__":
    plot()
