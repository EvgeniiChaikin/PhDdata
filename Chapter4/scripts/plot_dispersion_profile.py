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
                snapshot_min = plot["snapshot_min"]
                snapshot_max = plot["snapshot_max"]
                time_min, time_max = 0.0, 0.0

                print("Min snp", snapshot_min, "N snp", snapshot_max - snapshot_min)

                fig, ax = plot_style(8, 6)

                bins = 20
                edges_kpc = np.linspace(x_min, x_max, bins)  # kpc
                centers_kpc = 0.5 * (edges_kpc[1:] + edges_kpc[:-1])
                values_HI = np.zeros_like(centers_kpc)
                values_H2 = np.zeros_like(centers_kpc)

                for counter, (key, value) in enumerate(dict_sim.items()):

                    for snp in range(snapshot_min, snapshot_max):
                        print("snp", snp)

                        f = h5.File(value + "/output_{:04d}.hdf5".format(snp), "r")

                        unit_length_in_cgs = f["/Units"].attrs[
                            "Unit length in cgs (U_L)"
                        ]
                        unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
                        unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

                        boxsize_kpc = (
                            f["/Header"].attrs["BoxSize"]
                            * unit_length_in_cgs
                            / constants["PARSEC_IN_CGS"]
                            / 1e3
                        )
                        centre_kpc = boxsize_kpc / 2.0

                        if snp == snapshot_min:
                            time = f["/Header"].attrs["Time"]
                            time_min = (
                                time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6
                            )
                        elif snp == snapshot_max - 1:
                            time = f["/Header"].attrs["Time"]
                            time_max = (
                                time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6
                            )

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

                        gas_HI_mass_Msun = mass_Msun * XH * HI_frac
                        gas_H2_mass_Msun = mass_Msun * XH * H2_frac * 2.0

                        sigma2 = (
                            f["/PartType0/VelocityDispersions"][:]
                            * (unit_length_in_cgs / 1e5 / unit_time_in_cgs) ** 2
                        )  # [ (km/sec)**2]

                        pos_kpc[:, 0] -= centre_kpc[0]
                        pos_kpc[:, 1] -= centre_kpc[1]
                        pos_kpc[:, 2] -= centre_kpc[2]

                        MAX_HEIGHT_kpc = 0.25  # kpc

                        mask = np.where(
                            np.logical_and(
                                sigma2 < 1e20, np.abs(pos_kpc[:, 2]) < MAX_HEIGHT_kpc
                            )
                        )

                        r_kpc = np.sqrt(pos_kpc[:, 0] ** 2 + pos_kpc[:, 1] ** 2)

                        M_sigma2_HI, _, _ = stats.binned_statistic(
                            r_kpc[mask],
                            gas_HI_mass_Msun[mask] * sigma2[mask],
                            statistic="sum",
                            bins=edges_kpc,
                        )

                        M_sigma2_H2, _, _ = stats.binned_statistic(
                            r_kpc[mask],
                            gas_H2_mass_Msun[mask] * sigma2[mask],
                            statistic="sum",
                            bins=edges_kpc,
                        )

                        M_HI, _, _ = stats.binned_statistic(
                            r_kpc[mask],
                            gas_HI_mass_Msun[mask],
                            statistic="sum",
                            bins=edges_kpc,
                        )
                        M_H2, _, _ = stats.binned_statistic(
                            r_kpc[mask],
                            gas_H2_mass_Msun[mask],
                            statistic="sum",
                            bins=edges_kpc,
                        )

                        sigma2_HI = M_sigma2_HI / M_HI
                        sigma2_H2 = M_sigma2_H2 / M_H2

                        sigma_1D_HI = np.sqrt(sigma2_HI / 3.0)
                        sigma_1D_H2 = np.sqrt(sigma2_H2 / 3.0)

                        values_HI += sigma_1D_HI
                        values_H2 += sigma_1D_H2

                    # Finish calculating of the average across snapshots
                    values_HI /= snapshot_max - snapshot_min
                    values_H2 /= snapshot_max - snapshot_min

                    if split == 1:
                        colors = color2
                        lw = lw2
                    else:
                        colors = line_properties["colour"]
                        lw = line_properties["linewidth"]

                    ax.plot(
                        centers_kpc,
                        values_HI,
                        label=key.replace("_", "\_"),
                        color=colors[counter],
                        lw=lw[counter],
                        dashes=(17, 4),
                    )

                    ax.plot(
                        centers_kpc,
                        values_H2,
                        color=colors[counter],
                        lw=lw[counter],
                        dashes=(5, 5),
                    )

                ax.set_xlim(x_min, x_max)
                ax.set_yticks([0, 10, 20, 30, 40])
                ax.set_ylim(y_min, y_max)

                ax.xaxis.set_tick_params(labelsize=33)
                ax.yaxis.set_tick_params(labelsize=33)

                ax.set_xlabel("Radial distance $r$ [kpc]", fontsize=33)
                ax.set_ylabel(
                    "$\\sigma_{\\rm turb} \\, [\\rm km \\, s^{-1}]$", fontsize=33,
                )

                (l0,) = ax.plot([100, 120], [100, 120], lw=4, color="k", dashes=(17, 4))
                (l1,) = ax.plot([100, 120], [100, 120], lw=4, color="k", dashes=(5, 5))

                leg2 = plt.legend(
                    [l0, l1],
                    [r"H\textsc{i}", "H$_2$",],
                    loc="upper right",
                    frameon=False,
                    fontsize=30.0,
                )

                if split != 1:
                    print("Setting legend")
                    leg1 = ax.legend(
                        fontsize=28.0,
                        ncol=2,
                        bbox_to_anchor=(-0.20, 1.33),
                        loc="upper left",
                        handlelength=0,
                        handletextpad=0,
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
                        text.set_color(line_properties["colour"][counter])

                    plt.gca().add_artist(leg2)

                ax.text(
                    0.05,
                    0.96,
                    "${:.1f} < t < {:.1f}$ Gyr".format(
                        time_min[0] / 1e3, time_max[0] / 1e3
                    ),
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=27,
                )

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)
                plt.close()


if __name__ == "__main__":
    plot()
