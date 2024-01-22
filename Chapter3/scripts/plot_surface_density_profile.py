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
                TIME_WINDOW_Myr = plot["timewindow"]

                fig, ax = plot_style(8, 8)

                for counter, (key, value) in enumerate(dict_sim.items()):

                    f = h5.File(value + "/output_{:04d}.hdf5".format(snapshot), "r")

                    unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
                    unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
                    unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

                    boxsize = (
                        f["/Header"].attrs["BoxSize"]
                        * unit_length_in_cgs
                        / constants["PARSEC_IN_CGS"]
                        / 1e3
                    )
                    centre = boxsize / 2.0

                    time = f["/Header"].attrs["Time"]
                    time_snp = time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6

                    bins = 16
                    edges = np.linspace(x_min, x_max, bins)  # kpc
                    centers = 0.5 * (edges[1:] + edges[:-1])
                    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

                    ages = np.where(
                        np.abs(
                            f["/PartType4/BirthTimes"][:]
                            * unit_time_in_cgs
                            / constants["YEAR_IN_CGS"]
                            / 1e6
                            - time_snp[0]
                        )
                        <= TIME_WINDOW_Myr
                    )

                    for part_idx in [0, 4]:
                        print("part idx", part_idx)

                        pos = (
                            f[f"/PartType{part_idx}/Coordinates"][:, :]
                            * unit_length_in_cgs
                            / constants["PARSEC_IN_CGS"]
                            / 1e3
                        )
                        mass = (
                            f[f"/PartType{part_idx}/Masses"][:]
                            * unit_mass_in_cgs
                            / constants["SOLAR_MASS_IN_CGS"]
                        )

                        pos[:, 0] -= centre[0]
                        pos[:, 1] -= centre[0]
                        r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)

                        if part_idx == 0:

                            values, _, _ = stats.binned_statistic(
                                r, mass, statistic="sum", bins=edges
                            )
                            Sigma_Msun_pc2 = values / areas / (1e3) ** 2.0

                            ax.plot(
                                centers,
                                Sigma_Msun_pc2,
                                lw=line_properties["linewidth"][counter],
                                color=line_properties["colour"][counter],
                                label=key.replace("_", "\_"),
                            )
                        else:

                            print(np.shape(pos[ages]), np.shape(pos))
                            values, _, _ = stats.binned_statistic(
                                r[ages], mass[ages], statistic="sum", bins=edges
                            )
                            Sigma_Msun_pc2 = values / areas / (1e3) ** 2.0

                            ax.plot(
                                centers,
                                Sigma_Msun_pc2,
                                lw=line_properties["linewidth"][counter],
                                color=line_properties["colour"][counter],
                                dashes=(4, 4),
                            )

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_yscale("log")

                ax.xaxis.set_tick_params(labelsize=27)
                ax.yaxis.set_tick_params(labelsize=27)

                ax.set_xlabel("Radial distance $r$ [kpc]", fontsize=27)
                ax.set_ylabel(
                    "$\\Sigma \\, [\\rm M_\\odot \\, pc^{-2}]$",
                    labelpad=0,
                    fontsize=27,
                )

                leg1 = plt.legend(loc="upper right", fontsize=19, frameon=False)

                (line1,) = ax.plot([1e5, 1e6], [1e5, 1e6], lw=2.5, color="k")
                (line2,) = ax.plot(
                    [1e5, 1e6], [1e5, 1e6], lw=2.5, color="k", dashes=(4, 4)
                )

                leg2 = plt.legend(
                    [line1, line2],
                    [
                        "Gas ($t={:.1f}$ Gyr)".format(time_snp[0] / 1e3),
                        "Stars (${:.1f}".format(
                            time_snp[0] / 1e3 - TIME_WINDOW_Myr / 1e3
                        )
                        + "< t_{\\rm Birth}"
                        + "< {:.1f}$ Gyr)".format(time_snp[0] / 1e3),
                    ],
                    loc="lower left",
                    fontsize=15.5,
                    frameon=False,
                )

                plt.gca().add_artist(leg1)

                fixlogax(ax, "y")

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)
                plt.close()


if __name__ == "__main__":
    plot()
