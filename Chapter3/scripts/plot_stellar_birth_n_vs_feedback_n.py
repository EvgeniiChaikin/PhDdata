import numpy as np
import h5py as h5
import sys
from scipy import stats
from plot_style import *
from constants import *
import json
from scipy import stats
from matplotlib import ticker


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

                print(script_name)

                output = plot["output_file"]
                dict_sim = plot["data"]
                idx = plot["snapshot"]
                split = plot["split"]

                fig, ax = plot_style(8, 8)

                bin_edges = np.logspace(-2.5, 5.5, 61)
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

                for counter, (key, value) in enumerate(dict_sim.items()):

                    print(key)

                    f = h5.File(value + "/output_{:04d}.hdf5".format(idx), "r")

                    unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
                    unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
                    unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

                    # Hydrogen fractions
                    stars_XH = f["/PartType4/ElementMassFractions"][:, 0]
                    gas_XH = f["/PartType0/ElementMassFractions"][:, 0]

                    # Birth density
                    birth_n = (
                        f["/PartType4/BirthDensities"][:]
                        * stars_XH
                        / unit_length_in_cgs ** 3
                        * unit_mass_in_cgs
                        / constants["PROTON_MASS_IN_CGS"]
                    )

                    # SNII density
                    snii_n_stars = (
                        f["/PartType4/LastSNIIFeedbackDensities"][:]
                        * stars_XH
                        / unit_length_in_cgs ** 3
                        * unit_mass_in_cgs
                        / constants["PROTON_MASS_IN_CGS"]
                    )

                    mask = np.where(snii_n_stars > 0.0)
                    N_snii_binned, _, _ = stats.binned_statistic(
                        birth_n[mask],
                        snii_n_stars[mask],
                        bins=bin_edges,
                        statistic="median",
                    )

                    if split:
                        if split == 1:
                            labels = ["M4", "M5", "M6"]
                        else:
                            labels = ["Without EOS", "With EOS"]

                        c_size = len(dict_sim.items()) // 2
                        colors = color2[: len(labels)]

                        if counter < len(labels):
                            print(labels[counter])
                            ax.plot(
                                bin_centers,
                                N_snii_binned,
                                lw=3.0,
                                color=colors[counter],
                                label=key.replace("_", "\_"),
                                zorder=4,
                            )
                        else:
                            ax.plot(
                                bin_centers,
                                N_snii_binned,
                                lw=4.0,
                                dashes=(10, 5),
                                color=colors[counter - len(labels)],
                                label=key.replace("_", "\_"),
                                zorder=4,
                            )

                    else:
                        if len(dict_sim.items()) == 5:
                            if counter < 3:
                                ax.plot(
                                    bin_centers,
                                    N_snii_binned,
                                    lw=line_properties["linewidth"][counter],
                                    color=line_properties["colour"][counter],
                                    alpha=line_properties["alpha"][counter],
                                    label=key.replace("_", "\_"),
                                )
                            elif counter == 3:
                                N_snii_binned_3 = N_snii_binned
                            else:
                                ax.fill_between(
                                    bin_centers,
                                    N_snii_binned_3,
                                    N_snii_binned,
                                    edgecolor="grey",
                                    alpha=0.2,
                                    hatch="XXXX",
                                    zorder=-2,
                                    label="IG\_M5\_\{min,max\}\_density",
                                )
                                ax.plot(bin_centers, N_snii_binned_3, dashes = (tuple(d for d in dashesMMD[0])), color = colorMMD, lw=lwMMD, zorder=-1, alpha = alphaMMD)
                                ax.plot(bin_centers, N_snii_binned, dashes = (tuple(d for d in dashesMMD[1])), color = colorMMD, lw=lwMMD, zorder=-1, alpha = alphaMMD)

                        else:
                            ax.plot(
                                bin_centers,
                                N_snii_binned,
                                lw=line_properties["linewidth"][counter],
                                color=line_properties["colour"][counter],
                                alpha=line_properties["alpha"][counter],
                                label=key.replace("_", "\_"),
                            )

                    time = f["/Header"].attrs["Time"]
                    # In Myr
                    time_snp = time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6

                ax.plot(
                    [1e-7, 1e7],
                    [1e-7, 1e7],
                    color="darkred",
                    alpha=0.8,
                    lw=2.5,
                    dashes=(16, 4),
                    zorder=-1,
                )

                leg1 = ax.legend(loc="upper left", fontsize=20.5, frameon=False)

                ax.xaxis.set_tick_params(labelsize=33)
                ax.yaxis.set_tick_params(labelsize=33)

                ax.set_yscale("log")
                ax.set_xscale("log")

                ax.xaxis.set_ticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
                ax.yaxis.set_ticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])

                locmin = ticker.LogLocator(
                    base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=10
                )

                ax.xaxis.set_minor_locator(locmin)
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.yaxis.set_minor_locator(locmin)
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())

                ax.set_xlim(10 ** (-2.2), 10 ** (4.2))
                ax.set_ylim(10 ** (-2.2), 10 ** (4.2))

                fixlogax(ax, "x")
                fixlogax(ax, "y")

                ax.set_xlabel(
                    "Stellar birth density $n_{\\rm H,birth}$ [cm$^{-3}$]", fontsize=30
                )
                ax.set_ylabel(
                    "SN feedback density $n_{\\rm H, SN}$ [cm$^{-3}$]", fontsize=30
                )

                ax.text(
                    0.97,
                    0.04,
                    "$t<\\rm {:.0f}$ Gyr".format(time_snp[0] / 1e3),
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=27,
                )

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)
                plt.close()


plot()
