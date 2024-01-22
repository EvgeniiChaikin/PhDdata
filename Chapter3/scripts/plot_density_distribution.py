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

                output = plot["output_file"]
                dict_sim = plot["data"]
                split = plot["split"]
                idx = plot["snapshot"]
                dt = plot["timewindow"]
                print(output)

                plt.rcParams["ytick.direction"] = "in"
                plt.rcParams["xtick.direction"] = "in"
                plt.rcParams["axes.linewidth"] = 2
                plt.rc("text", usetex=True)
                plt.rc("font", family="serif")

                fig = plt.subplots(figsize=(8, 8))

                ax1 = plt.subplot2grid((2, 1), (0, 0))
                ax2 = plt.subplot2grid((2, 1), (1, 0))

                ax = [ax1, ax2]

                bin_edges = np.logspace(-3.5, 5.5, 101)
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

                    time = f["/Header"].attrs["Time"]
                    # In Myr
                    time_snp = time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6

                    gas_SNII_times = np.where(
                        np.abs(
                            f["/PartType0/LastSNIIThermalFeedbackTimes"][:]
                            * unit_time_in_cgs
                            / constants["YEAR_IN_CGS"]
                            / 1e6
                            - time_snp[0]
                        )
                        <= dt
                    )
                    stars_SNII_times = np.where(
                        np.abs(
                            f["/PartType4/LastSNIIThermalFeedbackTimes"][:]
                            * unit_time_in_cgs
                            / constants["YEAR_IN_CGS"]
                            / 1e6
                            - time_snp[0]
                        )
                        <= dt
                    )
                    birth_times = np.where(
                        np.abs(
                            f["/PartType4/BirthTimes"][:]
                            * unit_time_in_cgs
                            / constants["YEAR_IN_CGS"]
                            / 1e6
                            - time_snp[0]
                        )
                        <= dt
                    )

                    # Birth density
                    birth_n = (
                        f["/PartType4/BirthDensities"][:]
                        * stars_XH
                        / unit_length_in_cgs ** 3
                        * unit_mass_in_cgs
                        / constants["PROTON_MASS_IN_CGS"]
                    )[birth_times]
                    # SNII density
                    snii_n_gas = (
                        f["/PartType0/DensitiesAtLastSupernovaEvent"][:]
                        * gas_XH
                        / unit_length_in_cgs ** 3
                        * unit_mass_in_cgs
                        / constants["PROTON_MASS_IN_CGS"]
                    )[gas_SNII_times]
                    snii_n_stars = (
                        f["/PartType4/DensitiesAtLastSupernovaEvent"][:]
                        * stars_XH
                        / unit_length_in_cgs ** 3
                        * unit_mass_in_cgs
                        / constants["PROTON_MASS_IN_CGS"]
                    )[stars_SNII_times]

                    snii_n = np.concatenate([snii_n_gas, snii_n_stars])

                    N_birth_binned, _, _ = stats.binned_statistic(
                        birth_n,
                        np.ones_like(birth_n) / np.size(birth_n),
                        bins=bin_edges,
                        statistic="count",
                    )
                    N_snii_binned, _, _ = stats.binned_statistic(
                        snii_n,
                        np.ones_like(snii_n) / np.size(snii_n),
                        bins=bin_edges,
                        statistic="count",
                    )

                    if len(list(dict_sim.items())) == 5:

                        if counter < 3:

                            ax[0].plot(
                                bin_centers,
                                np.cumsum(N_birth_binned / np.sum(N_birth_binned)),
                                lw=line_properties["linewidth2"][counter],
                                color=line_properties["colour"][counter],
                                alpha=line_properties["alpha"][counter],
                                ls=line_properties["ls"][counter],
                            )

                            ax[1].plot(
                                bin_centers,
                                np.cumsum(N_snii_binned / np.sum(N_snii_binned)),
                                lw=line_properties["linewidth2"][counter],
                                color=line_properties["colour"][counter],
                                alpha=line_properties["alpha"][counter],
                                label=key.replace("_", "\_"),
                            )
                        elif counter == 3:
                            y_star_c3 = np.cumsum(
                                N_birth_binned / np.sum(N_birth_binned)
                            )
                            y_snii_c3 = np.cumsum(N_snii_binned / np.sum(N_snii_binned))
                        else:
                            ax[0].fill_between(
                                bin_centers,
                                np.cumsum(N_birth_binned / np.sum(N_birth_binned)),
                                y_star_c3,
                                edgecolor="grey",
                                alpha=0.2,
                                hatch="XXXX",
                                zorder=-2,
                            )
                            ax[0].plot(bin_centers, y_star_c3, dashes = (tuple(d for d in dashesMMD[0])), color = colorMMD, lw=lwMMD, zorder=-1, alpha = alphaMMD)
                            ax[0].plot(bin_centers, np.cumsum(N_birth_binned / np.sum(N_birth_binned)), 
                                                   dashes = (tuple(d for d in dashesMMD[1])), color = colorMMD, lw=lwMMD, zorder=-1, alpha = alphaMMD)

                            ax[1].fill_between(
                                bin_centers,
                                np.cumsum(N_snii_binned / np.sum(N_snii_binned)),
                                y_snii_c3,
                                edgecolor="grey",
                                hatch="XXXX",
                                alpha=0.20,
                                zorder=-2,
                                label="IG\_M5\_\{min,max\}\_density",
                            )
                            ax[1].plot(bin_centers, y_snii_c3, dashes = (tuple(d for d in dashesMMD[0])), color = colorMMD, lw=lwMMD, zorder=-1, alpha = alphaMMD)
                            ax[1].plot(bin_centers, np.cumsum(N_snii_binned / np.sum(N_snii_binned)), 
                                                   dashes = (tuple(d for d in dashesMMD[1])), color = colorMMD, lw=lwMMD, zorder=-1, alpha = alphaMMD)

                    else:
                        if split:
                            if split == 1:
                                labels = ["M4", "M5", "M6"]
                            else:
                                labels = ["Without EOS", "With EOS"]

                            colors = color2[: len(labels)]
                            print(counter, len(labels))
                            c_size = len(dict_sim.items()) // 2

                            if counter < len(labels):
                                print(labels[counter])

                                ax[0].plot(
                                    bin_centers,
                                    np.cumsum(N_birth_binned / np.sum(N_birth_binned)),
                                    lw=2.5,
                                    color=colors[counter],
                                    alpha=line_properties["alpha"][counter],
                                    ls=line_properties["ls"][counter],
                                )

                                ax[1].plot(
                                    bin_centers,
                                    np.cumsum(N_snii_binned / np.sum(N_snii_binned)),
                                    lw=2.5,
                                    color=colors[counter],
                                    alpha=line_properties["alpha"][counter],
                                    label=key.replace("_", "\_"),
                                )

                            else:
                                ax[0].plot(
                                    bin_centers,
                                    np.cumsum(N_birth_binned / np.sum(N_birth_binned)),
                                    lw=3,
                                    color=colors[counter - len(labels)],
                                    alpha=line_properties["alpha"][counter],
                                    dashes=(7, 3),
                                    ls=line_properties["ls"][counter],
                                )

                                ax[1].plot(
                                    bin_centers,
                                    np.cumsum(N_snii_binned / np.sum(N_snii_binned)),
                                    lw=3,
                                    color=colors[counter - len(labels)],
                                    dashes=(7, 3),
                                    alpha=line_properties["alpha"][counter],
                                    label=key.replace("_", "\_"),
                                )

                        else:
                            ax[0].plot(
                                bin_centers,
                                np.cumsum(N_birth_binned / np.sum(N_birth_binned)),
                                lw=line_properties["linewidth"][counter],
                                color=line_properties["colour"][counter],
                                alpha=line_properties["alpha"][counter],
                                ls=line_properties["ls"][counter],
                            )

                            ax[1].plot(
                                bin_centers,
                                np.cumsum(N_snii_binned / np.sum(N_snii_binned)),
                                lw=line_properties["linewidth"][counter],
                                color=line_properties["colour"][counter],
                                alpha=line_properties["alpha"][counter],
                                label=key.replace("_", "\_"),
                            )

                for i, plot_name in enumerate(
                    ["Stellar birth \n gas densities", "SN feedback \n gas densities"]
                ):

                    ax[i].set_yscale("log")
                    ax[i].set_xscale("log")
                    ax[i].xaxis.set_tick_params(labelsize=33)
                    ax[i].yaxis.set_tick_params(labelsize=33)

                    ax[i].xaxis.set_ticks(
                        [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
                    )
                    ax[i].yaxis.set_ticks(
                        [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
                    )


                    locmin = ticker.LogLocator(
                        base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=20
                    )

                    ax[i].xaxis.set_minor_locator(locmin)
                    ax[i].xaxis.set_minor_formatter(ticker.NullFormatter())

                    locmin = ticker.LogLocator(
                        base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=20
                    )

                    ax[i].yaxis.set_minor_locator(locmin)
                    ax[i].yaxis.set_minor_formatter(ticker.NullFormatter())

                    ax[i].set_xlim(10 ** (-3.2), 10 ** (4.2))
                    ax[i].set_ylim(1e-3, 2)

                    fixlogax(ax[i], "x")
                    fixlogax(ax[i], "y")

                    if i == 1:
                        ax[i].set_xlabel("$n_{\\rm H}$ [cm$^{-3}$]", fontsize=33)
                    else:
                        ax[i].set_xticklabels([])

                    ax[i].set_ylabel("Fraction of particles", labelpad=0, fontsize=28)

                    plt.rcParams.update({"figure.autolayout": True})
                    plt.rcParams["ytick.direction"] = "in"
                    plt.rcParams["xtick.direction"] = "in"
                    plt.rcParams["axes.linewidth"] = 2

                    ax[i].tick_params(which="both", width=1.7)
                    ax[i].tick_params(which="major", length=9)
                    ax[i].tick_params(which="minor", length=5)
                    ax[i].tick_params(
                        axis="both",
                        which="both",
                        pad=8,
                        left=True,
                        right=True,
                        top=True,
                        bottom=True,
                    )
                    ax[i].text(
                        0.04,
                        0.94,
                        plot_name,
                        ha="left",
                        va="top",
                        transform=ax[i].transAxes,
                        fontsize=23,
                    )

                ax[1].legend(fontsize=20.0, markerfirst=False, frameon=False)
                ax[0].text(
                        0.97,
                        0.04,
                        "${:.1f} < t < {:.1f}$ Gyr".format(
                        (time_snp[0] - dt) / 1e3, time_snp[0] / 1e3
                        ),
                        ha="right",
                        va="bottom",
                        transform=ax[0].transAxes,
                        fontsize=25,
                    )

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)
                plt.close()


plot()
