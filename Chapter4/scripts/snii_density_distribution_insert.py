import numpy as np
import sys
import matplotlib.patheffects as pe
import json
import h5py as h5
from scipy import stats
from plot_style import *
from constants import *
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
                idx = plot["snapshot"]
                dt = plot["timewindow"]
                runs_per_plot = len(dict_sim) // 2
                print("Runs per plot:", runs_per_plot)
                split = plot["split"]

                print(output)

                bin_edges = np.logspace(-3.5, 5.5, 101)
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

                fig, (ax, ax2) = plt.subplots(
                    2, 1, figsize=(8.0, 8.0), sharex=True, sharey=False
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

                for counter, (key, value) in enumerate(dict_sim.items()):
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

                    snii_n_gas = (
                        f["/PartType0/DensitiesAtLastSupernovaEvent"][:]
                        * gas_XH
                        / unit_length_in_cgs**3
                        * unit_mass_in_cgs
                        / constants["PROTON_MASS_IN_CGS"]
                    )[gas_SNII_times]
                    snii_n_stars = (
                        f["/PartType4/DensitiesAtLastSupernovaEvent"][:]
                        * stars_XH
                        / unit_length_in_cgs**3
                        * unit_mass_in_cgs
                        / constants["PROTON_MASS_IN_CGS"]
                    )[stars_SNII_times]

                    snii_n = np.concatenate([snii_n_gas, snii_n_stars])

                    N_snii_binned, _, _ = stats.binned_statistic(
                        snii_n,
                        np.ones_like(snii_n) / np.size(snii_n),
                        bins=bin_edges,
                        statistic="count",
                    )

                    if counter < runs_per_plot:
                        c = counter
                    else:
                        c = counter - runs_per_plot

                    if split == 0:
                        color = line_properties["colour"][c]
                        dashes = (3, 0)
                        lw = line_properties["linewidth"][c]
                    else:
                        color = color4[c]
                        dashes = tuple(d for d in dashes4[c])
                        lw = 2.3

                    if counter < runs_per_plot:
                        ax.plot(
                            bin_centers,
                            np.cumsum(N_snii_binned / np.sum(N_snii_binned)),
                            lw=lw,
                            dashes=dashes,
                            color=color,
                            label=key.replace("_", "\_"),
                        )
                    else:
                        ax2.plot(
                            bin_centers,
                            np.cumsum(N_snii_binned / np.sum(N_snii_binned)),
                            lw=lw,
                            color=color,
                            label=key.replace("_", "\_"),
                            dashes=dashes,
                        )

                ax.legend(fontsize=14, frameon=False, loc="lower right")
                ax2.legend(fontsize=14, frameon=False, loc="lower right")

                for a in [ax, ax2]:
                    a.set_ylabel("Fr. of particles", fontsize=32)

                    a.set_yscale("log")
                    a.set_xscale("log")
                    a.xaxis.set_tick_params(labelsize=28)
                    a.yaxis.set_tick_params(labelsize=28)

                    a.xaxis.set_ticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
                    a.yaxis.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4])

                    locmin = ticker.LogLocator(
                        base=10.0,
                        subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                        numticks=20,
                    )

                    a.xaxis.set_minor_locator(locmin)
                    a.xaxis.set_minor_formatter(ticker.NullFormatter())

                    a.set_xlim(10 ** (-3.2), 10 ** (4.2))
                    a.set_ylim(1e-3, 2)

                    fixlogax(a, "y")

                fixlogax(ax2, "x")
                ax2.set_xlabel("$n_{\\rm H, snii}$ [cm$^{-3}$]", fontsize=32)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    plot()
