import numpy as np
import sys
import matplotlib.patheffects as pe
import json
from scipy import stats
from matplotlib import ticker
from plot_style import *


def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def plot_SFR():
    runs = read_sim_data("main.json")

    for script_name, plots in runs.items():
        print(script_name)

        if script_name == sys.argv[0]:
            print("FOUND")

            for plot in plots:
                output = plot["output_file"]
                dict_sim = plot["data"]
                runs_per_plot = len(dict_sim) // 2
                print("Runs per plot:", runs_per_plot)
                y_min, y_max = plot["ylims_main"]
                y_min_sub, y_max_sub = plot["ylims_insert"]
                split = plot["split"]

                print(output)

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
                    x_minor_locator = AutoMinorLocator(5)
                    y_minor_locator = AutoMinorLocator(5)
                    a.xaxis.set_minor_locator(x_minor_locator)
                    a.yaxis.set_minor_locator(y_minor_locator)

                for counter, (key, value) in enumerate(dict_sim.items()):
                    SFR_file = np.loadtxt(f"{value}/SFR.txt", skiprows=25)
                    time = SFR_file[:, 1] * 9.778131e02 / 1e3  # Gyr
                    total_SFR = SFR_file[:, -1] * 1.022690e01  # M_sol / yr

                    N_bins = 500  # 51
                    dt = 0.025  # window of 2x25 Myr = 50 Myr

                    bin_centers = np.linspace(0.0 + dt, 1.0 - dt, N_bins)
                    bin_values = np.zeros_like(bin_centers)

                    for i in range(len(bin_centers)):
                        mask = np.where(np.abs(time - bin_centers[i]) < dt)[0]
                        bin_values[i] = np.mean(total_SFR[mask])

                    if counter < runs_per_plot:
                        c = counter
                    else:
                        c = counter - runs_per_plot

                    if split == 0:
                        color = line_properties["colour"]
                        dashes = (3, 0)
                        lw = line_properties["linewidth"][c]
                    else:
                        color = color4
                        dashes = tuple(d for d in dashes4[c])
                        lw = 2.3

                    if counter < runs_per_plot:
                        ax.plot(
                            bin_centers,
                            bin_values,
                            label=key.replace("_", "\_"),
                            lw=lw,
                            color=color[c],
                        )
                    else:
                        ax2.plot(
                            bin_centers,
                            bin_values,
                            label=key.replace("_", "\_"),
                            color=color[c],
                            lw=lw,
                        )

                leg1 = ax.legend(
                    columnspacing=0.25,
                    borderaxespad=0.20,
                    labelspacing=0.3,
                    loc="lower center",
                    frameon=False,
                    ncol=2,
                    handlelength=0,
                    handletextpad=0,
                    prop={"size": 22},
                )
                hl_dict = {handle.get_label(): handle for handle in leg1.legendHandles}
                for k in hl_dict:
                    hl_dict[k].set_color("white")
                for counter, text in enumerate(leg1.get_texts()):
                    text.set_color(color[counter])

                leg2 = ax2.legend(
                    fontsize=21,
                    columnspacing=0.25,
                    borderaxespad=0.20,
                    labelspacing=0.3,
                    loc="upper center",
                    frameon=False,
                    ncol=2,
                    handlelength=0,
                    handletextpad=0,
                    prop={"size": 22},
                )
                hl_dict = {handle.get_label(): handle for handle in leg2.legendHandles}

                for k in hl_dict:
                    hl_dict[k].set_color("white")
                for counter, text in enumerate(leg2.get_texts()):
                    text.set_color(color[counter])

                for a in [ax, ax2]:
                    a.set_yscale("log")
                    a.set_yticks([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])

                    locmin = ticker.LogLocator(
                        base=10.0,
                        subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                        numticks=10,
                    )

                    a.yaxis.set_minor_locator(locmin)
                    a.yaxis.set_minor_formatter(ticker.NullFormatter())

                    a.xaxis.set_tick_params(labelsize=30)
                    a.yaxis.set_tick_params(labelsize=30)

                    a.set_xlim(-0.04, 1.04)
                    a.set_ylabel("SFR [M$_{\\odot}$ yr$^{-1}$]", fontsize=32)
                    fixlogax(a, "y")

                ax.set_ylim(y_min, y_max)
                ax2.set_ylim(y_min_sub, y_max_sub)

                ax2.set_xlabel("Time [Gyr]", fontsize=32)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    plot_SFR()
