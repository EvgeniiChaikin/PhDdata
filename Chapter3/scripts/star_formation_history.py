import numpy as np
import sys
import matplotlib.patheffects as pe
import json
from scipy import stats
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
                split = plot["split"]

                print(output)

                fig, ax = plot_style(8, 8)

                for counter, (key, value) in enumerate(dict_sim.items()):

                    SFR_file = np.loadtxt(f"{value}/SFR.txt", skiprows=25)
                    time = SFR_file[:, 1] * 9.778131e02 / 1e3  # Gyr
                    total_SFR = SFR_file[:, -1] * 1.022690e01  # M_sol / yr

                    N_bins = 100
                    bin_edges = np.linspace(0.0, 1.0, N_bins)
                    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

                    bin_values, _, _ = stats.binned_statistic(
                        time, total_SFR, statistic="median", bins=bin_edges
                    )

                    if split > 0:
                        if split == 1:
                            colors = color2
                            dashes = dashes2
                        # All 9 reuns
                        elif split == 9:
                            colors = color4
                            dashes = dashes4
                        else:
                            colors = color3
                            dashes = dashes3
 
                        print(len(colors), counter)
                        ax.plot(
                            bin_centers,
                            bin_values,
                            lw=3,
                            color=colors[counter],
                            zorder=4,
                            dashes = (tuple(d for d in dashes[counter])),
                            label=key.replace("_", "\_"),
                            )

                    else:
                        if len(list(dict_sim.items())) == 5:
                            if counter < 3:
                                ax.plot(
                                    bin_centers,
                                    bin_values,
                                    label=key.replace("_", "\_"),
                                    lw=line_properties["linewidth"][counter],
                                    color=line_properties["colour"][counter],
                                    alpha=line_properties["alpha"][counter],
                                    ls=line_properties["ls"][counter],
                                    zorder=4,
                                )
                            elif counter == 3:
                                bin_values_run4 = bin_values
                            else:

                                if "M5" in key:
                                    label="IG\_M5\_\{min,max\}\_density"
                                elif "M3" in key:
                                    label="IGD\_M3\_\{min,max\}\_density"
                                else:
                                    label="IG\_M6\_\{min,max\}\_density"
                                print(label)

                                ax.fill_between(
                                    bin_centers,
                                    bin_values_run4,
                                    bin_values,
                                    edgecolor="grey",
                                    alpha=0.2,
                                    hatch="XXXX",
                                    zorder=-2,
                                    label=label,
                                )
                                ax.plot(bin_centers, bin_values_run4, dashes = (tuple(d for d in dashesMMD[0])), color = colorMMD, lw=lwMMD, zorder=-1, alpha = alphaMMD)
                                ax.plot(bin_centers, bin_values, dashes = (tuple(d for d in dashesMMD[1])), color = colorMMD, lw=lwMMD, zorder=-1, alpha = alphaMMD)

                        else:
                            ax.plot(
                                bin_centers,
                                bin_values,
                                label=key.replace("_", "\_"),
                                lw=line_properties["linewidth"][counter],
                                color=line_properties["colour"][counter],
                                alpha=line_properties["alpha"][counter],
                                ls=line_properties["ls"][counter],
                                zorder=4,
                            )

                if 0 < split < 9:
                    leg1 = ax.legend(fontsize=24, loc="lower left", frameon=False)
                elif split == 9:
                    # too many runs -> small legend
                    leg1 = ax.legend(fontsize=17, loc="lower left", frameon=False)
                elif split < 0:
                    leg1 = ax.legend(fontsize=27, loc="upper center", frameon=False)
                else:
                    leg1 = ax.legend(fontsize=27, loc="lower center", frameon=False)

                # Axes & labels
                ax.xaxis.set_tick_params(labelsize=33)
                ax.yaxis.set_tick_params(labelsize=33)
                ax.set_xlim(-0.02, 1.02)
             

                # Y log scale
                ax.set_yscale("log")
                plt.yticks([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
                fixlogax(ax, "y")
                if split >= 0:
                    ax.set_ylim(0.08, 25)
                else:
                    ax.set_ylim(0.08 / 1e2, 25 / 1e2)

                ax.set_xlabel("Time [Gyr]", fontsize=33)
                ax.set_ylabel("SFR [M$_{\\odot}$ yr$^{-1}$]", fontsize=33)
                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)


plot_SFR()
