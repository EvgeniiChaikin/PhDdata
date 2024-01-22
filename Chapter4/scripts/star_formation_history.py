import numpy as np
import sys
import matplotlib.patheffects as pe
from matplotlib import ticker
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
                y_min, y_max = plot["ylims"]

                try:
                    fig_size_x, fig_size_y = plot["fig_size"]
                except KeyError:
                    fig_size_x, fig_size_y = 8, 8

                print(output)

                fig, ax = plot_style(fig_size_x, fig_size_y)

                for counter, (key, value) in enumerate(dict_sim.items()):

                    SFR_file = np.loadtxt(f"{value}/SFR.txt", skiprows=25)
                    time = SFR_file[:, 1] * 9.778131e02 / 1e3  # Gyr
                    total_SFR = SFR_file[:, -1] * 1.022690e01  # M_sol / yr

                    N_bins = 500 # 51
                    dt = 0.025 # window of 2x25 Myr = 50 Myr

                    bin_centers = np.linspace(0.0 + dt, 1.0 - dt, N_bins)
                    bin_values = np.zeros_like(bin_centers)

                    for i in range(len(bin_centers)):
                        mask = np.where(np.abs(time-bin_centers[i]) < dt)[0]
                        bin_values[i] = np.mean(total_SFR[mask])

                    if split == 0 or split == 9:
                        colors = line_properties["colour"]
                        lw = line_properties["linewidth"]
                        dashes = line_properties["dashes"]
                    if split == 1:
                        colors = color2
                        lw = lw2
                        dashes = dashes2
                    elif split == 3:
                        colors = color3
                        lw = lw3
                        dashes = dashes3
                    elif split == 4:
                        colors = color4
                        lw = lw4
                        dashes = dashes4
                    elif split == 5:
                        colors = color5
                        lw = lw5
                        dashes = dashes5
                    elif split == 6:
                        colors = color6
                        lw = lw6
                        dashes = dashes6
                    elif split == 7:
                        colors = color7
                        lw = lw7
                        dashes = dashes7

                    if key.find("nolegend")>=0:
                        label = "_nolegend_"
                    else:
                        label = key.replace("_", "\_")
                        label = label.replace("{", "\{")
                        label = label.replace("}", "\}")

                    ax.plot(
                        bin_centers,
                        bin_values,
                        lw=lw[counter],
                        color=colors[counter],
                        dashes=tuple(d for d in dashes[counter]),
                        zorder=4,
                        label=label,
                    )

                if split == 1:
                    leg1 = ax.legend(
                        fontsize=27.5,
                        ncol=2,
                        bbox_to_anchor=(-0.20, 1.6),
                        loc="upper left",
                        handlelength=0,
                        handletextpad=0,
                        columnspacing=0.25,
                        borderaxespad=0.20,
                        labelspacing=0.3,
                    )

                elif split == 3:
                    leg1 = ax.legend(
                        fontsize=22,
                        loc="lower center",
                        frameon=False,
                        columnspacing=0.25,
                        borderaxespad=0.20,
                        labelspacing=0.3,
                    )
                elif split != 6:
                    leg1 = ax.legend(
                        fontsize=28,
                        loc="lower center",
                        frameon=False,
                        columnspacing=0.25,
                        borderaxespad=0.20,
                        labelspacing=0.3,
                    )
                else:
                    leg1 = ax.legend(
                        fontsize=23.4,
                        ncol=1,
                        bbox_to_anchor=(-0.18, 1.7),
                        loc="upper left",
                        columnspacing=0.25,
                        borderaxespad=0.20,
                        labelspacing=0.3,
                    )

                    #leg1 = ax.legend(fontsize=19, loc="upper center", frameon=False,
                    #columnspacing=0.25,
                    #borderaxespad=0.20,
                    #labelspacing=0.3
                    #)

                if split != 6: 
                    hl_dict = {
                        handle.get_label(): handle for handle in leg1.legendHandles
                    }
                    for k in hl_dict:
                        hl_dict[k].set_color("white")

                    for counter, text in enumerate(leg1.get_texts()):
                        text.set_color(colors[counter])

                l0, = ax.plot([1e4,1e5],[1e4,1e5], lw=3, color="k")
                l1, = ax.plot([1e4,1e5],[1e4,1e5], lw=3, color="k", dashes=(4, 4))
                
                if split == 3:
                    leg2 = plt.legend(
                    [l0, l1],
                    [
                        "Gravitational instability criterion",
                        "Temperature-density criterion"
                    ],
                    loc="upper right",
                    frameon=False,
                    fontsize=LEGEND_SIZE * 0.9,
                    )

                    plt.gca().add_artist(leg1)

                # Axes & labels
                ax.xaxis.set_tick_params(labelsize=31)
                ax.yaxis.set_tick_params(labelsize=31)
                ax.set_xlim(-0.02, 1.02)

                # Y log scale
                ax.set_yscale("log")
                plt.yticks([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
                fixlogax(ax, "y")
                ax.set_ylim(y_min, y_max)

                if split != 6 and split != 1:
                    ax.set_xlabel("Time [Gyr]", fontsize=31)
                ax.set_ylabel("SFR [M$_{\\odot}$ yr$^{-1}$]", fontsize=31)
                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    plot_SFR()
