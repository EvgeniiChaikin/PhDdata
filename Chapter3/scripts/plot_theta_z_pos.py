import numpy as np
import sys
from scipy import stats
from plot_style import *
import json
from swiftsimio import load


def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def plot():
    percentiles = [16, 84]
    bin_edges = np.linspace(-0.5, 0.5, 20)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    runs = read_sim_data("main.json")

    for script_name, plots in runs.items():
        if script_name == sys.argv[0]:
            print("FOUND")

            for plot in plots:
                output = plot["output_file"]
                dict_sim = plot["data"]
                split = plot["split"]
                idx = plot["snapshot"]
                TIME_WINDOW_MYR = plot["timewindow"]

                print(output)

                fig, ax = plot_style(8, 8)

                for counter, (key, value) in enumerate(dict_sim.items()):
                    print(counter, key)

                    # Load data
                    data = load(value + "/output_{:04d}.hdf5".format(idx))

                    # Snapshot time
                    snp_time_Myr = data.metadata.time.to("Myr")
                    birth_times = data.stars.birth_times.to("Myr")

                    mask_with_feedback = (
                        data.stars.last_sniifeedback_densities
                        > 0.0 * data.stars.last_sniifeedback_densities.units
                    )
                    mask_time = np.logical_and(
                        birth_times > 0.0 * birth_times.units,
                        np.abs(birth_times.value - snp_time_Myr.value)
                        < TIME_WINDOW_MYR,
                    )
                    mask = np.logical_and(mask_with_feedback, mask_time)

                    center = data.metadata.boxsize * 0.5
                    z_pos = data.stars.last_sniistar_position[mask][:, 2] - center[2]
                    theta = data.stars.last_sniithermal_feedback_star_theta[mask]

                    if len(z_pos) > 0:
                        print(len(z_pos))

                        cos_theta_part = np.cos(theta)

                        medians = []
                        deviations = []

                        for i in range(len(bin_centers)):
                            bin_idx = np.where(
                                np.logical_and(
                                    z_pos >= bin_edges[i], z_pos < bin_edges[i + 1]
                                )
                            )

                            y_values_in_this_bin = cos_theta_part[bin_idx]
                            medians.append(np.mean(y_values_in_this_bin))
                            # deviations.append(
                            #    np.percentile(y_values_in_this_bin, percentiles)
                            # )

                        medians = np.array(medians)
                        # deviations = np.array(np.abs(np.array(deviations).T - medians))
                        # down, up = deviations

                        # Add medians

                        if len(dict_sim.items()) == 5:
                            if counter < 3:
                                ax.plot(
                                    bin_centers,
                                    medians,
                                    label=key.replace("_", "\_"),
                                    ls=line_properties["ls"][counter],
                                    color=line_properties["colour"][counter],
                                    lw=line_properties["linewidth"][counter],
                                    zorder=5 - counter,
                                )
                            elif counter == 3:
                                medians_3 = medians
                            else:
                                ax.fill_between(
                                    bin_centers,
                                    medians_3,
                                    medians,
                                    edgecolor="grey",
                                    alpha=0.2,
                                    hatch="XXXX",
                                    zorder=-2,
                                    label="IG\_M5\_\{min,max\}\_density",
                                )
                                ax.plot(
                                    bin_centers,
                                    medians_3,
                                    dashes=(tuple(d for d in dashesMMD[0])),
                                    color=colorMMD,
                                    lw=lwMMD,
                                    zorder=-1,
                                    alpha=alphaMMD,
                                )
                                ax.plot(
                                    bin_centers,
                                    medians,
                                    dashes=(tuple(d for d in dashesMMD[1])),
                                    color=colorMMD,
                                    lw=lwMMD,
                                    zorder=-1,
                                    alpha=alphaMMD,
                                )

                        elif len(dict_sim.items()) == 6:
                            colors = color2
                            dashes = dashes2
                            ax.plot(
                                bin_centers,
                                medians,
                                label=key.replace("_", "\_"),
                                color=colors[counter],
                                dashes=(tuple(d for d in dashes[counter])),
                                lw=3.5,
                                zorder=3,
                            )
                        else:
                            ax.plot(
                                bin_centers,
                                medians,
                                label=key.replace("_", "\_"),
                                ls=line_properties["ls"][counter],
                                color=line_properties["colour"][counter],
                                lw=line_properties["linewidth"][counter],
                                zorder=5 - counter,
                            )

                # Decorate
                plt.xticks([-0.75, -0.5, -0.25, 0, 0.25, 0.50, 0.75])
                plt.yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.50, 0.75])
                ax.xaxis.set_tick_params(labelsize=33)
                ax.yaxis.set_tick_params(labelsize=33)

                ax.set_xlabel("$z$ [kpc]", fontsize=33)
                ax.set_ylabel(
                    "$\\langle \\cos \\theta \\rangle$ (SN feedback)",
                    fontsize=33,
                )

                plt.xlim(-0.5, 0.5)
                plt.ylim(-0.85, 0.85)

                plt.legend(loc="lower left", fontsize=22.0, frameon=False)
                ax.text(
                    0.96,
                    0.96,
                    "${:.1f}<\,$".format((snp_time_Myr.value - TIME_WINDOW_MYR) / 1e3)
                    + "$t<\\rm {:.1f}$ Gyr".format(snp_time_Myr.value / 1e3),
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=27,
                )

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)
                plt.close()


plot()
