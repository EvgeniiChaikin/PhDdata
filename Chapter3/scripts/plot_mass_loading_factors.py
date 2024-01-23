import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
import sys
import h5py as h5
from constants import *
from plot_style import *
from scipy import stats
from scipy.interpolate import interp1d
from tqdm import tqdm
import json


def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def loaddata(snapshots, dict_sim, save_file_name):
    # Simulatins
    for model_counter, value in enumerate(dict_sim.values()):
        print(model_counter)

        # Snapshots
        for i in tqdm(range(snapshots)):
            # Loading data
            f = h5.File(f"{value}" + "/output_{:04d}.hdf5".format(i), "r")

            # Units
            unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
            unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
            unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

            time = f["/Header"].attrs["Time"]
            time_arr[i, model_counter] = (
                time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6
            )

            # Box parameters
            boxsize = f["/Header"].attrs["BoxSize"]
            centre = boxsize / 2.0

            gas_pos = f["/PartType0/Coordinates"][:, 2]
            gas_mass = f["/PartType0/Masses"][:]
            # Z direction
            gas_v_z = f["/PartType0/Velocities"][:, 2]
            gas_pos -= centre[2]

            # mass outflows
            z_area = z_area_0 * constants["PARSEC_IN_CGS"] / unit_length_in_cgs  # kpc
            delta_z = delta_z_0 * constants["PARSEC_IN_CGS"] / unit_length_in_cgs  # kpc

            args_outflow_p = np.where(
                np.logical_and(
                    np.logical_and(
                        gas_pos >= z_area - delta_z / 2, gas_pos <= z_area + delta_z / 2
                    ),
                    gas_v_z > 0.0,
                )
            )

            if len(args_outflow_p[0]) > 0:
                outflows_p[i, model_counter] = (
                    np.sum(gas_mass[args_outflow_p] * gas_v_z[args_outflow_p])
                    / delta_z
                    / unit_time_in_cgs
                    * constants["YEAR_IN_CGS"]
                    * unit_mass_in_cgs
                    / constants["SOLAR_MASS_IN_CGS"]
                )

            args_outflow_m = np.where(
                np.logical_and(
                    np.logical_and(
                        gas_pos >= -z_area - delta_z / 2,
                        gas_pos <= -z_area + delta_z / 2,
                    ),
                    gas_v_z < 0.0,
                )
            )

            if len(args_outflow_m[0]) > 0:
                outflows_m[i, model_counter] = (
                    np.sum(gas_mass[args_outflow_m] * gas_v_z[args_outflow_m])
                    / delta_z
                    / unit_time_in_cgs
                    * constants["YEAR_IN_CGS"]
                    * unit_mass_in_cgs
                    / constants["SOLAR_MASS_IN_CGS"]
                )

            f.close()

    print("Saving files")
    np.savez(save_file_name, outflows_m, outflows_p, time_arr)


def plot():
    global time_arr, outflows_p, outflows_m, z_area_0, delta_z_0

    runs = read_sim_data("main.json")

    for script_name, plots in runs.items():
        print(script_name)

        if script_name == sys.argv[0]:
            print("FOUND")

            for plot in plots:
                output = plot["output_file"]
                dict_sim = plot["data"]

                snapshots = plot["snapshot"]
                save_file_name = output.strip(".pdf")
                N_models = len(dict_sim)

                print(f"Number of models: {N_models}")

                # Arrays to fill in
                outflows_p = np.zeros((snapshots, N_models))
                outflows_m = np.zeros((snapshots, N_models))
                time_arr = np.zeros((snapshots, N_models))

                z_area_0 = 10000.0  # pc (distance from the disk)
                delta_z_0 = 100.0  # pc

                try:
                    input_file = np.load(f"{save_file_name}.npz")
                    outflows_m, outflows_p, time_arr = (
                        input_file["arr_0"],
                        input_file["arr_1"],
                        input_file["arr_2"],
                    )
                    print(f"data have been loaded for {save_file_name}")
                except IOError:
                    loaddata(snapshots, dict_sim, save_file_name)

                print(output, save_file_name)

                fig, ax = plot_style(8, 8)

                for counter, (key, value) in enumerate(dict_sim.items()):
                    SFR_file = np.loadtxt(f"{value}/SFR.txt", skiprows=25)
                    time = SFR_file[:, 1] * 9.778131e02
                    total_SFR = SFR_file[:, -1] * 1.022690e01  # M_sol / yr

                    sfh_bins = 100
                    sfh_edges = np.linspace(0.0, 1.0e3, sfh_bins)  # Myr
                    sfh_centers = 0.5 * (sfh_edges[1:] + sfh_edges[:-1])

                    sfh_values, _, _ = stats.binned_statistic(
                        time, total_SFR, statistic="median", bins=sfh_edges
                    )

                    SFR_f = interp1d(
                        sfh_centers, sfh_values, fill_value=0.0, bounds_error=False
                    )

                    N_bins = 40
                    bin_edges = np.linspace(0.0, 1.0e3, N_bins)  # Myr
                    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                    bin_values = np.zeros(N_bins - 1)

                    for i in range(N_bins - 1):
                        n_snp = 0
                        for c, t in enumerate(time_arr[:, counter]):
                            if bin_edges[i] < t <= bin_edges[i + 1]:
                                n_snp += 1  # in bin
                                bin_values[i] += (
                                    outflows_p[c, counter] - outflows_m[c, counter]
                                )
                        if n_snp:
                            bin_values[i] /= n_snp
                            bin_values[i] /= SFR_f(bin_centers[i])

                    if len(dict_sim.items()) == 5:
                        if counter < 3:
                            ax.plot(
                                bin_centers / 1e3,
                                bin_values,
                                label=key.replace("_", "\_"),
                                color=line_properties["colour"][counter],
                                alpha=line_properties["alpha"][counter],
                                ls=line_properties["ls"][counter],
                                lw=line_properties["linewidth"][counter],
                            )
                        elif counter == 3:
                            y3 = bin_values
                        else:
                            ax.fill_between(
                                bin_centers / 1e3,
                                bin_values,
                                y3,
                                edgecolor="grey",
                                alpha=0.2,
                                hatch="XXXX",
                                zorder=-2,
                                label="IG\_M5\_\{min,max\}\_density",
                            )
                            ax.plot(
                                bin_centers / 1e3,
                                y3,
                                dashes=(tuple(d for d in dashesMMD[0])),
                                color=colorMMD,
                                lw=lwMMD,
                                zorder=-1,
                                alpha=alphaMMD,
                            )
                            ax.plot(
                                bin_centers / 1e3,
                                bin_values,
                                dashes=(tuple(d for d in dashesMMD[1])),
                                color=colorMMD,
                                lw=lwMMD,
                                zorder=-1,
                                alpha=alphaMMD,
                            )

                    else:
                        colors = color2
                        dashes = dashes2
                        ax.plot(
                            bin_centers / 1e3,
                            bin_values,
                            label=key.replace("_", "\_"),
                            color=colors[counter],
                            lw=2.5,
                            dashes=(tuple(d for d in dashes[counter])),
                        )

                ax.xaxis.set_tick_params(labelsize=33)
                ax.yaxis.set_tick_params(labelsize=33)
                plt.xlabel("Time [Gyr]", fontsize=33)
                plt.ylabel("Wind mass loading factor $\\eta$", fontsize=33)

                plt.legend(loc="lower left", fontsize=25)
                plt.ylim(0.08, 14)
                plt.xlim(-0.01, 1.01)
                plt.yscale("log")
                fixlogax(ax, "y")

                ax.text(
                    0.04,
                    0.96,
                    f"$d = {z_area_0/1e3:.0f}$ kpc, $\\Delta d = {delta_z_0:.0f}$ pc",
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=27,
                )

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    plot()
