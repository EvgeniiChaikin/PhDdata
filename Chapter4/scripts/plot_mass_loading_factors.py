import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import ticker
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

    print("Loading..")

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
                gas_v_z > 0.0)
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
                    gas_pos >= -z_area - delta_z / 2, gas_pos <= -z_area + delta_z / 2
                ),
                gas_v_z < 0.0)
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

                print(plot)

                output = plot["output_file"]
                dict_sim = plot["data"]
                split = plot["split"]
                snapshots = plot["snapshot"]
                y_min, y_max = plot["ylims"]
                save_file_name = output.strip(".pdf")
                N_models = len(dict_sim)

                try:
                    fig_size_x, fig_size_y = plot["fig_size"]
                except KeyError:
                    fig_size_x, fig_size_y = 8, 8

                print(f"Number of models: {N_models}")

                # Arrays to fill in
                outflows_p = np.zeros((snapshots, N_models))
                outflows_m = np.zeros((snapshots, N_models))
                time_arr = np.zeros((snapshots, N_models))

                z_area_0 = 10000.0  # pc (distance from the disk)
                delta_z_0 = 1000.0  # pc

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

                fig, ax = plot_style(fig_size_x, fig_size_y)

                for counter, (key, value) in enumerate(dict_sim.items()):

                    SFR_file = np.loadtxt(f"{value}/SFR.txt", skiprows=25)
                    time = SFR_file[:, 1] * 9.778131e02
                    total_SFR = SFR_file[:, -1] * 1.022690e01  # M_sol / yr

                    N_bins = 500 
                    dt = 0.025 * 1e3 # window of 2x25 Myr = 50 Myr

                    sfh_centers = np.linspace(0.0, 1.0e3, N_bins)
                    sfh_values = np.zeros_like(sfh_centers)

                    for i in range(len(sfh_centers)):
                        mask = np.where(np.abs(time-sfh_centers[i]) < dt)[0]
                        sfh_values[i] = np.mean(total_SFR[mask])

                    SFR_f = interp1d(
                        sfh_centers, sfh_values, fill_value=0.0, bounds_error=False
                    )

                    N_bins = 51
                    bin_edges = np.linspace(0.0, 1.0e3, N_bins) # Myr
                    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1]) 
                    bin_values = np.zeros(N_bins-1)

                    for i in range(N_bins-1):
                        n_snp = 0
                        for c, t in enumerate(time_arr[:, counter]):
                            if bin_edges[i] < t <= bin_edges[i+1]:
                                n_snp += 1 # in bin
                                bin_values[i] += (outflows_p[c, counter] - outflows_m[c, counter])
                        if n_snp:
                            print(n_snp)
                            bin_values[i] /= n_snp
                            bin_values[i] /= SFR_f(bin_centers[i])
                    
                    if split < 7:

                        if split == 1:
                            colors = color2
                            lw = lw2
                            dashes = dashes2
                        elif split == 4:
                            colors = color4
                            lw = lw4
                            dashes = dashes4

                        ax.plot(
                            bin_centers / 1e3,
                            bin_values,
                            label=key.replace("_", "\_"),
                            color=colors[counter],
                            lw=lw[counter],
                            dashes = tuple(d for d in dashes[counter]),
                        )

                    else:
                        ax.plot(
                            bin_centers / 1e3,
                            bin_values,
                            label=key.replace("_", "\_"),
                            color=line_properties["colour"][counter],
                            alpha=line_properties["alpha"][counter],
                            ls=line_properties["ls"][counter],
                            lw=2.5,
                        )

                ax.xaxis.set_tick_params(labelsize=31)
                ax.yaxis.set_tick_params(labelsize=31)

                plt.xlabel("Time [Gyr]", fontsize=31)
                plt.ylabel("Mass loading $\\eta$", fontsize=31, labelpad=0.15)

                print(y_min, y_max)

                plt.xlim(-0.01, 1.01)
                plt.yscale("log")
                ax.set_yticks([1e-2,1e-1,1e0,1e1])

                locmin = ticker.LogLocator(
                base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=10
                )

                ax.yaxis.set_minor_locator(locmin)
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())

                ax.set_ylim(y_min, y_max)
                fixlogax(ax, "y")

                plt.savefig(
                    f"./images/{output}", bbox_inches="tight", pad_inches=0.1
                )


if __name__ == "__main__":
    plot()
