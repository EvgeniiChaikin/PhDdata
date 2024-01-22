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

            E_inj = f['PartType4']["InjectedKineticEnergy"][:] * 0.5
            E_recv = f['PartType4']["InjectedKineticEnergyReceived"][:] 

            energy_expected[i, model_counter] = np.sum(E_inj)
            energy_actual[i, model_counter] = np.sum(E_recv)

            f.close()

    print("Saving files")
    np.savez(save_file_name, energy_expected, energy_actual, time_arr)


def plot():

    global time_arr, energy_actual, energy_expected

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
                energy_expected = np.zeros((snapshots, N_models))
                energy_actual = np.zeros((snapshots, N_models))
                time_arr = np.zeros((snapshots, N_models))

                try:
                    input_file = np.load(f"{save_file_name}.npz")
                    energy_expected, energy_actual, time_arr = (
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
              
                    dE_actual = energy_actual[1:, counter] - energy_actual[:-1, counter]
                    dE_expected =  energy_expected[1:, counter] - energy_expected[:-1, counter]

                    dE_ratio = np.zeros(len(dE_actual))
                    mask = dE_expected != 0.0
                    dE_ratio[mask] = dE_actual[mask] / dE_expected[mask]

                    center = 0.5 * (time_arr[1:, counter]  + time_arr[:-1, counter])
                    ax.plot(
                        center / 1e3,
                        dE_ratio,
                        lw=lw[counter],
                        color=colors[counter],
                        dashes=tuple(d for d in dashes[counter]),
                        zorder=4,
                        label=key.replace("_", "\_"),
                    )


                ax.xaxis.set_tick_params(labelsize=31)
                ax.yaxis.set_tick_params(labelsize=31)

                plt.xlabel("Time [Gyr]", fontsize=31)
                plt.ylabel("$E_{\\rm kin}$ (recv) / $E_{\\rm kin}$ (released)", fontsize=LABEL_SIZE * 0.85, labelpad=38)

                #leg1 = ax.legend(fontsize=19, loc="upper center", frameon=False,
                #    columnspacing=0.25,
                #        borderaxespad=0.20,
                #        labelspacing=0.3
                #)

                print(y_min, y_max)

                plt.xlim(-0.01, 1.01)

                ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])

                ax.set_ylim(y_min, y_max)

                plt.savefig(
                    f"./images/{output}", bbox_inches="tight", pad_inches=0.1
                )


if __name__ == "__main__":
    plot()
