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


def loaddata(snapshots, dict_runs, save_name, gal_counter):

    print("Loading")

    # Simulatins
    for model_counter, value in enumerate(dict_runs.values()):

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
            time_arr[i, model_counter, gal_counter] = (
                time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6
            )

            # Box parameters
            boxsize_kpc = (
                f["/Header"].attrs["BoxSize"]
                * unit_length_in_cgs
                / constants["PARSEC_IN_CGS"]
                / 1e3
            )
            centre_kpc = boxsize_kpc / 2.0

            gas_pos_kpc = (
                f["/PartType0/Coordinates"][:, 2]
                * unit_length_in_cgs
                / constants["PARSEC_IN_CGS"]
                / 1e3
            )

            gas_mass_Msun = (
                f["/PartType0/Masses"][:]
                * unit_mass_in_cgs
                / constants["SOLAR_MASS_IN_CGS"]
            )

            # Z direction
            gas_v_z_kms = (
                f["/PartType0/Velocities"][:, 2]
                * unit_length_in_cgs
                / 1e5
                / unit_time_in_cgs
            )
            gas_pos_kpc -= centre_kpc[2]

            # mass outflows
            args_outflow_p = np.where(
                np.logical_and(
                    np.logical_and(
                        gas_pos_kpc >= z_area_kpc - delta_z_kpc / 2,
                        gas_pos_kpc <= z_area_kpc + delta_z_kpc / 2,
                    ),
                    gas_v_z_kms > 0.0,
                )
            )

            if len(args_outflow_p[0]) > 0:
                outflows_p[i, model_counter, gal_counter] = (
                    np.sum(gas_mass_Msun[args_outflow_p] * gas_v_z_kms[args_outflow_p])
                    / delta_z_kpc
                    / constants["PARSEC_IN_CGS"]
                    / 1e3
                    * 1e5
                    * constants["YEAR_IN_CGS"]
                )

            args_outflow_m = np.where(
                np.logical_and(
                    np.logical_and(
                        gas_pos_kpc >= -z_area_kpc - delta_z_kpc / 2,
                        gas_pos_kpc <= -z_area_kpc + delta_z_kpc / 2,
                    ),
                    gas_v_z_kms < 0.0,
                )
            )

            if len(args_outflow_m[0]) > 0:
                outflows_m[i, model_counter, gal_counter] = (
                    np.sum(gas_mass_Msun[args_outflow_m] * gas_v_z_kms[args_outflow_m])
                    / delta_z_kpc
                    / constants["PARSEC_IN_CGS"]
                    / 1e3
                    * 1e5
                    * constants["YEAR_IN_CGS"]
                )

            f.close()

    print("Saving files")
    np.savez(
        save_name,
        outflows_m[:, :, gal_counter],
        outflows_p[:, :, gal_counter],
        time_arr[:, :, gal_counter],
    )
    return


def plot():

    global time_arr, outflows_p, outflows_m, z_area_kpc, delta_z_kpc

    runs = read_sim_data("main.json")

    for script_name, plots in runs.items():

        print(script_name)

        if script_name == sys.argv[0]:

            print("FOUND")

            for plot in plots:

                save_file_name = plot["loaddata_files"]
                output = plot["output_file"]
                dict_sim = plot["data"]
                snapshots = plot["snapshot"]
                split = plot["split"]
                y_min, y_max = plot["ylims_main"]
                y_min_sub, y_max_sub = plot["ylims_insert"]

                print(output)

                N_models = len(dict_sim) // 2
                print(f"Number of models per plot: {N_models}")

                # Arrays to fill in
                outflows_p = np.zeros((snapshots, N_models, 2))
                outflows_m = np.zeros((snapshots, N_models, 2))
                time_arr = np.zeros((snapshots, N_models, 2))

                z_area_kpc = 10.0
                delta_z_kpc = 1.0

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

                for gal_c, file_name in enumerate(save_file_name):

                    if gal_c == 0:
                        runs = {
                            key: value
                            for (key, value) in dict_sim.items()
                            if "H12" in value
                        }
                    else:
                        runs = {
                            key: value
                            for (key, value) in dict_sim.items()
                            if "H10" in value
                        }

                    print(file_name, "runs (new dict)", runs)

                    try:
                        input_file = np.load(f"{file_name}.npz")
                        (
                            outflows_m[:, :, gal_c],
                            outflows_p[:, :, gal_c],
                            time_arr[:, :, gal_c],
                        ) = (
                            input_file["arr_0"],
                            input_file["arr_1"],
                            input_file["arr_2"],
                        )
                        print(f"data have been loaded for {file_name}")

                    except IOError:
                        loaddata(snapshots, runs, file_name, gal_c)

                    for run_c, (key, value) in enumerate(runs.items()):

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
                        bin_edges = np.linspace(0.0, 1.0e3, N_bins)  # Myr
                        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                        bin_values = np.zeros(N_bins - 1)

                        for i in range(N_bins - 1):

                            n_snp = 0
                            for c, t in enumerate(time_arr[:, run_c, gal_c]):
                                if bin_edges[i] < t <= bin_edges[i + 1]:
                                    n_snp += 1  # in bin
                                    bin_values[i] += (
                                        outflows_p[c, run_c, gal_c]
                                        - outflows_m[c, run_c, gal_c]
                                    )
                            if n_snp:
                                print(n_snp)
                                bin_values[i] /= n_snp
                                bin_values[i] /= SFR_f(bin_centers[i])

                        if split == 0:
                            color = line_properties["colour"]
                            # dashes = (3, 0)
                            lw = line_properties["linewidth"][run_c]
                        else:
                            color = color4
                            # dashes = tuple(d for d in dashes4[run_c])
                            lw = 2.3

                        if gal_c == 0:
                            ax.plot(
                                bin_centers / 1e3,
                                bin_values,
                                label=key.replace("_", "\_"),
                                lw=lw,
                                color=color[run_c],
                            )
                        else:
                            ax2.plot(
                                bin_centers / 1e3,
                                bin_values,
                                label=key.replace("_", "\_"),
                                lw=lw,
                                color=color[run_c],
                            )

                leg1 = ax.legend(
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

                hl_dict = {handle.get_label(): handle for handle in leg1.legendHandles}

                for k in hl_dict:
                    hl_dict[k].set_color("white")
                for counter, text in enumerate(leg1.get_texts()):
                    text.set_color(color[counter])

                leg2 = ax2.legend(
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

                    a.set_yticks([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
                    a.set_ylabel("Mass loading $\\eta$", fontsize=30)

                    fixlogax(a, "y")

                ax.set_ylim(y_min, y_max)
                ax2.set_ylim(y_min_sub, y_max_sub)

                ax2.set_xlabel("Time [Gyr]", fontsize=32)
                plt.setp(ax.get_xticklabels(), visible=False)

                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    plot()
