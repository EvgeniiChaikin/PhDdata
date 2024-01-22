import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
import h5py as h5
from scipy import stats
from constants import *
from plot_style import *
from scipy.interpolate import interp1d
from tqdm import tqdm


def do_sedov_analytic(time_bins, dens, nmodel):

    for i in range(time_bins):

        if time_plot_arr[i] > 0.0:

            r_s, P_s, rho_s, v_s, r_shock, _, _, _, _ = sedov(
                time_plot_arr[i] * constants["YEAR_IN_CGS"] * 1e6,
                1e51,
                dens / (constants["H_mass_fraction"] / constants["PROTON_MASS_IN_CGS"]),
                5.0 / 3.0,
                analytic_space_bins,
                3,
            )

            r_analytical[i, nmodel] = r_shock / constants["PARSEC_IN_CGS"] / 1e3


def process_data(name, n_snapshot_max, nmodel):

    print("Begin to process data...")

    for idx in tqdm(range(0, n_snapshot_max, 1)):

        f = h5.File(f"{name}" + "/output_{:04d}.hdf5".format(idx), "r")

        if idx == 0:

            unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
            unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
            unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

            box_size = f["/Header"].attrs["BoxSize"]
            centre = box_size / 2.0

        gas_pos = f["/PartType0/Coordinates"][:, :]
        gas_pos[:, 0] -= centre[0]
        gas_pos[:, 1] -= centre[1]
        gas_pos[:, 2] -= centre[2]
        r = np.sqrt(np.sum(gas_pos * gas_pos, axis=1))

        gas_vel = f["/PartType0/Velocities"][:, :]
        v_r = np.sum(gas_pos * gas_vel, axis=1) / r

        gas_density_f = f["/PartType0/Densities"][:]
        smoothing_length = f["/PartType0/SmoothingLengths"][:]

        # time in Myr
        time_arr[idx, nmodel] = (
            f["/Header"].attrs["Time"]
            * unit_time_in_cgs
            / constants["YEAR_IN_CGS"]
            / 1e6
        )

        h, _, _ = stats.binned_statistic(
            r,
            smoothing_length * unit_length_in_cgs / constants["PARSEC_IN_CGS"],
            statistic="mean",
            bins=magnitudes,
        )
        rho_v_r_bin, _, _ = stats.binned_statistic(
            r,
            gas_density_f
            * unit_mass_in_cgs
            / unit_length_in_cgs ** 3
            / constants["PROTON_MASS_IN_CGS"]
            * v_r
            * 1.0e5,
            statistic="mean",
            bins=magnitudes,
        )

        f.close()

        h[np.isnan(h)] = 0.0
        rho_v_r_bin[np.isnan(rho_v_r_bin)] = 0.0

        n_bin = np.argmax(rho_v_r_bin)
        h_numerical[idx, nmodel] = h[n_bin]
        r_numerical[idx, nmodel] = magnitudes_centres[n_bin]
        if idx > 0:
            r_numerical[idx, nmodel] = max(
                r_numerical[idx, nmodel], r_numerical[idx - 1, nmodel]
            )

    print("Finishing loading data")


def plot_data():

    fig, ax = plot_style(8, 8)

    for idx, key in enumerate(dict_data.keys()):

        ax.plot(
            time_arr[:, idx],
            1e3 * r_numerical[:, idx],
            label=key,
            lw=line_properties["linewidth"][idx],
            color=line_properties["colour"][idx],
            zorder=3,
        )

        if idx == 1:

            print("Plot analytic")

            ax.plot(
                time_plot_arr,
                1e3 * r_analytical[:, idx],
                color="k",
                lw=3,
                dashes=(3, 3),
                label="ST solution",
                zorder=4,
            )

            plt.axvline(
                x=t_sf(1e51, densities[idx]),
                color="grey",
                lw=2,
                dashes=(8, 2),
                alpha=0.8,
                zorder=1,
            )

    ax.legend(loc="lower right", fontsize=LEGEND_SIZE, frameon=False)

    ax.set_xlabel("Time since SN [Myr]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Blastwave position [pc]", fontsize=LABEL_SIZE)
    ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

    ax.set_ylim(-5, 240)
    ax.set_xlim(-0.1, 3.9)

    plt.savefig("./images/radius_vs_time_3_runs.pdf", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":

    densities = np.array([0.1, 0.1])

    dict_data = {
        "high\_res\_n01": "../run01_diffusion_new_one_p_with_centre_higres_sphere",
        "high\_res\_n01\_nocooling": "../run01_diffusion_new_one_p_with_centre_higres_nocooling_sphere_lowT",
    }

    N_models = len(dict_data)
    N_bins = 301
    r_max = 0.3
    n_snapshot_max_arg = 134

    print("Max number of snapshots: {:d}".format(n_snapshot_max_arg))
    print("Number of radial bins: {:d}".format(N_bins))
    print("Max radial coordinate: {:.1f} [pc]".format(r_max * 1e3))

    # Bins for numerical solution
    magnitudes = np.linspace(0.0, r_max, N_bins)
    magnitudes_centres = 0.5 * (magnitudes[:-1] + magnitudes[1:])
    magnitudes_widths = magnitudes[1:] - magnitudes[:-1]

    # Bins for ST solution
    analytic_space_bins = 1000
    analytic_time_bins = 1000
    time_plot_arr = np.linspace(0.0, 4.0, analytic_time_bins)
    r_analytical = np.zeros((analytic_time_bins, N_models))

    magnitudes_analyt = np.linspace(0.0, r_max, analytic_space_bins + 1)
    magnitudes_centres_analyt = 0.5 * (magnitudes_analyt[:-1] + magnitudes_analyt[1:])

    save_file_name = "./data/radius_vs_time_data"

    for counter in range(len(dict_data)):
        do_sedov_analytic(analytic_time_bins, densities[counter], counter)

    try:
        input_file = np.load(f"{save_file_name}.npz")
        r_numerical, h_numerical, time_arr = (
            input_file["arr_0"],
            input_file["arr_1"],
            input_file["arr_2"],
        )
        assert np.shape(r_numerical) == (n_snapshot_max_arg, N_models)

    except (IOError, AssertionError):

        time_arr = np.zeros((n_snapshot_max_arg, N_models))
        h_numerical = np.zeros((n_snapshot_max_arg, N_models))
        r_numerical = np.zeros((n_snapshot_max_arg, N_models))

        for counter, value in enumerate(dict_data.values()):
            print(
                "Counter: {:d}, path: {:s}, density: {:.1e} [cm^-3]".format(
                counter, value, densities[counter])
            )
            process_data(value, n_snapshot_max_arg, counter)

        np.savez(save_file_name, r_numerical, h_numerical, time_arr)

    plot_data()
