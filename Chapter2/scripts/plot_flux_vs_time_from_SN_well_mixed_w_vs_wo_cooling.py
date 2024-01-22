import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py as h5
from cycler import cycler
from scipy import stats
from constants import *
from plot_style import *
from scipy import interpolate
from tqdm import tqdm
from numba import njit


def do_sedov_analytic(time_bins, dens, M_Fe60=1e-4, tau_Fe60=2.6):

    for i in range(time_bins):

        if time_plot_arr[i] > 0.0:

            print("Sedov (analytic): {:d}".format(i))

            r_s, P_s, rho_s, v_s, r_shock, _, _, _, _ = sedov(
                time_plot_arr[i] * constants["YEAR_IN_CGS"] * 1e6,
                1e51,
                dens / (constants["H_mass_fraction"] / constants["PROTON_MASS_IN_CGS"]),
                5.0 / 3.0,
                analytic_space_bins,
                3,
            )

            r_s = np.insert(r_s, np.size(r_s), [r_shock + 1e-15, r_shock * 2])
            v_s = np.insert(v_s, np.size(v_s), [0, 0])
            rho_s = np.insert(
                rho_s,
                np.size(rho_s),
                [
                    dens
                    / (constants["H_mass_fraction"] / constants["PROTON_MASS_IN_CGS"]),
                    dens
                    / (constants["H_mass_fraction"] / constants["PROTON_MASS_IN_CGS"]),
                ],
            )

            rho_v_r_bin, _, _ = stats.binned_statistic(
                r_s / constants["PARSEC_IN_CGS"] / 1e3,
                rho_s * v_s / constants["PROTON_MASS_IN_CGS"],
                statistic="mean",
                bins=magnitudes_analyt,
            )
            rho_v_r_bin[np.isnan(rho_v_r_bin)] = 0.0

            # Compute mass in the shell
            Volume = 4.0 * np.pi / 3.0 * r_shock ** 3
            M_shell = (
                dens
                / (constants["H_mass_fraction"] / constants["PROTON_MASS_IN_CGS"])
                * Volume
                / constants["SOLAR_MASS_IN_CGS"]
            )

            print("Mass in the shock {:.4e} [Msun]".format(M_shell))
            print("M_Fe60 / M_sw {:.4e}".format(M_Fe60 / M_shell))

            # Compute decay
            f_decay = np.exp(-np.log(2.0) * time_plot_arr[i] / tau_Fe60)
            print("Fe60 decay factor {:.4e}".format(f_decay))

            # Flux density: rho * v
            rhov_sedov_analytic[i, :] = rho_v_r_bin * M_Fe60 / M_shell * f_decay


def process_data(
    name, count, n_snapshot_max, save_file_name, M_Fe60=1e-4, tau_Fe60=2.6
):

    print("Begin to process data...")

    # For computing mass in the shell (ensures the shell is always expanding)
    n_thr_min = 1

    for idx in tqdm(range(n_snapshot_max)):

        with h5.File(f"{name}" + "/output_{:04d}.hdf5".format(idx), "r") as f:

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
            gas_m = (
                f["/PartType0/Masses"][:]
                * unit_mass_in_cgs
                / constants["SOLAR_MASS_IN_CGS"]
            )
            v_r = np.sum(gas_pos * gas_vel, axis=1) / r
            v2 = np.sum(gas_vel * gas_vel, axis=1)

            gas_density_f = f["/PartType0/Densities"][:]

            time_arr[idx, count] = (
                f["/Header"].attrs["Time"]
                * unit_time_in_cgs
                / constants["YEAR_IN_CGS"]
                / 1e6
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
            rho_v_r_bin[np.isnan(rho_v_r_bin)] = 0.0

            # Find maximum (the shock position)
            n_max = np.argmax(rho_v_r_bin)

            # Bin velocities
            v_bin, _, _ = stats.binned_statistic(
                r, v_r, statistic="mean", bins=magnitudes
            )  # km/s
            v_bin[np.isnan(v_bin)] = 1e10

            # Find the end of the shock by radial velocity cut
            vel_thr = 0.5  # km/s 0.5
            n_thr = np.argmin(np.abs(v_bin[n_max:] - vel_thr)) + n_max
            n_thr = max(n_thr, n_thr_min)
            if n_thr > n_thr_min:
                n_thr_min = n_thr

            M_bin, _, _ = stats.binned_statistic(
                r, gas_m, statistic="sum", bins=magnitudes
            )

            # Consistency check
            E_kin = 0.5 * gas_m * v2
            E_kin_bin, _, _ = stats.binned_statistic(
                r, E_kin, statistic="sum", bins=magnitudes
            )

            print("Bin position of the shock max and end:", n_max, n_thr)
            print(
                "Fraction of kinetic energy in the shock: {:.4e}".format(
                    np.sum(E_kin_bin[:n_thr]) / np.sum(E_kin)
                )
            )

            # Mass in the shell
            M = np.sum(M_bin[:n_thr])
            print("Mass in the shock {:.4e} [Msun]".format(M))
            print("M_Fe60 / M_sw {:.4e}".format(M_Fe60 / M))

            # Compute Decay
            f_decay = np.exp(-np.log(2.0) * time_arr[idx, count] / tau_Fe60)
            print("Fe60 decay factor {:.4e}".format(f_decay))

        rho_v_r_bin[np.isnan(rho_v_r_bin)] = 0.0
        flux_profiles[idx, :, count] = rho_v_r_bin * M_Fe60 / M * f_decay

    print("Finishing loading data")
    np.savez(save_file_name, flux_profiles, time_arr)

    return


def plot_data(runs):

    print("Performing interpolation...")
    flux_profiles[np.where(flux_profiles < 0.0)] = 0.0

    # Interpolation of the analytic solution
    flux_interp_f_analyt = interpolate.interp2d(
        magnitudes_centres_analyt, time_plot_arr, rhov_sedov_analytic, kind="linear"
    )

    fig, ax = plot_style(8, 8)

    for count, name in enumerate(runs.keys()):

        flux_interp_f = interpolate.interp2d(
                magnitudes_centres, time_arr[:, count], flux_profiles[:, :, count], kind="linear"
        )
        colormap = cmm.viridis
        colors = [colormap(i) for i in np.linspace(0, 1, 4)]
        ax.set_prop_cycle(cycler("color", colors[::-1]))

        for c, dist in enumerate(
            [
                50.0,
                100.0,
                125.0,
                150.0,
            ]
        ):

            print("Distance from SN is {:.1f}".format(dist))

            # Numerical solution
            x_arr = time_plot_arr
            # cgs
            y_arr = flux_interp_f(dist / 1e3, x_arr)
            # Divide by 1e5 becasue we want velocity in km/s = 1e5 cm/sec

            efficiency = 0.05

            if count == 0:
                print(dist, count)
                (line,) = plt.plot(
                    x_arr,
                    y_arr / 60 * constants["YEAR_IN_CGS"] * 0.25 * efficiency,
                    label="$D={:.0f}$ pc".format(dist),
                    lw=3.5,
                )
            else:
                print(dist, count)
                (line,) = plt.plot(
                    x_arr,
                    y_arr / 60 * constants["YEAR_IN_CGS"] * 0.25 * efficiency,
                    dashes=(10,2,2,2),
                    lw=3.5
                )

            color = color = line.get_color()

            # Analytical solution
            y_arr = flux_interp_f_analyt(dist / 1e3, x_arr)
            plt.plot(
                    x_arr,
                    y_arr / 60 * constants["YEAR_IN_CGS"] * 0.25 * efficiency,
                    color=color,
                    lw=1.5,
                    dashes=(3, 3),
            )

    # Cooling time
    plt.axvline(x=t_sf(1e51, dens), color="grey", lw=3, dashes=(8, 2), alpha=0.5)

    leg1 = ax.legend(loc="upper right", fontsize=LEGEND_SIZE * 0.8, frameon=False)

    (line1,) = plt.plot([-30,-40],[10,20], color="k", lw=3.5)
    (line2,) = plt.plot([-30,-40],[10,20], color="k", lw=3.5, dashes=(10,2,2,2))
    (line3,) = plt.plot([-30,-40],[10,20], color="k", lw=1.5, dashes=(3, 3))
    lines = [line1, line2, line3]
    key_list = list(runs.keys())
    key_list.append("ST solution")

    leg2 = plt.legend(lines, key_list,
                      loc="center right",
                      frameon=False,
                      fontsize=LEGEND_SIZE * 0.8)

    plt.gca().add_artist(leg1)

    ax.set_xlabel("Time since SN [Myr]", fontsize=LABEL_SIZE * 1.2)
    ax.set_ylabel("$^{60}$Fe flux [atoms cm$^{-2}$ yr$^{-1}$]", fontsize=LABEL_SIZE * 1.2)

    ax.xaxis.set_tick_params(labelsize=LABEL_SIZE * 1.2)
    ax.yaxis.set_tick_params(labelsize=LABEL_SIZE * 1.2)

    plt.xlim(0, 3.9)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yscale("log")
    plt.ylim(0.5, 1e3)

    fixlogax(ax, a="y")

    plt.savefig(
        "./images/flux_at_fixed_dist_well_mixed.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )


if __name__ == "__main__":

    N_bins = 301  # 1pc bins
    r_max = 0.3
    n_snapshot_max_arg = 134
    # path = "../run01_diffusion_new_one_p_with_centre_higres_sphere/"

    runs = {"high\_res\_n01": "../run01_diffusion_new_one_p_with_centre_higres_sphere/",
            "high\_res\_n01\_nocooling": "../run01_diffusion_new_one_p_with_centre_higres_nocooling_sphere_lowT/"}

    dens = 0.1  # H/cc3

    print("Max number of snapshots: {:d}".format(n_snapshot_max_arg))
    print("Number of radial bins: {:d}".format(N_bins))
    print("Max radial coordinate: {:.1f} [pc]".format(r_max * 1e3))
    print("Box density: {:.2e} [cm^-3]".format(dens))

    magnitudes = np.linspace(0.0, r_max, N_bins)
    magnitudes_centres = (magnitudes[:-1] + magnitudes[1:]) / 2
    magnitudes_widths = magnitudes[1:] - magnitudes[:-1]

    time_arr = np.zeros((n_snapshot_max_arg, len(runs)))
    flux_profiles = np.zeros((n_snapshot_max_arg, N_bins - 1, len(runs)))

    # Parameters for the analytic solution
    analytic_space_bins = 500
    analytic_time_bins = 500
    time_plot_arr = np.linspace(0.0, 3.9, analytic_time_bins)
    rhov_sedov_analytic = np.zeros((analytic_time_bins, analytic_space_bins))
    magnitudes_analyt = np.linspace(0.0, r_max, analytic_space_bins + 1)
    magnitudes_centres_analyt = 0.5 * (magnitudes_analyt[:-1] + magnitudes_analyt[1:])

    save_file_name = "./data/Fe60_Flux_vs_time_at_fixed_distance_well_mixed"

    do_sedov_analytic(analytic_time_bins, dens)

    try:
        input_file = np.load(f"{save_file_name}.npz")
        flux_profiles, time_arr = (input_file["arr_0"], input_file["arr_1"])
    except IOError:
        for count, path in enumerate(runs.values()):
            print(count, path)
            process_data(path, count, n_snapshot_max_arg, save_file_name)

    plot_data(runs)
