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


def R0(Esn = 1e51, Psw = 2e-8):
    # pc
    return 10.0 * np.power(Esn / 1.0e51, 1./3.) * np.power(Psw / 2e-8, -1./3.)


def Rfade(nH = 0.1, Esn = 1e51, cs = 10.0):
    # pc
    return 160.0 * np.power(Esn / 1e51, 0.32) * np.power(nH / 0.1, -0.37) * np.power(cs / 10.0, -2./5.)


def do_sedov_analytic(time_bins, dens, c):

    for i in range(time_bins):
        if time_plot_analyt[i] > 0.0:

            print("Sedov (analytic): {:d}".format(i))

            r_s, P_s, rho_s, v_s, r_shock, _, _, _, _ = sedov(
                time_plot_analyt[i] * constants["YEAR_IN_CGS"] * 1e6,
                1e51,
                dens / (constants["H_mass_fraction"] / constants["PROTON_MASS_IN_CGS"]),
                5.0 / 3.0,
                analytic_space_bins,
                3,
            )

            r_analytical[i, c] = r_shock / constants["PARSEC_IN_CGS"] / 1e3


def process_data(name, n_model, n_snapshot_max, M_Fe60=1e-4, tau_Fe60=2.6):

    print("Begin to process data...")
    print(f"Path: {name}")
    print(f"N model: {n_model}")
    print("Velocity threshold for shock detection: {:.3f}".format(vel_thr[n_model]))

    # For computing mass in the shell (ensures the shell is always expanding)
    n_thr_min = 1

    for idx in tqdm(range(n_snapshot_max)):

        f = h5.File(f"{name}" + "/output_{:04d}.hdf5".format(idx), "r")

        print("Snapshot number: {:d}".format(idx))

        if idx == 0:

            unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
            unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
            unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

            box_size = f["/Header"].attrs["BoxSize"]
            centre = box_size / 2.0

        time_arr[idx, n_model] = (
            f["/Header"].attrs["Time"]
            * unit_time_in_cgs
            / constants["YEAR_IN_CGS"]
            / 1e6
        )

        # in kpc
        gas_pos = f["/PartType0/Coordinates"][:, :]
        gas_pos[:, 0] -= centre[0]
        gas_pos[:, 1] -= centre[1]
        gas_pos[:, 2] -= centre[2]
        r = np.sqrt(np.sum(gas_pos * gas_pos, axis=1))

        # in km/s
        gas_vel = f["/PartType0/Velocities"][:, :]

        # in solar masses
        gas_m = (
            f["/PartType0/Masses"][:]
            * unit_mass_in_cgs
            / constants["SOLAR_MASS_IN_CGS"]
        )
        v_r = np.sum(gas_pos * gas_vel, axis=1) / r
        v2 = np.sum(gas_vel * gas_vel, axis=1)

        # in Msun/ kpc**3
        gas_density_f = f["/PartType0/Densities"][:]
        
        # in Myr
        time_arr[idx, n_model] = (
            f["/Header"].attrs["Time"]
            * unit_time_in_cgs
            / constants["YEAR_IN_CGS"]
            / 1e6
        )

        # in cgs units
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
        n_thr = np.argmin(np.abs(v_bin[n_max:] - vel_thr[n_model])) + n_max
        n_thr = max(n_thr, n_thr_min)
        if n_thr > n_thr_min:
            n_thr_min = n_thr

        M_bin, _, _ = stats.binned_statistic(r, gas_m, statistic="sum", bins=magnitudes)

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
        f_decay = np.exp(-np.log(2.0) * time_arr[idx, n_model] / tau_Fe60)
        print("Fe60 decay factor {:.4e}".format(f_decay))

        rho_v_r_bin[np.isnan(rho_v_r_bin)] = 0.0
 
        # in cgs
        flux_profiles[idx, :, n_model] = rho_v_r_bin * M_Fe60 / M * f_decay

        f.close()

    print("Finishing loading data")


def plot_data():

    # Remove weird values
    flux_profiles[np.where(flux_profiles < 0.0)] = 0.0

    # Time axis
    time_bins = 1000
    time_plot_arr = np.linspace(0.0, 3.9, time_bins)

    # Create figure

    COL_SIZE, ROW_SIZE = 2, 3
    fig, ax = plt.subplots(
        COL_SIZE, ROW_SIZE, figsize=(12, 10), sharex=True
    )
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["axes.linewidth"] = 2

    for j in range(ROW_SIZE):
        for i in range(COL_SIZE):
            ax[i, j].tick_params(which="both", width=1.7)
            ax[i, j].tick_params(which="major", length=9)
            ax[i, j].tick_params(which="minor", length=5)
            ax[i, j].xaxis.set_minor_locator(AutoMinorLocator(5))
            ax[i, j].yaxis.set_minor_locator(AutoMinorLocator(5))
            ax[i, j].tick_params(
                axis="both",
                which="both",
                pad=8,
                left=True,
                right=True,
                top=True,
                bottom=True,
            )

            ax[i, j].set_xlim(0.0, 3.8)
            ax[i, j].xaxis.set_tick_params(labelsize=LABEL_SIZE)
            ax[i, j].yaxis.set_tick_params(labelsize=LABEL_SIZE)

            if i == 1:
                ax[i, j].set_xlabel("Time since SN [Myr]", fontsize=LABEL_SIZE * 0.85)
                ax[i, j].set_xticks([0.5, 1.5, 2.5, 3.5])

            if j == 0 and i == 0:
                ax[i, j].set_ylabel("Distance to SN [pc]", fontsize=LABEL_SIZE * 0.85)
            elif j == 0:
                ax[i, j].set_ylabel("$^{60}$Fe flux [atoms cm$^{-2}$ yr$^{-1}$]", fontsize=LABEL_SIZE * 0.85)
            else:
                ax[i, j].get_yaxis().set_ticklabels([])

        ax[0, j].set_ylim(0, 300)
        ax[1, j].set_ylim(0, 39)

    D = np.array([[25, 50, 75, 100], [50, 100, 125, 150], [100, 150, 200, 250]])

    for num, name in enumerate(dict_data.keys()):

        print(name, num)
        print("Performing interpolation...")

        # Create new arrays for interpolation
        fluxes_run = flux_profiles[:, :, num]
        times_run = time_arr[:, num]

        flux_interp_f = interpolate.interp2d(
            magnitudes_centres, times_run, fluxes_run, kind="linear"
        )

        # Number of bins in the image
        im_n_bins = 400

        # Distance to SN [kpc]
        y = np.linspace(0, 300.0, im_n_bins)

        # Time since SN [Myr]
        x = np.linspace(0, 5.0, im_n_bins)

        # 1 percent efficiency
        # Flux in cgs units is divided by Fe60 atomic number and multiplied by yr
        # because we want units [N of Fe60 atoms per cm2 per yr
        # Factor of 0.25 is because pi R**2 / 4pi R**2
        efficiency = 0.05
        flux_Fe60 = flux_interp_f(y / 1e3, x) / 60 * constants["YEAR_IN_CGS"] * 0.25 * efficiency

        extent = [x[0], x[-1], y[0], y[-1]]

        mappable0 = ax[0, num].imshow(
            np.swapaxes(np.log10(flux_Fe60 + 1e-16), 0, 1),
            cmap=cmm.Blues,
            extent=[extent[0], extent[1], extent[2], extent[3]],
            origin="lower",
            vmin=-0.25,
            vmax=3.25,
            aspect="auto",
            zorder=-1,
        )

        CS = ax[0, num].contour(
            x,
            y,
            flux_Fe60.T,
            levels=[1, 10],
            linewidths=[1.0],
            colors=["k"],
            zorder=3,
            linestyles=["dashed"],
        )

        R0_val = R0()
        Rfade_val = Rfade(nH = dens[num])

        Rkill_min = [0.0] * 2  # pc
        Rkill_max = [R0_val] * 2  # pc

        Rfade_min = [Rfade_val] * 2 # pc
        Rfade_max = [500] * 2 # pc

        x_R = [-0.1, 4.1]

        ax[0, num].fill_between(x_R, Rkill_min, Rkill_max, color='red', alpha=0.08, hatch = "\\\\\\\\")
        ax[0, num].fill_between(x_R, Rfade_min, Rfade_max, color='red', alpha=0.08, hatch = "\\\\\\\\")

        colormap = cmm.viridis
        colors = [colormap(i) for i in np.linspace(0, 1, 4)]
        ax[1, num].set_prop_cycle(cycler("color", colors[::-1]))

        for c, dist in enumerate(
            D[num, :]
        ):

            print("Distance from SN is {:.1f}".format(dist))

            # Numerical solution
            x_arr = time_plot_arr
            y_arr = flux_interp_f(dist / 1e3, x_arr)

            # The original flux is in cgs units, we do the conversion from sec to yr
            (line,) = ax[1, num].plot(
                x_arr,
                y_arr / 60 * constants["YEAR_IN_CGS"] * 0.25 *  efficiency,
                label="$D={:.0f}$ pc".format(dist),
                lw=2.5,
            )

            color = line.get_color()
            ax[0, num].axhline(y=dist, color=color, dashes=(1, 2), lw=2.0, alpha=1.0)

        ax[0, num].plot(
            time_plot_analyt,
            r_analytical[:, num] * 1e3,
            color="orange",
            lw=2,
            dashes=(3, 3),
            zorder=10,
        )

        ax[1, num].legend(loc="upper right", fontsize=LEGEND_SIZE*0.5, frameon=False)

        ax[0, num].text(
            0.05,
            0.96,
            name,
            ha="left",
            va="top",
            transform=ax[0, num].transAxes,
            fontsize=LABEL_SIZE*0.8,
            color="black",
            zorder=50
        )

    fig.subplots_adjust(top=0.92)
    cbar_ax = fig.add_axes([0.3, 0.95, 0.42, 0.025])

    cbar = fig.colorbar(mappable0, cax=cbar_ax, orientation="horizontal", ticks=[0, 1, 2, 3],extend='both')
    cbar_ax.xaxis.set_tick_params(labelsize=20)
    cbar.ax.tick_params(which='major', width=1.7,length=17)
    cbar.ax.set_xlabel('log $^{60}$Fe flux [atoms cm$^{-2}$ yr$^{-1}$]', rotation=0, fontsize=21, labelpad=-70)

    plt.savefig(
        "./images/flux_at_fixed_dist_well_mixed_3runs.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )


if __name__ == "__main__":

    N_bins = 301  # 1pc bins
    r_max = 0.3
    n_snapshot_max_arg = 130

    dict_data = {
        "high\_res\_n1": "../run01_diffusion_new_one_p_with_centre_higres_1_sphere",
        "high\_res\_n01": "../run01_diffusion_new_one_p_with_centre_higres_sphere",
        "high\_res\_n001": "../run01_diffusion_new_one_p_with_centre_higres_001_sphere",
    }

    dens = [1, 0.1, 0.01]
    # To compute shock front
    vel_thr = [0.25, 0.5, 1.0]

    print("Max number of snapshots: {:d}".format(n_snapshot_max_arg))
    for value in dict_data.values():
        print(f"Path to snapshots: {value}")
    print("Number of radial bins: {:d}".format(N_bins))
    print("Max radial coordinate: {:.1f} [pc]".format(r_max * 1e3))
    for n in dens:
        print("Box density: {:.2e} [cm^-3]".format(n))

    magnitudes = np.linspace(0.0, r_max, N_bins)
    magnitudes_centres = (magnitudes[:-1] + magnitudes[1:]) / 2
    magnitudes_widths = magnitudes[1:] - magnitudes[:-1]

    save_file_name = "./data/Fe60_flux_at_2p2Myr_well_mixed"

    # Parameters for the analytic solution
    analytic_space_bins = 1000
    analytic_time_bins = 1000
    time_plot_analyt = np.linspace(0.0, 3.9, analytic_time_bins)

    r_analytical = np.zeros((analytic_time_bins, len(dens)))
    magnitudes_analyt = np.linspace(0.0, r_max, analytic_space_bins + 1)
    magnitudes_centres_analyt = 0.5 * (magnitudes_analyt[:-1] + magnitudes_analyt[1:])

    for i, rho in enumerate(dens):
        do_sedov_analytic(analytic_time_bins, rho, i)

    try:
        input_file = np.load(f"{save_file_name}.npz")
        flux_profiles, time_arr = (
            input_file["arr_0"],
            input_file["arr_1"],
        )
        print(np.shape(time_arr), (n_snapshot_max_arg, len(dens)))
        assert np.shape(time_arr) == (n_snapshot_max_arg, len(dens))

    except (IOError, AssertionError):
        time_arr = np.zeros((n_snapshot_max_arg, len(dens)))
        flux_profiles = np.zeros((n_snapshot_max_arg, N_bins - 1, len(dens)))

        for num, (key, path) in enumerate(dict_data.items()):
            process_data(path, num, n_snapshot_max_arg)

        np.savez(save_file_name, flux_profiles, time_arr)

    plot_data()
