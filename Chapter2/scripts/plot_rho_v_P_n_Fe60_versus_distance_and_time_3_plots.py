import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py as h5
from scipy import stats
from constants import *
from plot_style import *
from scipy import interpolate
from tqdm import tqdm


def intersection_volume_vector(rp, hp, w, r1, r2):
    """
    Parameters:
    -------------------------
    rp: float
    Particle radius

    hp: float
    Particle smoothing length

    r1: float
    radius of the inner boundary of the bin

    r2: float
    radius of the outer boundary of the bin

    Returns
    -------------------------
    output: float
    Volume coefficient: \in [0, 1]
    """

    Volumep = 4.0 * np.pi / 3.0 * hp ** 3

    output = np.zeros_like(rp)

    for (sign, rb) in zip([-1.0, 1.0], [r1, r2]):

        case2 = np.logical_and(rp + hp > rb, np.abs(rp - hp) < rb)
        case3 = np.logical_and(
            np.logical_and(rp + hp <= rb, rp - hp >= -rb), np.logical_not(case2)
        )
        case4 = np.logical_and(
            np.logical_or(rp + hp < rb, rp - hp < -rb),
            np.logical_not(np.logical_or(case3, case2)),
        )

        # Case2
        alphab = (rb ** 2 + rp[case2] ** 2 - hp[case2] ** 2) / (2.0 * rp[case2] * rb)
        alphap = (hp[case2] ** 2 + rp[case2] ** 2 - rb ** 2) / (
            2.0 * rp[case2] * hp[case2]
        )

        heightb = rb * (1.0 - alphab)
        heightp = hp[case2] * (1.0 - alphap)

        cap_volumeb = np.pi * heightb ** 2 / 3.0 * (3.0 * rb - heightb)
        cap_volumep = np.pi * heightp ** 2 / 3.0 * (3.0 * hp[case2] - heightp)

        output[case2] += sign * (cap_volumeb + cap_volumep) / Volumep[case2] * w[case2]

        # Case3
        output[case3] += sign * w[case3]

        # Case4
        Volumeb = 4.0 * np.pi / 3.0 * rb ** 3
        output[case4] += sign * Volumeb / Volumep[case4] * w[case4]

    # Particle is inside the bin
    case1 = np.logical_and(rp - hp >= r1, rp + hp <= r2)
    output[case1] = 1.0 * w[case1]

    return output


def bin_distribute(r, h, w):

    r_min = r - h
    r_max = r + h

    bin_values = np.zeros_like(magnitudes_centres)

    for bin_idx in range(len(magnitudes) - 1):
        in_bin = np.where(
            np.logical_and(r_min < magnitudes[bin_idx + 1], r_max > magnitudes[bin_idx])
        )

        if len(in_bin[0]):
            weight = intersection_volume_vector(
                rp=r[in_bin],
                hp=h[in_bin],
                w=w[in_bin],
                r1=magnitudes[bin_idx],
                r2=magnitudes[bin_idx + 1],
            )

            bin_values[bin_idx] += np.sum(weight)

    return bin_values


def do_sedov_analytic(time_bins, dens):

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

            r_analytical[i] = r_shock / constants["PARSEC_IN_CGS"] / 1e3


def process_data(name, n_snapshot_max, save_file_name):

    print("Begin to process data...")

    for idx in tqdm(range(n_snapshot_max)):

        f = h5.File(f"{name}" + "/output_{:04d}.hdf5".format(idx), "r")

        # If 1st snp, get metadata
        if idx == 0:

            unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
            unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
            unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

            box_size = f["/Header"].attrs["BoxSize"]
            centre = box_size / 2.0

        # Coords
        gas_pos = f["/PartType0/Coordinates"][:, :]
        gas_pos[:, 0] -= centre[0]
        gas_pos[:, 1] -= centre[1]
        gas_pos[:, 2] -= centre[2]
        r = np.sqrt(np.sum(gas_pos * gas_pos, axis=1))

        time_arr[idx] = (
            f["/Header"].attrs["Time"]
            * unit_time_in_cgs
            / constants["YEAR_IN_CGS"]
            / 1e6
        )

        r = np.sqrt(np.sum(gas_pos * gas_pos, axis=1))  # Distance from the explosion

        # Compute gas velocity
        gas_vel = f["/PartType0/Velocities"][:, :]
        v_r = np.sum(gas_pos * gas_vel, axis=1) / r

        # Gas mass [Msun]
        gas_m = f["/PartType0/Masses"][:]

        # Smoothing length
        gas_h = f["/PartType0/SmoothingLengths"][:]

        # Density
        gas_density_f = f["/PartType0/Densities"][:]

        # Ejecta mass fraction
        ejecta = f["/PartType0/ElementMassFractions"][:, -1]

        # We want to normalise ejecta becasue the value of Fe60 ejected is aribtrary chosen
        ejecta_normalisation = np.sum(ejecta[np.where(ejecta > 0.0)])

        # We want 1e-4 Msun total in Fe60
        ejecta_mass = 1e-4 / np.mean(gas_m) * ejecta / ejecta_normalisation * gas_m

        # cgs units
        bin_volume = (
            4.0
            * np.pi
            / 3.0
            * (magnitudes[1:] ** 3 - magnitudes[:-1] ** 3)
            * (constants["PARSEC_IN_CGS"] * 1e3) ** 3
        )

        # Compute density with smoothing
        m_bin = bin_distribute(r, gas_h, ejecta_mass)
        rhov_bin = bin_distribute(r, gas_h, gas_m * v_r)
        rhov2_bin = bin_distribute(r, gas_h, gas_m * v_r * v_r)

        # Divide by total ejecta mass by volume then by A=60, then from Msola to grams then divide m_p
        n_Fe60_bin = (
            m_bin
            / bin_volume
            * constants["SOLAR_MASS_IN_CGS"]
            / 60.0
            / constants["PROTON_MASS_IN_CGS"]
        )


        rho_v_r_bin = ( rhov_bin 
            / bin_volume
            * constants["SOLAR_MASS_IN_CGS"]
            / constants["PROTON_MASS_IN_CGS"]
            * 1.0e5
        )


        rho_v_r2_bin = ( rhov2_bin 
            / bin_volume
            * constants["SOLAR_MASS_IN_CGS"]
            / constants["PROTON_MASS_IN_CGS"]
            * 1.0e5 * 1e5
        )


        profiles[idx, :, 0] = rho_v_r_bin
        profiles[idx, :, 1] = rho_v_r2_bin
        profiles[idx, :, 2] = n_Fe60_bin
        f.close()

    np.savez(save_file_name, profiles, time_arr)

    return


def plot_data():

    print("Performing interpolation...")
    profiles[np.where(profiles < 0.0)] = 0.0

    # Three fields
    rhov_interp = interpolate.interp2d(
        magnitudes_centres, time_arr, profiles[:, :, 0], kind="linear"
    )

    rhov2_interp = interpolate.interp2d(
        magnitudes_centres, time_arr, profiles[:, :, 1], kind="linear"
    )

    nFe60_interp = interpolate.interp2d(
        magnitudes_centres, time_arr, profiles[:, :, 2], kind="linear"
    )

    # Number of bins in the image
    im_n_bins = 400

    # Distance to SN [kpc]
    y = np.linspace(0, 230.0, im_n_bins)

    # Time since SN [Myr]
    x = np.linspace(0, 5.0, im_n_bins)

    # Perfor interpolating [kpc, Myr]
    Fe60 = nFe60_interp(y / 1e3, x)
    rhov = rhov_interp(y / 1e3, x)
    rhov2 = rhov2_interp(y / 1e3, x)

    # Create plot
    # plt.rc_context({"xtick.color": "white", "ytick.color": "white"})
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    fig, ax = plt.subplots(1, 3, figsize=(14, 6.05), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for i in range(3):
        ax[i].tick_params(
            axis="both",
            which="both",
            pad=8,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )
        ax[i].tick_params(which="both", width=1.0)
        ax[i].tick_params(which="major", length=6)
        ax[i].tick_params(which="minor", length=3)

    extent = [x[0], x[-1], y[0], y[-1]]

    mappable0 = ax[0].imshow(
        np.swapaxes(np.log10(rhov / 1e5 + 1e-16), 0, 1),
        cmap=cmm.viridis,
        extent=[extent[0], extent[1], extent[2], extent[3]],
        origin="lower",
        vmin=0,
        vmax=2,
        aspect="auto",
        zorder=1,
    )

    mappable1 = ax[1].imshow(
        np.swapaxes(np.log10(rhov2 * constants["PROTON_MASS_IN_CGS"] + 1e-35), 0, 1),
        cmap=cmm.viridis,
        extent=[extent[0], extent[1], extent[2], extent[3]],
        origin="lower",
        vmin=-12.75,
        vmax=-9.25,
        aspect="auto",
        zorder=1,
    )

    mappable2 = ax[2].imshow(
        np.swapaxes(np.log10(1e-16 + Fe60), 0, 1),
        cmap=cmm.Blues,
        extent=[extent[0], extent[1], extent[2], extent[3]],
        origin="lower",
        vmin=-12.5,
        vmax=-8.5,
        aspect="auto",
        zorder=1,
    )

    # Left
    divider = make_axes_locatable(ax[0])
    cax0 = divider.append_axes("top", size="5%", pad=-0.05)
    cbar0 = plt.colorbar(
        mappable0,
        cax=cax0,
        use_gridspec=True,
        ticks=[0, 1, 2.0],
        orientation="horizontal",
    )
    cax0.xaxis.set_ticks_position("top")
    cax0.xaxis.set_ticklabels(
        ["$0$", "$1$", "$2$"],
        fontsize=LABEL_SIZE*0.9,
        color="black",
    )
    cax0.set_xlabel(
        "log $\\rho \\, v$ [$10^{5}$ m$_{\\rm p}$ cm$^{-2}$ s$^{-1}$]",
        fontsize=LABEL_SIZE*0.9,
        labelpad=-80,
    )
    cax0.xaxis.set_tick_params(color="black", length=10.0)

    # Middle
    divider = make_axes_locatable(ax[1])
    cax1 = divider.append_axes("top", size="5%", pad=-0.05)
    cbar1 = plt.colorbar(
        mappable1,
        cax=cax1,
        use_gridspec=True,
        ticks=[-12.5, -11.5, -10.5, -9.5],
        orientation="horizontal",
    )
    cax1.xaxis.set_ticks_position("top")
    cax1.xaxis.set_ticklabels(
        ["$-12.5$", "$-11.5$", "$-10.5$", "$-9.5$"],
        fontsize=LABEL_SIZE*0.9,
        color="black",
    )
    cax1.set_xlabel(
        "log $\\rho \, v^2$ [dyne cm$^{-2}$]",
        fontsize=LABEL_SIZE*0.9,
        labelpad=-80,
    )
    cax1.xaxis.set_tick_params(color="black", length=10.0)

    # Right
    divider = make_axes_locatable(ax[2])
    cax2 = divider.append_axes("top", size="5%", pad=-0.05)
    cbar2 = plt.colorbar(
        mappable2,
        cax=cax2,
        use_gridspec=True,
        ticks=[-12, -11, -10, -9],
        orientation="horizontal",
    )
    cax2.xaxis.set_ticks_position("top")
    cax2.xaxis.set_ticklabels(
        ["$-12$", "$-11$", "$-10$", "$-9$"],
        fontsize=LABEL_SIZE*0.9,
        color="black",
    )
    cax2.set_xlabel(
        "log $n_{\\rm ^{60}Fe}$ [atoms cm$^{-3}$]",
        fontsize=LABEL_SIZE*0.9,
        labelpad=-80,
    )
    cax2.xaxis.set_tick_params(color="black", length=10.0)

    for i in range(3):
        ax[i].axvline(
            x=t_sf(1e51, dens), color="deepskyblue", lw=1.5, dashes=(8, 2), alpha=1.0
        )
        ax[i].plot(
            time_plot_arr,
            r_analytical * 1e3,
            color="orange",
            lw=3,
            dashes=(3, 3),
            zorder=10,
        )

    ax[0].text(
        0.15,
        0.96,
        f"{value}",
        ha="left",
        va="top",
        transform=ax[0].transAxes,
        fontsize=LABEL_SIZE,
        color="white",
    )

    ax[0].set_ylabel("Distance to SN [pc]", fontsize=LABEL_SIZE)
    ax[0].yaxis.set_tick_params(labelsize=LABEL_SIZE, labelcolor="black")

    for i in range(3):
        ax[i].set_xlim(0, 3.8)
        ax[i].set_ylim(0, 230)
        ax[i].set_xticks([0.5, 1.5, 2.5, 3.5])
        ax[i].xaxis.set_tick_params(labelsize=LABEL_SIZE, labelcolor="black")
        ax[i].set_xlabel("Time since SN [Myr]", fontsize=LABEL_SIZE)

    plt.savefig("./images/time_dist_flux_map.pdf", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":

    N_bins = 151
    r_max = 0.3  # kpc
    n_snapshot_max_arg = 133
    path = "../run01_diffusion_new_one_p_with_centre_higres_sphere/"
    dens = 0.1  # H/cm3
    value = "high\_res\_n01"

    print("Max number of snapshots: {:d}".format(n_snapshot_max_arg))
    print(f"Path to snapshots: {path}")
    print("Number of radial bins: {:d}".format(N_bins))
    print("Max radial coordinate: {:.1f} [pc]".format(r_max * 1e3))
    print("Box density: {:.2e} [cm^-3]".format(dens))

    magnitudes = np.linspace(0.0, r_max, N_bins)
    magnitudes_centres = 0.5 * (magnitudes[:-1] + magnitudes[1:])
    magnitudes_widths = magnitudes[1:] - magnitudes[:-1]

    # For analytical solution
    analytic_time_bins = 500
    analytic_space_bins = 500
    time_plot_arr = np.linspace(0.0, 3.9, analytic_time_bins)
    r_analytical = np.zeros(analytic_time_bins)

    save_file_name = "./data/Flux_vs_time_and_distance_img"

    do_sedov_analytic(analytic_time_bins, dens)

    try:
        input_file = np.load(f"{save_file_name}.npz")
        profiles, time_arr = (input_file["arr_0"], input_file["arr_1"])
        print(np.shape(time_arr), n_snapshot_max_arg)
        assert np.shape(time_arr) == (n_snapshot_max_arg,)

    except (IOError, AssertionError):
        # For numerical solution
        time_arr = np.zeros(n_snapshot_max_arg)
        profiles = np.zeros((n_snapshot_max_arg, N_bins - 1, 3))
        process_data(path, n_snapshot_max_arg, save_file_name)

    plot_data()
