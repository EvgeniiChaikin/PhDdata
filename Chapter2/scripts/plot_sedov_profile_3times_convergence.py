import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
import h5py as h5
from constants import *
from plot_style import *
from scipy.interpolate import interp1d
from scipy import stats
from tqdm import tqdm


def wendlandC2(u):
    # Normalised
    return (
        21.0
        / (2 * np.pi)
        * np.where(
            u > 1.0,
            0.0,
            4.0 * u**5 - 15.0 * u**4 + 20.0 * u**3 - 10 * u**2 + 1.0,
        )
    )


# 20
def intersection_volume_vector(rp, hp_full, w, r1, r2, n_sample: int = 20):
    """
    Parameters:
    -------------------------
    rp: float
    Particle radius

    hp_full: float
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

    r_edges = np.linspace(0.0, 1.0, n_sample + 1)
    r_points = 0.5 * (r_edges[:-1] + r_edges[1:])
    kernel_at_r = wendlandC2(r_points)
    Volumep_full = 4.0 * np.pi / 3.0 * hp_full**3

    v_shell_overlap = np.zeros((np.size(rp), n_sample))

    for n_subshell, (r_point, r_edge_right) in enumerate(zip(r_points, r_edges[1:])):
        hp = hp_full * r_edge_right

        Volumep = 4.0 * np.pi / 3.0 * hp**3

        for sign, rb in zip([-1.0, 1.0], [r1, r2]):
            case2 = np.logical_and(rp + hp > rb, np.abs(rp - hp) < rb)
            case3 = np.logical_and(
                np.logical_and(rp + hp <= rb, rp - hp >= -rb), np.logical_not(case2)
            )
            case4 = np.logical_and(
                np.logical_or(rp + hp < rb, rp - hp < -rb),
                np.logical_not(np.logical_or(case3, case2)),
            )

            # Case2
            alphab = (rb**2 + rp[case2] ** 2 - hp[case2] ** 2) / (
                2.0 * rp[case2] * rb
            )
            alphap = (hp[case2] ** 2 + rp[case2] ** 2 - rb**2) / (
                2.0 * rp[case2] * hp[case2]
            )

            heightb = rb * (1.0 - alphab)
            heightp = hp[case2] * (1.0 - alphap)

            cap_volumeb = np.pi * heightb**2 / 3.0 * (3.0 * rb - heightb)
            cap_volumep = np.pi * heightp**2 / 3.0 * (3.0 * hp[case2] - heightp)

            v_shell_overlap[case2, n_subshell] += (
                sign * (cap_volumeb + cap_volumep) * w[case2]
            )

            # Case3
            v_shell_overlap[case3, n_subshell] += sign * w[case3] * Volumep[case3]

            # Case4
            Volumeb = 4.0 * np.pi / 3.0 * rb**3
            v_shell_overlap[case4, n_subshell] += sign * Volumeb * w[case4]

        # Particle is inside the bin
        case1 = np.logical_and(rp - hp >= r1, rp + hp <= r2)
        v_shell_overlap[case1, n_subshell] = 1.0 * w[case1] * Volumep[case1]

    for i in range(n_sample - 1, 0, -1):
        v_shell_overlap[:, i] -= v_shell_overlap[:, i - 1]

    output = (
        np.sum(4.0 * np.pi / 3.0 * v_shell_overlap * kernel_at_r, axis=1) / Volumep_full
    )

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
                hp_full=h[in_bin],
                w=w[in_bin],
                r1=magnitudes[bin_idx],
                r2=magnitudes[bin_idx + 1],
            )

            bin_values[bin_idx] += np.sum(weight)

    return bin_values


def plot_profile(dens=0.101, gamma_WendlandC2: float = 1.936492):
    global magnitudes, magnitudes_centres, magnitudes_widths

    # Create figure

    COL_SIZE, ROW_SIZE = 3, 3
    fig, ax = plt.subplots(COL_SIZE, ROW_SIZE, figsize=(15, 13), sharex=True)
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

    for j in range(ROW_SIZE):
        ax[2, j].set_xlabel("Distance from SN [pc]", fontsize=LABEL_SIZE)

    ax[0, 0].set_ylabel("$n_{\\rm H} \\, \\rm [cm^{-3}]$", fontsize=LABEL_SIZE)
    ax[1, 0].set_ylabel("$v \\, \\rm [km \\, s^{-1}]$", fontsize=LABEL_SIZE)
    ax[2, 0].set_ylabel(
        "log $n_{\\rm ^{60}Fe} \\rm \\, [cm^{-3}]$", fontsize=LABEL_SIZE
    )

    # Increase label sizes
    for j in range(ROW_SIZE):
        ax[2, j].xaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax[1, j].yaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax[2, j].yaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax[0, j].yaxis.set_tick_params(labelsize=LABEL_SIZE)

        ax[0, j].set_xlim(-2, 242)
        ax[0, j].set_ylim(-0.02, 1.02)
        ax[1, j].set_xlim(-2, 242)
        ax[1, j].set_ylim(-15.2, 65.2)
        ax[2, j].set_xlim(-2, 242)
        ax[2, j].set_ylim(-12.2, -8.8)

        if j:
            plt.setp(ax[0, j].get_yticklabels(), visible=False)
            plt.setp(ax[1, j].get_yticklabels(), visible=False)
            plt.setp(ax[2, j].get_yticklabels(), visible=False)

    # For numerical methids
    dict_sim = {
        "../run01_diffusion_new_one_p_with_centre_higres_sphere": "high\_res\_n01",
        "../run01_diffusion_new_one_p_with_centre_midres_sphere": "mid\_res\_n01",
        "../run01_diffusion_new_one_p_with_centre_lowres_sphere": "low\_res\_n01",
    }

    for sim, (key, value) in enumerate(dict_sim.items()):
        print(sim, key, value)
        for counter, idx in enumerate([17, 17 * 4, 17 * 7]):  # 119
            print(counter, idx)
            with h5.File(f"{key}" + "/output_{:04d}.hdf5".format(idx), "r") as f:
                # Internal units
                unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
                unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
                unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]
                unit_energy_in_cgs = (
                    unit_mass_in_cgs * unit_length_in_cgs**2 / unit_time_in_cgs**2
                )

                # Snapshot data
                time = (
                    f["/Header"].attrs["Time"]
                    * unit_time_in_cgs
                    / constants["YEAR_IN_CGS"]
                    / 1e3
                )  # kyr

                boxsize = f["/Header"].attrs["BoxSize"]
                centre = boxsize / 2.0

                # Get the median H mass function in the ISM in the simulation
                X_gas = f["/PartType0/ElementMassFractions"][:, 0]
                X_gas_median = np.median(X_gas)

                # Binning
                N_bins = 301
                r_max = 300.0  # pc
                magnitudes = np.linspace(0.0, r_max, N_bins)
                magnitudes_centres = (magnitudes[:-1] + magnitudes[1:]) / 2
                magnitudes_widths = magnitudes[1:] - magnitudes[:-1]

                print("Snapshot time: \t {:.3f} kyr".format(time[0]))
                print("Hydrogen mass fratcion: \t {:.3f}".format(X_gas_median))
                print("Bin witdh: {:.2f} [pc]".format(np.mean(magnitudes_widths)))

                # Get the analytical solution
                r_s, P_s, rho_s, v_s, r_shock, _, _, _, _ = sedov(
                    time
                    * constants["YEAR_IN_CGS"]
                    * 1e3,  # Time since SN went off [kpc]
                    1e51,  # Energy [erg]
                    dens
                    / (
                        X_gas_median / constants["PROTON_MASS_IN_CGS"]
                    ),  # Density of the ISM [H part per cm^3]
                    5.0 / 3.0,  # Gamma (equation of state)
                    1000,  # Binning
                    3,
                )  # Dimentions

                # The solution is only within the shock-heated region
                # Append two points to extend the solution beyond the shock
                r_s = np.append(r_s, r_shock + 0.0001)
                r_s = np.append(r_s, r_shock * 20.0)

                rho_s = np.insert(
                    rho_s,
                    np.size(rho_s),
                    [
                        dens / (X_gas_median / constants["PROTON_MASS_IN_CGS"]),
                        dens / (X_gas_median / constants["PROTON_MASS_IN_CGS"]),
                    ],
                )

                v_s = np.insert(
                    v_s,
                    np.size(v_s),
                    [
                        0.0,
                        0.0,
                    ],
                )

                # Analytical solution
                ax[0, sim].plot(
                    r_s / constants["PARSEC_IN_CGS"],
                    rho_s * (X_gas_median / constants["PROTON_MASS_IN_CGS"]),
                    lw=2,
                    color=line_properties["colour"][counter],
                    dashes=(3, 3),
                    zorder=10,
                )

                # Analytical solution
                ax[1, sim].plot(
                    r_s / constants["PARSEC_IN_CGS"],
                    v_s / 1e5,
                    lw=2,
                    color=line_properties["colour"][counter],
                    dashes=(3, 3),
                )

                ax[2, sim].axvline(
                    x=r_shock / constants["PARSEC_IN_CGS"],
                    lw=2,
                    color=line_properties["colour"][counter],
                    dashes=(3, 3),
                )

                # Gas positions
                gas_pos = (
                    f["/PartType0/Coordinates"][:, :]
                    * unit_length_in_cgs
                    / constants["PARSEC_IN_CGS"]
                )
                gas_pos[:, 0] -= (
                    centre[0] * unit_length_in_cgs / constants["PARSEC_IN_CGS"]
                )
                gas_pos[:, 1] -= (
                    centre[1] * unit_length_in_cgs / constants["PARSEC_IN_CGS"]
                )
                gas_pos[:, 2] -= (
                    centre[2] * unit_length_in_cgs / constants["PARSEC_IN_CGS"]
                )

                gas_density = (
                    f["/PartType0/Densities"][:]
                    * unit_mass_in_cgs
                    / unit_length_in_cgs**3
                    * X_gas
                    / constants["PROTON_MASS_IN_CGS"]
                )

                median_gas_density = np.median(gas_density)
                print("Median gas density", median_gas_density)

                r = np.sqrt(
                    np.sum(gas_pos * gas_pos, axis=1)
                )  # Distance from the explosion

                # Gas mass
                gas_m = f["/PartType0/Masses"][:]

                # Velocity
                gas_v = f["/PartType0/Velocities"][:]
                r = np.sqrt(
                    np.sum(gas_pos * gas_pos, axis=1)
                )  # Distance from the explosion
                v_r = np.sum(gas_pos * gas_v, axis=1) / r

                # Smoothing length
                gas_h = (
                    f["/PartType0/SmoothingLengths"][:]
                    * unit_length_in_cgs
                    / constants["PARSEC_IN_CGS"]
                )

                # Ejecta mass fraction
                ejecta = f["/PartType0/ElementMassFractions"][:, -1]

                # We want to normalise ejecta becasue the value of Fe60 ejected is aribtrary chosen
                ejecta_normalisation = np.sum(ejecta[np.where(ejecta > 0.0)])

                # We want 1e-4 Msun total in Fe60
                ejecta_mass = (
                    1e-4 / np.mean(gas_m) * ejecta / ejecta_normalisation * gas_m
                )

                # Gas mass
                m_bin_full = bin_distribute(r, gas_h * gamma_WendlandC2, gas_m)
                print("Computed gas mass")

                # Ejecta mass
                m_bin = bin_distribute(r, gas_h * gamma_WendlandC2, ejecta_mass)
                print("Computed ejecta mass")

                # Radial velocity
                v_bin = (
                    bin_distribute(r, gas_h * gamma_WendlandC2, v_r * gas_m)
                    / m_bin_full
                )
                print("Computed gas velocity")

                # cgs units
                bin_volume = (
                    4.0
                    * np.pi
                    / 3.0
                    * (magnitudes[1:] ** 3 - magnitudes[:-1] ** 3)
                    * constants["PARSEC_IN_CGS"] ** 3
                )

                rho_bin = (
                    m_bin_full
                    / bin_volume
                    * X_gas_median
                    * constants["SOLAR_MASS_IN_CGS"]
                    / constants["PROTON_MASS_IN_CGS"]
                )

                # Divide by total ejecta mass by volume then by A=60, then from Msola to grams then divide m_p
                n_Fe60_bin = (
                    m_bin
                    / bin_volume
                    / 60.0
                    * constants["SOLAR_MASS_IN_CGS"]
                    / constants["PROTON_MASS_IN_CGS"]
                )

                # Numerical solutuion (binned)
                ax[0, sim].plot(
                    magnitudes_centres,
                    rho_bin,
                    lw=line_properties["linewidth"][counter],
                    color=line_properties["colour"][counter],
                )
                # Gas mass
                ax[1, sim].plot(
                    magnitudes_centres,
                    v_bin,
                    lw=line_properties["linewidth"][counter],
                    color=line_properties["colour"][counter],
                    label=value,
                )

                args = np.where(n_Fe60_bin > 0.0)
                if sim:
                    ax[2, sim].plot(
                        magnitudes_centres[args],
                        np.log10(n_Fe60_bin[args]),
                        lw=line_properties["linewidth"][counter],
                        color=line_properties["colour"][counter],
                        label=f"$t = {time[0]/1e3:.1f}$ Myr",
                    )
                else:
                    ax[2, sim].plot(
                        magnitudes_centres[args],
                        np.log10(n_Fe60_bin[args]),
                        lw=line_properties["linewidth"][counter],
                        color=line_properties["colour"][counter],
                    )

        ax[0, sim].text(
            0.04,
            0.96,
            f"{value}",
            ha="left",
            va="top",
            transform=ax[0, sim].transAxes,
            fontsize=LEGEND_SIZE * 1.0,
        )

    ax[2, 0].plot([-30, -40], [-30, -40], lw=3, color="k", label="Numerical")
    ax[2, 0].plot(
        [-30, -40], [-30, -40], lw=3, color="k", dashes=(3, 3), label="ST solution"
    )

    ax[2, 0].legend(loc="upper right", fontsize=LEGEND_SIZE * 0.75)
    ax[2, 2].legend(loc="upper right", fontsize=LEGEND_SIZE * 0.75)

    plt.savefig(
        "./images/sedov_radial_profile_3_times_conv.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )


if __name__ == "__main__":
    plot_profile()
