import matplotlib.pylab as plt
import numpy as np
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


def plot_profile(dens=0.101348, gamma_WendlandC2: float = 1.936492):
    global magnitudes, magnitudes_centres, magnitudes_widths

    # Create figure

    COL_SIZE, ROW_SIZE = 2, 1
    fig, ax = plt.subplots(
        COL_SIZE,
        ROW_SIZE,
        figsize=(8, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.5]},
    )
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["axes.linewidth"] = 2

    for i in range(COL_SIZE):
        ax[i].tick_params(which="both", width=1.7)
        ax[i].tick_params(which="major", length=9)
        ax[i].tick_params(which="minor", length=5)
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].tick_params(
            axis="both",
            which="both",
            pad=8,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )

    ax[0].set_ylabel("log $n_{\\rm ^{60}Fe} \\rm \\, [cm^{-3}]$", fontsize=LABEL_SIZE)
    ax[0].set_ylim(-12.2, -8.8)
    ax[0].yaxis.set_tick_params(labelsize=LABEL_SIZE)

    ax[1].set_ylabel(
        "$M_{\\rm ^{60}Fe}(<D) / M_{\\rm ^{60}Fe, tot}$", fontsize=LABEL_SIZE
    )
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_xlabel("Distance from SN $D$ [pc]", fontsize=LABEL_SIZE)
    ax[1].xaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax[1].yaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax[1].set_xlim(-2, 242)

    # For numerical methods
    dict_sim = {
        "../run01_diffusion_new_one_p_with_centre_higres_highdiff": "high\_res\_n01\_highdiff",
        "../run01_diffusion_new_one_p_with_centre_higres_sphere": "high\_res\_n01",
        "../run01_diffusion_new_one_p_with_centre_higres_lowerdiff": "high\_res\_n01\_lowdiff",
        "../run01_diffusion_new_one_p_with_centre_higres_nodiff_sphere": "high\_res\_n01\_nodiff",
    }

    idx = 17 * 7  # t = 3.5 Myr

    for counter, (key, value) in enumerate(dict_sim.items()):
        with h5.File(f"{key}" + "/output_{:04d}.hdf5".format(idx), "r") as f:
            # Internal units
            unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
            unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

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
            r_max = 300.0  # 1 pc per bin
            magnitudes = np.linspace(0.0, r_max, N_bins)
            magnitudes_centres = (magnitudes[:-1] + magnitudes[1:]) / 2
            magnitudes_widths = magnitudes[1:] - magnitudes[:-1]

            print("Snapshot time: \t {:.3f} kyr".format(time[0]))
            print("Hydrogen mass fratcion: \t {:.3f}".format(X_gas_median))
            print("Bin witdh: {:.2f} [pc]".format(np.mean(magnitudes_widths)))

            # Gas positions
            gas_pos = (
                f["/PartType0/Coordinates"][:, :]
                * unit_length_in_cgs
                / constants["PARSEC_IN_CGS"]
            )
            gas_pos[:, 0] -= centre[0] * unit_length_in_cgs / constants["PARSEC_IN_CGS"]
            gas_pos[:, 1] -= centre[1] * unit_length_in_cgs / constants["PARSEC_IN_CGS"]
            gas_pos[:, 2] -= centre[2] * unit_length_in_cgs / constants["PARSEC_IN_CGS"]

            r = np.sqrt(
                np.sum(gas_pos * gas_pos, axis=1)
            )  # Distance from the explosion

            # Gas mass
            gas_m = f["/PartType0/Masses"][:]

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
            ejecta_mass = 1e-4 / np.mean(gas_m) * ejecta / ejecta_normalisation * gas_m

            m_bin = bin_distribute(r, gas_h * gamma_WendlandC2, ejecta_mass)
            m_bin_tot = np.sum(m_bin)

            print("Total Mass to bin", np.sum(ejecta_mass))
            print("Total Mass in Fe60", m_bin_tot)
            print("Sum of ejecta mass fractions", np.sum(ejecta))

            # cgs units
            bin_volume = (
                4.0
                * np.pi
                / 3.0
                * (magnitudes[1:] ** 3 - magnitudes[:-1] ** 3)
                * constants["PARSEC_IN_CGS"] ** 3
            )

            # Divide by total ejecta mass by volume then by A=60, then from Msola to grams then divide m_p
            n_Fe60_bin = (
                m_bin
                / bin_volume
                / 60.0
                * constants["SOLAR_MASS_IN_CGS"]
                / constants["PROTON_MASS_IN_CGS"]
            )

            # Numerical solution (binned)
            ax[0].plot(
                magnitudes_centres,
                np.log10(n_Fe60_bin),
                lw=line_properties["linewidth"][counter],
                color=line_properties["colour"][counter],
                label=value,
            )

            # Numerical solution (binned)
            ax[1].plot(
                magnitudes_centres,
                np.cumsum(m_bin) / m_bin_tot,
                lw=line_properties["linewidth"][counter],
                color=line_properties["colour"][counter],
            )

            if counter == 0:
                m_bin = bin_distribute(r, gas_h * gamma_WendlandC2, gas_m)

                # Divide by total ejecta mass by volume then by A=60, then from Msolar to grams then divide m_p
                n_bin = m_bin / bin_volume
                n_bin /= np.max(n_bin)
                n_bin *= 1e-9

                ax[0].plot(
                    magnitudes_centres, np.log10(n_bin), lw=2, color="k", dashes=(3, 3)
                )

    ax[0].legend(loc="lower right", fontsize=LEGEND_SIZE * 0.75, frameon=False)

    plt.savefig("./images/nFe_conv.pdf", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    plot_profile()
    # n_sample = 10
    # r_edges = np.linspace(0.0, 1.0, n_sample + 1)
    # r_points = 0.5 * (r_edges[:-1] + r_edges[1:])
    # kernel_at_r = wendlandC2(r_points)
    # dV = 4 * np.pi * r_points[:] ** 2 * (r_edges[1:] - r_edges[:-1])
    # print(np.sum(dV * kernel_at_r))
