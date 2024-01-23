import numpy as np
import h5py as h5
import sys
from scipy import stats
from plot_style import *
from constants import *
import json
from scipy import stats, interpolate
from matplotlib import ticker

from swiftsimio.visualisation.projection import scatter as scatter

# from swiftsimio.visualisation.projection_backends.histogram import (
#    scatter as scatter_hist,
# )


def load_dib_2006_data(path="../Dib2006data.dat"):
    dib_data = np.loadtxt(path)
    print(np.shape(dib_data))
    x, y = dib_data[:, 0], dib_data[:, 1]
    sigma_th = 12.0  # km/s
    y = np.sqrt(y**2 + sigma_th**2.0)
    return x, y


def load_zhou_2017_data(path="../Zhou2017data.dat"):
    zh_data = np.loadtxt(path)
    print(np.shape(zh_data))
    x, x_err, y, y_err = zh_data[:, 2], zh_data[:, 3], zh_data[:, 0], zh_data[:, 1]
    return x, x_err, y, y_err


def load_law_2022_resolved(path="../Law2022data.dat"):
    law_data = np.loadtxt(path)
    print(np.shape(law_data))
    x, y, y_err = law_data[:, 0], law_data[:, 1], law_data[:, 2]
    return x, y, y_err


def load_zhou_2017_resolved(path="../Zhou2017Resolved.dat"):
    zh_data = np.loadtxt(path)
    print(np.shape(zh_data))
    x, y = zh_data[:, 0], zh_data[:, 1]
    size = int(len(x) / 2)

    min_x = np.max([x[0], x[size]])
    max_x = np.min([x[-1 - size], x[-1]])

    print(min_x, max_x)

    top_f = interpolate.interp1d(x[:size], y[:size])
    bottom_f = interpolate.interp1d(x[size:], y[size:])

    x_sample = np.logspace(np.log10(min_x), np.log10(max_x), 10)
    y_bottom = bottom_f(x_sample)
    y_top = top_f(x_sample)

    return x_sample, y_bottom, y_top


def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def plot():
    gamma = 2.018932

    bin_edges = np.arange(-4, -0.55, 0.1)  # log Sigma SFR
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    runs = read_sim_data("main.json")

    for script_name, plots in runs.items():
        print(script_name)

        if script_name == sys.argv[0]:
            print("FOUND")

            for plot in plots:
                output = plot["output_file"]
                dict_sim = plot["data"]
                split = plot["split"]
                snapshot_min = plot["snapshot_min"]
                snapshot_max = plot["snapshot_max"]

                # dib_x, dib_y = load_dib_2006_data()
                zh_x, zh_y_b, zh_y_t = load_zhou_2017_resolved()
                law_x, law_y, law_y_err = load_law_2022_resolved()

                fig, ax = plot_style(8, 8)

                l1 = ax.fill_between(
                    np.log10(zh_x), zh_y_b, zh_y_t, color="grey", alpha=0.4
                )

                l2 = ax.errorbar(
                    law_x,
                    law_y,
                    yerr=law_y_err,
                    fmt="o-",
                    c="maroon",
                    markersize=10,
                )

                output_dict = {}

                for counter, (key, value) in enumerate(dict_sim.items()):
                    x_full = np.array([])
                    y_full = np.array([])

                    for snp in range(snapshot_min, snapshot_max):
                        print("snp", snp)

                        f = h5.File(value + "/output_{:04d}.hdf5".format(snp), "r")

                        unit_length_in_cgs = f["/Units"].attrs[
                            "Unit length in cgs (U_L)"
                        ]
                        unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
                        unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

                        if snp == snapshot_min:
                            time = f["/Header"].attrs["Time"]
                            time_min = (
                                time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6
                            )
                            print("SETTING TMIN", time_min)
                        elif snp == snapshot_max - 1:
                            time = f["/Header"].attrs["Time"]
                            time_max = (
                                time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6
                            )
                            print("SETTING TMAX", time_max)

                        boxsize_pc = (
                            f["/Header"].attrs["BoxSize"]
                            * unit_length_in_cgs
                            / constants["PARSEC_IN_CGS"]
                        )
                        centre_pc = boxsize_pc / 2.0

                        gas_pos_pc = (
                            f["/PartType0/Coordinates"][:, :]
                            * unit_length_in_cgs
                            / constants["PARSEC_IN_CGS"]
                        )
                        gas_pos_pc -= centre_pc

                        gas_hsml_pc = (
                            f["/PartType0/SmoothingLengths"][:]
                            * unit_length_in_cgs
                            / constants["PARSEC_IN_CGS"]
                            * gamma
                        )
                        gas_mass_Msun = (
                            f["/PartType0/Masses"][:]
                            * unit_mass_in_cgs
                            / constants["SOLAR_MASS_IN_CGS"]
                        )

                        # X_H = f["/PartType0/ElementMassFractions"][:, 0]

                        HI_frac = f["/PartType0/SpeciesFractions"][:, 0]
                        gas_HI_M_Msun = gas_mass_Msun * HI_frac

                        gas_SFR = (
                            f["/PartType0/StarFormationRates"][:]
                            * unit_mass_in_cgs
                            / unit_time_in_cgs
                            * constants["YEAR_IN_CGS"]
                            / constants["SOLAR_MASS_IN_CGS"]
                        )
                        gas_SFR[gas_SFR < 0.0] = 0.0

                        gas_VZ_kms = f["/PartType0/Velocities"][:, 2] * (
                            unit_length_in_cgs / 1e5 / unit_time_in_cgs
                        )

                        size_pc = 21.0e3  # pc
                        pixel_pc = 1.0e3  # pc

                        N_pix = int(size_pc * 2 / pixel_pc)
                        print("Npix", N_pix)

                        mask_x = np.logical_and(
                            gas_pos_pc[:, 0] > -size_pc, gas_pos_pc[:, 0] < size_pc
                        )
                        mask_y = np.logical_and(
                            gas_pos_pc[:, 1] > -size_pc, gas_pos_pc[:, 1] < size_pc
                        )
                        mask_z = np.logical_and(
                            gas_pos_pc[:, 2] > -size_pc * 200,
                            gas_pos_pc[:, 2] < size_pc * 200,
                        )
                        mask_xy = np.logical_and(mask_x, mask_y)
                        mask_xyz = np.logical_and(mask_xy, mask_z)
                        mask_xyz = np.logical_and(
                            mask_xyz, np.abs(gas_VZ_kms) < 1e7
                        )  # km/s
                        mask_xyz = np.logical_and(mask_xyz, HI_frac > 1e-7)

                        # Mapping from [-size_pc, size_pc] to [0, 1]
                        gas_x = gas_pos_pc[:, 0][mask_xyz] / size_pc / 2.0 + 0.5
                        gas_y = gas_pos_pc[:, 1][mask_xyz] / size_pc / 2.0 + 0.5
                        # Rescale smoothing length in the same way
                        gas_h = gas_hsml_pc[mask_xyz] / size_pc / 2.0

                        # Select only the gas that satisfies the masking criteria
                        gas_HI_M_Msun_masked = gas_HI_M_Msun[mask_xyz]
                        gas_sfr_masked = gas_SFR[mask_xyz]
                        gas_vz_masked = gas_VZ_kms[mask_xyz]

                        x_val = scatter(
                            x=gas_x, y=gas_y, h=gas_h, m=gas_sfr_masked, res=N_pix
                        )
                        # Rescale back to kpc ** 2
                        x_val = x_val / (size_pc * 2.0 / 1e3) ** 2

                        # Count number of particles per pixel
                        counts = scatter(
                            x=gas_x,
                            y=gas_y,
                            h=np.zeros_like(gas_h),
                            m=gas_mass_Msun[mask_xyz],
                            res=N_pix,
                        )

                        print("min count (before corr)", np.min(counts[counts > 0.0]))

                        counts /= N_pix**2.0
                        mask_gas = (
                            counts > 1e5 * 20.0
                        )  # 20 particels with mass 1e5 Msun

                        print("min count (after corr)", np.min(counts[counts > 0.0]))
                        print(
                            "counts",
                            np.size(counts[mask_gas]),
                            np.size(counts[mask_gas]) / N_pix**2.0,
                        )

                        mass_density = scatter(
                            x=gas_x,
                            y=gas_y,
                            h=np.zeros_like(gas_h),
                            m=gas_HI_M_Msun_masked,
                            res=N_pix,
                        )

                        vel_mean_mass = scatter(
                            x=gas_x,
                            y=gas_y,
                            h=np.zeros_like(gas_h),
                            m=gas_vz_masked * gas_HI_M_Msun_masked,
                            res=N_pix,
                        )
                        vel2_mass = scatter(
                            x=gas_x,
                            y=gas_y,
                            h=np.zeros_like(gas_h),
                            m=(gas_vz_masked**2) * gas_HI_M_Msun_masked,
                            res=N_pix,
                        )

                        vel_mean = vel_mean_mass[mask_gas] / mass_density[mask_gas]
                        vel2 = vel2_mass[mask_gas] / mass_density[mask_gas]
                        x_val = x_val[mask_gas]

                        sigma_th2 = 9.0**2.0  # (km/s)**2
                        y_val_raw = vel2 + sigma_th2 - vel_mean**2.0

                        print(np.mean(vel_mean), np.max(vel_mean))

                        y_val = np.sqrt(y_val_raw)
                        x_val = x_val.flatten()
                        y_val = y_val.flatten()

                        x_full = np.concatenate([x_full, x_val])
                        y_full = np.concatenate([y_full, y_val])

                    bin_values, _, _ = stats.binned_statistic(
                        np.log10(x_full),
                        y_full,
                        bins=bin_edges,
                        statistic="median",
                    )

                    deviations = []
                    mask_bins = []
                    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
                        mask = np.logical_and(
                            np.log10(x_full) > left, np.log10(x_full) <= right
                        )
                        vals = y_full[mask]
                        if np.size(vals) >= 2:
                            deviations.append(np.percentile(vals, [16, 84]))
                            mask_bins.append(True)
                        else:
                            mask_bins.append(False)

                    deviations = np.array(deviations)

                    if split == 2:
                        colors = color2
                        lw = lw2
                        dashes = dashes2
                    elif split == 4:
                        colors = color4
                        lw = lw4
                        dashes = dashes4
                    else:
                        colors = line_properties["colour"]
                        lw = line_properties["linewidth"]
                        dashes = line_properties["dashes"]

                    ax.plot(
                        bin_centers[mask_bins],
                        bin_values[mask_bins],
                        label=key.replace("_", "\_"),
                        color=colors[counter],
                        lw=lw[counter],
                        dashes=tuple(d for d in dashes[counter]),
                    )

                    if counter == 1:
                        ax.fill_between(
                            bin_centers[mask_bins],
                            deviations[:, 0],
                            deviations[:, 1],
                            edgecolor=colors[counter],
                            alpha=0.60,
                            hatch=r"\ ",
                            facecolor="none",
                        )

                    output_dict[key] = [
                        bin_centers[mask_bins],
                        bin_values[mask_bins],
                        deviations[:, 0],
                        deviations[:, 1],
                    ]

                leg1 = plt.legend(
                    loc="upper left",
                    fontsize=27.0,
                    scatterpoints=3,
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    columnspacing=0.25,
                    borderaxespad=0.20,
                    labelspacing=0.3,
                )

                print("Setting legend")
                hl_dict = {handle.get_label(): handle for handle in leg1.legendHandles}

                for k in hl_dict:
                    hl_dict[k].set_color("white")

                for counter, text in enumerate(leg1.get_texts()):
                    text.set_color(colors[counter])

                leg2 = plt.legend(
                    [l1, l2],
                    ["Zhou et al (2017)", "Law et al (2022)"],
                    loc="lower right",
                    frameon=False,
                    fontsize=20.0,
                    scatterpoints=3,
                )

                plt.gca().add_artist(leg1)

                # Axes & labels
                ax.xaxis.set_tick_params(labelsize=30)
                ax.yaxis.set_tick_params(labelsize=30)

                ax.set_ylim(0, 55)
                ax.set_xlim(-4, -0.6)

                ax.set_xlabel(
                    "log $\\Sigma_{\\rm SFR}$ [$\\rm M_\\odot$ yr$^{-1}$ kpc$^{-2}$]",
                    fontsize=LABEL_SIZE,
                )

                ax.set_ylabel(
                    "$\\sigma_{\\rm gas, obs}$ [km s$^{-1}$]",
                    fontsize=30,
                )

                ax.text(
                    0.04,
                    0.04,
                    "${:.1f} < t < {:.1f}$ Gyr".format(
                        time_min[0] / 1e3, time_max[0] / 1e3
                    ),
                    ha="left",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=24,
                )

                np.savez(output.replace(".pdf", ".npz"), **output_dict)
                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    plot()
