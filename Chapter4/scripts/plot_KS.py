import numpy as np
import h5py as h5
import sys
from scipy import stats
from plot_style import *
from constants import *
import json
from scipy import stats
from matplotlib import ticker
from swiftsimio.visualisation.projection import scatter
from loadObservationalData import read_obs_data


def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# Get the default KS relation for correct IMF
def KS(sigma_g, n, A):
    return A * sigma_g**n


def plot():
    bin_edges = np.arange(-1, 3, 0.25)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    runs = read_sim_data("main.json")

    for script_name, plots in runs.items():
        print(script_name)

        if script_name == sys.argv[0]:
            print("FOUND")

            for plot in plots:
                output = plot["output_file"]
                dict_sim = plot["data"]
                snapshot_min = plot["snapshot_min"]
                snapshot_max = plot["snapshot_max"]
                split = plot["split"]

                time_min, time_max = 0.0, 0.0

                print("Min snp", snapshot_min, "N snp", snapshot_max - snapshot_min)

                fig, ax = plot_style(8, 8)

                Sigma_g = np.logspace(1, 3, 1000)
                Sigma_star = KS(Sigma_g, 1.4, 1.515e-4)

                (l0,) = ax.plot(
                    np.log10(Sigma_g),
                    np.log10(Sigma_star),
                    color="grey",
                    dashes=(10, 3, 2, 3),
                    lw=2,
                )

                observational_data = read_obs_data()

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

                        boxsize = (
                            f["/Header"].attrs["BoxSize"]
                            * unit_length_in_cgs
                            / constants["PARSEC_IN_CGS"]
                        )
                        centre = boxsize / 2.0

                        gas_pos = (
                            f["/PartType0/Coordinates"][:, :]
                            * unit_length_in_cgs
                            / constants["PARSEC_IN_CGS"]
                        )
                        gas_pos -= centre

                        gas_hsml = (
                            f["/PartType0/SmoothingLengths"][:]
                            * unit_length_in_cgs
                            / constants["PARSEC_IN_CGS"]
                            * 2.0
                        )

                        gas_mass = (
                            f["/PartType0/Masses"][:]
                            * unit_mass_in_cgs
                            / constants["SOLAR_MASS_IN_CGS"]
                        )
                        gas_HI = (
                            f["/PartType0/AtomicHydrogenMasses"][:]
                            * unit_mass_in_cgs
                            / constants["SOLAR_MASS_IN_CGS"]
                        )
                        gas_H2 = (
                            f["/PartType0/MolecularHydrogenMasses"][:]
                            * unit_mass_in_cgs
                            / constants["SOLAR_MASS_IN_CGS"]
                        )

                        gas_SFR = (
                            f["/PartType0/StarFormationRates"][:]
                            * unit_mass_in_cgs
                            / unit_time_in_cgs
                            * constants["YEAR_IN_CGS"]
                            / constants["SOLAR_MASS_IN_CGS"]
                        )
                        gas_SFR[gas_SFR < 0.0] = 0.0

                        size = 21e3  # pc
                        pixel = 0.75e3  # pc
                        N_pix = int(size * 2 / pixel)
                        print("NPIXELS", N_pix)

                        mask_x = np.logical_and(
                            gas_pos[:, 0] > -size, gas_pos[:, 0] < size
                        )
                        mask_y = np.logical_and(
                            gas_pos[:, 1] > -size, gas_pos[:, 1] < size
                        )
                        mask_z = np.logical_and(
                            gas_pos[:, 2] > -size * 1e2, gas_pos[:, 2] < size * 1e2
                        )
                        mask_xy = np.logical_and(mask_x, mask_y)
                        mask_xyz = np.logical_and(mask_xy, mask_z)

                        gas_x = gas_pos[:, 0][mask_xyz] / size / 2.0 + 0.5
                        gas_y = gas_pos[:, 1][mask_xyz] / size / 2.0 + 0.5
                        gas_h = gas_hsml[mask_xyz] / size / 2.0

                        gas_cold = gas_HI[mask_xyz] + gas_H2[mask_xyz]
                        gas_sfr = gas_SFR[mask_xyz]

                        x_val = scatter(
                            x=gas_x, y=gas_y, h=gas_h, m=gas_cold, res=N_pix
                        )
                        y_val = scatter(x=gas_x, y=gas_y, h=gas_h, m=gas_sfr, res=N_pix)
                        x_val = x_val / (size * 2.0) ** 2
                        y_val = y_val / (size * 2.0) ** 2 * 1e6
                        x_val = x_val.flatten()
                        y_val = y_val.flatten()

                        x_full = np.concatenate([x_full, x_val])
                        y_full = np.concatenate([y_full, y_val])

                        print(np.max(x_full), np.max(y_full))

                    y_binned, _, _ = stats.binned_statistic(
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

                    if split == 0:
                        color = line_properties["colour"]
                        lws = line_properties["linewidth"]
                        dashes = line_properties["dashes"]
                    elif split == 2:
                        color = color2
                        lws = lw2
                        dashes = dashes2
                    else:
                        color = color4
                        lws = lw4
                        dashes = dashes4

                    ax.plot(
                        bin_centers[mask_bins],
                        np.log10(y_binned[mask_bins] + 1e-15),
                        lw=lws[counter],
                        label=key.replace("_", "\_"),
                        color=color[counter],
                        dashes=tuple(d for d in dashes[counter]),
                        zorder=4,
                    )

                    if counter == 1:
                        ax.fill_between(
                            bin_centers[mask_bins],
                            np.log10(deviations[:, 0] + 1e-15),
                            np.log10(deviations[:, 1] + 1e-15),
                            edgecolor=color[counter],
                            alpha=0.60,
                            hatch=r"\ ",
                            facecolor="none",
                        )

                ax.text(
                    0.96,
                    0.96,
                    "${:.1f} < t < {:.1f}$ Gyr".format(
                        time_min[0] / 1e3, time_max[0] / 1e3
                    ),
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=22,
                )

                for index, observation in enumerate(observational_data):
                    if observation.gas_surface_density is not None:
                        if observation.description == "Bigiel et al. (2008) inner":
                            data = observation.bin_data_KS(np.arange(-1, 3, 0.25), 0.4)
                            l1 = plt.errorbar(
                                data[0],
                                data[1],
                                yerr=[data[2], data[3]],
                                fmt="o",
                                c="lightcoral",
                                markersize=10,
                            )
                        elif observation.description == "Bigiel et al. (2010) outer":
                            data2 = observation.bin_data_KS(np.arange(-1, 3, 0.25), 0.4)
                            l2 = plt.errorbar(
                                data2[0],
                                data2[1],
                                yerr=[data2[2], data2[3]],
                                fmt="o",
                                c="maroon",
                                markersize=10,
                            )

                leg1 = plt.legend(
                    loc="lower right",
                    fontsize=LEGEND_SIZE,
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
                    text.set_color(color[counter])

                leg2 = plt.legend(
                    [l0, l1, l2],
                    [
                        "KS law (Kennicutt 98)",
                        "Bigiel et al. (2008) inner",
                        "Bigiel et al. (2010) outer",
                    ],
                    loc="upper left",
                    frameon=False,
                    fontsize=LEGEND_SIZE * 0.80,
                )

                plt.gca().add_artist(leg1)

                # Axes & labels
                ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
                ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

                ax.set_ylim(-5.5, 1.5)
                ax.set_xlim(0, 3)

                ax.set_xlabel(
                    "log $\\left( \\Sigma_{\\rm H_2} + \\Sigma_{\\rm HI} \\right)$ [$\\rm M_\\odot$ pc$^{-2}$]",
                    fontsize=LABEL_SIZE,
                )
                ax.set_ylabel(
                    "log $\\Sigma_{\\rm SFR}$ [$\\rm M_\\odot$ yr$^{-1}$ kpc$^{-2}$]",
                    fontsize=LABEL_SIZE,
                )
                # plt.show()
                plt.savefig(f"./images/{output}", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    plot()
