import numpy as np

from plot_style import *
import matplotlib.pylab as plt
from matplotlib import cm as cmm
from matplotlib.ticker import AutoMinorLocator
from unyt import unyt_array

# from swiftsimio.visualisation.projection import scatter
from swiftsimio.visualisation.projection_backends.subsampled_extreme import scatter
from swiftsimio.visualisation.smoothing_length_generation import (
    generate_smoothing_lengths,
)
import h5py as h5
import sys
import json

# Physical constants
year_in_cgs = 3600.0 * 24 * 365.25
Msun_in_cgs = 1.98848e33
pc_in_cgs = 3.08567758e18
gamma = 2.018932  # Quartic spline


def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def plot(dict_sim, snapshot):
    ROW_SIZE = 2
    COL_SIZE = 4
    fig, ax = plt.subplots(COL_SIZE, ROW_SIZE, figsize=(8.25, 14.3), sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    for idx, (key, value) in enumerate(dict_sim.items()):
        col = idx

        if True:
            # Loading data
            f = h5.File(value + "/output_{:04d}.hdf5".format(snapshot), "r")

            # Units
            unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
            unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
            unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

            # Box size
            boxsize_kpc = (
                f["/Header"].attrs["BoxSize"] * unit_length_in_cgs / pc_in_cgs / 1e3
            )
            centre_kpc = boxsize_kpc / 2.0

            # print("Centre", centre_kpc)

            time_Myr = f["/Header"].attrs["Time"] * unit_time_in_cgs / year_in_cgs / 1e6

            gas_pos_kpc = (
                f["/PartType0/Coordinates"][:, :] * unit_length_in_cgs / pc_in_cgs / 1e3
            )
            star_pos_kpc = (
                f["/PartType4/Coordinates"][:, :] * unit_length_in_cgs / pc_in_cgs / 1e3
            )

            gas_mass_Msun = f["/PartType0/Masses"][:] * unit_mass_in_cgs / Msun_in_cgs
            star_mass_Msun = f["/PartType4/Masses"][:] * unit_mass_in_cgs / Msun_in_cgs

            gas_H_kpc = (
                f["/PartType0/SmoothingLengths"][:]
                * gamma
                * unit_length_in_cgs
                / pc_in_cgs
                / 1e3
            )

            stellar_ages = (
                f["/PartType4/BirthTimes"][:] * unit_time_in_cgs / year_in_cgs / 1e6
            )
            ages_mask = stellar_ages > 0.0

            star_H = generate_smoothing_lengths(
                unyt_array(star_pos_kpc[ages_mask, :], "kpc"),
                unyt_array(boxsize_kpc, "kpc"),
                kernel_gamma=gamma,
                neighbours=65,
                speedup_fac=1,
                dimension=3,
            )
            star_H_kpc = star_H.to("kpc").value

            gas_pos_kpc[:, 0] -= centre_kpc[0]
            gas_pos_kpc[:, 1] -= centre_kpc[1]
            gas_pos_kpc[:, 2] -= centre_kpc[2]

            star_pos_kpc[:, 0] -= centre_kpc[0]
            star_pos_kpc[:, 1] -= centre_kpc[1]
            star_pos_kpc[:, 2] -= centre_kpc[2]

            # selecting the galaxy in the very big box
            if DWARF:
                output_name = "morph_main_M10_{:04d}.pdf".format(snapshot)
                size = 30.0 * (1e-2 ** (1.0 / 3.0))
                pixel = 0.1 * (1e-2 ** (1.0 / 3.0))
            else:
                output_name = "morph_main_M12_{:04d}.pdf".format(snapshot)
                size = 30.0
                pixel = 0.1  # kpc

            mask_ISM = np.abs(gas_pos_kpc[:, 2]) < height_kpc * 5
            mask_winds = ~mask_ISM
            print("HEIGHT", height_kpc)

            M_ISM = np.sum(gas_mass_Msun[mask_ISM])
            M_winds = np.sum(gas_mass_Msun[mask_winds])
            M_stars = np.sum(star_mass_Msun[ages_mask])
            M_total = M_ISM + M_winds + M_stars

            print("ISM", "WINDS", "STARS")
            print(np.log10(M_ISM), np.log10(M_winds), np.log10(M_stars))
            print("___________________")

            M_ISM_fr = M_ISM / M_total * (2.0 * size)
            M_winds_fr = M_winds / M_total * (2.0 * size)
            M_stars_fr = M_stars / M_total * (2.0 * size)

            gas_T_K = f["/PartType0/Temperatures"][:]
            top_T = 1e5
            bottom_T = 1e3
            mask_hot = np.logical_and(mask_ISM, gas_T_K > top_T)
            mask_warm = np.logical_and(
                mask_ISM, np.logical_and(gas_T_K > bottom_T, gas_T_K <= top_T)
            )
            mask_cold = np.logical_and(mask_ISM, gas_T_K <= bottom_T)

            M_hot = np.sum(gas_mass_Msun[mask_hot])
            M_warm = np.sum(gas_mass_Msun[mask_warm])
            M_cold = np.sum(gas_mass_Msun[mask_cold])

            M_hot_fr = 10 * M_hot / (M_ISM + 9.0 * M_hot) * (2.0 * size)
            M_warm_fr = M_warm / (M_ISM + 9.0 * M_hot) * (2.0 * size)
            M_cold_fr = M_cold / (M_ISM + 9.0 * M_hot) * (2.0 * size)

            print(
                M_ISM_fr / (2.0 * size) * 100.0,
                M_winds_fr / (2.0 * size) * 100.0,
                M_stars_fr / (2.0 * size) * 100.0,
            )
            print(
                M_hot_fr / (2.0 * size) * 100.0,
                M_warm_fr / (2.0 * size) * 100.0,
                M_cold_fr / (2.0 * size) * 100.0,
            )

            N_pix = int(size * 2 / pixel)
            print("NPIXELS", N_pix)
            print("SIZE", size)

            mask_x = np.logical_and(gas_pos_kpc[:, 0] > -size, gas_pos_kpc[:, 0] < size)
            mask_y = np.logical_and(gas_pos_kpc[:, 1] > -size, gas_pos_kpc[:, 1] < size)
            mask_z = np.logical_and(
                gas_pos_kpc[:, 2] > -size * 1e2, gas_pos_kpc[:, 2] < size * 1e2
            )
            mask_xy = np.logical_and(mask_x, mask_y)
            mask_xyz = np.logical_and(mask_xy, mask_z)

            gas_x = gas_pos_kpc[:, 0][mask_xyz] / size / 2.0 + 0.5
            gas_y = gas_pos_kpc[:, 1][mask_xyz] / size / 2.0 + 0.5
            gas_H = gas_H_kpc[mask_xyz] / size / 2.0
            gas_mass_Msun = gas_mass_Msun[mask_xyz]

            mask_x_s = np.logical_and(
                star_pos_kpc[ages_mask, 0] > -size, star_pos_kpc[ages_mask, 0] < size
            )
            mask_y_s = np.logical_and(
                star_pos_kpc[ages_mask, 1] > -size, star_pos_kpc[ages_mask, 1] < size
            )
            mask_z_s = np.logical_and(
                star_pos_kpc[ages_mask, 2] > -size * 1e2,
                star_pos_kpc[ages_mask, 2] < size * 1e2,
            )
            mask_xy_s = np.logical_and(mask_x_s, mask_y_s)
            mask_xyz_s = np.logical_and(mask_xy_s, mask_z_s)

            star_x = star_pos_kpc[ages_mask, 0][mask_xyz_s] / size / 2.0 + 0.5
            star_y = star_pos_kpc[ages_mask, 1][mask_xyz_s] / size / 2.0 + 0.5
            star_H = star_H_kpc[mask_xyz_s] / size / 2.0
            star_mass_Msun = star_mass_Msun[ages_mask][mask_xyz_s]

            x_val = scatter(x=gas_x, y=gas_y, h=gas_H, m=gas_mass_Msun, res=N_pix)
            x_val_s = scatter(x=star_x, y=star_y, h=star_H, m=star_mass_Msun, res=N_pix)

            x_val = x_val / (size * 2.0 * 1e3) ** 2
            x_val_s = x_val_s / (size * 2.0 * 1e3) ** 2

            x_val_s[x_val_s < 10.0**min_v_s] = 10.0**min_v_s

            extent = [-size, size, -size, size]

            print("time", time_Myr)

            im2 = ax[col, 0].imshow(
                np.log10(x_val),
                extent=extent,
                origin="lower",
                cmap=cmm.viridis,
                vmin=min_v,
                vmax=max_v,
            )

            ax[col, 1].bar(
                size, M_stars_fr, size * 0.2, color="black", align="edge", bottom=-size
            )
            ax[col, 1].bar(
                size,
                M_winds_fr,
                size * 0.2,
                color="sienna",
                align="edge",
                bottom=-size + M_stars_fr,
            )
            ax[col, 1].bar(
                size,
                M_ISM_fr,
                size * 0.2,
                color="lightgrey",
                align="edge",
                bottom=-size + M_winds_fr + M_stars_fr,
            )

            ax[col, 0].bar(
                -1.2 * size,
                M_cold_fr,
                size * 0.2,
                color="blue",
                align="edge",
                bottom=-size,
            )
            ax[col, 0].bar(
                -1.2 * size,
                M_warm_fr,
                size * 0.2,
                color="orange",
                align="edge",
                bottom=-size + M_cold_fr,
            )
            ax[col, 0].bar(
                -1.2 * size,
                M_hot_fr,
                size * 0.2,
                color="red",
                align="edge",
                bottom=-size + M_warm_fr + M_cold_fr,
            )

            ax[col, 0].set_xlim(-size * 1.2, size)
            ax[col, 0].set_ylim(-size, size)

            im = ax[col, 1].imshow(
                np.log10(x_val_s),
                extent=extent,
                origin="lower",
                cmap=cmm.Purples,
                vmin=min_v_s,
                vmax=max_v_s,
            )

            ax[col, 1].set_xlim(-size, size * 1.2)
            ax[col, 1].set_ylim(-size, size)

            if col == 0:
                ax[col, 0].text(
                    0.95,
                    0.95,
                    "$t = {:.2f}$ Gyr".format(time_Myr[0] / 1e3),
                    ha="right",
                    va="top",
                    color="white",
                    fontsize=20,
                    transform=ax[col, 0].transAxes,
                    bbox=dict(
                        facecolor="black",
                        edgecolor="black",
                        boxstyle="round",
                        alpha=0.5,
                    ),
                )

            for i in range(2):
                ax[col, i].text(
                    1.0 - float(i),
                    0.05,
                    key.replace("_", "\_"),
                    ha="center",
                    va="bottom",
                    color="black",
                    transform=ax[col, i].transAxes,
                    fontsize=25,
                    bbox=dict(
                        facecolor="orange",
                        edgecolor="black",
                        boxstyle="round",
                        alpha=0.7,
                    ),
                )

            if DWARF:
                if col == 0:
                    offset = 3.5
                    value = 3
                    ax[col, 1].plot(
                        [-2 - offset, -2 + value - offset],
                        [1.25 + offset, 1.25 + offset],
                        lw=3,
                        alpha=0.9,
                    )
                    ax[col, 1].text(
                        -2.0 - offset,
                        1.75 + offset,
                        f"${value}$ kpc",
                        fontsize=20,
                        alpha=0.9,
                    )
            else:
                if col == 0:
                    offset = 8.0
                    value = 15
                    ax[col, 1].plot(
                        [-16 - offset, -16 + value - offset],
                        [14.2 + offset, 14.2 + offset],
                        lw=3,
                        alpha=0.9,
                    )
                    ax[col, 1].text(
                        -16.5 - offset,
                        16.5 + offset,
                        f"${value}$ kpc",
                        fontsize=20,
                        alpha=0.9,
                    )

        for i in range(2):
            plt.setp(ax[col, i].get_yticklabels(), visible=False)
            plt.setp(ax[col, i].get_xticklabels(), visible=False)
            ax[col, i].tick_params(which="major", length=0)
            ax[col, i].tick_params(which="minor", length=0)

    fig.subplots_adjust(bottom=0.09, top=0.93)

    cbar_ax = fig.add_axes([0.25, 0.06, 0.50, 0.018])
    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="horizontal",
        ticks=[-1.5, -1.0, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
        extend="both",
    )
    cbar_ax.xaxis.set_tick_params(labelsize=22)
    cbar.ax.tick_params(which="major", width=1.7, length=20)
    cbar.ax.set_xlabel(
        "log $\\Sigma_{\\rm *} \\, \\rm [M_{\\odot} \\, pc^{-2}]$",
        rotation=0,
        fontsize=24,
        labelpad=3.2,
    )

    cbar_ax2 = fig.add_axes([0.25, 0.98, 0.50, 0.018])
    cbar2 = fig.colorbar(
        im2,
        cax=cbar_ax2,
        orientation="horizontal",
        ticks=[-1.5, -1.0, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
        extend="both",
    )
    cbar_ax2.xaxis.set_tick_params(labelsize=22)
    cbar2.ax.tick_params(which="major", width=1.7, length=20)
    cbar2.ax.set_xlabel(
        "log $\\Sigma_{\\rm gas} \\, \\rm [M_{\\odot} \\, pc^{-2}]$",
        rotation=0,
        fontsize=24,
        labelpad=3.2,
    )

    plt.savefig(
        output_name,
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=100,
    )


if __name__ == "__main__":
    file_names = ["main.json", "main_dwarf.json"]

    for min_v, max_v, min_v_s, max_v_s, DWARF, name, height_kpc in zip(
        [-0.5, -1.0],
        [2.5, 2.0],
        [-0.5, -1.5],
        [2.5, 1.5],
        [0, 1],
        file_names,
        [0.430865, 0.0719907],
    ):
        print(DWARF, name)
        snapshot = 70  # 70
        dict_sim = read_sim_data(name)
        plot(dict_sim, snapshot)
