import numpy as np

from plot_style import *
import matplotlib.pylab as plt
from matplotlib import cm as cmm
from matplotlib.ticker import AutoMinorLocator
import h5py as h5
import sphviewer as sph
import sys
import json

# Physical constants
k_in_cgs = 1.38064852e-16
mH_in_cgs = 1.6737236e-24
year_in_cgs = 3600.0 * 24 * 365.25
Msun_in_cgs = 1.98848e33
G_in_cgs = 6.67259e-8
pc_in_cgs = 3.08567758e18
gamma = 2.018932 # Quartic spline

# defining figure

ROW_SIZE = 5
COL_SIZE = 1
fig, ax = plt.subplots(COL_SIZE, ROW_SIZE, figsize=(15, 4.10), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0, wspace=0)
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


snapshot = 120  # 10 = 50 Myr
dt = 200.0

dict_sim = read_sim_data("main.json")
(len(dict_sim))

for idx, (key, value) in enumerate(dict_sim.items()):

    col, row = idx % ROW_SIZE, idx // ROW_SIZE
    print(col, row, idx)

    if True:

        print(col, row)

        # Loading data
        f = h5.File(value + "/output_{:04d}.hdf5".format(snapshot), "r")

        # Units
        unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
        unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
        unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

        # Box size
        boxsize = f["/Header"].attrs["BoxSize"] * unit_length_in_cgs / pc_in_cgs / 1e3
        centre = boxsize / 2.0

        time = f["/Header"].attrs["Time"] * unit_time_in_cgs / year_in_cgs / 1e6

        # loading gas-particle data
        gas_pos = (
            f["/PartType0/Coordinates"][:, :] * unit_length_in_cgs / pc_in_cgs / 1e3
        )
        gas_mass = f["/PartType0/Masses"][:]
        gas_hsml = f["/PartType0/SmoothingLengths"][:]

        gas_pos[:, 0] -= centre[0]
        gas_pos[:, 1] -= centre[1]
        gas_pos[:, 2] -= centre[2]
        gas_pos[:, :2] = np.swapaxes(np.vstack([gas_pos[:, 0], gas_pos[:, 2]]), 0, 1)
        gas_SNII_times = np.where(
            np.abs(
                f["/PartType0/LastSNIIThermalFeedbackTimes"][:]
                * unit_time_in_cgs
                / year_in_cgs
                / 1e6
                - time[0]
            )
            <= dt
        )

        # selecting the galaxy in the very big box
        size = boxsize[0] / 150

        # particle density
        Particles = sph.Particles(
            gas_pos[gas_SNII_times],
            gas_mass[gas_SNII_times],
            gas_hsml[gas_SNII_times] * gamma,
        )
        extent = [-size, size, -size, size]
        Camera = sph.Camera(
            r="infinity",
            t=0,
            p=0,
            roll=0,
            xsize=500,
            ysize=500,
            x=0.0,
            y=0.0,
            z=0.0,
            extent=extent,
        )

        Scene = sph.Scene(Particles, Camera)
        Render = sph.Render(Scene)
        extent = Render.get_extent()

        # plot 1
        density = Render.get_image()
        data1 = density * unit_mass_in_cgs / Msun_in_cgs / (1e3) ** 2
        data1[np.where(data1 == 0.0)] = np.min(data1[np.where(data1 > 0.0)])
        data1 /= np.sum(data1)

        print("time", time)
        cmap = cmm.get_cmap('cividis', 12)
        im = ax[col].imshow(
            np.log10(data1),
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=-6.0,
            vmax=-3.0,
            zorder=-1,
        )

        ax[col].set_xlim(-size, size)
        ax[col].set_ylim(-size, size)

        print(np.max(data1))
        CS = ax[col].contour(
            data1,
            extent=extent,
            origin="lower",
            levels=[5e-6],
            linewidths=[1.0],
            colors=["k", "deepskyblue"],
            zorder=3,
            linestyles=["dashed", "dashed"],
            dashes=[(5, 2), (2, 2)],
        )

        if col == 4:
            offset = 7.5
            ax[col].plot(
                [-16 - offset, -6 - offset],
                [14.2 + offset, 14.2 + offset],
                lw=3,
                color="white",
                alpha=0.9,
            )
            ax[col].text(
                -16.5 - offset,
                16.4 + offset,
                "$10$ kpc",
                fontsize=16.0,
                color="white",
                alpha=0.9,
            )

        if col == 0:
            ax[col].text(
                0.05,
                0.95,
                "${:.1f} < t < {:.1f}$ Gyr".format( (time[0]-dt) / 1e3, time[0] / 1e3),
                ha="left",
                va="top",
                color="white",
                fontsize=16.5,
                transform=ax[col].transAxes,
                bbox=dict(
                    facecolor="black", edgecolor="black", boxstyle="round", alpha=0.5
                ),
            )

        ax[col].text(
            0.5,
            0.05,
            key.replace("_", "\_"),
            ha="center",
            va="bottom",
            color="white",
            transform=ax[col].transAxes,
            fontsize=16.3,
            bbox=dict(
                facecolor="black", edgecolor="black", boxstyle="round", alpha=0.5
            ),
        )

        plt.setp(ax[col].get_yticklabels(), visible=False)
        plt.setp(ax[col].get_xticklabels(), visible=False)
        ax[col].tick_params(which="major", length=0)
        ax[col].tick_params(which="minor", length=0)

fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.3, 0.11, 0.42, 0.07])

cbar = fig.colorbar(
    im, cax=cbar_ax, orientation="horizontal", ticks=[-6, -5, -4, -3], extend="both"
)
cbar_ax.xaxis.set_tick_params(labelsize=23)
cbar.ax.tick_params(which="major", width=1.7, length=21)
cbar.ax.set_xlabel(
    "log Mass fraction of heated gas per pixel", rotation=0, fontsize=23, labelpad=3
)

plt.savefig(
    "snii_heated_parts_faceon_3dt.pdf", bbox_inches="tight", pad_inches=0.1, dpi=100
)
