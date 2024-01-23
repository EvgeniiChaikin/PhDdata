import numpy as np

from plot_style import *
import matplotlib.pylab as plt
from matplotlib import cm as cmm
from matplotlib.ticker import AutoMinorLocator
import h5py as h5
import sphviewer as sph
import sys
import json


def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# Physical constants
k_in_cgs = 1.38064852e-16
mH_in_cgs = 1.6737236e-24
year_in_cgs = 3600.0 * 24 * 365.25
Msun_in_cgs = 1.98848e33
G_in_cgs = 6.67259e-8
pc_in_cgs = 3.08567758e18
gamma = 2.018932  # Quartic spline

# defining figure

ROW_SIZE = 5
COL_SIZE = 1
fig, ax = plt.subplots(
    COL_SIZE, ROW_SIZE, figsize=(15, 4.025), sharex=True, sharey=True
)
fig.subplots_adjust(hspace=0, wspace=0)
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


snapshot = 100
dict_sim = read_sim_data("main.json")

for idx, (key, value) in enumerate(dict_sim.items()):
    col, row = idx % ROW_SIZE, idx // ROW_SIZE
    print(col, row, idx)

    if True:
        # Loading data
        f = h5.File(value + "/output_{:04d}.hdf5".format(snapshot), "r")

        # Units
        unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
        unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
        unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

        # Box size
        boxsize = f["/Header"].attrs["BoxSize"] * unit_length_in_cgs / pc_in_cgs / 1e3
        centre = boxsize / 2.0

        # loading gas-particle data
        gas_pos = (
            f["/PartType0/Coordinates"][:, :] * unit_length_in_cgs / pc_in_cgs / 1e3
        )
        gas_mass = f["/PartType0/Masses"][:]
        gas_hsml = f["/PartType0/SmoothingLengths"][:]

        gas_pos[:, 0] -= centre[0]
        gas_pos[:, 1] -= centre[1]
        gas_pos[:, 2] -= centre[2]
        gas_pos[:, :2] = np.swapaxes(np.vstack([gas_pos[:, 1], gas_pos[:, 0]]), 0, 1)

        # selecting the galaxy in the very big box
        size = boxsize[0] / 170

        # particle density
        Particles = sph.Particles(gas_pos, gas_mass, gas_hsml * gamma)
        extent = [-size, size, -size, size]
        Camera = sph.Camera(
            r="infinity",
            t=0,
            p=0,
            roll=0,
            xsize=750,
            ysize=750,
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
        args = np.where(data1 == 0)
        data1[args] = np.min(data1[np.where(data1 > 0)])

        time = f["/Header"].attrs["Time"] * unit_time_in_cgs / year_in_cgs / 1e6

        print("time", time)
        # cmap = cmm.get_cmap('cividis', 30)

        im = ax[col].imshow(
            np.log10(data1),
            extent=extent,
            origin="lower",
            cmap=cmm.cividis,
            vmin=-0.5,
            vmax=2.0,
        )

        ax[col].set_xlim(-size, size)
        ax[col].set_ylim(-size, size)

        if col == 4:
            shift = 5
            ax[col].plot(
                [-16 - shift, -6 - shift],
                [14.2 + shift, 14.2 + shift],
                lw=1.5,
                color="white",
                alpha=0.9,
            )
            ax[col].text(
                -16.1 - shift,
                16.0 + shift,
                "$10$ kpc",
                fontsize=16,
                color="white",
                alpha=0.9,
            )

        if col == 0:
            ax[col].text(
                0.05,
                0.95,
                "$t = {:.1f}$ Gyr".format(time[0] / 1e3),
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

plt.savefig(
    "galaxy_face_on_{:04d}.pdf".format(snapshot),
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=100,
)
