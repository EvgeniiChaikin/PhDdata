import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py as h5
from constants import *
from plot_style import *
from tqdm import tqdm
from matplotlib import cm as cmm
import sphviewer as sph


def plot(snp, z_aver_width=10.0 / 1e3, gamma=1.936492, X_H=0.73738788833, FONTSIZE=28):
    # Plotting (make a four-panel figure)
    fig, ax = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.set_size_inches(18, 19)

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    dict_sim = {
        "../run01_diffusion_new_one_p_with_centre_higres_1_sphere": "high\_res\_n1",
        "../run01_diffusion_new_one_p_with_centre_higres_sphere": "high\_res\_n01",
        "../run01_diffusion_new_one_p_with_centre_higres_001_sphere": "high\_res\_n001",
    }

    for count, (key, value) in enumerate(dict_sim.items()):
        with h5.File(
            f"{key}" + "/output_{:04d}.hdf5".format(snp),
            "r",
        ) as f:
            print(count)
            boxsize = f["/Header"].attrs["BoxSize"]

            # Units
            unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
            unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
            unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]

            # Box centre
            centre = boxsize / 2.0

            # Load Fields (in the correct units)
            gas_pos = f["/PartType0/Coordinates"][:, :]
            gas_T = f["/PartType0/Temperatures"][:]
            gas_mass = f["/PartType0/Masses"][:]
            gas_hsml = f["/PartType0/SmoothingLengths"][:]
            ejecta = f["/PartType0/ElementMassFractions"][:, -1]

            # Shift particles with respect to the box centre
            gas_pos[:, 0] -= centre[0]
            gas_pos[:, 1] -= centre[1]
            gas_pos[:, 2] -= centre[2]

            # Compute distance with respect to the centre
            r = np.sqrt(np.sum(gas_pos * gas_pos, axis=1))

            # We want to normalise ejecta becasue the value of Fe60 ejected is aribtrary chosen
            ejecta_normalisation = np.sum(ejecta[np.where(ejecta > 0.0)])

            # We want 1e-4 Msun total in Fe60
            ejecta_mass = 1e-4 / np.mean(gas_mass)

            # Define a slab to do the average
            arg = np.where(np.abs(gas_pos[:, 2]) < z_aver_width)

            # Number of bins along x and y coordinates of a 2D plot
            N_bins = 500

            size = boxsize[0]

            # Create standard camera
            Camera = sph.Camera(
                r="infinity",
                t=0,
                p=0,
                roll=0,
                xsize=N_bins,
                ysize=N_bins,
                x=0.0,
                y=0.0,
                z=0.0,
                extent=[-size / 2, size / 2, -size / 2, size / 2],
            )

            # Scene 1 (Density field; need to provide only coords, masses, and smoothing lenghts)
            Particles = sph.Particles(
                gas_pos[arg], gas_mass[arg], gas_hsml[arg] * gamma
            )
            Scene = sph.Scene(Particles, Camera)
            Render = sph.Render(Scene)
            extent = Render.get_extent()
            density = Render.get_image()
            const = (
                X_H
                / z_aver_width
                / 2
                * unit_mass_in_cgs
                / unit_length_in_cgs**3
                / constants["PROTON_MASS_IN_CGS"]
            )

            # Scene 2 (Mass weighted temperature)
            Particles = sph.Particles(
                gas_pos[arg], gas_mass[arg] * gas_T[arg], gas_hsml[arg] * gamma
            )
            Scene = sph.Scene(Particles, Camera)
            Render = sph.Render(Scene)
            temperature = Render.get_image() / density

            # Scene 4 (Ejecta density is computed in the same way as the gas density expect that we use the Fe60 mass
            Particles = sph.Particles(
                gas_pos[arg], gas_mass[arg] * ejecta[arg], gas_hsml[arg] * gamma
            )
            Scene = sph.Scene(Particles, Camera)
            Render = sph.Render(Scene)
            ej_schene = Render.get_image()
            ej_schene[np.where(ej_schene == 0.0)] = np.min(
                ej_schene[np.where(ej_schene > 0.0)]
            )

            # Divide by 60 because Fe60
            const_ej = (
                ejecta_mass
                / ejecta_normalisation
                / 60
                / z_aver_width
                / 2
                * constants["SOLAR_MASS_IN_CGS"]
                / unit_length_in_cgs**3
                / constants["PROTON_MASS_IN_CGS"]
            )

            # Panel 1 (density)
            im = ax[0, count].imshow(
                np.log10(density * const),
                extent=extent,
                origin="lower",
                cmap=cmm.coolwarm,
                vmin=-3.0,
                vmax=1.0,
            )

            ax[0, count].set_xticks([])
            ax[0, count].set_yticks([])

            divider = make_axes_locatable(ax[0, count])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(
                im,
                cax=cax,
                ticks=[-3, -2, -1, 0, 1],
                # orientation="horizontal",
                use_gridspec=True,
            )

            if count == 2:
                cax.yaxis.set_ticklabels(
                    ["$-3.0$", "$-2.0$", "$-1.0$", "$0.0$", "$1.0$"],
                    fontsize=FONTSIZE,
                )
                cax.set_ylabel(
                    "$\\mathrm{log}\\, n_{\\rm H} \\, \\rm [cm^{-3}]$",
                    fontsize=FONTSIZE * 1.2,
                    labelpad=20,
                )  # , rotation=270
            else:
                cax.yaxis.set_ticklabels([])

            time = (
                f["/Header"].attrs["Time"]
                * unit_time_in_cgs
                / constants["YEAR_IN_CGS"]
                / 1e6
            )
            print(time)

            # Panel 2 (temperature)
            im2 = ax[1, count].imshow(
                np.log10(temperature),
                extent=extent,
                origin="lower",
                cmap=cmm.coolwarm,
                vmin=4,
                vmax=7,
            )

            divider = make_axes_locatable(ax[1, count])
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            ax[1, count].set_xticks([])
            ax[1, count].set_yticks([])

            cbar1 = plt.colorbar(
                im2,
                cax=cax1,
                ticks=[4, 5, 6, 7],
                use_gridspec=True,
            )
            if count == 2:
                cax1.yaxis.set_ticklabels(
                    ["$4.0$", "$5.0$", "$6.0$", "$7.0$"], fontsize=FONTSIZE
                )
                cax1.set_ylabel(
                    "$\\mathrm{log}\\, T_{\\rm K}$ [K]",
                    fontsize=FONTSIZE * 1.2,
                    labelpad=20,
                )  # , rotation=270
            else:
                cax1.yaxis.set_ticklabels([])

            for i in range(3):
                ax[i, count].text(
                    -0.5 * boxsize[0] * 0.91,
                    -0.5 * boxsize[0] * 0.90,
                    "$100$ pc",
                    fontsize=22,
                    color="black",
                )
                ax[i, count].plot(
                    [-0.5 * boxsize[0] * 0.91, -0.5 * boxsize[0] * 0.91 + 100.0 / 1e3],
                    [-0.5 * boxsize[0] * 0.95, -0.5 * boxsize[0] * 0.95],
                    lw=3,
                    color="black",
                )

            # Panel 3 (ejecta density)
            im = ax[2, count].imshow(
                np.log10(ej_schene * const_ej),
                extent=extent,
                origin="lower",
                cmap=cmm.Purples,
                vmin=-12.25,
                vmax=-9.75,
            )
            divider = make_axes_locatable(ax[2, count])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax[2, count].set_xticks([])
            ax[2, count].set_yticks([])
            cbar = plt.colorbar(
                im,
                cax=cax,
                use_gridspec=True,
                ticks=[-13.0, -12.0, -11.0, -10.0, -9.0],
            )
            if count == 2:
                cax.yaxis.set_ticklabels(
                    ["$-13$", "$-12$", "$-11$", "$-10$", "$-9$"], fontsize=FONTSIZE
                )
                cax.set_ylabel(
                    "log $n_{\\rm ^{60}Fe} \\rm \\, [cm^{-3}]$",
                    fontsize=FONTSIZE * 1.2,
                    labelpad=20,
                )
            else:
                cax.yaxis.set_ticklabels([])

            ax[0, count].text(
                0.96,
                0.96,
                value,
                ha="right",
                va="top",
                transform=ax[0, count].transAxes,
                fontsize=FONTSIZE,
                bbox=dict(
                    facecolor="skyblue", edgecolor="black", boxstyle="round", alpha=0.8
                ),
            )

    plt.savefig(
        "./remnant_image3_{:04d}.png".format(snp), bbox_inches="tight", pad_inches=0.1
    )
    plt.savefig(
        "./remnant_image3_{:04d}.pdf".format(snp), bbox_inches="tight", pad_inches=0.1
    )

    plt.close()


if __name__ == "__main__":
    for i in range(17 * 4, 17 * 4 + 1):
        print("Current snapshot number is {}".format(i))
        plot(i)
