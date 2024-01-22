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

    with h5.File(
        #"../run01_diffusion_new_one_p_with_centre_higres_1_sphere/output_{:04d}.hdf5".format(
        "../run01_diffusion_new_one_p_with_centre_higres_sphere/output_{:04d}.hdf5".format(
            snp
        ),
        "r",
    ) as f:

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
        gas_P = f["/PartType0/Pressures"][:] * unit_mass_in_cgs / unit_length_in_cgs / unit_time_in_cgs **2 / constants["BOLTZMANN_IN_CGS"]
        gas_mass = f["/PartType0/Masses"][:]
        gas_hsml = f["/PartType0/SmoothingLengths"][:]
        ejecta = f["/PartType0/ElementMassFractions"][:, -1]

        # Shift particles with respect to the box centre
        gas_pos[:, 0] -= centre[0]
        gas_pos[:, 1] -= centre[1]
        gas_pos[:, 2] -= centre[2]

        # Compute distance with respect to the centre
        r = np.sqrt(np.sum(gas_pos * gas_pos, axis=1))

        # Gas velocity
        gas_vel = (
            f["/PartType0/Velocities"][:, :] * unit_length_in_cgs / unit_time_in_cgs
        )

        v_r = np.sum(gas_pos * gas_vel, axis=1) / r

        # We want to normalise ejecta becasue the value of Fe60 ejected is aribtrary chosen
        ejecta_normalisation = np.sum(ejecta[np.where(ejecta > 0.0)])

        # We want 1e-4 Msun total in Fe60
        ejecta_mass = 1e-4 / np.mean(gas_mass)

        print("ejecta normalisation", ejecta_normalisation)
        print(
            "Total ejecta mass check:",
            np.sum(gas_mass * ejecta * ejecta_mass) / ejecta_normalisation,
        )

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
        Particles = sph.Particles(gas_pos[arg], gas_mass[arg], gas_hsml[arg] * gamma)
        Scene = sph.Scene(Particles, Camera)
        Render = sph.Render(Scene)
        extent = Render.get_extent()
        density = Render.get_image()
        const = (
            X_H
            / z_aver_width
            / 2
            * unit_mass_in_cgs
            / unit_length_in_cgs ** 3
            / constants["PROTON_MASS_IN_CGS"]
        )

        # Scene 2 (Mass weighted temperature)
        Particles = sph.Particles(
            gas_pos[arg], gas_mass[arg] * gas_T[arg], gas_hsml[arg] * gamma
        )
        Scene = sph.Scene(Particles, Camera)
        Render = sph.Render(Scene)
        temperature = Render.get_image() / density

        # Scene 3 (Mass weighted gas radial velocity)
        Particles = sph.Particles(
            gas_pos[arg], gas_mass[arg] * v_r[arg], gas_hsml[arg] * gamma
        )
        Scene = sph.Scene(Particles, Camera)
        Render = sph.Render(Scene)
        velocity = Render.get_image() / density

        # Pressure
        Particles = sph.Particles(
            gas_pos[arg], gas_mass[arg] * gas_P[arg], gas_hsml[arg] * gamma
        )
        Scene = sph.Scene(Particles, Camera)
        Render = sph.Render(Scene)
        pressure = Render.get_image() / density

        # Scene 3a (Mass weighted gas velocity_x)
        Particles = sph.Particles(
            gas_pos[arg], gas_mass[arg] * gas_vel[:, 0][arg], gas_hsml[arg] * gamma
        )
        Scene = sph.Scene(Particles, Camera)
        Render = sph.Render(Scene)
        velocity_x = Render.get_image() / density

        # Scene 3b (Mass weighted gas velocity_y)
        Particles = sph.Particles(
            gas_pos[arg], gas_mass[arg] * gas_vel[:, 1][arg], gas_hsml[arg] * gamma
        )
        Scene = sph.Scene(Particles, Camera)
        Render = sph.Render(Scene)
        velocity_y = Render.get_image() / density

        # Scene 4 (Ejecta density is computed in the same way as the gas density expect that we use the Fe60 mass
        Particles = sph.Particles(
            gas_pos[arg], gas_mass[arg] * ejecta[arg], gas_hsml[arg] * gamma
        )
        Scene = sph.Scene(Particles, Camera)
        Render = sph.Render(Scene)
        ej_schene = Render.get_image()

        # Divide by 60 because Fe60
        const_ej = (
            ejecta_mass
            / ejecta_normalisation
            / 60
            / z_aver_width
            / 2
            * constants["SOLAR_MASS_IN_CGS"]
            / unit_length_in_cgs ** 3
            / constants["PROTON_MASS_IN_CGS"]
        )

        # Plotting (make a four-panel figure)
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.set_size_inches(9.8*1.3, 9*1.3)

        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        # Panel 1 (density)
        im = ax[0, 0].imshow(
            np.log10(density * const),
            extent=extent,
            origin="lower",
            cmap=cmm.coolwarm,
            vmin=-3,
            vmax=1.,
        )

        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes("top", size="5%", pad=0.05)
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        cbar = plt.colorbar(
            im,
            cax=cax,
            ticks=[-3, -2, -1, 0, 1],
            orientation="horizontal",
            use_gridspec=True,
        )
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_ticklabels(
            ["$-3.0$", "$-2.0$", "$-1.0$", "$0.0$", "$1.0$"], fontsize=FONTSIZE
        )
        cax.set_xlabel(
            "$\\mathrm{log}\\, n_{\\rm H} \\, \\rm [cm^{-3}]$",
            fontsize=FONTSIZE,
            labelpad=-70,
        )  # , rotation=270

        time = (
            f["/Header"].attrs["Time"]
            * unit_time_in_cgs
            / constants["YEAR_IN_CGS"]
            / 1e6
        )

        header = (
            "This file contains: hydrogen number density (n_H) [units: N of N / cm3] \n"
            + "The width of the slice (along the Z axis) is {:.3f} pc \n".format(
                2.0 * z_aver_width * 1e3
            )
            + "Time of the snapshot: {:6.4f} Myr \n".format(time[0])
            + "X-axis limits: [{:.5f}, {:.5f}] kpc \n".format(0.0, boxsize[0])
            + "Y-axis limits: [{:.5f}, {:.5f}] kpc \n".format(0.0, boxsize[1])
            + "Number of uniformily-spaced bins along the X axis: {} \n".format(N_bins)
            + "Number of uniformily-spaced bins along the Y axis: {} \n".format(N_bins)
            + "The initial ISM density is equal to 0.01 hydrogen particles per cubic cm \n"
            + "The mass fraction of hydrogen is {:.3f} \n".format(X_H)
            + "Total Fe60 mass released by SN is 1e-4 Msun\n"
        )

        np.savetxt(
            "./output_Alexander2/density_{:03d}.dat".format(snp), density * const, fmt="%.5e", header=header
        )

        # Panel 2 (temperature)
        im2 = ax[0, 1].imshow(
            np.log10(temperature),
            extent=extent,
            origin="lower",
            cmap=cmm.coolwarm,
            vmin=4,
            vmax=7,
        )

        divider = make_axes_locatable(ax[0, 1])
        cax1 = divider.append_axes("top", size="5%", pad=0.05)
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])

        cbar1 = plt.colorbar(
            im2,
            cax=cax1,
            ticks=[4, 5, 6, 7],
            orientation="horizontal",
            use_gridspec=True,
        )  #
        cax1.xaxis.set_ticks_position("top")
        cax1.xaxis.set_ticklabels(
            ["$4.0$", "$5.0$", "$6.0$", "$7.0$"], fontsize=FONTSIZE
        )
        cax1.set_xlabel(
            "$\\mathrm{log}\\, T_{\\rm K}$ [K]", fontsize=FONTSIZE, labelpad=-75
        )  # , rotation=270

        # Panel 3 (velocity) divide by 1e5 because cm/sec -> km/sec
        im = ax[1, 0].imshow(
            np.log10(pressure),
            #velocity / 1e5,
            extent=extent,
            origin="lower",
            cmap=cmm.inferno,
            vmin=3,
            vmax=6,
        )

        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        ax[1, 0].set_xticks([])
        ax[1, 0].set_yticks([])
        cbar = plt.colorbar(
            im, cax=cax, ticks=[3, 4, 5, 6, 7], orientation="horizontal", use_gridspec=True
            #im, cax=cax, ticks=[0, 10, 20], orientation="horizontal", use_gridspec=True
        )
        cax.xaxis.set_ticks_position("bottom")
        #cax.xaxis.set_ticklabels(["$0$", "$10$", "$20$"], fontsize=FONTSIZE)
        cax.xaxis.set_ticklabels(["$3$", "$4$", "$5$", "$6$", "$7$"], fontsize=FONTSIZE)
        cax.set_xlabel("$P/ \\mathrm{k_B}$ [K cm$^{-3}$]", fontsize=FONTSIZE, labelpad=11)  # , rotation=270
        #cax.set_xlabel("$v$ [km s$^{-1}$]", fontsize=FONTSIZE, labelpad=11)  # , rotation=270
        ax[1, 0].plot(
            [-0.5 * boxsize[0] * 0.91, -0.5 * boxsize[0] * 0.91 + 100.0 / 1e3],
            [-0.5 * boxsize[0] * 0.95, -0.5 * boxsize[0] * 0.95],
            lw=3,
            color="white",
        )
        ax[1, 0].text(
            -0.5 * boxsize[0] * 0.91,
            -0.5 * boxsize[0] * 0.90,
            "$100$ pc",
            fontsize=20,
            color="white",
        )

        # Panel 4 (ejecta density)
        im = ax[1, 1].imshow(
            np.log10(ej_schene * const_ej),
            extent=extent,
            origin="lower",
            cmap=cmm.Purples,
            vmin=-12,
            vmax=-9,
        )

        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        cbar = plt.colorbar(
            im,
            cax=cax,
            orientation="horizontal",
            use_gridspec=True,
            ticks=[-12, -11, -10, -9],
        )
        cax.xaxis.set_ticks_position("bottom")
        cax.xaxis.set_ticklabels(["$-12$", "$-11$", "$-10$", "$-9$"], fontsize=FONTSIZE)
        cax.set_xlabel(
            "log $n_{\\rm ^{60}Fe} \\rm \\, [cm^{-3}]$", fontsize=FONTSIZE, labelpad=11
        )

        ax[1, 1].text(
            0.96,
            0.04,
            "$t = {:.1f}$ Myr".format(time[0]),
            ha="right",
            va="bottom",
            transform=ax[1, 1].transAxes,
            fontsize=FONTSIZE,
            bbox=dict(
                facecolor="skyblue", edgecolor="black", boxstyle="round", alpha=0.8
            ),
        )

        header = (
            "This file contains: ejecta density (n_{^{60}Fe}) [units: N of ^{60}Fe particles / cm3]]. \n"
            + "The width of the slice (along the Z axis) is {:.3f} pc \n".format(
                2.0 * z_aver_width * 1e3
            )
            + "Time of the snapshot: {:6.4f} Myr \n".format(time[0])
            + "X-axis limits: [{:.5f}, {:.5f}] kpc \n".format(0.0, boxsize[0])
            + "Y-axis limits: [{:.5f}, {:.5f}] kpc \n".format(0.0, boxsize[1])
            + "Number of uniformily-spaced bins along the X axis: {} \n".format(N_bins)
            + "Number of uniformily-spaced bins along the Y axis: {} \n".format(N_bins)
            + "The initial ISM density is equal to 0.01 hydrogen particles per cubic cm \n"
            + "The mass fraction of hydrogen is {:.3f} \n".format(X_H)
            + "Total Fe60 mass released by SN is 1e-4 Msun\n"
        )

        np.savetxt(
            "./output_Alexander2/Fe60_density_{:03d}.dat".format(snp),
            ej_schene * const_ej,
            fmt="%.5e",
            header=header,
        )

        header = (
            "This file contains: gas radial velocity (v_r) [units: km/s]. \n"
            + "The width of the slice (along the Z axis) is {:.3f} pc \n".format(
                2.0 * z_aver_width * 1e3
            )
            + "Time of the snapshot: {:6.4f} Myr \n".format(time[0])
            + "X-axis limits: [{:.5f}, {:.5f}] kpc \n".format(0.0, boxsize[0])
            + "Y-axis limits: [{:.5f}, {:.5f}] kpc \n".format(0.0, boxsize[1])
            + "Number of uniformily-spaced bins along the X axis: {} \n".format(N_bins)
            + "Number of uniformily-spaced bins along the Y axis: {} \n".format(N_bins)
            + "The initial ISM density is equal to 0.01 hydrogen particles per cubic cm \n"
            + "The mass fraction of hydrogen is {:.3f} \n".format(X_H)
            + "Total Fe60 mass released by SN is 1e-4 Msun\n"
        )

        np.savetxt(
            "./output_Alexander2/velocity_r_{:03d}.dat".format(snp),
            velocity / 1e5,
            fmt="%.5e",
            header=header,
        )

        header = (
            "This file contains: gas velocity along X axis (v_x) [units: km/s]. \n"
            + "The width of the slice (along the Z axis) is {:.3f} pc \n".format(
                2.0 * z_aver_width * 1e3
            )
            + "Time of the snapshot: {:6.4f} Myr \n".format(time[0])
            + "X-axis limits: [{:.5f}, {:.5f}] kpc \n".format(0.0, boxsize[0])
            + "Y-axis limits: [{:.5f}, {:.5f}] kpc \n".format(0.0, boxsize[1])
            + "Number of uniformily-spaced bins along the X axis: {} \n".format(N_bins)
            + "Number of uniformily-spaced bins along the Y axis: {} \n".format(N_bins)
            + "The initial ISM density is equal to 0.01 hydrogen particles per cubic cm \n"
            + "The mass fraction of hydrogen is {:.3f} \n".format(X_H)
            + "Total Fe60 mass released by SN is 1e-4 Msun\n"
        )

        np.savetxt(
            "./output_Alexander2/velocity_x_{:03d}.dat".format(snp),
            velocity_x / 1e5,
            fmt="%.5e",
            header=header,
        )

        header = (
            "This file contains: gas velocity along Y axis (v_y) [units: km/s]. \n"
            + "The width of the slice (along the Z axis) is {:.3f} pc \n".format(
                2.0 * z_aver_width * 1e3
            )
            + "Time of the snapshot: {:6.4f} Myr \n".format(time[0])
            + "X-axis limits: [{:.5f}, {:.5f}] kpc \n".format(0.0, boxsize[0])
            + "Y-axis limits: [{:.5f}, {:.5f}] kpc \n".format(0.0, boxsize[1])
            + "Number of uniformily-spaced bins along the X axis: {} \n".format(N_bins)
            + "Number of uniformily-spaced bins along the Y axis: {} \n".format(N_bins)
            + "The initial ISM density is equal to 0.01 hydrogen particles per cubic cm \n"
            + "The mass fraction of hydrogen is {:.3f} \n".format(X_H)
            + "Total Fe60 mass released by SN is 1e-4 Msun\n"
        )

        np.savetxt(
            "./output_Alexander2/velocity_y_{:03d}.dat".format(snp),
            velocity_y / 1e5,
            fmt="%.5e",
            header=header,
        )

        plt.savefig(
            "./output_Alexander2/remnant_image_{:04d}.png".format(snp), bbox_inches="tight", pad_inches=0.1
        )
        plt.close()


if __name__ == "__main__":

    for i in range(34,35):
        print("Current snapshot number is {}".format(i))
        plot(i)
