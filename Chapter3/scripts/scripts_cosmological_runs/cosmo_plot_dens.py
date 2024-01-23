import numpy as np
import matplotlib.pylab as plt
import h5py

from velociraptor.observations import load_observations
from velociraptor.observations.objects import ObservationalData

import os
from astropy.cosmology import Planck15 as cosmo
from swiftsimio import load
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
from tqdm import tqdm
from typing import List
from matplotlib import ticker


def fixlogax(ax, a="x"):
    if a == "x":
        labels = [item.get_text() for item in ax.get_xticklabels()]
        positions = ax.get_xticks()
        for i in range(len(positions)):
            labels[i] = "$10^{" + str(int(np.log10(positions[i]))) + "}$"
        if np.size(np.where(positions == 1)) > 0:
            labels[np.where(positions == 1)[0][0]] = "$1$"
        if np.size(np.where(positions == 10)) > 0:
            labels[np.where(positions == 10)[0][0]] = "$10$"
        if np.size(np.where(positions == 0.1)) > 0:
            labels[np.where(positions == 0.1)[0][0]] = "$0.1$"
        ax.set_xticklabels(labels)
    if a == "y":
        labels = [item.get_text() for item in ax.get_yticklabels()]
        positions = ax.get_yticks()
        for i in range(len(positions)):
            labels[i] = "$10^{" + str(int(np.log10(positions[i]))) + "}$"
        if np.size(np.where(positions == 1)) > 0:
            labels[np.where(positions == 1)[0][0]] = "$1$"
        if np.size(np.where(positions == 10)) > 0:
            labels[np.where(positions == 10)[0][0]] = "$10$"
        if np.size(np.where(positions == 0.1)) > 0:
            labels[np.where(positions == 0.1)[0][0]] = "$0.1$"
        ax.set_yticklabels(labels)


EOS = 0

if not EOS:
    line_properties = {
        "colour": ["black", "orange", "dodgerblue", "sienna", "lightgrey"],
        "lw": [3.0, 2.0, 2.5, 2.0, 2.0, 6],
        "alpha": [1.0, 1.0, 1.0, 1.0, 1.0, 0.2],
        "ls": ["-", "-", "-", "-", "-", "-"],
        "marker": ["^", "s", "*", "o", "d"],
        "ms": [200 * 1.3, 230 * 1.3, 350 * 1.3, 220 * 1.3, 200],
    }
else:
    colors = plt.cm.cividis(np.linspace(0, 1, 2))
    line_properties = {
        "colour": [colors[0], colors[1], colors[0], colors[1]],
        "linewidth": [2, 3, 1.8, 2.8],
        "alpha": [1.0, 1.0, 1.0, 1.0],
        "ls": ["-", "-", "--", "--"],
        "marker": ["o", "o", "^", "^"],
    }


path_to_Berhoozi_smhm = "/cosma7/data/dp004/dc-chai1/pipeline-configs/observational_data/data/GalaxyStellarMassHaloMass/Behroozi2019Ratio.hdf5"
berhoozi_smhm_data = load_observations(
    path_to_Berhoozi_smhm, redshift_bracket=[-0.1, 0.1]
)

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["axes.linewidth"] = 2
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

redshift_ticks = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0])
redshift_labels = [
    "$0$",
    "$1$",
    "$2$",
    "$3$",
    "$5$",
    "$10$",
]
a_ticks = 1.0 / (redshift_ticks + 1.0)


label_size = 27
N_obj = 4


class SimInfo:
    def __init__(self, folder, snap_num, output_path, num_of_galaxies: int = 1):
        self.output_path = output_path
        self.num_of_galaxies = num_of_galaxies
        self.num_galaxies = num_of_galaxies
        self.snap_num = snap_num

        # Path to snapshot
        snapshot = os.path.join(folder, f"colibre_{snap_num:04d}.hdf5")

        print(f"colibre_{snap_num:04d}.hdf5")
        print(snapshot)
        self.snapshot = os.path.join(folder, f"colibre_{snap_num:04d}.hdf5")

        # Path to properties
        properties = os.path.join(folder, f"halo_{snap_num:04d}.properties.0")
        if os.path.exists(properties):
            self.subhalo_properties = os.path.join(
                folder, f"halo_{snap_num:04d}.properties.0"
            )
        else:
            self.subhalo_properties = os.path.join(
                folder, f"halo_{snap_num:04d}.properties"
            )

        # Path to group catalogue
        catalog = os.path.join(folder, f"halo_{snap_num:04d}.catalog_groups.0")
        if os.path.exists(catalog):
            self.catalog_groups = os.path.join(
                folder, f"halo_{snap_num:04d}.catalog_groups.0"
            )
        else:
            self.catalog_groups = os.path.join(
                folder, f"halo_{snap_num:04d}.catalog_groups"
            )

        # Path to particle catalogue
        catalog_particles = os.path.join(
            folder, f"halo_{snap_num:04d}.catalog_particles.0"
        )
        if os.path.exists(catalog_particles):
            self.catalog_particles = os.path.join(
                folder, f"halo_{snap_num:04d}.catalog_particles.0"
            )
        else:
            self.catalog_particles = os.path.join(
                folder, f"halo_{snap_num:04d}.catalog_particles"
            )

        with h5py.File(self.snapshot, "r") as snapshot_file:
            self.boxSize = snapshot_file["/Header"].attrs["BoxSize"][0]  # Mpc
            self.a = snapshot_file["/Header"].attrs["Scale-factor"][0]
            self.name = snapshot_file["/Header"].attrs["RunName"].decode()


class HaloCatalogue:
    def __init__(self, siminfo: SimInfo):
        with h5py.File(siminfo.subhalo_properties, "r") as properties:
            stellar_mass = properties["Aperture_mass_star_30_kpc"][:] * 1e10  # msun
            gas_mass = properties["Aperture_mass_gas_30_kpc"][:] * 1e10  # m sun
            halo_mass = properties["Mass_BN98"][:] * 1e10  # msun
            halo_mass_gas = properties["Mass_BN98_gas"][:] * 1e10  # msun
            halo_mass_star = properties["Mass_BN98_star"][:] * 1e10  # msun

            Xminpot = properties["Xcminpot"][:]
            Yminpot = properties["Ycminpot"][:]
            Zminpot = properties["Zcminpot"][:]

            half_mass_radius_star = properties["R_HalfMass_star"][:] * 1e3  # kpc
            half_mass_radius_gas = properties["R_HalfMass_gas"][:] * 1e3  # kpc

            sfr = (
                properties["Aperture_SFR_gas_30_kpc"][:] * 10227144.8879616 / 1e9
            )  # Msun/yr

            N_of_obj = 0
            args_sort_by_halo_mass = np.argsort(halo_mass)
            structure_type = properties["Structuretype"][:]

            for n_trial in range(siminfo.num_of_galaxies, siminfo.num_of_galaxies * 3):
                catalogue = args_sort_by_halo_mass[-n_trial:]
                centrals = np.where(structure_type[catalogue] == 10)[0]
                catalogue = catalogue[centrals]
                N_of_obj = len(catalogue)

                print(f"N trial: {n_trial}")
                print(f"Actual number of galaxies to show: {N_of_obj} \n")

                if N_of_obj == siminfo.num_of_galaxies:
                    break

            self.num_of_haloes = N_of_obj

            print(f"Expected number of galaxies to show: {siminfo.num_of_galaxies}")

            self.halo_index = [catalogue[i] for i in range(self.num_of_haloes)]

            self.log_stellar_mass_Msun = np.log10(stellar_mass[catalogue])
            self.log_gas_mass_Msun = np.log10(gas_mass[catalogue])
            self.log_halo_mass_Msun = np.log10(halo_mass[catalogue])
            self.log_halo_mass_gas_Msun = np.log10(halo_mass_gas[catalogue])
            self.log_halo_mass_star_Msun = np.log10(halo_mass_star[catalogue])

            self.halfmass_radius_star_kpc = half_mass_radius_star[catalogue]
            self.halfmass_radius_gas_kpc = half_mass_radius_gas[catalogue]

            self.X_pos_kpc = Xminpot[catalogue]
            self.Y_pos_kpc = Yminpot[catalogue]
            self.Z_pos_kpc = Zminpot[catalogue]

            self.star_formation_rate_Msun_yr = sfr[catalogue]

            self.birth_scale_factors = []
            self.stellar_masses_Msun = []
            self.birth_densities_H_cm3 = []
            self.snii_feedback_densities_H_cm3 = []

            print("Haloes' stellar masses", self.log_stellar_mass_Msun)


def make_masks(siminfo, halo):
    group_file = h5py.File(siminfo.catalog_groups, "r")
    particles_file = h5py.File(siminfo.catalog_particles, "r")
    snapshot_file = h5py.File(siminfo.snapshot, "r")

    star_ids = snapshot_file["/PartType4/ParticleIDs"][:]
    gas_ids = snapshot_file["/PartType0/ParticleIDs"][:]

    halo_start_position = group_file["Offset"][halo]
    halo_end_position = group_file["Offset"][halo + 1]

    particle_ids_in_halo = particles_file["Particle_IDs"][
        halo_start_position:halo_end_position
    ]

    _, _, mask_stars = np.intersect1d(
        particle_ids_in_halo,
        star_ids,
        assume_unique=True,
        return_indices=True,
    )

    _, _, mask_gas = np.intersect1d(
        particle_ids_in_halo,
        gas_ids,
        assume_unique=True,
        return_indices=True,
    )

    return mask_gas[mask_gas > 0], mask_stars[mask_stars > 0]


def plot_main(
    siminfo_list: List[SimInfo],
    halo_catalogue_data_list: List[HaloCatalogue],
    labels: List[str],
):
    bin_edges = np.logspace(-3.5, 5.5, 101)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    fig = plt.subplots(figsize=(20, 10))

    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((2, 4), (0, 2))
    ax3 = plt.subplot2grid((2, 4), (0, 3))
    ax4 = plt.subplot2grid((2, 4), (1, 2))
    ax5 = plt.subplot2grid((2, 4), (1, 3))

    ax = [ax1, ax2, ax3, ax4, ax5]

    plt.tight_layout(pad=8.0)

    ax[0].set_xlim(10**10.7, 2e13)
    ax[0].set_ylim(10.0**-3.1, 10.0**-0.9)
    ax[0].set_xlabel(
        r"$M_{\rm vir}$ $\left[\rm{M}_\odot\right]$", fontsize=label_size * 1.5
    )
    ax[0].set_ylabel(r"$M_* / M_{\rm vir}$", fontsize=label_size * 1.5)

    for idx, (label, catalogue) in enumerate(zip(labels, halo_catalogue_data_list)):
        print(idx, label, catalogue)

        Mhalo = 10.0**catalogue.log_halo_mass_Msun
        Mstar = 10.0**catalogue.log_stellar_mass_Msun
        Mratio = Mstar / Mhalo

        ax[0].scatter(
            Mhalo[-N_obj:],
            Mratio[-N_obj:],
            s=line_properties["ms"][idx],
            color=line_properties["colour"][idx],
            label=label.replace("_", "\_"),
            marker=line_properties["marker"][idx],
            lw=2,
            edgecolor="black",
            zorder=10 - idx,
        )

        # ax[0].scatter(
        #    Mhalo[:-N_obj],
        #    Mratio[:-N_obj],
        #    s=30,
        #    color=line_properties["colour"][idx],
        #    marker=line_properties["marker"][idx],
        # )

        for gal_plt, gal in enumerate(
            range(number_of_galaxies - 1, number_of_galaxies - 1 - N_obj, -1)
        ):
            print(
                gal,
                gal_plt,
                np.log10(np.sum(catalogue.stellar_masses_Msun[gal])),
                catalogue.log_stellar_mass_Msun[gal],
            )
            print(
                "POS (",
                np.round(catalogue.X_pos_kpc[gal], 3),
                np.round(catalogue.Y_pos_kpc[gal], 3),
                np.round(catalogue.Z_pos_kpc[gal], 3),
                ")",
            )

            scale_factors = catalogue.birth_scale_factors[gal]
            birth_n = catalogue.birth_densities_H_cm3[gal]
            snii_n = catalogue.snii_feedback_densities_H_cm3[gal]

            print("SIZE", np.size(snii_n))
            N_birth_binned, _, _ = stats.binned_statistic(
                birth_n,
                np.ones_like(birth_n) / np.size(birth_n),
                bins=bin_edges,
                statistic="count",
            )
            N_snii_binned, _, _ = stats.binned_statistic(
                snii_n,
                np.ones_like(snii_n) / np.size(snii_n),
                bins=bin_edges,
                statistic="count",
            )

            ax[gal_plt + 1].plot(
                bin_centers,
                np.cumsum(N_birth_binned / np.sum(N_birth_binned)),
                lw=line_properties["lw"][idx],
                color=line_properties["colour"][idx],
                alpha=line_properties["alpha"][idx],
                ls=line_properties["ls"][idx],
            )
            # label=label.replace("_","\_"))

            ax[gal_plt + 1].plot(
                bin_centers,
                np.cumsum(N_snii_binned / np.sum(N_snii_binned)),
                lw=2.5,
                color=line_properties["colour"][idx],
                alpha=line_properties["alpha"][idx],
                dashes=(5, 2),
            )

            if idx == 0:
                ax[gal_plt + 1].text(
                    0.75,
                    0.04,
                    "log$\\frac{M_{\\rm vir}}{\\mathrm{M_\\odot}}$"
                    + f"= \n {catalogue.log_halo_mass_Msun[gal]:.1f}",
                    ha="center",
                    va="bottom",
                    color="black",
                    alpha=1.0,
                    transform=ax[gal_plt + 1].transAxes,
                    fontsize=25.5,
                )

    line1 = ax[0].errorbar(
        berhoozi_smhm_data[0].x,
        berhoozi_smhm_data[0].y,
        berhoozi_smhm_data[0].y_scatter,
        color="grey",
        lw=3.0,
        dashes=(10, 2),
    )

    leg2 = ax[0].legend(
        [
            line1,
        ],
        [berhoozi_smhm_data[0].citation],
        loc="upper left",
        fontsize=25,
        frameon=False,
    )

    ax[0].add_artist(leg2)
    ax[0].legend(fontsize=25.5, loc="lower right", frameon=True)

    (line1,) = ax[2].plot([1e-7, 1e-8], [1e-7, 1e-8], lw=2.5, color="black")
    (line2,) = ax[2].plot(
        [1e-7, 1e-8], [1e-7, 1e-8], lw=2.5, dashes=(5.0, 2.0), color="black"
    )

    leg2 = ax[4].legend(
        [line1, line2],
        ["Stellar birth densities", "SN feedback densities"],
        loc="upper left",
        fontsize=23,
        ncol=2,
        bbox_to_anchor=(-1.5, 1.3),
    )

    if EOS:
        ax[3].plot(
            [1e3, 1e4], [1e3, 1e4], color="grey", ls="-", label="Without pressure floor"
        )
        ax[3].plot(
            [1e3, 1e4], [1e3, 1e4], color="grey", ls="--", label="With pressure floor"
        )
        ax[3].legend(fontsize=16, loc="center left", frameon=False)

    for i in range(5):
        ax[i].set_yscale("log")
        ax[i].set_xscale("log")

        if i > 0:
            ax[i].xaxis.set_tick_params(labelsize=label_size * 1.1)
            ax[i].yaxis.set_tick_params(labelsize=label_size * 1.1)
        else:
            ax[i].xaxis.set_tick_params(labelsize=label_size * 1.65)
            ax[i].yaxis.set_tick_params(labelsize=label_size * 1.65)

        if i > 0:
            ax[i].xaxis.set_ticks([1e-2, 1e0, 1e2, 1e4])
            ax[i].yaxis.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])

            ax[i].set_xlim(10 ** (-3.2), 10 ** (4.2))
            ax[i].set_ylim(1e-3, 2)

            fixlogax(ax[i], "x")
            fixlogax(ax[i], "y")

            locminx = ticker.LogLocator(base=100.0, subs=(0.05, 0.1, 0.5), numticks=10)
            locminy = ticker.LogLocator(
                base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=10
            )

            ax[i].xaxis.set_minor_locator(locminx)
            ax[i].yaxis.set_minor_locator(locminy)
            ax[i].xaxis.set_minor_formatter(ticker.NullFormatter())
            ax[i].yaxis.set_minor_formatter(ticker.NullFormatter())
            if i == 3 or i == 4:
                ax[i].set_xlabel("$n_{\\rm H}$ [cm$^{-3}$]", fontsize=label_size * 1.1)

            if i == 1 or i == 3:
                ax[i].set_ylabel("Fraction of particles", fontsize=label_size * 1.1)

        else:
            locmin = ticker.LogLocator(
                base=10.0,
                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                numticks=10,
            )

            ax[i].xaxis.set_minor_locator(locmin)
            ax[i].xaxis.set_minor_formatter(ticker.NullFormatter())

        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["axes.linewidth"] = 2
        ax[i].tick_params(which="both", width=1.7)
        if i == 0:
            ax[i].tick_params(which="major", length=14)
            ax[i].tick_params(which="minor", length=7)
        else:
            ax[i].tick_params(which="major", length=7)
            ax[i].tick_params(which="minor", axis="x", length=3.5)
            ax[i].tick_params(which="minor", axis="y", length=3.5)

        ax[i].tick_params(
            axis="both",
            which="both",
            pad=8,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )

    if EOS:
        plt.savefig("cosmo_gal_eos.pdf", bbox_inches="tight", pad_inches=0.1)
    else:
        plt.savefig("cosmo_gal.pdf", bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    if not EOS:
        runs = {
            "COS_M5_isotropic": "ISO_MIDREF_ALL/",
            "COS_M5_min_distance": "MDI_MIDREF_ALL/",
            "COS_M5_mass_weighted": "RND_MIDREF_ALL/",
            "COS_M5_min_density": "MDE_MIDREF_ALL/",
            "COS_M5_max_density": "MAX_MIDREF_ALL/",
        }
    else:
        runs = {
            "COS_M5_isotropic": "ISO_MIDREF_ALL/",
            "COS_M5_mass_weighted": "RND_MIDREF_ALL/",
            "COS_M5_isotropic_eos": "ISO_MIDREF_ALL_EOS/",
            "COS_M5_mass_weighted_eos": "RND_MIDREF_ALL_EOS/",
        }

    num = 23

    number_of_galaxies = 10
    path_out = "./output/"

    siminfo_array = []
    halo_data_array = []
    labels_array = []

    for name, path_in in runs.items():
        print(name, path_in)

        siminfo = SimInfo(
            folder=path_in,
            snap_num=num,
            output_path=path_out,
            num_of_galaxies=number_of_galaxies,
        )

        print(f"Run's name: {siminfo.name:s}")
        print(f"Run's scale-factor: {siminfo.a:.3f}")

        halo_data = HaloCatalogue(siminfo)
        data = load(siminfo.snapshot)

        Mstar = data.stars.masses.to("Msun").value
        a_birth = data.stars.birth_scale_factors.value

        n_birth = data.stars.birth_densities.to("g/cm**3").value / 1.673e-24

        X_gas = data.gas.element_mass_fractions.hydrogen.value
        X_stars = data.stars.element_mass_fractions.hydrogen.value
        n_snii_stars = (
            data.stars.last_sniifeedback_densities.to("g/cm**3").value
            / 1.673e-24
            * X_stars
        )

        print("Snapshot is loaded!")

        # Loop over the sample to calculate morphological parameters
        for i in tqdm(range(halo_data.num_of_haloes)):
            # Read particle data
            mask_gas, mask_stars = make_masks(siminfo, halo_data.halo_index[i])

            print(f"Number of stellar particles in halo {i} is: {len(mask_stars)}")

            if len(mask_stars) == 0:
                print(f"Warning!!! Halo {halo_data.halo_index[i]} has no stars data!!!")
                continue

            halo_data.birth_scale_factors.append(a_birth[mask_stars])
            halo_data.stellar_masses_Msun.append(Mstar[mask_stars])

            n_snii_gal = n_snii_stars[mask_stars]

            halo_data.birth_densities_H_cm3.append(n_birth[mask_stars])
            halo_data.snii_feedback_densities_H_cm3.append(
                (n_snii_gal[np.where(n_snii_gal > 0.0)])
            )

        siminfo_array.append(siminfo)
        halo_data_array.append(halo_data)
        labels_array.append(name)

    plot_main(siminfo_array, halo_data_array, labels_array)
