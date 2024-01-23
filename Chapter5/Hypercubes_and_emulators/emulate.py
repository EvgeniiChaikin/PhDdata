import swiftemulator as se
import numpy as np
import matplotlib.pylab as plt

from velociraptor.observations import load_observations
from velociraptor.observations.objects import ObservationalData

import matplotlib as mpl
import emcee
import corner
import unyt

from glob import glob
from pathlib import Path
from tqdm import tqdm
from matplotlib.colors import Normalize
from typing import Tuple, Any, Union

import os

from astropy.cosmology import Planck15 as cosmology

import yaml
import json

from matplotlib.ticker import MultipleLocator, AutoMinorLocator

from swiftemulator.io.swift import load_parameter_files, load_pipeline_outputs
from swiftemulator.emulators.gaussian_process import GaussianProcessEmulator
from swiftemulator.mean_models import LinearMeanModel
from swiftemulator.mocking import mock_hypercube
from swiftemulator.backend.model_specification import ModelSpecification
from swiftemulator.backend.model_parameters import ModelParameters
from swiftemulator.backend.model_values import ModelValues
from swiftemulator.sensitivity import cross_check

from swiftemulator.mocking import mock_sweep
from scipy import stats

from matplotlib.ticker import AutoMinorLocator
from scipy import interpolate

TICK_LENGTH_MAJOR = 9
TICK_LENGTH_MINOR = 5
TICK_WIDTH = 1.7
PLOT_SIZE = 8
LABEL_SIZE = 30
REREFENCE_RUN_NAME = "Reference"

plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["axes.linewidth"] = 2
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


class ObsData(object):
    def __init__(self, path_to_observational_data: str) -> None:
        # GSMF
        self.leja_gsmf = load_observations(
            f"{path_to_observational_data}/data/GalaxyStellarMassFunction/Leja_2020.hdf5",
            redshift_bracket=[1.9, 2.1],
        )
        self.driver_gsmf = load_observations(
            f"{path_to_observational_data}/data/GalaxyStellarMassFunction/Driver2021.hdf5",
            redshift_bracket=[-0.1, 0.1],
        )
        self.berhoozi_gsmf = load_observations(
            f"{path_to_observational_data}/data/GalaxyStellarMassFunction/Behroozi2019_all.hdf5",
            redshift_bracket=[-0.1, 0.1],
        )

        # SMHM
        self.berhoozi_smhm_data = load_observations(
            f"{path_to_observational_data}/data/GalaxyStellarMassHaloMass/Behroozi2019Ratio.hdf5",
            redshift_bracket=[-0.1, 0.1],
        )
        self.moster_smhm_data = load_observations(
            f"{path_to_observational_data}/data/GalaxyStellarMassHaloMass/Moster2018Ratio.hdf5",
            redshift_bracket=[-0.1, 0.1],
        )

        # Sizes
        self.lange_sizes = load_observations(
            f"{path_to_observational_data}/data/GalaxyStellarMassGalaxySize/Lange2015rBand_ms.hdf5",
            redshift_bracket=[-0.1, 0.1],
        )

        xGASS_raw = np.loadtxt(
            f"{path_to_observational_data}/data/GalaxyStellarMassGalaxySize/xGASS_sizes.dat",
        )
        self.hardwick_sizes = {
            "x": 10.0 ** xGASS_raw[:, 1],
            "y": 10.0 ** xGASS_raw[:, -2],
            "log_error": xGASS_raw[:, -1],
            "log_error_population": xGASS_raw[:, -1] / 1.253 * np.sqrt(xGASS_raw[:, 0]),
            "citation": "Hardwick et al. (2022, xGASS)",
        }
        self.hardwick_sizes["error"] = [
            -(10.0 ** (xGASS_raw[:, -2] - self.hardwick_sizes["log_error"]))
            + 10.0 ** (xGASS_raw[:, -2]),
            (10.0 ** (xGASS_raw[:, -2] + self.hardwick_sizes["log_error"]))
            - 10.0 ** (xGASS_raw[:, -2]),
        ]
        self.hardwick_sizes["error_population"] = [
            -(10.0 ** (xGASS_raw[:, -2] - self.hardwick_sizes["log_error_population"]))
            + 10.0 ** (xGASS_raw[:, -2]),
            (10.0 ** (xGASS_raw[:, -2] + self.hardwick_sizes["log_error_population"]))
            - 10.0 ** (xGASS_raw[:, -2]),
        ]

        # Cold gas
        self.Saintonge_h2 = load_observations(
            f"{path_to_observational_data}/data/GalaxyH2Fractions/Saintonge2017_abcissa_M_star.hdf5",
            redshift_bracket=[-0.1, 10.1],
        )
        self.Hunt_h2 = load_observations(
            f"{path_to_observational_data}/data/GalaxyH2Fractions/Hunt2020_Data.hdf5",
            redshift_bracket=[-0.1, 10.1],
        )
        self.Catinella_hi = load_observations(
            f"{path_to_observational_data}/data/GalaxyHIFractions/Catinella2018_abcissa_M_star.hdf5",
            redshift_bracket=[-0.1, 10.1],
        )
        self.Hunt_hi = load_observations(
            f"{path_to_observational_data}/data/GalaxyHIFractions/Hunt2020_Data.hdf5",
            redshift_bracket=[-0.1, 10.1],
        )

        class ObservationalDataInstance(object):
            """
            Holds observational data.
            """

            def __init__(self, scale_factor, sfr, error, description):
                self.scale_factor = scale_factor
                self.sfr = sfr
                self.error = error
                self.description = description

                if self.error is None:
                    self.fitting_formula = True
                else:
                    self.fitting_formula = False

        # SFR from Madau & Dickinson fitting formula (z < 10 and assuming h=0.7)
        obs15_a = np.logspace(np.log10(1.0 / (1.0 + 10.0)), 0, 100)
        obs15_z = 1.0 / obs15_a - 1.0
        obs15_rho = (
            0.015 * ((1.0 + obs15_z) ** 2.7) / (1.0 + ((1.0 + obs15_z) / 2.9) ** 5.6)
        )  # Msun / yr / Mpc^3
        obs15_rho /= 1.65  # Salpeter -> Chabrier correction

        self.sfh_madau = ObservationalDataInstance(
            obs15_a, obs15_rho, None, "Madau & Dickinson (2014)"
        )

        # Add radio data
        self.extra_sfh_observational_data_sample = load_observations(
            [
                f"{path_to_observational_data}/data/StarFormationRateHistory/Novak2017.hdf5",
                f"{path_to_observational_data}/data/StarFormationRateHistory/Gruppioni2020.hdf5",
                f"{path_to_observational_data}/data/StarFormationRateHistory/Enia2022.hdf5",
                f"{path_to_observational_data}/data/StarFormationRateHistory/Khusanova2021.hdf5",
                f"{path_to_observational_data}/data/StarFormationRateHistory/Cochrane2023.hdf5",
            ]
        )
        # Add Behroozi data
        self.sfh_behroozi2019 = load_observations(
            [
                f"{path_to_observational_data}/data/StarFormationRateHistory/Behroozi2019_true.hdf5",
                f"{path_to_observational_data}/data/StarFormationRateHistory/Behroozi2019_observed.hdf5",
            ]
        )

        return


class DataCube(object):
    def __init__(self, global_path: str = os.getcwd()) -> None:
        self.data = read_sim_data(f"{global_path}/config.json")

        self.parameters = self.data["input"]["parameters"]
        self.parameter_printable_names = self.data["input"]["parameter_printable_names"]
        self.parameter_limits = self.data["input"]["parameter_limits"]
        self.log_params = self.data["input"]["log_params"]
        self.log_params_short = [string.split(":")[-1] for string in self.log_params]

        self.path_to_params = self.data["input"]["paths"]["parameter_values"]
        self.path_to_data_z0p0 = self.data["input"]["paths"]["data_z0p0"]
        self.path_to_data_z2p0 = self.data["input"]["paths"]["data_z2p0"]
        self.path_to_data_sfh = self.data["input"]["paths"]["data_sfh"]
        self.path_to_reference_config = self.data["input"]["paths"][
            "reference_config_directory"
        ]
        self.path_to_reference_data = self.data["input"]["paths"][
            "reference_data_directory"
        ]
        self.path_to_data = self.data["input"]["paths"]["data_directory"]
        self.reference_config_file_name = self.data["input"][
            "reference_config_file_name"
        ]
        self.num_runs = self.data["input"]["number_of_runs"]

        self.path_to_output_plots = self.data["output"]["paths"]["path_to_plots"]

        self.emulator_parameters = self.data["emulator"]
        self.emulator_errors = {}
        self.plot_config = self.data["plots"]

        assert (
            self.data["plots"]["num_panels_x"] * self.data["plots"]["num_panels_y"]
            == self.num_runs
        ), "number of rows or/and columns is incorrect!"

        return

    def _load_params(self, path: str) -> Tuple[Any]:
        files = [Path(x) for x in glob(f"./{path}/*")]
        filenames = {filename.stem: filename for filename in files}

        e_spec_temp, e_params_temp = load_parameter_files(
            filenames=filenames,
            log_parameters=self.log_params,
            parameters=self.parameters,
            parameter_printable_names=self.parameter_printable_names,
        )
        return e_spec_temp, e_params_temp

    def load_params(self) -> None:
        self.e_spec, self.e_params = self._load_params(path=f"{self.path_to_params}")
        self.e_spec_ref, self.e_params_ref = self._load_params(
            path=f"{self.path_to_reference_config}"
        )

        self.runs = {}
        self.number_of_parameters = len(self.parameters)

        for run in os.scandir(f"./{self.path_to_params}/"):
            with open(f"./{self.path_to_params}/{run.name}") as file:
                data = yaml.full_load(file)

                self.runs[run.name] = []
                for param_idx in range(self.number_of_parameters):
                    section, param = self.parameters[param_idx].split(":")

                    if param in self.log_params_short:
                        self.runs[run.name].append(np.log10(data[section][param]))
                    else:
                        self.runs[run.name].append(data[section][param])

        return

    def plot_params(self) -> None:
        params = [
            [self.runs[run][i] for run in self.runs]
            for i in range(self.number_of_parameters)
        ]
        print(params)

        fig, ax = plt.subplots(
            self.number_of_parameters, self.number_of_parameters, figsize=(13, 13)
        )

        label_size = 16

        for row in range(self.number_of_parameters):
            for column in range(self.number_of_parameters):
                if row >= column:
                    if row > column:
                        ax[row, column].scatter(
                            [
                                self.e_params_ref.model_parameters[
                                    self.reference_config_file_name.removesuffix(".yml")
                                ][self.parameters[column]]
                            ],
                            [
                                self.e_params_ref.model_parameters[
                                    self.reference_config_file_name.removesuffix(".yml")
                                ][self.parameters[row]]
                            ],
                            s=180,
                            color="red",
                            marker="^",
                            zorder=30 * 2,
                        )

                if row < column:
                    ax[row, column].axis("off")
                elif column < row:
                    ax[row, column].scatter(
                        params[column],
                        params[row],
                        s=25 * 2,
                        color="orange",
                        marker="+",
                    )
                else:
                    (n, bins, patches) = ax[row, column].hist(
                        params[column],
                        color="orange",
                        histtype="step",
                        lw=3,
                        bins=12,
                        zorder=3,
                        alpha=0.7,
                    )

                if row < self.number_of_parameters - 1:
                    pass
                    ax[row, column].set_xticklabels([])
                else:
                    ax[row, column].set_xlabel(
                        self.parameter_printable_names[column], fontsize=label_size
                    )
                    ax[row, column].xaxis.set_tick_params(labelsize=label_size)

                if column > 0:
                    pass
                    ax[row, column].set_yticklabels([])
                else:
                    ax[row, column].set_ylabel(
                        self.parameter_printable_names[row], fontsize=label_size
                    )
                    ax[row, column].yaxis.set_tick_params(labelsize=label_size)

        fig.tight_layout()

        ax[0, -1].text(
            0.75,
            0.75,
            "COLIBRE CALIBRATION \n PARAMETER SPACE",
            ha="right",
            va="top",
            transform=ax[0, -1].transAxes,
            fontsize=40,
        )
        plt.savefig(
            f"{self.path_to_output_plots}/parameter_space_Colibre_main.pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def _load_data(self, path: str) -> Tuple[Any]:
        value_files_z0p0 = [
            Path(x) for x in glob(f"./{path}/{self.path_to_data_z0p0}/*")
        ]
        value_files_z2p0 = [
            Path(x) for x in glob(f"./{path}/{self.path_to_data_z2p0}/*")
        ]
        value_files_sfr = [Path(x) for x in glob(f"./{path}/{self.path_to_data_sfh}/*")]

        filenames_z0p0 = {filename.stem: filename for filename in value_files_z0p0}
        filenames_z2p0 = {filename.stem: filename for filename in value_files_z2p0}
        filenames_sfr = {filename.stem: filename for filename in value_files_sfr}

        valuesz0, unitsz0 = load_pipeline_outputs(
            filenames=filenames_z0p0,
            scaling_relations=[
                "stellar_mass_halo_mass_MBN98_centrals_ratio_50",
                "stellar_mass_projected_galaxy_size_50",
                "stellar_mass_specific_sfr_all_50",
                "adaptive_stellar_mass_function_with_scatter_50",
            ],
            log_independent=[
                "stellar_mass_halo_mass_MBN98_centrals_ratio_50",
                "stellar_mass_projected_galaxy_size_50",
                "stellar_mass_specific_sfr_all_50",
                "adaptive_stellar_mass_function_with_scatter_50",
            ],
            log_dependent=[
                "stellar_mass_halo_mass_MBN98_centrals_ratio_50",
                "stellar_mass_projected_galaxy_size_50",
                "stellar_mass_specific_sfr_all_50",
                "adaptive_stellar_mass_function_with_scatter_50",
            ],
        )

        valuesz0_z2p0, unitsz0_z2p0mf = load_pipeline_outputs(
            filenames=filenames_z2p0,
            scaling_relations=["adaptive_stellar_mass_function_with_scatter_50"],
            log_independent=["adaptive_stellar_mass_function_with_scatter_50"],
            log_dependent=["adaptive_stellar_mass_function_with_scatter_50"],
        )

        values_sfh, unitsz0_sfh = load_pipeline_outputs(
            filenames=filenames_sfr,
            scaling_relations=["SFH"],
            log_independent=[],
            log_dependent=["SFH"],
        )

        smhms = valuesz0["stellar_mass_halo_mass_MBN98_centrals_ratio_50"]
        sizes = valuesz0["stellar_mass_projected_galaxy_size_50"]
        ssfr = valuesz0["stellar_mass_specific_sfr_all_50"]
        gsmfz0p0 = valuesz0["adaptive_stellar_mass_function_with_scatter_50"]
        gsmfz2p0 = valuesz0_z2p0["adaptive_stellar_mass_function_with_scatter_50"]
        sfhs = values_sfh["SFH"]

        return (smhms, sizes, ssfr, gsmfz0p0, gsmfz2p0, sfhs)

    def load_data(self) -> None:
        (
            self.smhms_ref,
            self.sizes_ref,
            self.ssfr_ref,
            self.gsmfz0p0_ref,
            self.gsmfz2p0_ref,
            self.sfhs_ref,
        ) = self._load_data(path=f"{self.path_to_reference_data}")

        (
            self.smhms,
            self.sizes,
            self.ssfr,
            self.gsmfz0p0,
            self.gsmfz2p0,
            self.sfhs,
        ) = self._load_data(path=f"{self.path_to_data}")
        return

    def _create_emulator(
        self,
        x_bins_edges: Union[None, np.ndarray],
        data: Any,
        x_min: float,
        x_max: float,
        error_magnitude=1.0,
        name=None,
    ) -> GaussianProcessEmulator:
        print("Creating emulator...")

        assert x_min < x_max, "Xmin has to be smaller than Xmax!"
        assert error_magnitude > 0.0, "Error magnitude has to be positive"

        data_filtered = {}

        if data is None:
            print("DATA IS NONE")
            return None, None

        for keyname in data.model_values:
            x_values = data[keyname]["independent"]
            x_mask = np.logical_and(x_values > x_min, x_values < x_max)

            x_values_masked = x_values[x_mask]
            y_values_masked = data.model_values[keyname]["dependent"][x_mask]
            y_values_err_masked = data.model_values[keyname]["dependent_error"][x_mask]

            data_onekey = {}

            if x_bins_edges is not None:
                bin_x = []
                bin_y = []
                bin_y_err = []

                for left, right in zip(x_bins_edges[:-1], x_bins_edges[1:]):
                    mask_bin = np.logical_and(
                        x_values_masked > left, x_values_masked < right
                    )

                    if True in mask_bin:
                        bin_x.append(np.mean(x_values_masked[mask_bin]))
                        bin_y.append(np.mean(y_values_masked[mask_bin]))
                        bin_y_err.append(np.mean(y_values_err_masked[mask_bin]))

                data_onekey["independent"] = np.array(bin_x)
                data_onekey["dependent"] = np.array(bin_y)

                if name == "sfhs":
                    data_onekey["dependent_error"] = np.array(bin_y) * error_magnitude
                else:
                    data_onekey["dependent_error"] = (
                        np.array(bin_y_err) * error_magnitude
                    )

            else:
                data_onekey["independent"] = x_values_masked
                data_onekey["dependent"] = y_values_masked
                data_onekey["dependent_error"] = (
                    np.array(y_values_err_masked) * error_magnitude
                )

            data_filtered[keyname] = data_onekey

        data_updated = ModelValues(data_filtered)
        emulator = GaussianProcessEmulator()
        emulator.fit_model(
            model_specification=self.e_spec,
            model_parameters=self.e_params,
            model_values=data_updated,
        )
        return emulator, data_updated

    def create_emulators(self):
        for key in self.emulator_parameters.keys():
            if key == "sfhs":
                x_bin_edges = np.concatenate(
                    [
                        np.linspace(
                            self.emulator_parameters[key]["x_min"],
                            1.0,
                            self.emulator_parameters[key]["num_bins"],
                        ),
                        np.linspace(
                            1.0,
                            self.emulator_parameters[key]["x_max"],
                            self.emulator_parameters[key]["num_bins"],
                        )[1:],
                    ]
                )
            elif self.emulator_parameters[key]["num_bins"] > 0:
                x_bin_edges = np.linspace(
                    self.emulator_parameters[key]["x_min"],
                    self.emulator_parameters[key]["x_max"],
                    self.emulator_parameters[key]["num_bins"] + 1,
                )
            else:
                x_bin_edges = None

            emulator, data_updated = self._create_emulator(
                x_bins_edges=x_bin_edges,
                data=self.__getattribute__(key),
                x_min=self.emulator_parameters[key]["x_min"],
                x_max=self.emulator_parameters[key]["x_max"],
                error_magnitude=self.emulator_parameters[key]["error_magnitude"],
                name=key,
            )
            self.__setattr__(f"gpe_{key}", emulator)

            if key == "gsmfz0p0" or key == "sizes":
                print(f"Doing cross check for {key}")

                ccheck = cross_check.CrossCheck()
                ccheck.build_emulators(
                    model_specification=self.e_spec,
                    model_parameters=self.e_params,
                    model_values=data_updated,
                )
                data_by_cc = ccheck.build_mocked_model_values_original_independent()

                emulator_error = np.array([])

                fig, ax = plt.subplots(1, 1)
                for unique_identifier in range(self.num_runs):
                    diff = (
                        data_by_cc[f"{unique_identifier}"]["dependent"]
                        - data_updated[f"{unique_identifier}"]["dependent"]
                    )
                    emulator_error = np.concatenate([emulator_error, diff])
                    ax.plot(
                        data_by_cc[f"{unique_identifier}"]["independent"],
                        diff,
                        lw=1,
                        alpha=0.5,
                    )

                ax.axhline(y=3.0 * np.std(emulator_error), color="k", lw=3)
                ax.axhline(y=-3.0 * np.std(emulator_error), color="k", lw=3)
                ax.axhline(y=np.std(emulator_error), color="k", lw=2, dashes=(3, 3))
                ax.axhline(y=-np.std(emulator_error), color="k", lw=2, dashes=(3, 3))

                ax.set_ylim(-0.4, 0.4)

                plt.title(
                    f"{key}: std-err ({np.std(emulator_error):.2f}), mean-abs-err ({np.mean(np.abs(emulator_error)):.2f})"
                )
                plt.savefig(f"emulator_error_{key}.pdf")
                plt.close()

                print(f"Finishing doing cross check for {key}")
                print("Num points", np.shape(emulator_error))
                print("Max error", np.max(np.abs(emulator_error)))
                print("Mean error", np.mean(emulator_error))
                print("Mean abs error", np.mean(np.abs(emulator_error)))
                print("Median abs error", np.median(np.abs(emulator_error)))
                print("Std error", np.std(emulator_error))
                print("3 * std error", 3.0 * np.std(emulator_error))
                print(" ")

                self.emulator_errors[key] = emulator_error

            print("setting key", f"{key}_updated", "...")
            self.__setattr__(f"{key}_updated", data_updated)

        return


class Plots(object):
    def __init__(self) -> None:
        path_to_obs_data = (
            "/cosma7/data/dp004/dc-chai1/pipeline-configs/observational_data/"
        )
        self.observational_data = ObsData(path_to_observational_data=path_to_obs_data)

        self.data_cube = DataCube()
        self.data_cube.load_params()
        self.data_cube.load_data()
        self.data_cube.create_emulators()
        self.data_cube.plot_params()

        pass

    def create_gsmfz2p0_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_SIZE, PLOT_SIZE))

        ax.set_ylabel(
            r"$\mathrm{d}n(M)/\mathrm{dlog10} \, M$ [Mpc$^{-3}$]", fontsize=LABEL_SIZE
        )
        ax.set_xlabel(r"$M_*$  $\left[\rm{M}_\odot\right]$", fontsize=LABEL_SIZE)

        ax.set_xscale("log")
        ax.set_yscale("log")

        (line_ref,) = plt.plot(
            10 ** self.data_cube.gsmfz2p0_ref["data_0000"]["independent"],
            10 ** self.data_cube.gsmfz2p0_ref["data_0000"]["dependent"],
            color="red",
            lw=3.5,
            dashes=(3, 3),
            zorder=10,
        )

        ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

        line_leja = ax.errorbar(
            self.observational_data.leja_gsmf[0].x,
            self.observational_data.leja_gsmf[0].y,
            self.observational_data.leja_gsmf[0].y_scatter,
            color="grey",
            lw=3.0,
            fmt="o",
            zorder=-3,
        )

        leg = plt.legend(
            [line_ref, line_leja],
            [REREFENCE_RUN_NAME, self.observational_data.leja_gsmf[0].citation],
            loc="lower left",
            frameon=False,
            fontsize=20,
        )

        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
        ax.set_xticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
        ax.set_ylim(1e-4, 1e-1)
        ax.set_xlim(1e7, 1e12)

        ax.tick_params(which="both", width=TICK_WIDTH)
        ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
        ax.tick_params(which="minor", length=TICK_LENGTH_MINOR)
        ax.tick_params(
            axis="both",
            which="both",
            pad=8,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )

        return fig, ax, leg

    def create_gsmfz0p0_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_SIZE, PLOT_SIZE))

        ax.set_ylabel(
            r"$\mathrm{d}n(M)/\mathrm{dlog10} \, M$ [Mpc$^{-3}$]", fontsize=LABEL_SIZE
        )
        ax.set_xlabel(r"$M_*$  $\left[\rm{M}_\odot\right]$", fontsize=LABEL_SIZE)

        ax.set_xscale("log")
        ax.set_yscale("log")

        (line_ref,) = plt.plot(
            10 ** self.data_cube.gsmfz0p0_ref["data_0001"]["independent"],
            10 ** self.data_cube.gsmfz0p0_ref["data_0001"]["dependent"],
            color="red",
            lw=3.5,
            dashes=(3, 3),
            zorder=10,
        )

        ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

        line_driver = ax.errorbar(
            self.observational_data.driver_gsmf[0].x,
            self.observational_data.driver_gsmf[0].y,
            self.observational_data.driver_gsmf[0].y_scatter,
            color="grey",
            lw=3.0,
            fmt="o",
            zorder=-3,
        )

        leg = plt.legend(
            [line_ref, line_driver],
            [REREFENCE_RUN_NAME, self.observational_data.driver_gsmf[0].citation],
            loc="lower left",
            frameon=False,
            fontsize=20,
        )

        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
        ax.set_xticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
        ax.set_ylim(0.5e-4, 5e-1)
        ax.set_xlim(1e7, 5e12)

        ax.tick_params(which="both", width=TICK_WIDTH)
        ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
        ax.tick_params(which="minor", length=TICK_LENGTH_MINOR)
        ax.tick_params(
            axis="both",
            which="both",
            pad=8,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )

        return fig, ax, leg

    def create_smhms_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_SIZE, PLOT_SIZE))

        ax.set_xlabel(
            r"$M_{\rm BN98}$ $\left[\rm{M}_\odot\right]$", fontsize=LABEL_SIZE
        )
        ax.set_ylabel(r"$M_* / M_{\rm BN98}$ ($50$ kpc)", fontsize=LABEL_SIZE)

        ax.set_xscale("log")
        ax.set_yscale("log")

        (line_ref,) = plt.plot(
            10 ** self.data_cube.smhms_ref["data_0001"]["independent"],
            10 ** self.data_cube.smhms_ref["data_0001"]["dependent"],
            color="red",
            lw=3.5,
            dashes=(3, 3),
            zorder=10,
        )

        ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

        line_moster = ax.errorbar(
            self.observational_data.moster_smhm_data[0].x,
            self.observational_data.moster_smhm_data[0].y,
            self.observational_data.moster_smhm_data[0].y_scatter,
            color="k",
            lw=3.0,
            fmt="o",
            zorder=-3,
        )
        line_berhoozi = ax.errorbar(
            self.observational_data.berhoozi_smhm_data[0].x,
            self.observational_data.berhoozi_smhm_data[0].y,
            self.observational_data.berhoozi_smhm_data[0].y_scatter,
            color="grey",
            lw=3.0,
            fmt=">",
            zorder=-3,
        )

        leg = plt.legend(
            [line_ref, line_moster, line_berhoozi],
            [
                REREFENCE_RUN_NAME,
                self.observational_data.moster_smhm_data[0].citation,
                self.observational_data.berhoozi_smhm_data[0].citation,
            ],
            loc="lower right",
            frameon=False,
            fontsize=20,
        )

        ax.set_yticks([1e-3, 1e-2, 1e-1])
        ax.set_xticks([1e10, 1e11, 1e12, 1e13, 1e14])
        ax.set_ylim(5e-4, 0.1)
        ax.set_xlim(5e10, 1e14)

        ax.tick_params(which="both", width=TICK_WIDTH)
        ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
        ax.tick_params(which="minor", length=TICK_LENGTH_MINOR)
        ax.tick_params(
            axis="both",
            which="both",
            pad=8,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )

        return fig, ax, leg

    def create_sizes_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_SIZE, PLOT_SIZE))

        ax.set_ylabel(r"Projected size [kpc]", fontsize=LABEL_SIZE)
        ax.set_xlabel(r"$M_*$  $\left[\rm{M}_\odot\right]$", fontsize=LABEL_SIZE)

        ax.set_xscale("log")
        ax.set_yscale("log")

        (line_ref,) = plt.plot(
            10 ** self.data_cube.sizes_ref["data_0001"]["independent"],
            10 ** self.data_cube.sizes_ref["data_0001"]["dependent"],
            color="red",
            lw=3.5,
            dashes=(3, 3),
            zorder=10,
        )

        ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

        line_hardwick = ax.errorbar(
            self.observational_data.hardwick_sizes["x"],
            self.observational_data.hardwick_sizes["y"],
            self.observational_data.hardwick_sizes["error"],
            color="grey",
            lw=3.0,
            fmt="-",
            zorder=-3,
        )

        ax.fill_between(
            self.observational_data.hardwick_sizes["x"],
            self.observational_data.hardwick_sizes["y"]
            - self.observational_data.hardwick_sizes["error_population"][0],
            self.observational_data.hardwick_sizes["y"]
            + self.observational_data.hardwick_sizes["error_population"][1],
            color="grey",
            alpha=0.25,
            zorder=-100,
        )

        leg = plt.legend(
            [line_ref, line_hardwick],
            [REREFENCE_RUN_NAME, self.observational_data.hardwick_sizes["citation"]],
            loc="lower right",
            frameon=False,
            fontsize=20,
        )

        ax.set_xticks([1e8, 1e9, 1e10, 1e11])
        ax.set_xlim(10**8.0, 10**11.5)
        ax.set_ylim(0.7, 10.0**1.0)

        ax.tick_params(which="both", width=TICK_WIDTH)
        ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
        ax.tick_params(which="minor", length=TICK_LENGTH_MINOR)
        ax.tick_params(
            axis="both",
            which="both",
            pad=8,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )

        return fig, ax, leg

    def create_sfhs_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_SIZE, PLOT_SIZE))

        ax.set_ylabel(r"Cosmic SFRD", fontsize=LABEL_SIZE)
        ax.set_xlabel(r"Redshift", fontsize=LABEL_SIZE)

        ax.set_xscale("log")
        ax.set_yscale("log")

        (line_ref,) = plt.plot(
            1.0 / (1.0 + self.data_cube.sfhs_ref["data_0001"]["independent"]),
            10 ** self.data_cube.sfhs_ref["data_0001"]["dependent"],
            color="red",
            lw=3.5,
            dashes=(3, 3),
        )

        ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

        (line_madau,) = ax.plot(
            self.observational_data.sfh_madau.scale_factor,
            self.observational_data.sfh_madau.sfr,
            lw=2,
            color="grey",
            dashes=(1, 1),
        )

        sfh_observation_lines = []
        sfh_observation_labels = []

        for extra_data, color, wavrange in zip(
            self.observational_data.extra_sfh_observational_data_sample,
            ["maroon", "goldenrod", "slategrey", "violet", "darkblue"],
            ["radio", "FIR", "radio", "FIR", "radio"],
        ):
            sfh_observation_lines.append(
                ax.errorbar(
                    extra_data.x.value,
                    extra_data.y.value,
                    xerr=None
                    if extra_data.x_scatter is None
                    else extra_data.x_scatter.value,
                    yerr=None
                    if extra_data.y_scatter is None
                    else extra_data.y_scatter.value,
                    linestyle="none",
                    marker="o",
                    color=color,
                    elinewidth=0.5,
                    markeredgecolor="none",
                    markersize=2,
                    zorder=-10,
                )
            )
            sfh_observation_labels.append(f"{extra_data.citation} [{wavrange}]")

        for Behroozi_data, color in zip(
            self.observational_data.sfh_behroozi2019, ["lime", "coral"]
        ):
            sfh_observation_lines.append(
                ax.fill_between(
                    Behroozi_data.x.value,
                    Behroozi_data.y.value - Behroozi_data.y_scatter[0].value,
                    Behroozi_data.y.value + Behroozi_data.y_scatter[1].value,
                    color=color,
                    zorder=-10000,
                    alpha=0.3,
                )
            )
            sfh_observation_labels.append(Behroozi_data.citation)

        leg = plt.legend(
            [line_ref, line_madau, *sfh_observation_lines],
            [
                "reference",
                self.observational_data.sfh_madau.description.replace("&", "\&"),
                *sfh_observation_labels,
            ],
            loc="lower center",
            frameon=False,
            fontsize=13,
        )

        ax.set_ylim(1.8e-4, 1.7)
        ax.set_xlim(1.02, 0.07)
        ax.set_xlim(1.02, 0.07)

        redshift_ticks = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
        redshift_labels = ["$0$", "$0.2$", "$0.5$", "$1$", "$2$", "$3$", "$5$", "$10$"]
        a_ticks = 1.0 / (redshift_ticks + 1.0)

        ax.set_xticks(a_ticks)
        ax.set_xticklabels(redshift_labels)

        ax.tick_params(which="both", width=TICK_WIDTH)
        ax.tick_params(which="major", length=TICK_LENGTH_MAJOR)
        ax.tick_params(which="minor", axis="y", length=TICK_LENGTH_MINOR)
        ax.tick_params(which="minor", axis="x", length=0)
        ax.tick_params(
            axis="both",
            which="both",
            pad=8,
            left=True,
            right=True,
            top=True,
            bottom=True,
        )

        return fig, ax, leg

    def create_emulator_fitting_diagnostics_plot_sfhs(self):
        label_size = 15

        redshifts = np.linspace(
            self.data_cube.emulator_parameters["sfhs"]["x_min"],
            self.data_cube.emulator_parameters["sfhs"]["x_max"],
            100,
        )

        n_col = self.data_cube.plot_config["num_panels_x"]
        n_row = self.data_cube.plot_config["num_panels_y"]

        fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3))

        for i in range(n_row):
            for j in range(n_col):
                ax[i, j].plot(
                    self.observational_data.sfh_madau.scale_factor,
                    self.observational_data.sfh_madau.sfr,
                    lw=2,
                    color="grey",
                    label=self.observational_data.sfh_madau.description,
                )

                number = j * n_row + i

                best_fit = self.data_cube.runs[f"{number}.yml"]
                best_fit_params = {
                    p: v for p, v in zip(self.data_cube.parameters, best_fit)
                }

                pred2, pred_var2 = self.data_cube.gpe_sfhs.predict_values(
                    redshifts, model_parameters=best_fit_params
                )

                ax[i, j].plot(
                    1.0 / (1.0 + redshifts),
                    10**pred2,
                    alpha=0.6,
                    color="deepskyblue",
                    lw=4,
                )
                ax[i, j].scatter(
                    1.0
                    / (
                        self.data_cube.sfhs_updated.model_values[f"{number}"][
                            "independent"
                        ]
                        + 1
                    ),
                    10
                    ** self.data_cube.sfhs_updated.model_values[f"{number}"][
                        "dependent"
                    ],
                    s=50,
                    marker="^",
                    color="deepskyblue",
                )

                ax[i, j].set_xscale("log")
                ax[i, j].set_yscale("log")

                ax[i, j].xaxis.set_tick_params(labelsize=label_size)
                ax[i, j].yaxis.set_tick_params(labelsize=label_size)

                plt.rcParams.update({"figure.autolayout": True})
                plt.rcParams["ytick.direction"] = "in"
                plt.rcParams["xtick.direction"] = "in"
                plt.rcParams["axes.linewidth"] = 2
                ax[i, j].tick_params(which="both", width=1.7)
                ax[i, j].tick_params(which="major", length=9)
                ax[i, j].tick_params(which="minor", length=5)
                ax[i, j].tick_params(
                    axis="both",
                    which="both",
                    pad=8,
                    left=True,
                    right=True,
                    top=True,
                    bottom=True,
                )
                ax[i, j].set_ylim(1.8e-4, 1.7)
                ax[i, j].set_xlim(1.02, 0.07)
                ax[i, j].set_xlim(1.02, 0.07)

                redshift_ticks = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
                redshift_labels = [
                    "$0$",
                    "$0.2$",
                    "$0.5$",
                    "$1$",
                    "$2$",
                    "$3$",
                    "$5$",
                    "$10$",
                ]
                a_ticks = 1.0 / (redshift_ticks + 1.0)

                ax[i, j].set_xticks(a_ticks)
                ax[i, j].set_xticklabels(redshift_labels)

                if j == 0:
                    ax[i, j].set_ylabel(r"Cosmic SFRD", fontsize=label_size)
                else:
                    ax[i, j].set_yticklabels([])

                if i == n_row - 1:
                    ax[i, j].set_xlabel(r"Redshift", fontsize=label_size)
                else:
                    ax[i, j].set_xticklabels([])

        plt.savefig(
            f"{self.data_cube.path_to_output_plots}/emulator_example_sfhs.pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def create_emulator_fitting_diagnostics_plot_gsmf_z0p0(self):
        label_size = 15

        Mstar = np.linspace(
            self.data_cube.emulator_parameters["gsmfz0p0"]["x_min"],
            self.data_cube.emulator_parameters["gsmfz0p0"]["x_max"],
            100,
        )

        n_col = self.data_cube.plot_config["num_panels_x"]
        n_row = self.data_cube.plot_config["num_panels_y"]

        fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3))

        for i in range(n_row):
            for j in range(n_col):
                number = j * n_row + i

                ax[i, j].errorbar(
                    self.observational_data.driver_gsmf[0].x,
                    self.observational_data.driver_gsmf[0].y,
                    self.observational_data.driver_gsmf[0].y_scatter,
                    color="grey",
                    lw=3.0,
                    fmt="o",
                    zorder=-3,
                    label=self.observational_data.driver_gsmf[0].citation,
                )

                best_fit = self.data_cube.runs[f"{number}.yml"]
                best_fit_params = {
                    p: v for p, v in zip(self.data_cube.parameters, best_fit)
                }

                pred2, pred_var2 = self.data_cube.gpe_gsmfz0p0.predict_values(
                    Mstar, model_parameters=best_fit_params
                )

                ax[i, j].fill_between(
                    10**Mstar,
                    10 ** (pred2 - np.sqrt(pred_var2)),
                    10 ** (pred2 + np.sqrt(pred_var2)),
                    color="deepskyblue",
                    alpha=0.2,
                )
                ax[i, j].plot(
                    10**Mstar, 10**pred2, alpha=0.6, color="deepskyblue", lw=4
                )
                ax[i, j].scatter(
                    10
                    ** self.data_cube.gsmfz0p0_updated.model_values[f"{number}"][
                        "independent"
                    ],
                    10
                    ** self.data_cube.gsmfz0p0_updated.model_values[f"{number}"][
                        "dependent"
                    ],
                    s=100,
                    marker="^",
                    color="deepskyblue",
                )

                ax[i, j].set_yscale("log")
                ax[i, j].set_xscale("log")

                ax[i, j].xaxis.set_tick_params(labelsize=label_size)
                ax[i, j].yaxis.set_tick_params(labelsize=label_size)

                plt.rcParams.update({"figure.autolayout": True})
                plt.rcParams["ytick.direction"] = "in"
                plt.rcParams["xtick.direction"] = "in"
                plt.rcParams["axes.linewidth"] = 2
                ax[i, j].tick_params(which="both", width=1.7)
                ax[i, j].tick_params(which="major", length=9)
                ax[i, j].tick_params(which="minor", length=5)
                ax[i, j].tick_params(
                    axis="both",
                    which="both",
                    pad=8,
                    left=True,
                    right=True,
                    top=True,
                    bottom=True,
                )
                ax[i, j].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
                ax[i, j].set_xticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
                ax[i, j].set_ylim(0.5e-4, 5e-1)
                ax[i, j].set_xlim(1e7, 1e12)

                if j == 0:
                    ax[i, j].set_ylabel(
                        r"$\mathrm{d}n(M)/\mathrm{dlog10} \, M$ [Mpc$^{-3}$]",
                        fontsize=0.7 * label_size,
                    )
                else:
                    ax[i, j].set_yticklabels([])

                if i == n_row - 1:
                    ax[i, j].set_xlabel(
                        r"$M_*$  $\left[\rm{M}_\odot\right]$", fontsize=label_size
                    )
                else:
                    ax[i, j].set_xticklabels([])

        plt.savefig(
            f"{self.data_cube.path_to_output_plots}/emulator_example_gsmf.pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def create_emulator_fitting_diagnostics_plot_gsmf_z2p0(self):
        label_size = 15

        Mstar = np.linspace(
            self.data_cube.emulator_parameters["gsmfz2p0"]["x_min"],
            self.data_cube.emulator_parameters["gsmfz2p0"]["x_max"],
            100,
        )

        n_col = self.data_cube.plot_config["num_panels_x"]
        n_row = self.data_cube.plot_config["num_panels_y"]

        fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3))

        for i in range(n_row):
            for j in range(n_col):
                number = j * n_row + i

                ax[i, j].errorbar(
                    self.observational_data.leja_gsmf[0].x,
                    self.observational_data.leja_gsmf[0].y,
                    self.observational_data.leja_gsmf[0].y_scatter,
                    color="grey",
                    lw=3.0,
                    label=self.observational_data.leja_gsmf[0].citation,
                    fmt="o-",
                    zorder=-3,
                )

                best_fit = self.data_cube.runs[f"{number}.yml"]
                best_fit_params = {
                    p: v for p, v in zip(self.data_cube.parameters, best_fit)
                }

                pred2, pred_var2 = self.data_cube.gpe_gsmfz2p0.predict_values(
                    Mstar, model_parameters=best_fit_params
                )

                ax[i, j].fill_between(
                    10**Mstar,
                    10 ** (pred2 - np.sqrt(pred_var2)),
                    10 ** (pred2 + np.sqrt(pred_var2)),
                    color="deepskyblue",
                    alpha=0.2,
                )
                ax[i, j].plot(
                    10**Mstar, 10**pred2, alpha=0.6, color="deepskyblue", lw=4
                )
                ax[i, j].scatter(
                    10
                    ** self.data_cube.gsmfz2p0_updated.model_values[f"{number}"][
                        "independent"
                    ],
                    10
                    ** self.data_cube.gsmfz2p0_updated.model_values[f"{number}"][
                        "dependent"
                    ],
                    s=100,
                    marker="^",
                    color="deepskyblue",
                )

                ax[i, j].set_yscale("log")
                ax[i, j].set_xscale("log")

                ax[i, j].xaxis.set_tick_params(labelsize=label_size)
                ax[i, j].yaxis.set_tick_params(labelsize=label_size)

                plt.rcParams.update({"figure.autolayout": True})
                plt.rcParams["ytick.direction"] = "in"
                plt.rcParams["xtick.direction"] = "in"
                plt.rcParams["axes.linewidth"] = 2
                ax[i, j].tick_params(which="both", width=1.7)
                ax[i, j].tick_params(which="major", length=9)
                ax[i, j].tick_params(which="minor", length=5)
                ax[i, j].tick_params(
                    axis="both",
                    which="both",
                    pad=8,
                    left=True,
                    right=True,
                    top=True,
                    bottom=True,
                )
                ax[i, j].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
                ax[i, j].set_xticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
                ax[i, j].set_ylim(1e-4, 1e-1)
                ax[i, j].set_xlim(1e7, 2e11)

                if j == 0:
                    ax[i, j].set_ylabel(
                        r"$\mathrm{d}n(M)/\mathrm{dlog10} \, M$ [Mpc$^{-3}$]",
                        fontsize=0.7 * label_size,
                    )
                else:
                    ax[i, j].set_yticklabels([])

                if i == n_row - 1:
                    ax[i, j].set_xlabel(
                        r"$M_*$  $\left[\rm{M}_\odot\right]$", fontsize=label_size
                    )
                else:
                    ax[i, j].set_xticklabels([])

        plt.savefig(
            f"{self.data_cube.path_to_output_plots}/emulator_example_gsmf_highz.pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def create_emulator_fitting_diagnostics_plot_smhms(self):
        label_size = 15

        Mhalo = np.linspace(
            self.data_cube.emulator_parameters["smhms"]["x_min"],
            self.data_cube.emulator_parameters["smhms"]["x_max"],
            100,
        )

        n_col = self.data_cube.plot_config["num_panels_x"]
        n_row = self.data_cube.plot_config["num_panels_y"]

        fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3))

        for i in range(n_row):
            for j in range(n_col):
                number = j * n_row + i

                ax[i, j].errorbar(
                    self.observational_data.moster_smhm_data[0].x,
                    self.observational_data.moster_smhm_data[0].y,
                    self.observational_data.moster_smhm_data[0].y_scatter,
                    label=self.observational_data.moster_smhm_data[0].citation,
                    color="k",
                    lw=3.0,
                    fmt="o",
                    zorder=-3,
                )
                ax[i, j].errorbar(
                    self.observational_data.berhoozi_smhm_data[0].x,
                    self.observational_data.berhoozi_smhm_data[0].y,
                    self.observational_data.berhoozi_smhm_data[0].y_scatter,
                    label=self.observational_data.berhoozi_smhm_data[0].citation,
                    color="grey",
                    lw=3.0,
                    fmt=">",
                    zorder=-3,
                )

                best_fit = self.data_cube.runs[f"{number}.yml"]
                best_fit_params = {
                    p: v for p, v in zip(self.data_cube.parameters, best_fit)
                }

                pred2, pred_var2 = self.data_cube.gpe_smhms.predict_values(
                    Mhalo, model_parameters=best_fit_params
                )

                ax[i, j].fill_between(
                    10**Mhalo,
                    10 ** (pred2 - np.sqrt(pred_var2)),
                    10 ** (pred2 + np.sqrt(pred_var2)),
                    color="deepskyblue",
                    alpha=0.2,
                )
                ax[i, j].plot(
                    10**Mhalo, 10**pred2, alpha=0.6, color="deepskyblue", lw=4
                )
                ax[i, j].scatter(
                    10
                    ** self.data_cube.smhms_updated.model_values[f"{number}"][
                        "independent"
                    ],
                    10
                    ** self.data_cube.smhms_updated.model_values[f"{number}"][
                        "dependent"
                    ],
                    s=100,
                    marker="^",
                    color="deepskyblue",
                )

                ax[i, j].set_yscale("log")
                ax[i, j].set_xscale("log")

                ax[i, j].xaxis.set_tick_params(labelsize=label_size)
                ax[i, j].yaxis.set_tick_params(labelsize=label_size)

                plt.rcParams.update({"figure.autolayout": True})
                plt.rcParams["ytick.direction"] = "in"
                plt.rcParams["xtick.direction"] = "in"
                plt.rcParams["axes.linewidth"] = 2
                ax[i, j].tick_params(which="both", width=1.7)
                ax[i, j].tick_params(which="major", length=9)
                ax[i, j].tick_params(which="minor", length=5)
                ax[i, j].tick_params(
                    axis="both",
                    which="both",
                    pad=8,
                    left=True,
                    right=True,
                    top=True,
                    bottom=True,
                )
                ax[i, j].set_xticks([1e10, 1e11, 1e12, 1e3])
                ax[i, j].set_xlim(1e10, 1e13)
                ax[i, j].set_ylim(10.0**-4.1, 10.0**-1.1)

                if j == 0:
                    ax[i, j].set_ylabel(r"$M_* / M_{\rm BN98}$ ", fontsize=label_size)
                else:
                    ax[i, j].set_yticklabels([])

                if i == n_row - 1:
                    ax[i, j].set_xlabel(
                        r"$M_{\rm BN98}$ $\left[\rm{M}_\odot\right]$",
                        fontsize=label_size,
                    )
                else:
                    ax[i, j].set_xticklabels([])

        plt.savefig(
            f"{self.data_cube.path_to_output_plots}/emulator_example_smhm.pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def create_emulator_fitting_diagnostics_plot_sizes(self):
        label_size = 15

        Mstar = np.linspace(
            self.data_cube.emulator_parameters["sizes"]["x_min"],
            self.data_cube.emulator_parameters["sizes"]["x_max"],
            100,
        )

        n_col = self.data_cube.plot_config["num_panels_x"]
        n_row = self.data_cube.plot_config["num_panels_y"]

        fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3))

        for i in range(n_row):
            for j in range(n_col):
                number = j * n_row + i

                ax[i, j].errorbar(
                    self.observational_data.hardwick_sizes["x"],
                    self.observational_data.hardwick_sizes["y"],
                    self.observational_data.hardwick_sizes["error"],
                    color="grey",
                    lw=2.0,
                    fmt="o-",
                    zorder=-3,
                )

                best_fit = self.data_cube.runs[f"{number}.yml"]
                best_fit_params = {
                    p: v for p, v in zip(self.data_cube.parameters, best_fit)
                }

                pred2, pred_var2 = self.data_cube.gpe_sizes.predict_values(
                    Mstar, model_parameters=best_fit_params
                )

                ax[i, j].fill_between(
                    10**Mstar,
                    10 ** (pred2 - np.sqrt(pred_var2)),
                    10 ** (pred2 + np.sqrt(pred_var2)),
                    color="deepskyblue",
                    alpha=0.2,
                )
                ax[i, j].plot(
                    10**Mstar, 10**pred2, alpha=0.6, color="deepskyblue", lw=4
                )
                ax[i, j].scatter(
                    10
                    ** self.data_cube.sizes_updated.model_values[f"{number}"][
                        "independent"
                    ],
                    10
                    ** self.data_cube.sizes_updated.model_values[f"{number}"][
                        "dependent"
                    ],
                    s=100,
                    marker="^",
                    color="deepskyblue",
                )

                ax[i, j].set_yscale("log")
                ax[i, j].set_xscale("log")

                ax[i, j].xaxis.set_tick_params(labelsize=label_size)
                ax[i, j].yaxis.set_tick_params(labelsize=label_size)

                plt.rcParams.update({"figure.autolayout": True})
                plt.rcParams["ytick.direction"] = "in"
                plt.rcParams["xtick.direction"] = "in"
                plt.rcParams["axes.linewidth"] = 2
                ax[i, j].tick_params(which="both", width=1.7)
                ax[i, j].tick_params(which="major", length=9)
                ax[i, j].tick_params(which="minor", length=5)
                ax[i, j].tick_params(
                    axis="both",
                    which="both",
                    pad=8,
                    left=True,
                    right=True,
                    top=True,
                    bottom=True,
                )
                ax[i, j].set_xticks([1e9, 1e10, 1e11])
                ax[i, j].set_xlim(10**8.75, 10**11.25)
                ax[i, j].set_ylim(10.0**-0.5, 10.0**1.0)

                if j == 0:
                    ax[i, j].set_ylabel(r"Size [kpc]", fontsize=label_size)
                else:
                    ax[i, j].set_yticklabels([])

                if i == n_row - 1:
                    ax[i, j].set_xlabel(
                        r"$M_*$  $\left[\rm{M}_\odot\right]$", fontsize=label_size
                    )
                else:
                    ax[i, j].set_xticklabels([])

        plt.savefig(
            f"{self.data_cube.path_to_output_plots}/emulator_example_sizes.pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def setup_mcmc(self):
        self.initial_guess = list(
            self.data_cube.e_params_ref.model_parameters[
                self.data_cube.reference_config_file_name.removesuffix(".yml")
            ].values()
        )
        self.param_keys = list(
            self.data_cube.e_params_ref.model_parameters[
                self.data_cube.reference_config_file_name.removesuffix(".yml")
            ].keys()
        )

        print(self.param_keys)
        print(self.initial_guess)

        # GSMF at z=0
        mass_low_gsmf_z0p0 = self.data_cube.emulator_parameters["gsmfz0p0"][
            "x_min_fitting"
        ]
        mass_high_gsmf_z0p0 = self.data_cube.emulator_parameters["gsmfz0p0"][
            "x_max_fitting"
        ]
        driver_x = np.log10(self.observational_data.driver_gsmf[0].x.value)
        mask_gsmf_z0p0 = np.logical_and(
            driver_x > mass_low_gsmf_z0p0, driver_x < mass_high_gsmf_z0p0
        )
        self.driver_gsmf_x = np.log10(
            self.observational_data.driver_gsmf[0].x.value[mask_gsmf_z0p0]
        )
        self.driver_gsmf_y = np.log10(
            self.observational_data.driver_gsmf[0].y.value[mask_gsmf_z0p0]
        )
        self.driver_gsmf_y_err = np.mean(
            [
                -np.log10(
                    self.observational_data.driver_gsmf[0].y
                    - self.observational_data.driver_gsmf[0].y_scatter[0, :]
                )
                + np.log10(self.observational_data.driver_gsmf[0].y),
                np.log10(
                    self.observational_data.driver_gsmf[0].y
                    + self.observational_data.driver_gsmf[0].y_scatter[1, :]
                )
                - np.log10(self.observational_data.driver_gsmf[0].y),
            ],
            axis=0,
        )[mask_gsmf_z0p0]

        print("Driver", self.driver_gsmf_y_err, np.shape(self.driver_gsmf_y_err))

        # SIZES at z=0
        mass_low_sizes = self.data_cube.emulator_parameters["sizes"]["x_min_fitting"]
        mass_high_sizes = self.data_cube.emulator_parameters["sizes"]["x_max_fitting"]
        mask_sizes = np.logical_and(
            np.log10(self.observational_data.hardwick_sizes["x"]) > mass_low_sizes,
            np.log10(self.observational_data.hardwick_sizes["x"]) < mass_high_sizes,
        )
        self.hardwick_sizes_x = np.log10(
            self.observational_data.hardwick_sizes["x"][mask_sizes]
        )
        self.hardwick_sizes_y = np.log10(
            self.observational_data.hardwick_sizes["y"][mask_sizes]
        )
        self.hardwick_sizes_y_err = self.observational_data.hardwick_sizes["log_error"][
            mask_sizes
        ]

        print(
            "Hardwick", self.hardwick_sizes_y_err, np.shape(self.hardwick_sizes_y_err)
        )

        # GSMF at z=2
        mass_low_gsmf_z2p0 = self.data_cube.emulator_parameters["gsmfz2p0"][
            "x_min_fitting"
        ]
        mass_high_gsmf_z2p0 = self.data_cube.emulator_parameters["gsmfz2p0"][
            "x_max_fitting"
        ]
        leja_x = np.log10(self.observational_data.leja_gsmf[0].x.value)
        mask_gsmf_z2p0 = np.logical_and(
            leja_x > mass_low_gsmf_z2p0, leja_x < mass_high_gsmf_z2p0
        )
        self.leja_gsmf_x = np.log10(self.observational_data.leja_gsmf[0].x.value)[
            mask_gsmf_z2p0
        ]
        self.leja_gsmf_y = np.log10(self.observational_data.leja_gsmf[0].y.value)[
            mask_gsmf_z2p0
        ]
        self.leja_gsmf_y_err = np.mean(
            [
                -np.log10(
                    self.observational_data.leja_gsmf[0].y
                    - self.observational_data.leja_gsmf[0].y_scatter[0, :]
                )
                + np.log10(self.observational_data.leja_gsmf[0].y),
                np.log10(
                    self.observational_data.leja_gsmf[0].y
                    + self.observational_data.leja_gsmf[0].y_scatter[1, :]
                )
                - np.log10(self.observational_data.leja_gsmf[0].y),
            ],
            axis=0,
        )[mask_gsmf_z2p0]

        print("Leja", self.leja_gsmf_y_err)

        berhoozi_x = np.log10(self.observational_data.berhoozi_smhm_data[0].x.value)

        mass_low_smhm = self.data_cube.emulator_parameters["smhms"]["x_min_fitting"]
        mass_high_smhm = self.data_cube.emulator_parameters["smhms"]["x_max_fitting"]

        mask_smhms = np.logical_and(
            berhoozi_x > mass_low_smhm, berhoozi_x < mass_high_smhm
        )

        self.Ber_smhm_x = np.log10(
            self.observational_data.berhoozi_smhm_data[0].x.value[mask_smhms]
        )
        self.Ber_smhm_y = np.log10(
            self.observational_data.berhoozi_smhm_data[0].y.value[mask_smhms]
        )

        self.Ber_smhm_y_err = np.mean(
            [
                -np.log10(
                    self.observational_data.berhoozi_smhm_data[0].y
                    - self.observational_data.berhoozi_smhm_data[0].y_scatter[0, :]
                )
                + np.log10(self.observational_data.berhoozi_smhm_data[0].y),
                np.log10(
                    self.observational_data.berhoozi_smhm_data[0].y
                    + self.observational_data.berhoozi_smhm_data[0].y_scatter[1, :]
                )
                - np.log10(self.observational_data.berhoozi_smhm_data[0].y),
            ],
            axis=0,
        )[mask_smhms]

        print("Berhoozi", self.Ber_smhm_y_err)

        return

    def ln_prior(self, params):
        priors = self.data_cube.parameter_limits
        for i in range(len(params)):
            # Explore params only within the hypercube
            if params[i] < priors[i][0] or params[i] > priors[i][1]:
                return -np.inf
        return 1.0

    def ln_likelihood(self, params, m1=1.0, m2=1.0, m3=0.0, chi=False):
        prior_likelihood = self.ln_prior(params)

        best_fit_params = {name: val for name, val in zip(self.param_keys, params)}

        model1, model1_errs = self.data_cube.gpe_gsmfz0p0.predict_values(
            self.driver_gsmf_x, model_parameters=best_fit_params
        )
        diff1 = self.driver_gsmf_y - model1
        error1 = np.sqrt(0.06**2.0 + (self.driver_gsmf_y_err) ** 2.0)

        model2, model2_errs = self.data_cube.gpe_sizes.predict_values(
            self.hardwick_sizes_x, model_parameters=best_fit_params
        )
        diff2 = self.hardwick_sizes_y - model2
        error2 = np.sqrt(0.06**2.0 + (self.hardwick_sizes_y_err) ** 2.0)

        model3, model3_errs = self.data_cube.gpe_gsmfz2p0.predict_values(
            self.leja_gsmf_x, model_parameters=best_fit_params
        )
        diff3 = self.leja_gsmf_y - model3
        error3 = np.sqrt(0.06**2.0 + (self.leja_gsmf_y_err) ** 2.0)

        npoints = (m3 * len(diff3) + m2 * len(diff2) + m1 * len(diff1)) / (m1 + m2 + m3)

        C = (
            -0.5 * m3 * ((diff3 / error3) ** 2).mean()
            - 0.5 * m2 * ((diff2 / error2) ** 2).mean()
            - 0.5 * m1 * ((diff1 / error1) ** 2).mean()
        )

        likelyhood = prior_likelihood + C * npoints

        if chi == True:
            print(best_fit_params)

            chisqr = (
                m3 * ((diff3 / error3) ** 2).sum()
                + m2 * ((diff2 / error2) ** 2).sum()
                + m1 * ((diff1 / error1) ** 2).sum()
            )

            dos = (
                m3 * np.size(diff3)
                + m2 * np.size(diff2)
                + m1 * np.size(diff1)
                - (np.size(params))
            )
            reduced_chisqr = chisqr / dos
            print("dos", dos)
            print("Nparams", np.size(params))
            print("reduced chisqr", reduced_chisqr)
            return reduced_chisqr

        return likelyhood

    def model_fit(
        self, initial_guess, m1, m2, m3, nSteps=1000, nDiscard=500, nwalkers=100
    ):
        ndim = len(initial_guess)
        pos = [
            initial_guess + np.random.normal(loc=0, scale=np.abs(initial_guess * 0.01))
            for j in range(nwalkers)
        ]
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.ln_likelihood, args=[m1, m2, m3, False]
        )

        for i in tqdm(
            sampler.sample(pos, iterations=nSteps, skip_initial_state_check=True),
            total=nSteps,
        ):
            pass

        samples = sampler.chain[:, nDiscard:, :].reshape((-1, ndim))
        lnprob = sampler.lnprobability[:, nDiscard:].reshape(-1)
        params_mcmc = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(samples, [16, 50, 84], axis=0)),
        )

        return params_mcmc, samples, lnprob


def main():
    plots = Plots()

    fig, ax, leg = plots.create_gsmfz2p0_plot()
    for i in range(plots.data_cube.num_runs):
        ax.plot(
            10 ** plots.data_cube.gsmfz2p0.model_values[str(i)]["independent"],
            10 ** plots.data_cube.gsmfz2p0.model_values[str(i)]["dependent"],
            color="blue",
            alpha=0.4,
            zorder=-1,
        )
    plt.savefig(
        f"{plots.data_cube.path_to_output_plots}/gsmf_highz_all.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    fig, ax, leg = plots.create_gsmfz0p0_plot()
    for i in range(plots.data_cube.num_runs):
        ax.plot(
            10 ** plots.data_cube.gsmfz0p0.model_values[str(i)]["independent"],
            10 ** plots.data_cube.gsmfz0p0.model_values[str(i)]["dependent"],
            color="blue",
            alpha=0.4,
            zorder=-1,
        )
    plt.savefig(
        f"{plots.data_cube.path_to_output_plots}/gsmf_all.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    fig, ax, leg = plots.create_smhms_plot()
    for i in range(plots.data_cube.num_runs):
        ax.plot(
            10 ** plots.data_cube.smhms.model_values[str(i)]["independent"],
            10 ** plots.data_cube.smhms.model_values[str(i)]["dependent"],
            color="blue",
            alpha=0.4,
            zorder=-1,
        )
    plt.savefig(
        f"{plots.data_cube.path_to_output_plots}/smhm_all.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    fig, ax, leg = plots.create_sizes_plot()
    for i in range(plots.data_cube.num_runs):
        ax.plot(
            10 ** plots.data_cube.sizes.model_values[str(i)]["independent"],
            10 ** plots.data_cube.sizes.model_values[str(i)]["dependent"],
            color="blue",
            alpha=0.4,
            zorder=-1,
        )
    plt.savefig(
        f"{plots.data_cube.path_to_output_plots}/sizes_all.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    fig, ax, leg = plots.create_sfhs_plot()
    for i in range(plots.data_cube.num_runs):
        ax.plot(
            1.0 / (1.0 + plots.data_cube.sfhs.model_values[str(i)]["independent"]),
            10 ** plots.data_cube.sfhs.model_values[str(i)]["dependent"],
            color="blue",
            alpha=0.4,
            zorder=-1,
        )
    plt.savefig(
        f"{plots.data_cube.path_to_output_plots}/sfh_all.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    print("making diagnostics plots...")
    plots.create_emulator_fitting_diagnostics_plot_sizes()
    plots.create_emulator_fitting_diagnostics_plot_smhms()
    plots.create_emulator_fitting_diagnostics_plot_gsmf_z0p0()
    plots.create_emulator_fitting_diagnostics_plot_gsmf_z2p0()
    plots.create_emulator_fitting_diagnostics_plot_sfhs()
    print("Finished making diagnostics plots...")

    plots.setup_mcmc()

    initial_guess = np.array(plots.initial_guess)

    print(plots.data_cube.parameter_limits)
    print(initial_guess)

    params_all, samples_all, lnprob_all = plots.model_fit(
        initial_guess,
        m1=1.0,
        m2=1.0,
        m3=0.0,
        nSteps=5000,
        nDiscard=200,
        nwalkers=30,
    )
    best_fit_all = samples_all[np.argmax(lnprob_all)]
    print(best_fit_all)
    # print("current best", plots.ln_likelihood(np.array([np.log10(2e4), 0.10, np.log10(7e3), 0.6]), m1=1.0, m2=1.0, m3=0.0, chi=True))
    chi_val_all = plots.ln_likelihood(best_fit_all, m1=1.0, m2=1.0, m3=0.0, chi=True)
    print(best_fit_all, "Chi_val", chi_val_all)
    chi_val_ref = plots.ln_likelihood(
        [np.log10(2e4), 0.1, np.log10(7e3), 0.6], m1=1.0, m2=1.0, m3=0.0, chi=True
    )
    print([np.log10(2e4), 0.1, np.log10(7e3), 0.6], "Chi_val", chi_val_ref)

    params_sizes, samples_sizes, lnprob_sizes = plots.model_fit(
        initial_guess,
        m1=0.0,
        m2=1.0,
        m3=0.0,
        nSteps=5000,
        nDiscard=200,
        nwalkers=30,
    )
    best_fit_sizes = samples_sizes[np.argmax(lnprob_sizes)]
    chi_val_sizes = plots.ln_likelihood(
        best_fit_sizes, m1=0.0, m2=1.0, m3=0.0, chi=True
    )
    print(best_fit_sizes, "Chi_val", chi_val_sizes)

    params_gsmf_z0p0, samples_gsmf_z0p0, lnprob_gsmf_z0p0 = plots.model_fit(
        initial_guess,
        m1=1.0,
        m2=0.0,
        m3=0.0,
        nSteps=5000,
        nDiscard=200,
        nwalkers=30,
    )
    best_fit_gsmf_z0p0 = samples_gsmf_z0p0[np.argmax(lnprob_gsmf_z0p0)]
    chi_val_gsmf_z0p0 = plots.ln_likelihood(
        best_fit_gsmf_z0p0, m1=1.0, m2=0.0, m3=0.0, chi=True
    )
    print(best_fit_gsmf_z0p0, "Chi_val", chi_val_gsmf_z0p0)

    params_gsmf_z2p0, samples_gsmf_z2p0, lnprob_gsmf_z2p0 = plots.model_fit(
        initial_guess,
        m1=1.0,
        m2=1.0,
        m3=1.0,
        nSteps=500,
        nDiscard=200,
        nwalkers=30,
    )
    best_fit_gsmf_z2p0 = samples_gsmf_z2p0[np.argmax(lnprob_gsmf_z2p0)]
    chi_val_gsmf_z2p0 = plots.ln_likelihood(
        best_fit_gsmf_z2p0, m1=1.0, m2=1.0, m3=1.0, chi=True
    )
    print(best_fit_gsmf_z2p0, "Chi_val", chi_val_gsmf_z2p0)

    np.savez(
        "bestfit_model",
        samples_smhm=samples_gsmf_z0p0,
        samples_gsmf_z2p0=samples_gsmf_z2p0,
        samples_sizes=samples_sizes,
        samples_all=samples_all,
        lnprob_smhm=lnprob_gsmf_z0p0,
        lnprob_gsmf_z2p0=lnprob_gsmf_z2p0,
        lnprob_sizes=lnprob_sizes,
        lnprob_all=lnprob_all,
        chi_val_smhm=chi_val_gsmf_z0p0,
        chi_val_gsmf_z2p0=chi_val_gsmf_z2p0,
        chi_val_sizes=chi_val_sizes,
        chi_val_all=chi_val_all,
        smhm_error=plots.data_cube.emulator_errors["gsmfz0p0"],
        sizes_error=plots.data_cube.emulator_errors["sizes"],
    )

    for i, fid_val in enumerate(
        list(
            plots.data_cube.e_params_ref.model_parameters[
                plots.data_cube.reference_config_file_name.removesuffix(".yml")
            ].values()
        )
    ):
        fig, ax = plt.subplots(1, 1)

        param_bin_edges = np.linspace(
            plots.data_cube.parameter_limits[i][0],
            plots.data_cube.parameter_limits[i][1],
            100,
        )
        param_bin_centers = 0.5 * (param_bin_edges[1:] + param_bin_edges[:-1])

        for label, samples, c in zip(
            [
                "Constraining by $z=0$ sizes",
                "Constraining by $z=0$ GSMF",
                "Constraining by $z=2$ GSMF",
                "Constraining by $z=0$ GSMF, sizes, and $z=2$ GSMF",
            ],
            [samples_sizes, samples_gsmf_z0p0, samples_gsmf_z2p0, samples_all],
            ["blue", "gold", "lawngreen", "black"],
        ):
            ax.axvline(x=fid_val, color="grey", lw=2, dashes=(3, 3))
            param_model, _, _ = stats.binned_statistic(
                samples[:, i], None, statistic="count", bins=param_bin_edges
            )
            ax.plot(
                param_bin_centers,
                param_model / np.sum(param_model),
                lw=5,
                label=label,
                color=c,
            )

        ax.legend(fontsize=15, frameon=False)

        ax.set_ylabel("Posterior (marginalized)", fontsize=30)
        ax.set_xlabel(plots.data_cube.parameter_printable_names[i], fontsize=30)

        ax.xaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_tick_params(labelsize=25)

        plt.savefig(
            f"{plots.data_cube.path_to_output_plots}/posterior_param_{i}.pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    for relation in ["smhms", "gsmfz0p0", "sizes", "sfhs", "gsmfz2p0"]:
        print(f"{relation=}")

        for i in range(plots.data_cube.number_of_parameters):
            fig, ax, leg = getattr(plots, f"create_{relation}_plot")()

            best_fit_params = {
                name: val for name, val in zip(plots.param_keys, best_fit_all)
            }

            Mock_values, Mock_parameters = mock_sweep(
                getattr(plots.data_cube, f"gpe_{relation}"),
                plots.data_cube.e_spec,
                6,
                plots.data_cube.parameters[i],
                best_fit_params,
            )

            col_index = np.linspace(0, 1, len(Mock_values.keys()))

            for index, mock_name in enumerate(Mock_values.keys()):
                label = (
                    plots.data_cube.parameter_printable_names[i]
                    + "$ = "
                    + str(
                        np.round(
                            Mock_parameters[mock_name][plots.data_cube.parameters[i]], 3
                        )
                    )
                    + "$"
                )

                col = plt.cm.viridis(col_index[index])

                if relation == "sfhs":
                    x = 1.0 / (1.0 + Mock_values[mock_name]["independent"])
                else:
                    x = 10.0 ** Mock_values[mock_name]["independent"]
                ax.plot(
                    x,
                    10.0 ** Mock_values[mock_name]["dependent"],
                    color=col,
                    label=label,
                    lw=3,
                )

            leg_runs = ax.legend(fontsize=17.0, frameon=False, loc="upper right")

            x_min = plots.data_cube.emulator_parameters[relation]["x_min"]
            x_max = plots.data_cube.emulator_parameters[relation]["x_max"]

            ax.axvline(x=10**x_min, color="grey", lw=2, alpha=0.5)
            ax.axvline(x=10**x_max, color="grey", lw=2, alpha=0.5)

            try:
                fit_x_min = plots.data_cube.emulator_parameters[relation][
                    "x_min_fitting"
                ]
                fit_x_max = plots.data_cube.emulator_parameters[relation][
                    "x_max_fitting"
                ]

                ax.axvline(
                    x=10**fit_x_min, color="black", lw=2, alpha=0.5, dashes=(3, 3)
                )
                ax.axvline(
                    x=10**fit_x_max, color="black", lw=2, alpha=0.5, dashes=(3, 3)
                )

            except AttributeError:
                print(f"Plot `{relation}' was not used for fitting")

            ax.add_artist(leg)

            plt.savefig(
                f"{plots.data_cube.path_to_output_plots}/{relation}_param_{i}.pdf",
                bbox_inches="tight",
                pad_inches=0.1,
            )

    labels = [
        "Constraining by $z=0$ sizes",
        "Constraining by $z=0$ GSMF",
        "Constraining by $z=2$ GSMF",
        "Constraining by $z=0$ SMHM, sizes, and $z=2$ GSMF",
    ]
    params_4_main_models = [
        best_fit_sizes,
        best_fit_gsmf_z0p0,
        best_fit_gsmf_z2p0,
        best_fit_all,
    ]
    colors = ["blue", "darkorange", "lawngreen", "black"]

    for relation in ["smhms", "gsmfz0p0", "sizes", "sfhs", "gsmfz2p0"]:
        print(f"{relation=}")

        fig, ax, leg = getattr(plots, f"create_{relation}_plot")()

        x_min = plots.data_cube.emulator_parameters[relation]["x_min"]
        x_max = plots.data_cube.emulator_parameters[relation]["x_max"]

        if relation == "sfhs":
            x_values_raw = np.linspace(x_min, x_max, 100)
            x_values = np.log10(1.0 / (1.0 + x_values_raw))
        else:
            x_values_raw = np.linspace(x_min, x_max, 100)
            x_values = x_values_raw

        for label, best_fit, c in zip(labels, params_4_main_models, colors):
            best_fit_params = best_fit_params = {
                name: val for name, val in zip(plots.param_keys, best_fit)
            }

            pred2, pred_var2 = getattr(
                plots.data_cube, f"gpe_{relation}"
            ).predict_values(x_values_raw, model_parameters=best_fit_params)

            ax.fill_between(
                10**x_values,
                10 ** (pred2 - np.sqrt(pred_var2)),
                10 ** (pred2 + np.sqrt(pred_var2)),
                color=c,
                alpha=0.2,
            )
            ax.plot(10**x_values, 10**pred2, alpha=0.6, color=c, lw=4, label=label)

        leg_runs = ax.legend(fontsize=13.0, frameon=False, loc="upper right")

        ax.axvline(x=10**x_min, color="grey", lw=2, alpha=0.5)
        ax.axvline(x=10**x_max, color="grey", lw=2, alpha=0.5)

        try:
            fit_x_min = plots.data_cube.emulator_parameters[relation]["x_min_fitting"]
            fit_x_max = plots.data_cube.emulator_parameters[relation]["x_max_fitting"]

            ax.axvline(x=10**fit_x_min, color="black", lw=2, alpha=0.5, dashes=(3, 3))
            ax.axvline(x=10**fit_x_max, color="black", lw=2, alpha=0.5, dashes=(3, 3))

        except AttributeError:
            print(f"Plot `{relation}' was not used for fitting")

        ax.add_artist(leg)

        plt.savefig(
            f"{plots.data_cube.path_to_output_plots}/prediction_{relation}.pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    figure = corner.corner(
        samples_all[:, :],
        labels=plots.data_cube.parameter_printable_names,
        bins=40,
        hist_bin_factor=10,
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),
        color="k",
        smooth1d=1,
        smooth=True,
        max_n_ticks=5,
        plot_contours=True,
        plot_datapoints=True,
        plot_density=True,
        show_titles=True,
        title_kwargs={"fontsize": 15},
    )

    ndim = len(best_fit_all)
    axes = np.array(figure.axes).reshape((ndim, ndim))

    plt.savefig(
        f"{plots.data_cube.path_to_output_plots}/corner_all.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    figure = corner.corner(
        samples_gsmf_z0p0[:, :],
        labels=plots.data_cube.parameter_printable_names,
        bins=40,
        hist_bin_factor=10,
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),
        color="k",
        smooth1d=1,
        smooth=True,
        max_n_ticks=5,
        plot_contours=True,
        plot_datapoints=True,
        plot_density=True,
        show_titles=True,
        title_kwargs={"fontsize": 15},
    )

    ndim = len(best_fit_gsmf_z0p0)
    axes = np.array(figure.axes).reshape((ndim, ndim))

    plt.savefig(
        f"{plots.data_cube.path_to_output_plots}/corner_gsmfz0p0.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    figure = corner.corner(
        samples_sizes[:, :],
        labels=plots.data_cube.parameter_printable_names,
        bins=40,
        hist_bin_factor=10,
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),
        color="k",
        smooth1d=1,
        smooth=True,
        max_n_ticks=5,
        plot_contours=True,
        plot_datapoints=True,
        plot_density=True,
        show_titles=True,
        title_kwargs={"fontsize": 15},
    )

    ndim = len(best_fit_gsmf_z0p0)
    axes = np.array(figure.axes).reshape((ndim, ndim))

    plt.savefig(
        f"{plots.data_cube.path_to_output_plots}/corner_sizes.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    return


if __name__ == "__main__":
    main()
