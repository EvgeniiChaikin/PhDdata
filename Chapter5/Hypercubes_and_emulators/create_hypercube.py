from swiftemulator.design import latin
from swiftemulator.io.swift import write_parameter_files
from swiftemulator import ModelSpecification, ModelParameters
from pathlib import Path
import numpy as np
import os

np.random.seed(25)

spec = ModelSpecification(
    number_of_parameters=5,
    parameter_names=[
        "COLIBREAGN:subgrid_seed_mass_Msun",
        "COLIBREFeedback:SNII_f_kinetic",
        "COLIBREFeedback:SNII_energy_fraction_P_0_K_p_cm3",
        "COLIBREFeedback:SNII_delta_T_n_0_H_p_cm3",
        "COLIBREFeedback:SNIa_delta_T_n_0_H_p_cm3",
    ],
    parameter_printable_names=[
        "$M_{\\rm seed}$",
        "$f_{\\rm kin}$",
        "$P_{\\rm pivot}$",
        "$n_{\\rm pivot}$ [$\\rm cm^{-3}$]",
        "$n_{\\rm pivot}$ [$\\rm cm^{-3}$] (SNIa)",
    ],
    parameter_limits=[
        [3.7, 4.7],  # log10
        [0.0, 0.5],
        [3.2, 4.4],
        [0.05, 2.0],
        [0.05, 2.0],
    ],
)

number_of_simulations = 40

model_parameters = latin.create_hypercube(
    model_specification=spec,
    number_of_samples=number_of_simulations,
)


model_parameters_for_file = model_parameters.model_parameters.copy()

for i in range(number_of_simulations):
    model_parameters[f"{i}"][
        "COLIBREFeedback:SNIa_delta_T_n_0_H_p_cm3"
    ] = model_parameters_for_file[f"{i}"]["COLIBREFeedback:SNII_delta_T_n_0_H_p_cm3"]

parameter_transforms = {
    "COLIBREAGN:subgrid_seed_mass_Msun": lambda x: 10.0**x,
    "COLIBREFeedback:SNII_energy_fraction_P_0_K_p_cm3": lambda x: 10.0**x,
}

base_parameter_file = "colibre_50Mpc_376.yml"
output_path = "/cosma8/data/dp004/dc-chai1/paper4_hypercube_twelf_L50_HC4/"

write_parameter_files(
    filenames={
        key: output_path + f"{key}.yml"
        for key in model_parameters.model_parameters.keys()
    },
    model_parameters=model_parameters,
    parameter_transforms=parameter_transforms,
    base_parameter_file=base_parameter_file,
)

print("FINISHED")
