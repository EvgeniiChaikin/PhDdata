"""
Converts the cosmic star formation rate density from the the raw, .txt format into .yml format.
"""
import matplotlib

matplotlib.use("Agg")

import unyt
import sys
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats

from swiftsimio import load

# This parameter may need to be changed to 24 or 32 depending on the
# hypercube this script is applied to.
NUMBER_OF_RUNS_IN_THE_HYPERCUBE = 40

sfr_output_units = unyt.msun / (unyt.year * unyt.Mpc**3)

simulations = {i: i for i in range(0, NUMBER_OF_RUNS_IN_THE_HYPERCUBE)}

print(simulations)


def load_data(simulation, counter):
    global initial_snapshot

    print(counter)

    filename = f"./SFR_raw/SFR_{counter}.txt"

    if counter == 0:
        snapshot = "0/colibre_0001.hdf5"
        initial_snapshot = load(snapshot)

    data = np.genfromtxt(filename).T

    units = initial_snapshot.units
    boxsize = initial_snapshot.metadata.boxsize
    print(boxsize)
    box_volume = boxsize[0] * boxsize[1] * boxsize[2]

    sfr_units = initial_snapshot.gas.star_formation_rates.units

    # a, Redshift, SFR
    return data[2], data[3], (data[7] * sfr_units / box_volume).to(sfr_output_units)


simulation_data = {k: load_data(count, k) for count, k in enumerate(simulations.keys())}

magnitudes = np.logspace(-2, np.log10(20), 50)
magnitudes_centres = (magnitudes[:-1] + magnitudes[1:]) / 2.0
magnitudes_widths = magnitudes[1:] - magnitudes[:-1]

for simulation in simulation_data.keys():
    scale_factor, redshift, sfr = simulation_data[simulation]
    name = simulations[simulation]

    sfr_median, _, _ = stats.binned_statistic(
        1.0 / scale_factor - 1, sfr.value, statistic="median", bins=magnitudes
    )
    sfr_median[np.isnan(sfr_median)] = 0.0
    sfr_median[sfr_median == 0.0] = np.min(sfr_median[sfr_median > 0.0]) / 10.0
    print(sfr_median)

    data = {"SFH": {}}
    data["SFH"]["lines"] = {}
    data["SFH"]["lines"]["median"] = {}
    data["SFH"]["lines"]["median"]["centers"] = magnitudes_centres.tolist()
    data["SFH"]["lines"]["median"]["bins"] = magnitudes.tolist()
    data["SFH"]["lines"]["median"]["centers_units"] = "dimensionless"
    data["SFH"]["lines"]["median"]["values"] = sfr_median.tolist()
    data["SFH"]["lines"]["median"]["values_units"] = "Msun/yr/Mpc**3"

    with open(f"./SFR_processed/{simulation}.yml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
