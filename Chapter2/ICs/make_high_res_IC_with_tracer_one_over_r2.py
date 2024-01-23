###############################################################################
# This file is part of SWIFT.
# Copyright (c) 2016 Stefan Arridge (stefan.arridge@durhama.ac.uk)
#                    Matthieu Schaller (matthieu.schaller@durham.ac.uk)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

import h5py
import numpy as np
import sys
import matplotlib.pylab as plt

# Generates a SWIFT IC file with a constant density and pressure

# Read id, position and h from glass
res = int(sys.argv[1])
print("Qroot of the number of particles:", res)
dens = 0.1  # H/Acm3

glass = h5py.File("glassCube_{:d}.hdf5".format(res), "r")
pos = glass["/PartType0/Coordinates"][:, :]
arg = np.where(
    (pos[:, 1] < 0.0)
    | (pos[:, 2] < 0.0)
    | (pos[:, 0] < 0.0)
    | (pos[:, 1] > 1.0)
    | (pos[:, 2] > 1.0)
    | (pos[:, 0] > 1.0)
)

h = glass["/PartType0/SmoothingLength"][:]
numPart = np.size(h) - np.shape(arg[0])[0]
pos = np.swapaxes(
    np.array(
        [
            np.delete(pos[:, 0], arg),
            np.delete(pos[:, 1], arg),
            np.delete(pos[:, 2], arg),
        ]
    ),
    0,
    1,
)
h = np.delete(h, arg)

# Global parameters
T = 0.1  # Initial Temperature [K]
gamma = 5.0 / 3.0  # Gas adiabatic index
h_frac = 0.73738788833  # Hydrogen mass fraction
mu = 0.6  # Mean molecular weight
from_h_to_rho = 1 / (1.98848e33 / 3.08567758e21**3 / 1.67262e-24) / h_frac

print("From H to rho / 1e7: ", from_h_to_rho / 1e7)

rho = from_h_to_rho * dens  # Density of the gas in code units
mass = float(sys.argv[2]) / 100.0
boxSize = (mass / rho * numPart * 2 * 2 * 2) ** (1.0 / 3.0)

print("boxsize", boxSize)
print("cubic root of num part", pow(numPart, 1.0 / 3.0))

periodic = 1  # 1 For periodic box
fileName = "example_diffusion_{:d}_{:.1e}_within_sphere_one_over_r2.hdf5".format(
    res, mass
)  # Create name for the output file

# Definining units
m_h_cgs = 1.67e-24  # proton mass
k_b_cgs = 1.38e-16  # Boltzmann constant
Msun_in_cgs = 1.98848e33  # Solar mass
unit_length = 3.08567758e21  # kpc
unit_mass = 1.98848e33  # solar mass
unit_time = 3.0857e16  # ~ Gyr

pos *= boxSize
pos /= 2.0

h *= boxSize / 2.0

for i in [0, 1]:
    for j in [0, 1]:
        for k in [0, 1]:
            if i + j + k > 0:
                pos_try = np.swapaxes(
                    np.vstack(
                        [
                            pos[:numPart, 0] + boxSize / 2.0 * float(i),
                            pos[:numPart, 1] + boxSize / 2.0 * float(j),
                            pos[:numPart, 2] + boxSize / 2.0 * float(k),
                        ]
                    ),
                    0,
                    1,
                )

                print("i shifting particles by", boxSize / 2.0 * float(i))
                print("j shifting particles by", boxSize / 2.0 * float(j))
                print("k shifting particles by", boxSize / 2.0 * float(k))

                print("ijk", i, j, k)

                print("shape attachment array", np.shape(pos_try))
                print("shape pos", np.shape(pos))
                pos = np.append(pos, pos_try, axis=0)
                print("shape new pos", np.shape(pos))

                h = np.append(h, h[:numPart], axis=0)
                print("shape of new h", np.shape(h))

print("old Npar", numPart)
numPart = len(h)
print("new N par", numPart)

# Gas properties
r = np.sqrt(np.sum((pos - 0.5 * boxSize) * (pos - 0.5 * boxSize), axis=1))
v = np.zeros((numPart, 3))  # Velocity
u = np.zeros(numPart)  # Internal energy
m = np.zeros(numPart)  # Mass
abun = np.zeros((numPart, 10))  # Initial adundancies
Z = np.zeros(numPart)  # Initial metal fractions
ids = np.linspace(1, numPart, numPart) + 100000  # IDs

# Compute and save internal energy of the gas
internalEnergy = k_b_cgs * T * mu / ((gamma - 1.0) * m_h_cgs)
internalEnergy *= (unit_time / unit_length) ** 2
u[:] = internalEnergy

# Mass of the gas particles
m[:] = mass

# Creating single 1e51-erg supernova
E_SN = 1.0e51

# Target number of particles with E_SN and ejecta
N_part_with_E_SN = 57
# N_part_with_E_SN = 1

# t Sort according to distance from the centre
arg = np.argsort(r)

# For fiducial run, we want particles with ejecta and E_SN be within 5 pc
if dens == 0.1:  # 0.1
    # Part within inner 5pc
    for count, i in enumerate(arg):
        print(count, i, 1e3 * r[i], "[pc]")

        if 1e3 * r[i] > 5.0:  # 3.41  # 7.468372996966774
            print("find i max:", count, i)
            N_part_with_E_SN = count
            break
else:
    print("Number of particles with initial energy", N_part_with_E_SN)


# Inject energy
u[arg[:N_part_with_E_SN]] = (
    E_SN / N_part_with_E_SN / mass / Msun_in_cgs * (unit_time / unit_length) ** 2
)

print("New N part:", N_part_with_E_SN)
print("Max radius:", 1e3 * r[arg[N_part_with_E_SN - 1]], " [pc]")

# initial metal mass fractions:
mass_fr_init = {
    "init_abundance_metal": 0.0133714,
    "init_abundance_Hydrogen": 0.73738788833,
    "init_abundance_Helium": 0.24924186942,
    "init_abundance_Carbon": 0.0023647215,
    "init_abundance_Nitrogen": 0.0006928991,
    "init_abundance_Oxygen": 0.00573271036,
    "init_abundance_Neon": 0.00125649278,
    "init_abundance_Magnesium": 0.00070797838,
    "init_abundance_Silicon": 0.00066495154,
    "init_abundance_Iron": 0.00129199252,
    "init_abundance_Europium": 0.0,
}

radial_corr = 5.0**2 / (1e3 * r[arg[:N_part_with_E_SN]]) ** 2
radial_corr /= np.sum(radial_corr)

print("Radial corr")
print(r[arg[:N_part_with_E_SN]])
print(radial_corr * 57.0)
print(np.sum(radial_corr))

# Adding ejecta mass (in the form of europium)
ejecta_mass = mass / 1000.0 * radial_corr

# Update total particle mass
new_mass = mass + ejecta_mass

# Apply to the particles with E_SN and ejecta
m[arg[:N_part_with_E_SN]] = new_mass

# Loop over element fractions
for counter, key in enumerate(mass_fr_init.keys()):
    # Counter 0 corresponds to metallicity
    if counter == 0:
        Z[:] = mass_fr_init[key]
    # Everyting else is individual elements
    else:
        abun[:, counter - 1] = mass_fr_init[key]

    # Take care of different values of initial Z and Fe60
    if key == "init_abundance_metal" or key == "init_abundance_Europium":
        mass_fr_init[key] = (mass_fr_init[key] * mass + ejecta_mass) / new_mass
    # Otherwise just rescale according to the new mass
    else:
        mass_fr_init[key] *= mass / new_mass

    # Save the above values in the particle arrays
    if counter == 0:
        Z[arg[:N_part_with_E_SN]] = mass_fr_init[key]
    else:
        abun[arg[:N_part_with_E_SN], counter - 1] = mass_fr_init[key]

# File
f = h5py.File(fileName, "w")

# Header
grp = f.create_group("/Header")
grp.attrs["BoxSize"] = [boxSize, boxSize, boxSize]
grp.attrs["NumPart_Total"] = [numPart, 0, 0, 0, 0, 0]
grp.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
grp.attrs["NumPart_ThisFile"] = [numPart, 0, 0, 0, 0, 0]
grp.attrs["Time"] = [0.0]
grp.attrs["NumFilesPerSnapshot"] = [1]
grp.attrs["MassTable"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
grp.attrs["Flag_Entropy_ICs"] = [0]
grp.attrs["Dimension"] = 3

# Runtime parameters
grp = f.create_group("/RuntimePars")
grp.attrs["PeriodicBoundariesOn"] = periodic

# Units
grp = f.create_group("/Units")
grp.attrs["Unit length in cgs (U_L)"] = unit_length
grp.attrs["Unit mass in cgs (U_M)"] = unit_mass
grp.attrs["Unit time in cgs (U_t)"] = unit_time
grp.attrs["Unit current in cgs (U_I)"] = 1.0
grp.attrs["Unit temperature in cgs (U_T)"] = 1.0

print(np.max(u), np.min(u))
# Gas Particle group
grp = f.create_group("/PartType0")
grp.create_dataset("Coordinates", data=pos, dtype="d")
grp.create_dataset("Velocities", data=v, dtype="f")
grp.create_dataset("Masses", data=m, dtype="f")
grp.create_dataset("SmoothingLength", data=h, dtype="f")
grp.create_dataset("InternalEnergy", data=u, dtype="f")
grp.create_dataset("Metallicity", data=Z, dtype="f")
grp.create_dataset("ElementAbundance", data=abun, dtype="f")
grp.create_dataset("ParticleIDs", data=ids, dtype="L")

# Close the output file
f.close()

# Some output
print("Initial condition have been generated! \n")
print("-------------------------------------------------")
print("Output file name            : {:s}".format(fileName))
print("Box size                    : {:.3e} kpc".format(boxSize))
print("Number of gas particles     : {:d}".format(numPart))
print("Gas number density          : {:.3e} cm^-3".format(dens))
print("Average gas-particle mass   : {:.3e} M_\odot".format(np.mean(m)))
print("Min gas-particle mass       : {:.3e} M_\odot".format(np.min(m)))
print("Max gas-particle mass       : {:.3e} M_\odot".format(np.max(m)))
