#!/bin/bash

#SBATCH -J ISOFA
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -o out_%j.out
#SBATCH -e err_%j.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH -t 72:00:00
#SBATCH --exclusive


module load intel_comp/2020-update2
module load intel_mpi/2020-update2
module load ucx/1.8.1
module load parmetis/4.0.3-64bit
module load parallel_hdf5/1.10.6
module load fftw/3.3.8cosma7
module load gsl/2.5

# Run of our actual code
ulimit -c unlimited

../../swift-colibre_isolated_galaxy_fast/examples/swift \
    --threads=28 --external-gravity --cooling --stars --star-formation --feedback \
    --self-gravity --temperature --limiter --sync --hydro  --pin isolated_galaxy.yml
