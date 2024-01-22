#!/bin/bash

#SBATCH -J IG12R5
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=28 
#SBATCH -o out_%j.out
#SBATCH -e err_%j.err
#SBATCH -p cosma7
#SBATCH -A dp203
#SBATCH -t 24:00:00
#SBATCH --exclusive

module load intel_comp/2020-update2
module load intel_mpi/2020-update2
module load ucx/1.8.1
module load parmetis/4.0.3-64bit
module load parallel_hdf5/1.10.6
module load fftw/3.3.8cosma7
module load gsl/2.5
module load sundials/5.1.0_c7

# Run of our actual code
ulimit -c unlimited

../../swift/swift-colibre-paper3-more-output-correct-delay-reservoir/examples/swift \
    --external-gravity --self-gravity --stars --star-formation --cooling --temperature --hydro --feedback --limiter --sync  \
    --threads=28 --pin --dust -v 1 \
    isolated_galaxy.yml
