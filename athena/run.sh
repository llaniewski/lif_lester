#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --ntasks-per-socket 1
#SBATCH -t 360


set -e

module load GCC/14.3.0
module load CMake/4.0.3
module add Eigen/3.4.0

source spack/share/spack/setup-env.sh

spack env activate l3ster

srun build/lif $@
