#!/bin/bash
#SBATCH -c 24
#SBATCH -t 120

set -e

module load GCC/14.3.0
module load CMake/4.0.3
module add Eigen/3.4.0

if ! test -f spack/share/spack/setup-env.sh
then
	git clone -c feature.manyFiles=true -b releases/latest --depth 1 https://github.com/spack/spack.git
fi

. spack/share/spack/setup-env.sh

spack external find binutils cmake coreutils curl diffutils findutils git gmake openssh perl python sed tar

spack compiler find

spack env create l3ster
spack env activate l3ster

spack add cmake
spack add eigen intel-oneapi-tbb parmetis
spack add kokkos +openmp +serial
spack add trilinos cxxstd=17 +openmp +amesos2 +belos +tpetra +ifpack2
spack add pugixml
spack concretize
spack install

spack gc -y
spack clean -a
