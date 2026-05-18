#!/bin/bash
#SBATCH -c 24
#SBATCH -t 120

set -e

module load GCC/14.3.0
module load CMake/4.0.3
module add Eigen/3.4.0

source spack/share/spack/setup-env.sh

spack env activate l3ster

if ! test -d build
then
	mkdir build
fi
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_CXX_FLAGS="-Wno-interference-size" ../src/

make

