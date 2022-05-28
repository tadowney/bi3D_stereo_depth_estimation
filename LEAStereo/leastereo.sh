#!/bin/bash
#
#SBATCH --job-name=leastereo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=leastereo.out
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tdowney@scu.edu
#
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
/WAVE/users/unix/tdowney/DeepLearning/project/LEAStereo-master/LEAStereo-master/train_sf.sh
