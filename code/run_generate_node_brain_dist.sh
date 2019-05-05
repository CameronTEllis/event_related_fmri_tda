#!/usr/bin/env bash
# Create an unravelled distance matrix out of a node_brain volume
#
#SBATCH --output=./logs/generate_distance_volume-%j.out
#SBATCH -t 100
#SBATCH --mem 40000
#SBATCH -n 8

# Set up the environment
source code/setup_environment.sh

# Inputs are:
# 1: Participant (full path to node_brain file)
# 2: sl_rad (1 means 27 voxels)
# 3: dist_metric (anything for pdist, like 'euclidean', 'correlation', 'cityblock')
srun -n $SLURM_NTASKS --mpi=pmi2 python ./code/generate_node_brain_dist.py $1 $2 $3
