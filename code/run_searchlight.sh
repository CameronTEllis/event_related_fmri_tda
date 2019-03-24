#!/usr/bin/env bash
# Run searchlight analysis on the supplied data using the analysis type specified (refers to an analysis option in test_graph_structure.py)
#
#SBATCH --output=./logs/searchlight_analysis-%j.out
#SBATCH -t 360
#SBATCH --mem 10000
#SBATCH -n 4

# Set up the environment
source code/setup_environment.sh

# Inputs are:
# 1: Participant (full path to node_brain file)
# 2: Analysis type: loop_counter, loop_max, loop_ratio_$ratio
srun -n $SLURM_NTASKS --mpi=pmi2 python ./code/searchlight.py $1 $2
