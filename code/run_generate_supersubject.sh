#!/usr/bin/env bash
# Aggregate data across participants
#
#SBATCH --output=./logs/supersubject_aggregate-%j.out
#SBATCH -t 15
#SBATCH --mem 10000

# Set up the environment
source code/setup_environment.sh

# Inputs are:
# 1: Folder (full path to node_brain folder)
# 2: Condition
# 3: Permutation counter
python ./code/generate_supersubject.py $1 $2 $3
