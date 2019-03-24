#!/usr/bin/env bash
# run_generate_noise_validation.sh: bash script to launch generate_noise_validation.py for an individual participant.

#SBATCH --output=logs/generate_noise_validation-%j.out
#SBATCH -t 360
#SBATCH --mem 5000

source ./code/setup_environment.sh

# Inputs are (full paths):
# 1: Input noise dict
# 2: Template to be used
# 3: Output name for generated noise dict based on the simulation

python ./code/generate_noise_validation.py $1 $2 $3
