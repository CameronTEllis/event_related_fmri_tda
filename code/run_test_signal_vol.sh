#!/usr/bin/env bash
# Run the tests on the searchlight volumes. Different searchlight analyses have different tests that output to text files in the specified directory
#
#SBATCH --output=./logs/test_signal_vol-%j.out
#SBATCH -t 5
#SBATCH --mem 2000
#SBATCH -n 1

# Set up the environment
source code/setup_environment.sh

# Inputs are:
# 1: searchlight name
# 2: output dir
python ./code/test_signal_vol.py $1 $2
