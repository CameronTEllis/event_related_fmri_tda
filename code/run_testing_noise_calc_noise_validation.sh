#!/usr/bin/env bash
# bash script to launch testing_noise_calc_noise_validation.py for a single participant.

#SBATCH --output=logs/testing_noise_calc-%j.out
#SBATCH -t 20
#SBATCH --mem 5000

source code/setup_environment.sh

python ./code/testing_noise_calc_noise_validation.py
