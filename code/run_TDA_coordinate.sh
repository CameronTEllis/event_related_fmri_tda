#!/usr/bin/env bash
# Run TDA on just one voxel from a node_brain participant. Outputs a barcode that can be loaded in and used
#
#SBATCH --output=./logs/TDA_coordinate-%j.out
#SBATCH -t 30
#SBATCH --mem 1000
#SBATCH -n 1

# Set up the environment
source code/setup_environment.sh

# Inputs are:
# 1: Participant (full path to node_brain file)
# 2: Coordinates (e.g. "[x,y,z]" with no spaces)
# 3: Output file name 
python ./code/TDA_coordinate.py $1 $2 $3
