#!/usr/bin/env bash
#
# Generate the node_brain data for all participants

# Setup the environment
source code/setup_environment.sh

# Inputs are:
#1. Specify the structure type, community_structure, parallel_rings
#2. Define the signal magnitude
#3. Specify the signal properties specific to the structure
#4. Specify the timing properties, minimum isi, randomise and event duration
#5. What resampled image is it

for participant in `seq 0 19`
do

sbatch -p $short_partition ./code/run_generate_node_brain.sh $participant $1 $2 $3 $4 $5

done
