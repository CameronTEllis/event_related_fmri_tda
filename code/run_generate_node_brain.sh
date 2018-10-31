#!/usr/bin/env bash
# Generate a simulation of data containing topological signal. Specify the nature and magnitude of the signal with the following inputs. Output will go to simulated_data/node_brain
#
# There are many variables available for specification in this script:
# signal_structure= What is the structure you are inserting? Each structure has differe$
#       parallel_rings: 24 nodes arranged in N rings parallel to each other, M distance$
#       elipse: Create an elipse with 24 nodes of a certain form. Properties: [X_width,$
#       chain_rings: 24 nodes arranged in N rings at right angles to each other, M dist$
#	figure_eight: 24 nodes arranged in a figure 8. Properties: [N_rings, M_apart]
# signal_magnitude= Percent signal change of the signal being inserted
# signal_properties= Depends on the signal structure
# permutation= What resample iteration is this
#
#SBATCH --output=./logs/generate_data-%j.out
#SBATCH -t 120
#SBATCH --mem 20000

# Set up the environment
source code/setup_environment.sh

# Inputs are:
#1. Participant number (0 to 20) from the noise parameter files specified in ./simulator_parameters
#2. Specify the structure type
#3. Define the signal magnitude
#3. Specify the signal properties specific to the structure
#4. Specify the timing properties, minimum isi, randomise and event duration
#5. What resampled image is it  

python ./code/generate_node_brain.py $1 $2 $3 $4 $5 $6
