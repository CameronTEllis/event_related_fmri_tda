#!/bin/bash
#
# Setup the environment
# It is necessary to have the following importable in python:
# brainiak
# brainiak-extras
#
# Along with this you should get things like numpy and nibabel which are necessary.
# You also need to specify the 'short_partition' and 'long_partition' names

# Example of how to set up the brainiak environment (activating a conda environment)
module load python/3.6
source activate brainiak_extras

# Example of how to load in FSL tools (for running randomise)
module load FSL

# Specifying the partition names
short_partition=other
long_partition=other
