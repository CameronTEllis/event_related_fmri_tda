# Load in a distance matrix and perform a computation on every voxel. This is used to calculate the voxelwise persistent homology
#
#
# Import modules
import numpy as np
import nibabel
from brainiak.searchlight.searchlight import Searchlight
import os
import sys
from test_graph_structure import sl_kernel
from mpi4py import MPI
# What are the system inputs?

# What is the node_brain file you are running
brain_file = sys.argv[1]
analysis_type = sys.argv[2]

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Load in the data
nii = nibabel.load(brain_file)  # Load the participant
dimsize=nii.header.get_zooms()
node_brain = nii.get_data()

# Find the output name
if brain_file.find('dist') == -1:
    sub_idx = brain_file.find('_brain/') + 7
    subjectName = brain_file[sub_idx:]
else:
    if brain_file.find('supersubject_node_brain_dist/') > 0:
        sub_idx = brain_file.find('supersubject_node_brain_dist/') + 30
        subjectName = 'supersubject_' + brain_file[sub_idx:]
    else:
        sub_idx = brain_file.find('node_brain_dist/') + 16
        subjectName = brain_file[sub_idx:]

# Remove the nii.gz to put the analysis type on the end
subjectName = subjectName[:-7] + '_' + analysis_type + '.nii.gz'

# Where should the output go
output_name = './simulated_data/searchlights/' + subjectName

print('Creating ' + output_name)

# Make the mask
mask = node_brain != 0
mask = mask[:, :, :, 0]

# Create searchlight object
sl = Searchlight(sl_rad=1, max_blk_edge=5)

# Distribute data to processes
sl.distribute([node_brain], mask)
sl.broadcast(analysis_type)

# Run clusters
sl_outputs = sl.run_searchlight(sl_kernel)

# Print the rank of the data
print('Rank: ' + str(rank))

if rank == 0:

    # Convert the output into what can be used
    sl_outputs = sl_outputs.astype('double')
    sl_outputs[np.isnan(sl_outputs)] = 0

    # Save the volume
    sl_nii = nibabel.Nifti1Image(sl_outputs, nii.affine)
    hdr = sl_nii.header
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
    nibabel.save(sl_nii, output_name)  # Save
    print('Saved the searchlight')




