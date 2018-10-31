# Take the node brain data and then center a searchlight on every cube in the brain. In that cube make a distance matrix (node x node) that you then take the upper triangle of, unravel and store in the voxel at the center of the searchlight.

# Import modules
import numpy as np
import nibabel
import scipy.spatial.distance as sp_distance
from brainiak.searchlight.searchlight import Searchlight
import sys
from mpi4py import MPI

# What are the system inputs?

# What is the file name
if len(sys.argv) > 1:
    file = sys.argv[1]

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Load the participants and take a random sample
# Find the output name
path = file[0:file.find('_data/')+6]
participant_name = file[file.find('node_brain/')+11:]

output_folder = path + 'node_brain_dist/'
output_name = output_folder + participant_name

# Print progress
print('Loading in ' + file)

# Load in the data
nii = nibabel.load(file)  # Load the participant
dimsize=nii.header.get_zooms()
node_brain = nii.get_data()

# Make the mask
mask = node_brain != 0
mask = mask[:, :, :, 0]

# Create searchlight object
sl = Searchlight(sl_rad=1, max_blk_edge=5)

# Distribute data to processes
sl.distribute([node_brain], mask)
sl.broadcast(None)

# Define voxel function
def node2dist(data, mask, myrad, bcvar):
    # What are the dimensions of the data?
    dimsize = data[0].shape

    # Pull out the data
    mat = data[0].reshape((dimsize[0] * dimsize[1] * dimsize[2],
                           dimsize[3])).astype('double')

    # Calculate the distance matrix
    distance_matrix = sp_distance.squareform(
        sp_distance.pdist(np.transpose(mat)))

    # Take the upper triangle of the distance matrix and turn it into a vector
    dist_vect = distance_matrix[np.triu_indices(distance_matrix.shape[0])]

    # Remove the diagonal
    dist_vect = dist_vect[dist_vect != 0]

    # Store the results of the analyses
    return dist_vect

# Run clusters
sl_outputs = sl.run_searchlight(node2dist)

# Output the result
if rank == 0:

    # Convert the output into what can be used
    # The output of searchlight is an array within an array (except for
    # masked regions). There might be faster solutions but this works
    dim = sl_outputs.shape
    nodes = node_brain.shape[3]
    distances = int(((nodes ** 2) - nodes) / 2)
    output = np.zeros((dim[0], dim[1], dim[2], distances))
    for x in list(range(0, dim[0])):
        for y in list(range(0, dim[1])):
            for z in list(range(0, dim[2])):
                if sl_outputs[x, y, z] is not None:
                    output[x, y, z, :] = sl_outputs[x, y, z]

    output = output.astype('double')

    # Save the volume
    sl_nii = nibabel.Nifti1Image(output, nii.affine)
    hdr = sl_nii.header
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2], dimsize[3]))
    nibabel.save(sl_nii, output_name)  # Save
    print('Loading in ' + output_name)
