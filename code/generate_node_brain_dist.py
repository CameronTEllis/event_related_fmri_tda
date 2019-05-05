# Take the node brain data and then center a searchlight on every cube in the brain. In that cube make a distance matrix (node x node) that you then take the upper triangle of, unravel and store in the voxel at the center of the searchlight.

# Import modules
import numpy as np
import nibabel
import scipy.spatial.distance as sp_distance
from brainiak.searchlight.searchlight import Searchlight
import sys
from mpi4py import MPI
from scipy import stats

# What are the system inputs?

# What is the file name
if len(sys.argv) > 1:
    file = sys.argv[1]

# Do you want to specify a special searchlight radius. Default is 1 (27 voxels)
if len(sys.argv) > 2:
    sl_rad =  int(sys.argv[2])
else:
    sl_rad = 1

# What is the distance metric being used. This can be any distance compatible with scipy's pdist, like 'correlation' or 'cityblock'
if len(sys.argv) > 3:
    dist_metric = sys.argv[3]
else:
    dist_metric = 'euclidean'

# If you are using a different metric for computing the similarity matrix then say so here. This can be any distance compatible with scipy's pdist, like 'correlation' or 'cityblock'
if dist_metric == 'euclidean':
    dist_name = ''
else:
    dist_name = '_' + dist_metric

# If the SL radius is bigger than 1 then create a name
if sl_rad > 1:
	sl_name = '_rad_' + str(sl_rad)
else:
	sl_name = ''

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Load the participants and take a random sample
# Find the output name
path = file[0:file.find('_data/')+6]
participant_name = file[file.find('node_brain/')+11:-7]

output_folder = path + 'node_brain_dist/'
output_name = output_folder + participant_name + sl_name + dist_name + '.nii.gz'

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
sl = Searchlight(sl_rad=sl_rad, max_blk_edge=5)

# Distribute data to processes
sl.distribute([node_brain], mask)
sl.broadcast(dist_metric)

# Define voxel function
def node2dist(data, mask, myrad, bcvar):
    # What are the dimensions of the data?
    dimsize = data[0].shape

    # Pull out the data
    mat = data[0].reshape((dimsize[0] * dimsize[1] * dimsize[2],
                           dimsize[3])).astype('double')

    # Calculate the distance matrix. Use the bcvar as the metric name
    if bcvar == 'euclidean_norm':
        
        # If using this name, then first z score the data across voxels and then compute the euclidean distance
        mat = stats.zscore(mat, axis=0)
        dist_vect = sp_distance.pdist(np.transpose(mat), 'euclidean')
    else:
        dist_vect = sp_distance.pdist(np.transpose(mat), bcvar)

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
    print('Outputting in ' + output_name)
