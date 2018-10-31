# Define the tests to perform on this data. For the most part these are tests used in the searchlight computation

# Import modules
import numpy as np
import scipy.spatial.distance as sp_distance
from brainiak_extras.tda.rips import rips_filtration as rips
import sys

# Create a barcode by performing persistent homology. This uses the rips filtration technique, with PHAT as the workhorse
def make_persistence(distance_matrix,
                     max_dim=1,
                     max_scale=float('inf'),
                     ):

    # Perform the rips filtration to get a barcode
    barcode = rips(max_dim=max_dim,
                   max_scale=max_scale,
                   dist_mat=distance_matrix,
                   )

    # Convert to a np array
    barcode = np.asanyarray(barcode)

    return (barcode)


# What is the persistence of the biggest loop?
def test_tda_loop_max(distance_matrix,
                      max_dim=1,
                      max_scale=float('inf'),
                      persistence_diagram_name=None,
                     ):

    # Perform the rips filtration to get a barcode
    barcode = make_persistence(distance_matrix, max_dim, max_scale)
    
    # If there are no loops then store as zero but if there are loops then
    # store them here
    if (barcode[:, 2] == 1).sum() > 0:
        
        # What loops are there and what is the max duration
        loops = barcode[(barcode[:, 2] == 1), :]
        max_loop = (loops[:, 1] - loops[:, 0]).max()

    else:
        max_loop = 0

    # Return the mean death time of the clusters and duration of ring
    return max_loop


# How many loops are there?
def test_tda_loop_counter(distance_matrix,
                          max_dim=1,
                          max_scale=float('inf'),
                          persistence_diagram_name=None,
                          ):

    # Perform the rips filtration to get a barcode
    barcode = make_persistence(distance_matrix, max_dim, max_scale)
    
    # How many loops are there (betti 1 features
    loops = barcode[(barcode[:, 2] == 1), :]

    # Return the number of betti 1 features found
    return loops.shape[0]


# How many loops are there?
def test_tda_loop_threshold_ratio(distance_matrix,
                                  ratio_threshold=10,
                                  max_dim=1,
                                  max_scale=float('inf'),
                                  persistence_diagram_name=None,
                                  ):

    # Perform the rips filtration to get a barcode
    barcode = make_persistence(distance_matrix, max_dim, max_scale)
    
    # Get all the loops
    loops = barcode[(barcode[:, 2] == 1), :]

    # Determine whether the is one loop above the threshold
    if loops.shape[0] == 0:
        is_one_loop = 0
    elif loops.shape[0] == 1:
        is_one_loop = 1
    else:
        # What is the length of each loop 
        loop_lengths = loops[:, 1] - loops[:, 0]

        # Find the top two loop lengths
        largest_loops = np.sort(loop_lengths)[-2:]

        # Find the ratio in length
        loop_ratio = largest_loops[1] / largest_loops[0]

        # Does this ratio exceed the threshold
        if loop_ratio > ratio_threshold:
            is_one_loop = 1
        else:
            is_one_loop = 0

    # Return the number of betti 1 features found
    return is_one_loop
    

# Define voxel function
def sl_kernel(data, mask, myrad, bcvar):
    # What are the dimensions of the data?
    dimsize = data[0].shape

    # Convert the distance vector into a matrix
    mat = data[0][1, 1, 1, :] # Take only the centre data point
    nodes = int(np.ceil(np.sqrt(data[0].shape[3] * 2)))  # Unravel the data
    distance_matrix = np.zeros((nodes, nodes))
    x, y = np.triu_indices(nodes, 1)
    distance_matrix[x, y] = mat

    # Make it symmetrical
    distance_matrix = np.maximum(distance_matrix, distance_matrix.transpose())

    # Pass the distance matrix to the specified TDA test
    if bcvar == 'loop_counter':
        sl_outputs = test_tda_loop_counter(distance_matrix)
    elif bcvar == 'loop_max':
        sl_outputs = test_tda_loop_max(distance_matrix)
    elif bcvar == 'loop_ratio':
        sl_outputs = test_tda_loop_threshold_ratio(distance_matrix)
    else:
        print('No function called by ' + bcvar)
        raise
    
    # Store the results of the analyses
    return sl_outputs




