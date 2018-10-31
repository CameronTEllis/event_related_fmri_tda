# Create the graph structures to be used as signal, as well as some other utils functions.

# Import the modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sp_distance
import sklearn.manifold as manifold
from mpl_toolkits.mplot3d import Axes3D

# Perform an orthonormal transformation of the data (upsample the dimensionality of the data)
def orthonormal_transform(vector_size,
                          signal_coords):

    # Find normal vectors
    ortho_normal_matrix = np.zeros((signal_coords.shape[1], vector_size))
    for vec_counter in list(range(signal_coords.shape[1])):
        vector = np.random.randn(vector_size)

        # Combine vectors
        ortho_normal_matrix[vec_counter, :] = vector

    # Orthogonalize the vectors using the Gram-Schmidt method
    def gramschmidt(input):
        output = input[0:1, :].copy()
        for i in range(1, input.shape[0]):
            proj = np.diag(
                (input[i, :].dot(output.T) / np.linalg.norm(output,
                                                            axis=1) ** 2).flat).dot(
                output)
            output = np.vstack((output, input[i, :] - proj.sum(0)))
        output = np.diag(1 / np.linalg.norm(output, axis=1)).dot(output)
        return output.T

    ortho_normal_matrix = gramschmidt(ortho_normal_matrix)

    # Re store the coordinates in the new dimensional space
    new_coords = np.dot(ortho_normal_matrix, np.transpose(signal_coords))
    return np.transpose(new_coords)

# Create rings in 2d that are spaced in 3d. Can specify the number of rings and relative separation
def parallel_rings(nodes=100,
                   rings=2,
                   ring_separation=1, # How far apart are the rings (the radius of the rings is 1)
                   ):

    # Assume each ring gets the same number of nodes
    nodes_per_ring = int(nodes / rings)

    # How far apart are each of the elements in the ring
    arc_width = (2 * np.pi) / nodes_per_ring

    # What are the angles of the points
    node_angle = []
    for node_counter in list(range(0, nodes)):
        node_angle.append(node_counter * arc_width)

    # Calculate the coordinates for the rings
    signal_coords = np.zeros([nodes_per_ring * rings, 3])
    for ring_counter in list(range(rings)):
        for node_counter in list(range(0, nodes_per_ring)):

            node_idx = node_counter + (nodes_per_ring * ring_counter)
            x = np.cos(node_angle[node_counter])
            y = np.sin(node_angle[node_counter])
            z = ring_counter * ring_separation
            signal_coords[node_idx, :] = [x, y, z]

    return signal_coords


# Create a single elipse of (potentially) different height and widths. If both values are the same then this is a ring
def elipse(nodes=100,
           x_coef=0.5,
           y_coef=1,
           ):

    # How far apart are each of the elements in the ring
    arc_width = (2 * np.pi) / nodes

    # What are the angles of the points
    node_angle = []
    for node_counter in list(range(0, nodes)):
        node_angle.append(node_counter * arc_width)

    # Cycle through the nodes and make the plot
    signal_coords = np.zeros([nodes, 2])
    for node_counter in list(range(0, nodes)):


        x = x_coef * np.cos(node_angle[node_counter])
        y = y_coef * np.sin(node_angle[node_counter])
        signal_coords[node_counter, :] = [x, y]

    return signal_coords


# Rings that interlock perpendicular to one another
def chain_rings(nodes=100,
                rings=2,
                ring_separation=1,
                ):

    # Assume each ring gets the same number of nodes
    nodes_per_ring = int(nodes / rings)

    # How far apart are each of the elements in the ring
    arc_width = (2 * np.pi) / nodes_per_ring

    # What are the angles of the points
    node_angle = []
    for node_counter in list(range(0, nodes)):
        node_angle.append(node_counter * arc_width)

    # Calculate the coordinates for the rings
    signal_coords = np.zeros([nodes_per_ring * rings, 3])
    for ring_counter in list(range(rings)):
        for node_counter in list(range(0, nodes_per_ring)):

            node_idx = node_counter + (nodes_per_ring * ring_counter)
            dim_1 = np.cos(node_angle[node_counter])
            dim_2 = np.sin(node_angle[node_counter])

            # If you want equally spaced rings for >2 rings, you should use
            # a separation of 4/3

            y_shift = ring_separation * ring_counter

            # Change the orientation of the ring
            if np.mod(ring_counter,2) == 0:
                signal_coords[node_idx, :] = [dim_1, dim_2 + y_shift, 0]
            else:
                signal_coords[node_idx, :] = [0, dim_1 + y_shift, dim_2]

    return signal_coords


# Create rings that are in the same plane
def figure_eight(nodes=100,
                 rings=2,
                 ring_separation=2, # What is the separation between the centers of the rings
                 ):

    # Assume each ring gets the same number of nodes
    nodes_per_ring = int(nodes / rings)

    # How far apart are each of the elements in the ring
    arc_width = (2 * np.pi) / nodes_per_ring

    # What are the angles of the points
    node_angle = []
    for node_counter in list(range(0, nodes)):
        node_angle.append(node_counter * arc_width)

    # Calculate the coordinates for the rings
    signal_coords = np.zeros([nodes_per_ring * rings, 2])
    for ring_counter in list(range(rings)):
        for node_counter in list(range(0, nodes_per_ring)):

            node_idx = node_counter + (nodes_per_ring * ring_counter)
            dim_1 = np.cos(node_angle[node_counter])
            dim_2 = np.sin(node_angle[node_counter])

            # Use the separation differently depending on how many rings you
            x_shift = ring_separation * ring_counter

            signal_coords[node_idx, :] = [dim_1, dim_2 + x_shift]

    return signal_coords


# Convert a matrix (observation x coordinate) of coordinates into a vector
def coord2dist(signal_coords,
               ):
    # Calculate the distance matrix
    dist = sp_distance.squareform(sp_distance.pdist(signal_coords))

    return dist


# Make an mds plot of a distance matrix
def make_mds(dist,
             dim=2,
             mds_file=None,
             ):

    # Create an MDS plot
    mds = manifold.MDS(n_components=dim, dissimilarity='precomputed')  # Fit the
    # mds
    # object
    coords = mds.fit(dist).embedding_  # Find the mds coordinates
    coords = np.vstack(
        [coords, coords[0, :]])  # Duplicate first row for display
    
    if dim > 4:
        print('Can only plot in up to 4 dimensions, still using the specified dimensions')
    
    if dim == 2:
        plt.plot(coords[:, 0],
                 coords[:, 1],
                 'k--')
        plt.scatter(coords[:, 0], coords[:, 1], s=100)
    else:
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')
        
        if dim == 3:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
        else:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=coords[:, 3])
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

    # Save if a name is supplied
    if mds_file is not None:
        plt.savefig(mds_file)
