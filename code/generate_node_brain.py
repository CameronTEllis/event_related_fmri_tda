# Read in the data of each participant, calculate the noise properties of
# their brain, then input the signal somewhere in their brain. Then store it
# as an average representation per condition type

# Import modules
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import generate_graph_structure as graphs
import copy
import sys
from scipy.stats import zscore
from sklearn import linear_model
from brainiak.utils import fmrisim as sim
import logging
import os
from nilearn.image import smooth_img
np.seterr(divide='ignore')  # Not relevant for this script
        
# What are the system inputs?

# What participant are you analyzing
if len(sys.argv) > 1:
    participant_counter = int(sys.argv[1])

# Signal type
if len(sys.argv) > 2:
    signal_structure = sys.argv[2]
else:
    signal_structure = 'elipse'

# Signal magnitude
if len(sys.argv) > 3:
    signal_magnitude = float(sys.argv[3])
else:
    signal_magnitude = 0.5

# Signal properties
# What are the inputs to the generate_graph_structure for this signal type?
# Should be of the form [X,Y,Z]
if len(sys.argv) > 4:
    signal_properties = eval(sys.argv[4])
else:
    signal_properties = [1, 1]

# What are the timing properties?
# Specify the minimum
if len(sys.argv) > 5:
    timing_properties = eval(sys.argv[5])
else:
    timing_properties = [6.0,1,1.0,0,24]
    
# What resample is it of this participant
if len(sys.argv) > 6:
    resample = int(sys.argv[6])
else:
    resample = 0

logger = logging.getLogger(__name__)

# If it is not a resample, do you want to save some plots
save_signal_func = 0  # Save the plot of the signal function?
save_functional = 1  # Do you want to save the functional (with node_brain)

# Inputs for generate_signal
temporal_res = 100  # How many samples per second are there for timing files
tr_duration = 2  # Force this value
fwhm = 0 # What is the fwhm of the smoothing kernel
all_pos = 1  # Assume all activation is positive
bound_to_global_max = 0 # Is the signal range based on the global average, or is it voxel specific so that each voxel has the signal magnitude specified?

# Set analysis parameters
hrf_lag = 2 # How many TRs is the event offset by?
zscore_time = 1  # Do you want to z score each voxel in time
zscore_volume = 0  # Do you want to z score a volume

# What are the timing properties
minimum_isi = timing_properties[0]
randomise_timing = timing_properties[1]
event_durations = [timing_properties[2]]  # Assume the events are 1s long
repetitions_per_run = timing_properties[3]  # How many nodes are you simulating
nodes = timing_properties[4]  # How many nodes are to be created
deconvolution = timing_properties[5] # Do you want to use the deconvolution method for aggregating across events


## Set definitions

def regress_voxels(voxel_time,
                   design,
                   intercept=0,
                   ):
    # Takes in a voxel by time matrix and a TR by condition design matrix.
    # Assume that the timecourse used to create the design matrix was mean
    # centred
    
    # Calculate the coefficients for each condition
    voxels_reg = np.zeros((voxel_time.shape[0], design.shape[1] + intercept))
    voxels_err = np.zeros((voxel_time.shape[0], 1))
    voxels_r2 = np.zeros((voxel_time.shape[0], 1))
    ols = linear_model.LinearRegression()
    for voxel_counter in list(range(0, voxel_time.shape[0])):

        # Get this voxel
        voxel = voxel_time[voxel_counter, :]

        # If this is more than a single column design matrix go
        if design.shape[1] > 1:
            if np.all(np.isnan(voxel)) == 0:

                # What are the coefficients of each feature in the regression
                fit = ols.fit(design, voxel)
                coefs = fit.coef_

                # Calculate an R squared between the predicted response (based
                # on the coefficients) and the observed response
                predicted = np.mean((design * coefs) + fit.intercept_, 1)
                r2 = np.corrcoef(voxel, predicted)[0, 1] ** 2

                # Do you want to also add the intercept to the output?
                if intercept == 1:
                    voxels_reg[voxel_counter, :] = np.append(coefs, fit.intercept_)
                else:
                    voxels_reg[voxel_counter, :] = coefs

                voxels_err[voxel_counter, 0] = np.sqrt(np.sum((voxel - predicted) ** 2))
                voxels_r2[voxel_counter, 0] = r2

            else:
                voxels_reg[voxel_counter, :] = np.nan
                voxels_err[voxel_counter, 0] = np.nan
                voxels_r2[voxel_counter, 0] = np.nan
        else:

            # Correlate the voxels and the design
            voxels_r2[voxel_counter, 0] = np.corrcoef(voxel, design[:,
                                                             0])[0, 1]

    # # Convert all of the nans
    # voxels_reg[np.isnan(voxels_reg)] = 0

    # Output
    if design.shape[1] > 1:
        return voxels_reg, voxels_r2, voxels_err
    else:
        return voxels_r2


signal_properties_str = str(signal_properties)
signal_properties_str = signal_properties_str.replace('[', '_').replace(',',
                                                                      '_').replace(']', '').replace(' ', '')

timing_properties_str = str(timing_properties)
timing_properties_str = timing_properties_str.replace('[', '_').replace(',',
                                                                      '_').replace(']', '').replace(' ', '')

# What is the participant name with these parameters (change from default)
effect_name = '_' + signal_structure
effect_name = effect_name + '_s-' + str(signal_magnitude)
effect_name = effect_name + signal_properties_str
effect_name = effect_name + '_t' + timing_properties_str

# Only add this at the end
if resample > 0:
    effect_name = effect_name + '_resample-' + str(resample)

# Specify the paths and names
parameters_path = './simulator_parameters/'
simulated_data_path = './simulated_data/'
timing_path = parameters_path + 'timing/'

participant = 'sub-' + str(participant_counter + 1)
SigMask = parameters_path + '/real_results/significant_mask.nii.gz'
savename = participant + effect_name
node_brain_save = simulated_data_path + '/node_brain/' + savename + \
                  '.nii.gz'

# Quit if the file exists
if os.path.exists(node_brain_save):
    print('File exists, aborting')
    exit()

# Load data
print('Loading ' + participant)

# Load significant voxels
nii = nibabel.load(SigMask)
signal_mask = nii.get_data()

dimsize = nii.header.get_zooms()  # x, y, z and TR size

if signal_structure == 'community_structure':

    # Load the timing information
    nodes = 15  # overwrite
    onsets_runs = np.load(timing_path + participant + '.npy')

else:
    # Generate the timing for this participant
    base_runs = np.load(timing_path + participant + '.npy')
    runs = len(base_runs)
    onsets_runs = [-1] * runs
    for run_counter in list(range(runs)):
        
        # If the number of repetitions per run is zero then use Schapiro's data to calculate it
        if repetitions_per_run == 0:
            
            # How many events in this run for this participant
            events = sum(len(event) for event in base_runs)

            # How many events per node for this run?
            repetitions_per_run = int(events / nodes)
        
        current_time = 0  # Initialize
        onsets = [-1] * nodes  # Ignore first entry
        for hamilton_counter in list(range(0, repetitions_per_run)):

            # Pick on this hamilton the starting node and the direction
            node = np.random.randint(0, nodes - 1)
            direction = np.random.choice([-1, 1], 1)[0]

            # Loop through the events
            for node_counter in list(range(0, nodes)):

                # Append this time
                if np.all(onsets[node] == -1):
                    onsets[node] = np.array(current_time)
                else:
                    onsets[node] = np.append(onsets[node],
                                             current_time)

                # What is the isi?
                isi = minimum_isi + (np.random.randint(3) * tr_duration)

                # Increment the time
                current_time += event_durations[0] + isi

                # Update the node
                node += direction
                if node >= nodes:
                    node = 0
                elif node < 0:
                    node = (nodes - 1)

        # Store the runs
        print('There are %d onsets in run %d' % (len(onsets), run_counter))
        onsets_runs[run_counter] = onsets

# How many runs are there?
runs = len(onsets_runs)

# Generate the indexes of all voxels that will contain signal
vector_size = int(signal_mask.sum())

# Find all the indices that are significant
idx_list = np.where(signal_mask == 1)

idxs = np.zeros([vector_size, 3])
for idx_counter in list(range(0, len(idx_list[0]))):
    idxs[idx_counter, 0] = idx_list[0][idx_counter]
    idxs[idx_counter, 1] = idx_list[1][idx_counter]
    idxs[idx_counter, 2] = idx_list[2][idx_counter]

idxs = idxs.astype('int8')    
    
# What voxels are they
dimensions = signal_mask.shape

# Cycle through the runs and generate the data
node_brain = np.zeros([dimensions[0], dimensions[1], dimensions[2],
                       nodes, runs], dtype='double')  # Preset

# Generate the graph structure (based on the ratio)
if signal_structure == 'community_structure':
    signal_coords = graphs.community_structure(signal_properties[0],
                                              )
elif signal_structure == 'parallel_rings':
    signal_coords = graphs.parallel_rings(nodes,
                                          signal_properties[0],
                                          signal_properties[1],
                                         )
elif signal_structure == 'figure_eight':
    signal_coords = graphs.figure_eight(nodes,
                                        signal_properties[0],
                                        signal_properties[1],
                                       )
elif signal_structure == 'chain_rings':
    signal_coords = graphs.chain_rings(nodes,
                                       signal_properties[0],
                                       signal_properties[1],
                                      )
elif signal_structure == 'dispersed_clusters':
    signal_coords = graphs.dispersed_clusters(nodes,
                                              signal_properties[0],
                                              signal_properties[1],
                                              signal_properties[2],
                                             )
elif signal_structure == 'tendrils':
    signal_coords = graphs.tendrils(nodes,
                                    signal_properties[0],
                                    signal_properties[1],
                                   )
elif signal_structure == 'elipse':
    signal_coords = graphs.elipse(nodes,
                                  signal_properties[0],
                                  signal_properties[1],
                                 )


# Perform an orthonormal transformation of the data
if vector_size > signal_coords.shape[1]:
    signal_coords = graphs.orthonormal_transform(vector_size,
                                                      signal_coords)

# Do you want these coordinates to be all positive? This means that
# these coordinates are represented as different magnitudes of
# activation
if all_pos == 1:
    mins = np.abs(np.min(signal_coords, 0))
    for voxel_counter in list(range(0, len(mins))):
        signal_coords[:, voxel_counter] += mins[voxel_counter]

# Bound the value to have a max of 1 so that the signal magnitude is more interpretable
if bound_to_global_max == 1:
    signal_coords /= np.max(signal_coords.flatten())
else:
    signal_coords /= np.max(signal_coords, 0)

for run_counter in list(range(1, runs + 1)):

    # Get run specific names
    template_name = parameters_path + 'template/' + participant + '_r' + \
                    str(run_counter) + '.nii.gz'
    noise_dict_name = parameters_path + 'noise_dict/' + participant + '_r' + str(run_counter) + \
                      '.txt'
    nifti_save = simulated_data_path + 'nifti/' + participant + '_r' + str(run_counter)\
                 + effect_name + '.nii.gz'
    signal_func_save = './plots/' + participant + '_r' +\
                       str(run_counter) + effect_name + '.png'

    # Load the template (not yet scaled
    nii = nibabel.load(template_name)
    template = nii.get_data()  # Takes a while

    # Create the mask and rescale the template
    mask, template = sim.mask_brain(template,
                                    mask_self=True,
                                    )

    # Make sure all the signal voxels are within the mask
    signal_mask_run = signal_mask * mask

    # Pull out the onsets for this participant (copy it so you don't alter it)
    onsets = copy.deepcopy(onsets_runs[run_counter - 1])

    # Do you want to randomise the onsets (so that the events do not have a
    # fixed order)
    if randomise_timing == 1:

        # Extract all the timing information
        onsets_all = onsets[0]
        for node_counter in list(range(1, nodes)):
            onsets_all = np.append(onsets_all, onsets[node_counter])

        print('Randomizing timing across nodes')

        # Shuffle the onsets
        np.random.shuffle(onsets_all)

        # Insert the shuffled onsets back in
        for node_counter in list(range(0, nodes)):
            node_num = len(onsets[node_counter])
            onsets[node_counter] = np.sort(onsets_all[0:node_num])
            onsets_all = onsets_all[node_num:] # Remove the onsets

    # If you want to use different timing then take the order of the data
    # and then create a new timecourse
    if minimum_isi > 1:

        # Iterate through all the onset values
        onsets_all = []  # Reset
        for node_counter in list(range(0, nodes)):
            onsets_all = np.append(onsets_all, onsets[node_counter])

            # Take the idx of all of the elements in onset all
            Idxs = np.zeros((len(onsets[node_counter]), 2))

            Idxs[:, 0] = [node_counter] * len(onsets[node_counter])
            Idxs[:, 1] = list(range(0, len(onsets[node_counter])))

            # Append the indexes
            if node_counter == 0:
                onset_idxs = Idxs
            else:
                onset_idxs = np.concatenate((onset_idxs, Idxs))

        # Change the values of the onsets
        sorted_idxs = np.ndarray.argsort(onsets_all)
        cumulative_add = 0
        for onset_counter in list(range(0, len(onsets_all))):
            # What is the onset being considered
            onset = onsets_all[sorted_idxs[onset_counter]]

            # Add time to this onset
            onsets_all[sorted_idxs[onset_counter]] = onset + cumulative_add

            # Insert the onsets at the right time
            node_counter = int(onset_idxs[sorted_idxs[onset_counter], 0])
            idx_counter = int(onset_idxs[sorted_idxs[onset_counter], 1])
            onsets[node_counter][idx_counter] = onset + cumulative_add

            cumulative_add += minimum_isi - 1

    # How long should you model
    last_event = max(map(lambda x: x[-1], onsets))  # Find the max of maxs
    duration = int(last_event + 10) # Add a decay buffer

    # Specify the dimensions of the volume to be created
    dimensions = np.array([template.shape[0], template.shape[1],
                           template.shape[2], int(duration / tr_duration)])

    # Load the noise parameters in
    with open(noise_dict_name, 'r') as f:
        noise_dict = f.read()

    noise_dict = eval(noise_dict)

    # Preset brain size
    brain_signal = np.zeros([dimensions[0], dimensions[1], dimensions[2],
                             int(duration / tr_duration)], dtype='double')
    for node_counter in list(range(0, nodes)):

        print('Node ' + str(node_counter))

        # Preset value
        volume = np.zeros(dimensions[0:3])

        # Pull out the signal template
        signal_pattern = signal_coords[node_counter, :]
        onsets_node = onsets[node_counter]

        # Create the time course for the signal to be generated
        stimfunc = sim.generate_stimfunction(onsets=onsets_node,
                                             event_durations=event_durations,
                                             total_time=duration,
                                             temporal_resolution=temporal_res,
                                             )

        # Aggregate the timecourse
        if node_counter == 0:
            stimfunc_all = np.zeros((len(stimfunc), vector_size))
            for voxel_counter in list(range(0, vector_size)):
                stimfunc_all[:, voxel_counter] = np.asarray(
                    stimfunc).transpose() * signal_pattern[voxel_counter]
        else:

            # Add these elements together
            temp = np.zeros((len(stimfunc), vector_size))
            for voxel_counter in list(range(0, vector_size)):
                temp[:, voxel_counter] = np.asarray(stimfunc).transpose() * signal_pattern[voxel_counter]

            stimfunc_all += temp

    # After you have gone through all the nodes, convolve the HRF and
    # stimulation for each voxel
    print('Convolving HRF')
    
    # Decide if you are going to scale to global max (need to stop the convolve_hrf rescaling)
    if bound_to_global_max == 1:
        scale_function = False
    else:
        scale_function = True
    
    signal_func = sim.convolve_hrf(stimfunction=stimfunc_all,
                                   tr_duration=tr_duration,
                                   temporal_resolution=temporal_res,
                                   scale_function=scale_function,
                                   )
    
    # Do the final rescaling to the global max
    if bound_to_global_max == 1:
        signal_func /= np.max(signal_func.flatten())
    
    if save_signal_func == 1 and resample == 0 and run_counter == 1:
        plt.plot(stimfunc_all[::int(temporal_res * tr_duration), 0])
        plt.plot(signal_func[:,0])
        plt.xlim((0, 200))
        plt.ylim((-1, 5))
        plt.savefig(signal_func_save)

    # Convert the stim func into a binary vector of dim 1
    stimfunc_binary = np.mean(np.abs(stimfunc_all)>0, 1) > 0
    stimfunc_binary = stimfunc_binary[::int(tr_duration * temporal_res)]

    # Bound, can happen if the duration is not rounded to a TR
    stimfunc_binary = stimfunc_binary[0:signal_func.shape[0]]

    # Create the noise volumes (using the default parameters)
    noise = sim.generate_noise(dimensions=dimensions[0:3],
                               stimfunction_tr=stimfunc_binary,
                               tr_duration=tr_duration,
                               template=template,
                               mask=mask,
                               noise_dict=noise_dict,
                               )

    # Change the type of noise
    noise = noise.astype('double')

    # Create a noise function (same voxels for signal function as for noise)
    noise_function = noise[idxs[:, 0], idxs[:, 1], idxs[:, 2], :].T

    # Compute the signal magnitude for the data
    signal_func_scaled = sim.compute_signal_change(signal_function=signal_func,
                                                   noise_function=noise_function,
                                                   noise_dict=noise_dict,
                                                   magnitude=[
                                                       signal_magnitude],
                                                   method='PSC',
                                                   )
    
    # Multiply the voxels with signal by the HRF function
    brain_signal = sim.apply_signal(signal_function=signal_func_scaled,
                                    volume_signal=signal_mask,
                                    )

    # Convert any nans to 0s
    brain_signal[np.isnan(brain_signal)] = 0

    # Combine the signal and the noise (as long as the signal magnitude is above 0)
    if signal_magnitude >= 0:

        # Combine the signal and the noise
        brain = brain_signal + noise
        
    else:
        # Don't make signal just add it here
        brain = brain_signal

    # Save the participant data
    if save_functional == 1 and resample == 0:

        print('Saving ' + nifti_save)
        brain_nifti = nibabel.Nifti1Image(brain, nii.affine)

        hdr = brain_nifti.header
        hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2], 2.0))
        nibabel.save(brain_nifti, nifti_save)  # Save
    
    # Do you want to perform smoothing
    if fwhm > 0:
 
        # Convert to nifti
        brain_nifti = nibabel.Nifti1Image(brain, nii.affine)
        
        # Run the smoothing step

        sm_nii = smooth_img(brain_nifti, fwhm)

        # Pull out the data again and continue on
        brain = sm_nii.get_data()

    # Z score the data
    if zscore_time == 1:
        brain = zscore(brain, 3)

    # Mask brain
    brain = brain * mask.reshape((mask.shape[0], mask.shape[1], mask.shape[
        2], 1))
    
    # Find the representation of each node. You can use a deconvolution approach in which I create a design matrix, convolve it with an HRF and then regress each voxel against this matrix, storing the beta values. Alternatively you can just average the TRs N seconds after event onset
    if deconvolution == 1:
        
        # Preset
        design_mat = np.zeros((brain.shape[3], nodes))
        
        for node in list(range(0, nodes)):
            
            # Create a stim function for just this node
            stimfunc = sim.generate_stimfunction(onsets=onsets[node],
                                     event_durations=event_durations,
                                     total_time=duration,
                                     temporal_resolution=temporal_res,
                                     )
            
            # Convolve it with the HRF
            signal_func = sim.convolve_hrf(stimfunction=stimfunc,
                                           tr_duration=tr_duration,
                                           temporal_resolution=temporal_res,
                                           )
            
            # Store in the design matrix
            design_mat[:, node] = signal_func.T
        
        # Make a voxel by time array
        voxel_time = brain.reshape(np.prod(brain.shape[0:3]), brain.shape[3])
        
        # Regress each voxel in the brain with the design matrix
        voxels_reg, _, _ = regress_voxels(voxel_time,
                                          design_mat,
                                         )
        
        # Unravel the data in voxels_reg
        run_node_brain = voxels_reg.reshape(brain.shape[0], 
                                            brain.shape[1],
                                            brain.shape[2],
                                            nodes)
        
        # Store the TRs
        node_brain[:, :, :, :, run_counter - 1] = run_node_brain
        
    else:
        # Average the timepoints corresponding to each node
        for node in list(range(0, nodes)):

            node_trs = onsets[node]

            # Z score all the TRs where a node was created
            run_node_brain = np.zeros((dimensions[0], dimensions[1], dimensions[2],
                            len(node_trs)))
            for tr_counter in list(range(0, len(node_trs))):

                # When does it onset
                onset = int(np.round(node_trs[tr_counter] / tr_duration) +
                            hrf_lag)

                if onset < brain.shape[3]:
                    m = np.mean(brain[:, :, :, onset])
                    s = np.std(brain[:, :, :, onset])

                    # Do you want to z score the volumes
                    if zscore_volume == 1:
                        run_node_brain[:, :, :, tr_counter-1] = (brain[:, :, :, onset] - m) / s
                    else:
                        run_node_brain[:, :, :, tr_counter - 1] = brain[:, :, :, onset]

            # Average the TRs
            node_brain[:, :, :, node, run_counter - 1] = np.mean(run_node_brain, 3)

print('Wrapping up')

# average the brains across runs
node_brain = np.mean(node_brain, 4)

# Convert all NaNs to 0s for this participant
if signal_magnitude < 0:
    node_brain[np.isnan(node_brain)] = 0

# Mask again
node_brain = node_brain * mask.reshape(dimensions[0], dimensions[
    1], dimensions[2], 1)

# Save the volume
brain_nifti = nibabel.Nifti1Image(node_brain.astype('double'), nii.affine)

hdr = brain_nifti.header
hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2], tr_duration))
nibabel.save(brain_nifti, node_brain_save)  # Save
print('Saving ' + node_brain_save)
