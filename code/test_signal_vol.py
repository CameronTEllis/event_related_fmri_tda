# Calculate the summary information for a given searchlight volume. If the analysis was to count loops then this checks the proportion of voxels that are 1. If the analysis is to calculate the maximum persistence, this compares these values in the signal ROI with a symmetric ROI in the other hemisphere.

# Import the modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from scipy.optimize import curve_fit
import generate_graph_structure as graphs
import glob
import scipy.stats as stats
import fcntl
import time
from brainiak.utils.fmrisim import mask_brain

# Load in the searchlight volume
if len(sys.argv) > 1:
    searchlight_file = sys.argv[1]

# Specify the output dir
if len(sys.argv) > 2:
    output_dir = sys.argv[2]    

print('Loading in %s\nOutputting to %s' % (searchlight_file, output_dir))    
    
expected_loops = 1 # How many loops are expected

# Pull out the searchlight file    
start_idx = searchlight_file.rfind('/')
end_idx = searchlight_file.rfind('.nii.gz')
condition = searchlight_file[start_idx:end_idx]
    
# Load the searchlight volume    
searchlight_vol = nb.load(searchlight_file).get_data()    

# Load in the signal mask
parameters_path = './simulator_parameters/'
template_file = parameters_path + '/template/sub-1_r1.nii.gz'
signal_mask_file = parameters_path + '/real_results/significant_mask.nii.gz'

# Load in the masks
nii = nb.load(template_file)
template = (nii.get_data()).astype('float32')
wholebrain_mask, _ = mask_brain(template)

nii = nb.load(signal_mask_file)
signal_mask = (nii.get_data() > 0).astype('float32')

# Create a mirror image of the signal mask and then use that for comparison
flipped_mask = np.fliplr(signal_mask)

# Remove the voxels in the signal mask
wholebrain_mask -= signal_mask

# Pull out the voxels
whole_vox = searchlight_vol[wholebrain_mask == 1]
signal_vox = searchlight_vol[signal_mask == 1]
flipped_vox = searchlight_vol[flipped_mask == 1]

# Get the counts of the histograms
bins = np.linspace(0, searchlight_vol.max(), 100)
whole_counts, _ = np.histogram(whole_vox, bins)
signal_counts, _ = np.histogram(signal_vox, bins)
flipped_counts, _ = np.histogram(flipped_vox, bins)

# Scale the volume
scale = whole_counts.max() / signal_counts.max()
#plt.figure()
#plt.plot(bins[:-1], whole_counts)
#plt.plot(bins[:-1], flipped_counts * scale)
#plt.plot(bins[:-1], signal_counts * scale)
#plt.legend(('whole_brain', 'flipped', 'signal'))
#plt.savefig(output_dir + condition + '.png')

# To deal with the simultaneous file writing that will happen if you run this many thousands of times, wait up to 90 seconds to continue
time.sleep(np.random.randint(90))

if searchlight_file.find('loop_max') > -1:

    # Calculate the different in counts
    tstat_whole = stats.ttest_ind(signal_vox, whole_vox)
    tstat_flipped = stats.ttest_ind(signal_vox, flipped_vox)

    # Store the t statistic for the test
    with open(output_dir + 'signal_vs_whole.txt', 'a') as fid:
        fcntl.flock(fid, fcntl.LOCK_EX)
        fid.write('%s: %0.5f\n' % (condition, tstat_whole.statistic))
        fcntl.flock(fid, fcntl.LOCK_UN)
    
    with open(output_dir + 'signal_vs_flipped.txt', 'a') as fid:
        fcntl.flock(fid, fcntl.LOCK_EX)
        fid.write('%s: %0.5f\n' % (condition, tstat_flipped.statistic))
        fcntl.flock(fid, fcntl.LOCK_UN)
    
elif searchlight_file.find('loop_counter') > -1:

    # Calculate the proportion of voxels that equal 1
    proportion_signal = (signal_vox == expected_loops).mean()
    proportion_flipped = (flipped_vox == expected_loops).mean()

    # Store data
    with open(output_dir + 'signal_proportion.txt', 'a') as fid:
        fcntl.flock(fid, fcntl.LOCK_EX)
        fid.write('%s: %0.5f\n' % (condition, proportion_signal))
        fcntl.flock(fid, fcntl.LOCK_UN)
        
    with open(output_dir + 'flipped_proportion.txt', 'a') as fid:
        fcntl.flock(fid, fcntl.LOCK_EX)
        fid.write('%s: %0.5f\n' % (condition, proportion_flipped))
        fcntl.flock(fid, fcntl.LOCK_UN)
elif searchlight_file.find('loop_ratio') > -1:

    # Calculate the proportion of voxels that equal 1
    proportion_signal = (signal_vox == expected_loops).mean()
    proportion_flipped = (flipped_vox == expected_loops).mean()

    # Store data
    with open(output_dir + 'signal_ratio.txt', 'a') as fid:
        fcntl.flock(fid, fcntl.LOCK_EX)
        fid.write('%s: %0.5f\n' % (condition, proportion_signal))
        fcntl.flock(fid, fcntl.LOCK_UN)
        
    with open(output_dir + 'flipped_ratio.txt', 'a') as fid:
        fcntl.flock(fid, fcntl.LOCK_EX)
        fid.write('%s: %0.5f\n' % (condition, proportion_flipped))
        fcntl.flock(fid, fcntl.LOCK_UN)
        
print('Finished')