# Take in a real fMRI volume from ~/Validation/real_data/ and compute the noise properties of the volume. Then simulate a new volume with those same noise properties and store that. Finally, estimate the noise properties of the simulation and store those noise properties.
#
# This code can deal with some missing inputs. For instance if no output_name or no output_noise_dict_name is supplied then the steps to produce these will be skipped.
#
# The run time of this code is stored in ./Validation/simulation_timing.txt

import numpy as np
import nibabel
from brainiak.utils import fmrisim as sim
import sys
from os import path, remove
import logging

# Inputs are (full paths):
# 1: Input noise dict
# 2: Template to be used
# 3: Output name for generated noise dict based on the simulation

input_noise_dict_name = sys.argv[1]
input_template = sys.argv[2]
output_noise_dict_name = sys.argv[3]
match_noise = 1

# Get a nii file for reference
nii = nibabel.load(input_template)
dimsize = nii.header.get_zooms()
template = nii.get_data()

# Hard code
tr_duration = 2
trs = 311

dimensions = np.array(template.shape[0:3])  # What is the size of the brain

# Generate the continuous mask from the voxels
mask, template = sim.mask_brain(volume=template,
                                mask_self=True,
                               )

print('Num brain voxels:', np.sum(mask))

# Load the file name
with open(input_noise_dict_name, 'r') as f:
    noise_dict = f.read()

print('Loading ' + input_noise_dict_name)
noise_dict = eval(noise_dict)


print('Generating brain for this permutation ')
noise_dict['matched'] = match_noise
brain = sim.generate_noise(dimensions=dimensions,
                           stimfunction_tr=np.zeros((trs,1)),
                           tr_duration=int(tr_duration),
                           template=template,
                           mask=mask,
                           noise_dict=noise_dict,
                           )


# Calculate and save the output dict
print('Testing noise generation')
out_noise_dict = {'voxel_size': noise_dict['voxel_size'], 'matched': match_noise}
out_noise_dict = sim.calc_noise(volume=brain,
                            mask=mask,
                            template=template,
                            noise_dict=out_noise_dict,
                            )

# Remove file if it exists
if path.exists(output_noise_dict_name):
    remove(output_noise_dict_name)

# Save the file
with open(output_noise_dict_name, 'w') as f:
    f.write(str(out_noise_dict))

print('Complete')

print(noise_dict)
print(out_noise_dict)
