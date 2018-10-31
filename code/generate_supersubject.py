# Average the participants in the 4th dimension, either node_brain data or their node_brain_dist

# Import modules
import numpy as np
import nibabel
import sys
import glob

# What are the system inputs?

# What is the subject folder
if len(sys.argv) > 1:
    folder = sys.argv[1]

# What is the subject condition
if len(sys.argv) > 2:
    condition = sys.argv[2]

# What is the subject condition
if len(sys.argv) > 3:
    permutation = int(sys.argv[3])

# Load the participants and take a random sample
# Find the output name
path = folder[0:folder.find('_data/')+6]

if folder.find('dist') == -1:
    output_folder = path + 'supersubject_node_brain/'
else:
    output_folder = path + 'supersubject_node_brain_dist/'

if permutation == 0:
    output_name = output_folder + condition + '.nii.gz'
else:
    output_name = output_folder + condition + '_resample-' + str(permutation)\
                  + \
                  '.nii.gz'

# What are the brain volumes
brain_files = glob.glob(folder + '*' + condition + '.nii.gz')

# Load which permutation this is (all the searchlights must have the same)
if permutation > 0:
    with open(path + '/../supersubject_permutations.txt', 'r') as f:
        permutations = f.readlines()
    permutation_participants=permutations[permutation].split(',')


# Load all the participants data
print('Aggregating these brains')

for sample_counter in list(range(0, len(brain_files))):

    # What participant are you aggregating
    if permutation == 0:
        participant_counter = sample_counter
    else:
        participant_counter = int(permutation_participants[sample_counter])

    print(brain_files[participant_counter])

    # Load in the data
    nii = nibabel.load(brain_files[participant_counter])
    if sample_counter == 0:
        node_brain = nii.get_data()
    else:
        node_brain = node_brain + nii.get_data()

# Divide by the number of brains that were summed
node_brain = node_brain / len(brain_files)

# Convert the output into what can be used
node_brain = node_brain.astype('double')
node_brain[np.isnan(node_brain)] = 0

# Save the volume
output_nii = nibabel.Nifti1Image(node_brain, nii.affine)
hdr = output_nii.header
dimsize=nii.header.get_zooms()
hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2], dimsize[3]))
nibabel.save(output_nii, output_name)  # Save
