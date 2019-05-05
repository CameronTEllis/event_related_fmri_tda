#!/bin/bash
#
# Perform randomise on the the searchlight data (first by demeaning the volume in order to make the function work)
#
#SBATCH --output=./logs/randomise-%j.out
#SBATCH -p short
#SBATCH -t 5:00
#SBATCH --mem 2000

# Setup the environment
source code/setup_environment.sh

# Specify the inputs
condition=$1
analysis_type=$2

files=`ls simulated_data/searchlights/sub-*_${condition}_${analysis_type}.nii.gz`

# Delete all matching outputs
rm -f simulated_data/randomise/*${condition}_${analysis_type}.nii.gz

# Specify some file names
merged_file=simulated_data/randomise/sub_${condition}_${analysis_type}.nii.gz
output_file=simulated_data/randomise/sub_${condition}_${analysis_type}
temp_file=simulated_data/randomise/temp_${condition}_${analysis_type}.nii.gz
mask_file=simulator_parameters/real_results/intersect_whole_brain.nii.gz 
randomise_summary_dir=randomise_summary/

# Cycle through all the participants
for file in $files
do

# Duplicate the file so it can be edited
cp $file $temp_file

# If you are doing an an analysis of the loop number then only consider 1 loop cases
if [[ $analysis_type == "loop_counter" ]]
then
fslmaths $temp_file -thr 1 -uthr 1 $temp_file
fi

# Compute the mean of the brain voxels
brain_mean=`fslstats $temp_file -k $mask_file -m`

# Demean the volume
fslmaths $temp_file -sub $brain_mean $temp_file

# Merge the file (or create it if it doesn't exist)
if [ ! -e $merged_file ]
then
cp $temp_file $merged_file
else
fslmerge -t $merged_file $merged_file $temp_file
fi
done

# Run randomise on the data
randomise -i $merged_file -o ${output_file} -1 -n 1000 -T

# Create the average file from the merged
fslmaths $merged_file -Tmean ${output_file}_mean.nii.gz

# Remove the intermediate files file
#rm -f ${temp_file}
#rm -f ${merged_file}

# Perform the ROI based analysis of the tfce output in order to get relevant descriptive statistics
sbatch code/run_test_signal_vol.sh ${output_file}_tfce_corrp_tstat1.nii.gz ${randomise_summary_dir}
