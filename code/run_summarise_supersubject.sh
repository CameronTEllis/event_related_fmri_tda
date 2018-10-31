#!/bin/bash
#
# Specify the analysis metric, signal signal_strength and signal type to generate a txt document with the averaged score for all samples collected
#SBATCH --output=./logs/summarise_supersubject-%j.out
#SBATCH -p all
#SBATCH -t 30
#SBATCH --mem 2000

# break on an error
set -e

# Take inputs
analysis_metric=$1
condition=$2

# Are you doing
if [[ ${analysis_metric} == "loop_counter_"* ]]
then
    # Rename but then get the loop number you want to test
    signal_loops=${analysis_metric/"loop_counter_"/""}
    analysis_metric=loop_counter
fi

input_path=./simulated_data/searchlights/
results_path=./simulator_parameters/real_results/
output_path=./searchlight_histograms/

# Print the number of files being averaged
match_files=`ls ${input_path}/${condition}*${analysis_metric}.nii.gz | wc -l`
echo "Found $match_files files"

fslmerge -t ${input_path}/temp_${condition}_${analysis_metric}.nii.gz ${input_path}/${condition}*${analysis_metric}.nii.gz

fslmaths ${input_path}/temp_${condition}_${analysis_metric}.nii.gz -nan -mas ${results_path}/significant_mask.nii.gz ${input_path}/temp_${condition}_${analysis_metric}.nii.gz

if [ $analysis_metric == loop_counter ]
then

    #if [ ${condition:0:14} == parallel_rings ]
    #then
    #    # Pull out from the condition name, how many loops there ought to be
    #    signal_structure=${condition/_t_*/}
    #    signal_structure=${signal_structure/*_s-/}
    #    signal_structure=${signal_structure//_/ }
    #    signal_loops=`echo $signal_structure | cut -d " " -f 2` # Take second word of the paired down list
    #else
    #    signal_loops=3
    #fi

	fslmaths ${input_path}/temp_${condition}_${analysis_metric}.nii.gz -thr $signal_loops -uthr $signal_loops ${input_path}/temp_${condition}_${analysis_metric}.nii.gz
	fslstats -t ${input_path}/temp_${condition}_${analysis_metric}.nii.gz -m > ${output_path}/${condition}_${analysis_metric}_${signal_loops}.txt

else
	fslstats -t ${input_path}/temp_${condition}_${analysis_metric}.nii.gz -M > ${output_path}/${condition}_${analysis_metric}_.txt
fi

# Check the data
rm -f ${input_path}/temp_${condition}_${analysis_metric}.nii.gz
