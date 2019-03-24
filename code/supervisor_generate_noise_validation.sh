#!/bin/bash
# bash script to launch generate_noise_validation.py for all participants, all runs, as well as multiple iterations.

# Setup the environment
source ./code/setup_environment.sh

# Cycle through the participants and runs and make the files, one that is the template and many that are resampling.
resamples=10
for subj in `seq 1 20`
do
	for run in `seq 1 5`
	do

		input_noise_dict=$(pwd)/simulator_parameters/noise_dict/sub-${subj}_r${run}.txt
		template=$(pwd)/simulator_parameters/template/sub-${subj}_r${run}.nii.gz

		for resample in `seq 1 $resamples`
		do
			output_noise_dict_resample=$(pwd)/simulated_data/resampled_noise_dicts/sub-${subj}_r${run}_resample-${resample}.txt
			
			# Wait until the noise dict has been created (otherwise it may overwite)
			sbatch -p $short_partition ./code/run_generate_noise_validation.sh $input_noise_dict ${template} ${output_noise_dict_resample}

		done
	done
done
