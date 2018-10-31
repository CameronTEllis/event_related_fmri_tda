#!/bin/bash
#
# Run through the process of generating and doing super subject analyses on participants
#
# There are many variables available for specification in this script:
# signal_structure= What is the structure you are inserting? Each structure has different properties that also need to be specified. Below are the list of structures:
#	parallel_rings: 24 nodes arranged in N rings parallel to each other, M distance apart. Properties: [N_rings, M_apart]
#	elipse: Create an elipse with 24 nodes of a certain form. Properties: [X_width, Y_width]
#	chain_rings: 24 nodes arranged in N rings at right angles to each other, M distance apart. Properties: [N_rings, M_apart]
#	figure_eight: 24 nodes arranged in a figure 8. Properties: [N_rings, M_apart] 
# signal_magnitude= Percent signal change of the signal being inserted
# signal_properties= Depends on the signal structure
# permutation= What resample iteration is this
#
#SBATCH --output=./logs/supersubject_analysis-%j.out
#SBATCH -t 500
#SBATCH --mem 2000

# Setup environment
source code/setup_environment.sh

# Input the variables
if [ $# -eq 0 ]
then
	signal_structure='community_structure'
        signal_magnitude=2.3
	signal_properties=[1]
	permutation=0
else

	signal_structure=$1
	signal_magnitude=$2
	signal_properties=$3  # Should be a list structure (e.g. [1,2.0])
	permutation=$4
	repetitions_per_run=$5  # How many repetitions per run of each event do you want? If it is zero then you will use the same as Anna's (bound by the number of nodes)
	nodes=$6 # How many nodes do you want to simulate?
	deconvolution=$7 # Specify whether you deconvolve the data
fi

base_path=$(pwd)
code_path=${base_path}/code/
node_path=${base_path}/simulated_data/node_brain/
dist_path=${base_path}/simulated_data/node_brain_dist/
supersubject_path=${base_path}/simulated_data/supersubject_node_brain_dist/
searchlights_path=${base_path}/simulated_data/searchlights/
randomise_path=${base_path}/simulated_data/randomise/
real_data_path=${base_path}/simulator_parameters/real_results/
output_path=${base_path}/

# What are the assumed timing parameters
min_isi=5.0
randomise_timing=1
event_duration=1.0
timing_properties="${min_isi}_${randomise_timing}_${event_duration}_${repetitions_per_run}_${nodes}_${deconvolution}"

# Convert signal properties into appropriate format
signal_properties_str=${signal_properties/[/}
signal_properties_str=${signal_properties_str/]/}
signal_properties_str=${signal_properties_str//,/_}

# If there are more brackets then convert these
signal_properties_str=${signal_properties_str//[/_}
signal_properties_str=${signal_properties_str//]/}
# What is this condition
condition=${signal_structure}_s-${signal_magnitude}_${signal_properties_str}_t_$timing_properties

echo "Running $condition"

if [ $permutation != 0 ]
then
	condition=${condition}_resample-${permutation}
fi

# Wait to finish
participant_num=`ls $node_path/*${condition}.nii.gz | wc -l`

# Have you made all the participants already?
if [ ! -e ${supersubject_path}/${condition}.nii.gz ]
then
	if [ $participant_num -ne 20 ]
	then

		# Generate participants
		${code_path}/supervisor_generate_node_brain.sh ${signal_structure} ${signal_magnitude} ${signal_properties} [${min_isi},${randomise_timing},${event_duration},${repetitions_per_run},${nodes},${deconvolution}] $permutation

        while [ $participant_num -ne 20 ]
        do
                participant_num=`ls ${node_path}/*${condition}.nii.gz | wc -l`
                sleep 30s
        done
	fi
fi

# Wait to finish
participant_num=`ls ${dist_path}/*${condition}.nii.gz | wc -l`

# Have you made all the participants already?
if [[ ! -e ${supersubject_path}/${condition}.nii.gz ]]
then
	if [ $participant_num -ne 20 ]
	then

		# Convert nodes into distances
		for i in `seq 1 20`; 
		do 
			sbatch -p $short_partition ${code_path}/run_generate_node_brain_dist.sh $node_path/sub-${i}_${condition}.nii.gz;
		done

		while [ $participant_num -ne 20 ]
		do
		participant_num=`ls ${dist_path}/*${condition}.nii.gz | wc -l`
		sleep 30s
		done
		
	fi
	
	# Compute the searchlight on all of the participants
        for i in `seq 1 20`;
        do
		# Run the searchlight analysis on this data
		sbatch -p $short_partition ${code_path}/run_searchlight.sh ${dist_path}/sub-${i}_${condition}.nii.gz loop_max
		sbatch -p $short_partition ${code_path}/run_searchlight.sh ${dist_path}/sub-${i}_${condition}.nii.gz loop_counter
	done

	# Make super subject of the distance function
	sbatch -p $short_partition ${code_path}/run_generate_supersubject.sh ${dist_path} $condition 0

fi

#Wait until the above are done
while [ ! -e ${supersubject_path}/${condition}.nii.gz ]
do
	sleep 30s
done
sleep 20s

# Perform searchlights. All tests are for a single loop feature. You should add to test_graph_structure.py if you want more tests
sbatch ${code_path}/run_searchlight.sh ${supersubject_path}/${condition}.nii.gz loop_ratio
sbatch ${code_path}/run_searchlight.sh ${supersubject_path}/${condition}.nii.gz loop_counter
sbatch ${code_path}/run_searchlight.sh ${supersubject_path}/${condition}.nii.gz loop_max


#if [ $permutation != 0 ]
#then
#	rm -f $node_path/*${condition}.nii.gz
#	rm -f $dist_path/*${condition}.nii.gz
#fi
