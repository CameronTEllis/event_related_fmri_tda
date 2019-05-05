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
signal_structure=$1  # What graph structure are you using (e.g., elipse, parallel_rings). These options are specified in `generate_graph_structure`
signal_magnitude=$2  # How big, in percent signal change, does the signal structure evoke
signal_properties=$3  # What are the parameters for the choosen graph structure. For instance, if choosing an elipse then [1,1] will make it a unit cirtcle.
permutation=$4  # What resample is this. Useful if you want to make multiple simulations of the same condition
repetitions_per_run=$5  # How many repetitions per run of each event do you want? If it is zero then you will use the same as Anna's (bound by the number of nodes)
nodes=$6 # How many nodes do you want to simulate?
deconvolution=$7 # Specify whether you deconvolve the data
sl_rad=$8  # How big is the searchlight radius? 1 would mean 3x3x3, 3 would mean 7x7x7
dist_metric=$9 # What distance metric are you using. Default is to use euclidean distance after first norming within searchlight (e.g. euclidean, correlation, cityblock, euclidean_norm)

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

run_supersubject=0 # Do you want to perform supersubject analyses

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

echo "Generate the simulated data"
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

# Now that you have finished creating the participant, change the condition name to reflect what is needed

# Specify the searchlight radius
orig_condition=$condition
if [[ $sl_rad -ne 1 ]]
then
	condition=${condition}_rad_${sl_rad}
fi

# Specify the distance metric
if [[ $dist_metric != "euclidean" ]]
then
	condition=${condition}_${dist_metric}
fi

# Wait to finish
participant_num=`ls ${dist_path}/*${condition}.nii.gz | wc -l`

# Create the node brain dist
echo "Create a distance matrix inside each voxel"
if [ $participant_num -ne 20 ]
then

	# Convert nodes into distances
	for i in `seq 1 20`; 
	do 
		sbatch -p $short_partition ${code_path}/run_generate_node_brain_dist.sh $node_path/sub-${i}_${orig_condition}.nii.gz $sl_rad $dist_metric;
	done

	while [ $participant_num -ne 20 ]
	do
		participant_num=`ls ${dist_path}/*${condition}.nii.gz | wc -l`
		sleep 30s
	done	
	
fi


# Run the searchlight supervisor for each of the different conditions being considered
echo "Run the searchlights for all analysis conditions"
for analysis_type in loop_max loop_counter
do
	${code_path}/supervisor_searchlight.sh ${dist_path} ${condition} ${analysis_type}
done

# Wait for the searchlights to finish
echo "Waiting for searchlights ${condition} to finish"
for analysis_type in loop_max loop_counter
do
	participant_num=`ls ${searchlights_path}/*${condition}_${analysis_type}.nii.gz | wc -l`

	# Create the node brain dist
	while [ $participant_num -lt 20 ]
	do
		participant_num=`ls ${searchlights_path}/*${condition}_${analysis_type}.nii.gz | wc -l`
		sleep 30s
	done
done

# Run the ROI based analyses for these searchlights and also run the randomise script which itself does ROI analyses
echo "Performing ROI analyses"
for analysis_type in loop_max loop_counter
do
	files=`ls ${searchlights_path}/*${condition}_${analysis_type}.nii.gz`
	for file in $files
	do	
		# Perform the ROI analyses on each searchlight
		sbatch ${code_path}/run_test_signal_vol.sh $file searchlight_summary/
	done
	
	# Perform the randomise analyses on this condition
	sbatch ${code_path}/run_randomise_demean.sh ${condition} ${analysis_type}
done



# Run the supersubject analyses
if [[ $run_supersubject == 1 ]]
then
	echo "Performing supersubject analyses"

	# Have you made all the participants already?
	if [[ ! -e ${supersubject_path}/${condition}.nii.gz ]]
	then	

		echo Aggregating the distance matrices

		# Make super subject of the distance function
		sbatch -p $short_partition ${code_path}/run_generate_supersubject.sh ${dist_path} $condition 0

		#Wait until the above are done
		while [ ! -e ${supersubject_path}/${condition}.nii.gz ]
		do
			sleep 30s
		done
		sleep 20s
	fi

	# Run the searchlight analysis for each of the different conditions being considered
	for analysis_type in loop_counter loop_max loop_ratio_2 loop_ratio_5 loop_ratio_10
	do
		if [ ! -e ${searchlights_path}/supersubject_${condition}_${analysis_type}.nii.gz ]
		then
	    		sbatch ${code_path}/run_searchlight.sh ${supersubject_path}/${condition}.nii.gz ${analysis_type}
		fi
	done

fi



#if [ $permutation != 0 ]
#then
#	rm -f $node_path/*${condition}.nii.gz
#	rm -f $dist_path/*${condition}.nii.gz
#fi
