#!/usr/bin/env bash
# Run through all the searchlight analyses for these participants

path=$1 # Path to file
condition=$2
analysis_type=$3 # What type of searchlight would you like to run

# Setup environment
source code/setup_environment.sh

files=`ls $path*$condition*`
echo Running $analysis_type
for file in $files
do

    output_name="${file/node_brain_dist\//searchlights/}"
    output_name=${output_name%.nii.gz}_${analysis_type}.nii.gz

    if [ ! -e ${output_name} ]
    then

        #Print
        echo "Running $file"

        # Run the command
        sbatch -p $short_partition ./code/run_searchlight.sh $file $analysis_type
    fi

done
