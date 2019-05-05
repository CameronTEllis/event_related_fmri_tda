# event_related_fmri_tda
Repository for simulation of event-based fMRI for analysis with TDA. This is the code used in the manuscript: "Feasibility of Topological Data Analysis for event-related fMRI", by Cameron T. Ellis, Michael Lesnick, Greg Henselman-Petrusek, Bryn Keller, & Jonathan D. Cohen. The abstract for the manuscript is printed below:

>Recent fMRI research shows that perceptual and cognitive representations are instantiated in high-dimensional multi-voxel patterns in the brain. However, the methods for detecting these representations are limited. Topological Data Analysis (TDA) is a new approach, based on the mathematical field of topology, that can detect unique types of geometric features in patterns of data. Several recent studies have successfully applied TDA to study various forms of neural data; however, to our knowledge, TDA has not been successfully applied to data from event-related fMRI designs. Event-related fMRI is very common but limited in terms of the number of events that can be run within a practical time frame and the effect size that can be expected. Here, we investigate whether persistent homology — a popular TDA tool that identifies topological features in data and quantifies their robustness — can identify known signals given these constraints. We use fmrisim, a Python-based simulator of realistic fMRI data, to assess the plausibility of recovering a simple topological representation under a variety of conditions. Our results suggest that persistent homology can be used under certain circumstances to recover topological structure embedded in realistic fMRI data simulations. 

## Code organization

To run the steps of analysis use the "./code/supervisor_run_all.sh" script. This allows you to specify the parameters that are used to create the data. For instance, an example command would be:
'sbatch ./code/supervisor_run_all.sh elipse 0.5 [1,1] 1 12 5 1 1 euclidean_norm'

This would simulate and analyze a set of 20 participants that have an elipse inserted with 0.5 percent signal change. This elipse would be instantiated in the representation across 12 events that are repeated 5 times per run. There are additional parameters specified in this command that can be learned about from the comment section of this function.

The code follows the following steps for performing these analyses:

1. generate_node_brain.py  
Generates a realistic simulation of fMRI data according to the design and signal parameters specified. The necessary data to generate the noise simulation is found in './simulator_parameters/'. This script accepts many parameters to specify the design of the experiment and the nature of the signal/signal magnitude. Once the data has been simulated, a univariate GLM is run and the data is stored in './simulated_data/node_brain/' as averages of each condition type
Call using run_generate_node_brain.sh. 
Example command: 'sbatch ./code/run_generate_node_brain.sh 1 elipse 1.0 [1,1] [5.0,1,1.0,5,12,1] 0'

2. generate_node_brain_dist.py  
In order to enable the averaging of distance matrices between participants in an efficient manor, an unravelled distance matrix is created and stored. To calculate the distance matrix, a 3x3x3 searchlight is run on each voxel and the distance between the event types for the pattern of 27 voxels is compared. The searchlight computation creates an event type by event type distance matrix. The upper triangle of this matrix is unravelled to create a vector of values that are stored in that voxel's location.
Call using run_generate_node_brain_dist.sh. Run for all participants in a condition using supervisor_generate_node_brain_dist.sh
Example command: 'sbatch ./code/run_generate_node_brain_dist.sh $(pwd)/simulated_data/node_brain/sub-1_elipse_s-1.0_1_1_t_5.0_1_1.0_5_12_1.nii.gz 1 euclidean_norm'

3. searchlight.py  
This takes in the vectorized/unravelled distance matrices stored in each voxel and computes a specified metric depending on the desired analysis. For instance, this can compute the persistent homology of each voxel and then use it to evaluate the maximum persistence of loop features or to count the number of loops.
Call using run_searchlight.sh. Run all participants for a condition using supervisor_searchlight.sh
Example command: 'sbatch ./code/run_searchlight.sh $(pwd)/simulated_data/node_brain_dist/sub-1_elipse_s-1.0_1_1_t_5.0_1_1.0_5_12_1_euclidean_norm.nii.gz loop_counter'

4. run_randomise_demean.sh  
This takes in a condition name and analysis type and performs the ROI analyses, as well as the group analysis. Outputs text files in "./searchlight_summary" and "./randomise_summary/" with summary statistics for each volume.
Example command: 'sbatch ./code/run_randomise_demean.sh elipse_s-1.0_1_1_t_5.0_1_1.0_5_12_1_euclidean_norm loop_counter'

plot_TDA.ipynb is a jupyter notebook used to display and develop the visualizations for the outputs of the analyses. You need the ability to launch a jupyter notebook from your cluster in order to use this script.

The notebook also provides some explanation and examples of code to run to replicate the results from the paper. Hence for naive users it might be best to start with the notebook.

## Setup

This code assumes you will be submitting jobs on an High Performance Computing cluster using [SLURM](https://slurm.schedmd.com/overview.html). To run this code on your cluster you must change the file "code/setup_environments.sh" to allow you to set up the appropriate environment. In particular you must create an environment that contains both [brainiak](https://github.com/brainiak/brainiak) and [brainiak-extras](https://github.com/brainiak/brainiak-extras). You must also specify the partition in order to allow jobs to be submitted. You will also need [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) in order to perform some functions.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
