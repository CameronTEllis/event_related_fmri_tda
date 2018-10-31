# event_related_fmri_tda
Repository for simulation of event-based fMRI for analysis with TDA. This is the code used in the manuscript: "Limitations of Topological Data Analysis for event-related fMRI", by Cameron T. Ellis, Michael Lesnick, Greg Henselman, Bryn Keller, & Jonathan D. Cohen. The abstract for this manuscript is as follows:

>Recent fMRI research has shown that perceptual and cognitive representations are instantiated in high-dimensional multi-voxel patterns in the brain. However, the methods for detecting these representations are limited. Topological Data Analysis (TDA) is a new approach, based on the mathematical field of topology, that can detect unique types of geometric structures in patterns of data. Several recent studies have successfully applied TDA to the study of various forms of neural data; however, to our knowledge, TDA has not been successfully applied to data from event-related fMRI designs in ways that reveal novel structure. Event-related fMRI is very common but limited in terms of the number of events that can be run within a practical time frame and the effect size that can be expected. Here, we investigate whether TDA can identify known signals given these constraints. We use fmrisim, a Python-based simulator of realistic fMRI data, to assess the plausibility of recovering a simple topological representation under a variety of signal and design conditions. Our results suggest that TDA has limited usefulness for event-related fMRI using current methods, as fMRI data is too noisy to allow representations to be reliably identified using TDA.

## Code organization

To run the first 4 steps of analysis use the "./code/supervisor_supersubject.sh" script. This allows you to specify the parameters that are used to create the data. The code follows the following steps for performing these analyses:

1. generate_node_brain.py  
Generates a realistic simulation of fMRI data according to the design and signal parameters specified. The necessary data to generate the noise simulation is found in './simulator_parameters/'. This script accepts many parameters to specify the design of the experiment and the nature of the signal/signal magnitude. Once the data has been simulated, a univariate GLM is run and the data is stored in './simulated_data/node_brain/' as averages of each condition type

2. generate_node_brain_dist.py  
In order to enable the averaging of distance matrices between participants in an efficient manor, an unravelled distance matrix is created and stored. To calculate the distance matrix, a 3x3x3 searchlight is run on each voxel and the distance between the event types for the pattern of 27 voxels is compared. The searchlight computation creates an event type by event type distance matrix. The upper triangle of this matrix is unravelled to create a vector of values that are stored in that voxel's location.

3. generate_supersubject.sh  
This averages the voxelwise distance matrices across participants.

4. searchlight.py  
This takes in the vectorized/unravelled distance matrices stored in each voxel and computes a specified metric depending on the desired analysis. For instance, this can compute the persistent homology of each voxel and then use it to evaluate the maximum persistence of loop features or to count the number of loops.

5. run_summarise_supersubject.sh
This takes in all of the searchlight volumes stored in "simulated_data/searchlights/" and outputs text files in "./searchlight_summary" with summary statistics for each volume.

6. plot_TDA.ipynb
This is a jupyter notebook used to display and develop the visualizations for the outputs of the analyses. You need the ability to launch a jupyter notebook from your cluster in order to use this script.

## Setup

This code assumes you will be submitting jobs on an High Performance Computing cluster using [SLURM](https://slurm.schedmd.com/overview.html). To run this code on your cluster you must change the file "code/setup_environments.sh" to allow you to set up the appropriate environment. In particular you must create an environment that contains both [brainiak](https://github.com/brainiak/brainiak) and [brainiak-extras](https://github.com/brainiak/brainiak-extras). You must also specify the partition in order to allow jobs to be submitted. 

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
