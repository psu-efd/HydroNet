"""
Postprocess the results of simulations with the SRH_2D_Model and SRH_2D_Data classes, for the purpose of preparing the data for DeepONet training.

The postprocessing is done in the following steps:
1. Extract the results (h, u, v) at cell centers from the SRH-2D simulation results.
2. Split the data into training, validation, and testing sets.
3. Save the data into files.

"""

#if run in the cloud, need to specify the location of pyHMT2D. If pyHMT2D is installed
#with regular pip install, then the following is not necessary.
#import sys
#sys.path.append(r"C:\Users\Administrator\python_packages\pyHMT2D")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import multiprocessing
import os
import json
import platform
import time

import glob
import re

# Import from processing_common.py which is three levels up
import sys
import os
# Add the examples directory to the path (three levels up from this file)
examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)
from HydroNet.utils.data_processing_common import srh_to_pinn_points, convert_mesh_points_for_PINN, postprocess_results_for_DeepONet, verify_data_for_DeepONet, plot_profile_results


# Set the random seed before generating any random numbers
np.random.seed(123456)  # You can use any integer as the seed

plt.rc('text', usetex=False)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"
    


if __name__ == "__main__":

    #record the start time
    start_time = time.time()

    #read the run specs from the json config file
    print("Reading the configuration file ...")
    with open('simulations_config.json', 'r') as f:
        config = json.load(f)

    #get number of samples
    nSamples = config['parameter_specs']['n_samples']

    run_specs = config['run_specs']

    #The name of the case (used as the base to generate other file names, such as the control file name)
    case_name = run_specs['case_name']
    
    #simulation result file name to save the postprocessed results
    simulation_result_file = "postprocessed_simulation_results.json"

    #Simulation output unit (EN: English units, SI: SI units)
    output_unit = run_specs['output_unit']     

    #Postprocessing specifications
    postprocessing_specs = config['postprocessing_specs']

    #get the split fractions
    split_fractions = postprocessing_specs['split_fractions']
    training_fraction = split_fractions['training']
    validation_fraction = split_fractions['validation']
    test_fraction = split_fractions['test']

    #check if the sum of the split fractions is 1
    assert abs(training_fraction + validation_fraction + test_fraction - 1.0) < 1e-10, "Ratios must sum to 1"

    #Get the system name: Windows or Linux
    system_name = platform.system()
    
    # Define the flow variables to be postprocessed based on the output unit. The variables are in the order of (h, u, v), and additionally, Bed_Elev, ManningN, Sx, Sy
    if output_unit == "SI":
        flow_variables = ["Water_Depth_m", "Velocity_m_p_s", "Bed_Elev_meter", "ManningN", "Sx", "Sy"]
    elif output_unit == "EN":     #It seems that the name of the variables are slightly different for Windows and Linux. So depending on the system for which the code is run, the name of the variables are slightly different. Check the VTK files to see the actual names of the variables.
        if system_name == "Windows":
            flow_variables = ["Water_Depth_ft", "Velocity_ft_p_s", "Bed_Elev_ft", "ManningN", "Sx", "Sy"]
        elif system_name == "Linux":
            flow_variables = ["Water_Depth_ft", "Velocity_ft_p_s", "Bed_Elev_ft", "ManningN", "Sx", "Sy"]
    else:
        raise ValueError("Unsupported output unit: " + output_unit)


    #read the sampled parameters from the file 
    #get the sample file name from "simulations_config.json"
    sampled_parameters_file = run_specs['sampled_parameters_file']
    #load the sampled parameters from the file (the first row is the header)
    sampled_parameters = np.loadtxt(sampled_parameters_file, skiprows=1)

    #compute the mean and standard deviation of the sampled parameters
    sampled_parameters_mean = np.mean(sampled_parameters, axis=0)
    sampled_parameters_std = np.std(sampled_parameters, axis=0)

    print("sampled_parameters_mean = ", sampled_parameters_mean)
    print("sampled_parameters_std = ", sampled_parameters_std)
    
    #read the number of cells from the postprocessing specifications
    nCells = postprocessing_specs['nCells']

    #call srh_to_pinn_points to convert the SRH-2D mesh to points needed for using NN to solve PDEs
    # The control file name for the simulation case    
    if system_name == "Windows":
        srhcontrol_file = case_name+".srhhydro"
    elif system_name == "Linux":
        srhcontrol_file = case_name+"_SIF.dat"
    else:
        raise ValueError("Unsupported operating system: " + system_name)

    print("Converting the SRH-2D mesh to PINN points (PDE points and boundary points; result in mesh_points.json file) ...")
    srh_to_pinn_points("base_case/"+srhcontrol_file, refinement_pde=1, refinement_bc=1)

    #postprocess the results according to the postprocessing specifications
    print("Postprocessing the results ...")
    #nSamples = 100 #for testing (comment out for full postprocessing)
    all_DeepONet_stats = postprocess_results_for_DeepONet(nSamples, sampled_parameters, flow_variables, output_unit, postprocessing_specs)

    #verify the data for DeepONet
    #verify_data_for_DeepONet()

    #read all_DeepONet_stats from the file
    all_DeepONet_stats = json.load(open('data/DeepONet/all_DeepONet_stats.json', 'r'))

    #convert the mesh points in mesh_points.json to PINN points
    convert_mesh_points_for_PINN(postprocessing_specs, all_DeepONet_stats['all_data_stats'])

    #plot some results for visual checking
    #print("Plotting the results for DeepONet ...")
    #case_indices = [77, 372, 821, 522]    #1-based index
    #case_indices = [77]
    #for case_index in case_indices:
    #    plot_profile_results(case_index, "wse", output_unit)

    #record the end time
    end_time = time.time()
    
    #print the total time taken in hours
    print("Total time taken: ", (end_time - start_time)/3600, " hours")


    print("All done!")


