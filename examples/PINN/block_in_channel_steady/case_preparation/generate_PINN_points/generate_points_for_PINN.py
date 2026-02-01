#This script generates the points for the PINN (steady case; i.e., only x and y coordinates as input to PINN)
#  - It generates the points for the PDE points, boundary points and initial points from the Gmsh mesh
#  - It also generates the points for data points (vtk files of simulation results from SRH-2D)

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

import pygmsh
import gmsh

from vtk import vtkUnstructuredGridReader, vtkCellCenters
from vtk.util import numpy_support

# Get the project root directory (assumes this script is in examples/PINN/block_in_channel_steady/case_preparation/generate_PINN_points directory)
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)  # generate_PINN_points directory
case_prep_dir = os.path.dirname(script_dir)  # case_preparation directory
steady_dir = os.path.dirname(case_prep_dir)  # block_in_channel_steady directory
pinn_dir = os.path.dirname(steady_dir)  # PINN directory
examples_dir = os.path.dirname(pinn_dir)  # examples directory
project_root = os.path.dirname(examples_dir)  # project root directory

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from HydroNet.utils.data_processing_common import convert_mesh_points_for_PINN

import pyHMT2D 


plt.rc('text', usetex=False)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"


def create_data_files_from_SRH2D_results(srhcontrol_file, result_file, PINN_normalization_specs, bBoundary):
    """
    Create data files from SRH-2D vtk file.
    
    Parameters
    ----------
    srhcontrol_file : str
        Name of the SRH-2D srhcontrol file   
    result_file : str
        Name of the SRH-2D result file (e.g., block_in_channel_XMDFC.h5)
    PINN_normalization_specs (dict): The normalization specifications for the PINN data
    bBoundary : bool
        If True, the boundary points and their data will be exported too (currently only wall boundary is supported to enforce no-slip condition)
    """

    #check if the result file exists
    if not os.path.exists(result_file):
        print(f"Error: File {result_file} does not exist.")
        print("Please make sure you have the SRH-2D result file in the correct location.")
        exit()  

    #check the validity of the result file (it has to be in XMDFC HDF5 format)
    if not result_file.endswith('.h5') and not "XMDFC" in result_file:
        print(f"Error: File {result_file} is not in XMDFC HDF5 format.")
        print("Please make sure you have the SRH-2D result file in the correct location.")
        exit()

    #create the SRH_2D_Data object
    my_srh_2d_data = pyHMT2D.SRH_2D.SRH_2D_Data(srhcontrol_file)

    #read SRH-2D result in XMDF format (*.h5)
    #wether the XMDF result is nodal or cell center
    bNodal = False

    my_srh_2d_data.readSRHXMDFFile(result_file, bNodal)

    #convert the SRH-2D result to PINN data and return the statistics of the PINN data: all_PINN_data_stats.json
    all_PINN_points_stats_dict, all_PINN_data_stats_dict = my_srh_2d_data.outputXMDFDataToPINNData(bNodal, PINN_normalization_specs, bBoundary=bBoundary)

    return all_PINN_points_stats_dict, all_PINN_data_stats_dict

if __name__ == '__main__':

    #read the configuration specs from the json config file
    print("Reading the configuration file ...")
    with open('simulations_config.json', 'r') as f:
        config = json.load(f)  

    #Postprocessing specifications
    postprocessing_specs = config['postprocessing_specs']

    # Create data files from SRH-2D simulation result
    result_file = "block_in_channel_XMDFC.h5"   #Name of the SRH-2D result file (currently only XMDFC format is supported)
    bBoundary = False   #Use False for now. It does not work with True (not fully implemented yet for wall's water depth value; need to get internal water depth value from SRH-2D)
    all_PINN_points_stats_dict, all_PINN_data_stats_dict = create_data_files_from_SRH2D_results(postprocessing_specs['PINN_points_specs']['srhcontrol_file'], result_file, postprocessing_specs['PINN_normalization_specs'], bBoundary)
    
    # Convert SRH-2D mesh to PINN points:
    #  - create mesh_points.json file, equation_points.vtk, boundary_points.vtk files
    #  - convert the mesh_points.json file to npy files in "PINN" directory (to be loaded by PINNDataset)
    convert_mesh_points_for_PINN(postprocessing_specs, all_PINN_points_stats_dict, all_PINN_data_stats_dict, n_PDE_points_downsample=0)

    print("All done!")
