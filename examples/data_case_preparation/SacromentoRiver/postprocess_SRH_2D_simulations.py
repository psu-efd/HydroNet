"""
Postprocess the results of simulations with the SRH_2D_Model and SRH_2D_Data classes, for the purpose of preparing the data for DeepONet training.

The postprocessing is done in the following steps:
1. Extract the results (U, V, h) at cell centers from the SRH-2D simulation results.
2. Split the data into training, validation, and testing sets.
3. Save the data into numpy files.

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

from vtk.util import numpy_support as VN
import vtk

from sklearn.model_selection import train_test_split
import h5py    

# Set the random seed before generating any random numbers
np.random.seed(123456)  # You can use any integer as the seed

plt.rc('text', usetex=False)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def probeUnstructuredGridVTKOnPoints(pointVTK, readerUnstructuredGridVTK, varName,
                                         kernel="gaussian", radius=None, nullValue=None):
        """ Interpolate the data from the Unstructured Grid VTK onto given points.

            Currently, it simply call probeUnstructuredGridVTKOverLine(...) because it can handle points.

        Parameters
        ----------
        pointVTK : vtkPoints
            coordinates of points in vtkPoints format; the points don't need to ordered,
            thus they can be just a bunch of points

        """

        return probeUnstructuredGridVTKOverLine(pointVTK, readerUnstructuredGridVTK, varName,
                                              kernel, radius, nullValue)

def probeUnstructuredGridVTKOverLine(lineVTK, readerUnstructuredGridVTK, varName,
                                         kernel="gaussian", radius=None, nullValue=None):
        """ Interpolate the data from the Unstructured Grid VTK onto a line (profile).


        The unstructured grid VTK is supposed to be a 2D surface in 3D space, such as the mesh used in 2D hydraulics
        models.

        To probe on them, the surface has to be flattened first.

        Parameters
        ----------
        lineVTK : vtkLineSource or vtkPoints
            coordinates of points in the vtkLineSource; the points don't need to ordered,
            thus they can be just a bunch of points
        readerUnstructuredGridVTK : vtkUnstructuredGridReader
            Unstructured Grid VTK reader
        varName : str
            name of the variable to be probed
        kernel : str
            name of the kernel for interpolation (linear, gaussin, voronoi, Shepard"
        radius : float
            radius for interpolation kernels
        nullValue: float
            value to be assigned to invalid probing points


        Returns
        -------
        points: numpy arrays [number of points, 3]; points on the profile
        probed result array:
        elev: elevation (z) of points in the profile

        """

        # Get data from the Unstructured Grid VTK reader
        data = readerUnstructuredGridVTK.GetOutput()

        # make sure the data is stored at points (for smoother interpolation)
        cell2point = vtk.vtkCellDataToPointData()
        cell2point.SetInputData(data)
        cell2point.Update()
        data = cell2point.GetOutput()   #after this, all data are stored at points, not cell centers.

        bounds = data.GetBounds()

        #print("Unstructured Grid VTK bounds = ", bounds)
        #print("Unstructured Grid number of cells: ", data.GetNumberOfCells())
        #print("Unstructured Grid number of points: ", data.GetNumberOfPoints())

        if radius is None:
            boundingArea = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])   #assume 2D Grid
            averageCellArea =  boundingArea/data.GetNumberOfCells()            #average cell area
            radius = np.sqrt(averageCellArea)                                  #average size of cell
            radius = 2.0*radius                                                #double the search radius

        ### make a transform to set all Z values to zero ###
        flattener = vtk.vtkTransform()
        flattener.Scale(1.0, 1.0, 0.0)

        ### flatten the input in case it's not already flat ###
        i_flat = vtk.vtkTransformFilter()

        if isinstance(lineVTK, vtk.vtkLineSource):
            i_flat.SetInputConnection(lineVTK.GetOutputPort())
        elif isinstance(lineVTK, vtk.vtkPoints):
            polydata_temp = vtk.vtkPolyData()
            polydata_temp.SetPoints(lineVTK)
            i_flat.SetInputData(polydata_temp)
        else:
            raise Exception("lineVTK type,", type(lineVTK),", not supported. Only vtkLineSource and vtkPoints are supported.")

        i_flat.SetTransform(flattener)

        ### transfer z elevation values to the source's point scalar data ###
        s_elev = vtk.vtkElevationFilter()
        s_elev.SetInputData(data)
        s_elev.SetHighPoint(0, 0, bounds[5])
        s_elev.SetLowPoint(0, 0, bounds[4])
        s_elev.SetScalarRange(bounds[4], bounds[5])
        s_elev.Update()

        #print("s_elev = ", s_elev.GetUnstructuredGridOutput())

        ### flatten the source data; the Z elevations are already in the scalars data ###
        s_flat = vtk.vtkTransformFilter()
        s_flat.SetInputConnection(s_elev.GetOutputPort())
        s_flat.SetTransform(flattener)

        # build the probe using vtkPointInterpolator
        # construct the interpolation kernel
        if kernel == 'gaussian':
            kern = vtk.vtkGaussianKernel()
            kern.SetSharpness(2)
            kern.SetRadius(radius)
        elif kernel == 'voronoi':
            kern = vtk.vtkVoronoiKernel()
        elif kernel == 'linear':
            kern = vtk.vtkLinearKernel()
            kern.SetRadius(radius)
        elif kernel == 'Shepard':
            kern = vtk.vtkShepardKernel()
            kern.SetPowerParameter(2)
            kern.SetRadius(radius)
        else:
            raise Exception("The specified kernel is not supported.")

        probe = vtk.vtkPointInterpolator()
        probe.SetInputConnection(i_flat.GetOutputPort())
        probe.SetSourceConnection(s_flat.GetOutputPort())
        probe.SetKernel(kern)
        if nullValue is not None:
            probe.SetNullValue(nullValue)
        else:
            probe.SetNullPointsStrategyToClosestPoint()

        probe.Update()

        # (This approach of using vtkProbeFilter is replaced by vtkPointInterpolator for smoother result)
        # vtkProbeFilter, the probe line is the input, and the underlying dataset is the source.
        #probe = vtk.vtkProbeFilter()
        #probe.SetInputConnection(i_flat.GetOutputPort())
        #probe.SetSourceConnection(s_flat.GetOutputPort())
        #probe.Update()

        # get the data from the VTK-object (probe) to an numpy array
        #print("varName =", varName)
        #print(probe.GetOutput().GetPointData().GetArray(varName))

        #print("varName before special treatment = ", varName)
        #print("probe.GetOutput().GetPointData().GetArray(varName) = ", probe.GetOutput().GetPointData())

        #special treatment of the bed shear stress because its name is different for Windows and Linux
        #check whether the vtk file contains the variable "B_Stress_lb_p_ft" or "Strs_lb_p_ft2"
        if varName == "B_Stress_lb_p_ft" or varName == "Strs_lb_p_ft2":
            if probe.GetOutput().GetPointData().HasArray("B_Stress_lb_p_ft"):
                varName = "B_Stress_lb_p_ft"
            elif probe.GetOutput().GetPointData().HasArray("Strs_lb_p_ft2"):
                varName = "Strs_lb_p_ft2"

        #print("varName after special treatment = ", varName)

        varProbedValues = VN.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray(varName))

        numPoints = probe.GetOutput().GetNumberOfPoints()  # get the number of points on the line

        # get the elevation from the VTK-object (probe) to an numpy array
        elev = VN.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray("Elevation"))

        # intialise the points on the line
        x = np.zeros(numPoints)
        y = np.zeros(numPoints)
        z = np.zeros(numPoints)
        points = np.zeros((numPoints, 3))

        # get the coordinates of the points on the line
        for i in range(numPoints):
            x[i], y[i], z[i] = probe.GetOutput().GetPoint(i)
            points[i, 0] = x[i]
            points[i, 1] = y[i]
            points[i, 2] = z[i]

        return points, varProbedValues, elev

def sample_vtk_at_points(vtk_file, point_coordinates, variables):
    """Sample VTK data at specific points    

    Args:
        vtk_file: VTK file path
        point_coordinates (list): List of [x,y] coordinates
        variables (list): List of variable names to sample
    Returns:
        dict: Dictionary of sampled values
        numpy.ndarray: Numpy array of sampled values of all variables
    """

    # Read VTK file
    vtk_reader = vtk.vtkUnstructuredGridReader()
    vtk_reader.SetFileName(vtk_file)
    vtk_reader.Update()

    #print("points = ", points)
       
    # Create points object
    points_vtk = vtk.vtkPoints()
    for p in point_coordinates:
        points_vtk.InsertNextPoint(p[0], p[1], 0)

    #print("points_vtk = ", points_vtk)    

    # Get results
    result_dict = {}  #result as a dictionary
    result_array = np.zeros((len(point_coordinates), len(variables)+1))  #result as a numpy array

    index_var = 0
    for var in variables:
        var_name = str(var)

        #print("var = ", var_name)

        #probe the variable at the probing point
        _, array, elevation = probeUnstructuredGridVTKOnPoints(points_vtk, vtk_reader, var_name)

        #print("array = ", array)

        if array is not None:
            if "Velocity" in var:
                #extract the x and y components of the velocity vector
                u = array[:,0]
                v = array[:,1]

                #save the x and y components of the velocity vector to the result dictionary
                result_dict["Velocity_x"] = u
                result_dict["Velocity_y"] = v

                #save the x and y components of the velocity vector to the result array
                result_array[:, index_var] = u 

                index_var += 1

                result_array[:, index_var] = v

                index_var += 1
            else:
                result_dict[var_name] = array      
                result_array[:, index_var] = array
                index_var += 1
        else:
            raise ValueError("point sampling array is empty for variable = ", var_name)

    
    #print("result_dict = ", result_dict)
    #print("result_array = ", result_array)

    return result_dict, result_array, elevation


def sample_vtk_along_line(vtk_file, start_point, end_point, num_points, variables):

    """Sample VTK data along a line
    
    Args:
        vtk_file: VTK file path
        start_point (list): [x,y] coordinates of line start
        end_point (list): [x,y] coordinates of line end
        num_points (int): Number of sampling points along line
        variables (list): List of variable names to sample        

    Returns:
        dict: Dictionary of sampled values
    """
    
    # Create line points
    x = np.linspace(start_point[0], end_point[0], num_points)
    y = np.linspace(start_point[1], end_point[1], num_points)
    point_coordinates = np.column_stack((x, y))

    return sample_vtk_at_points(vtk_file, point_coordinates, variables)

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm
    
    Args:
        point: [x, y] coordinates of the point to check
        polygon: List of [x, y] coordinates defining the polygon vertices
        
    Returns:
        bool: True if point is inside polygon, False otherwise
    """
    x, y, z = point
    inside = False
    
    # Get the number of vertices
    n = len(polygon)
    
    # Check each edge of the polygon
    j = n - 1
    for i in range(n):
        # Get vertices of the edge
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        # Check if point crosses the edge
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
            
        j = i
    
    return inside

def get_cells_in_domain(vtk_reader):
    """Get cells whose centers are within the domain
    
    Args:
        vtk_reader: VTK reader object
        
    Returns:
        list: List of cell IDs that are inside the domain
        list: List of cell centers
    """
    
    # Get cell centers
    cell_ids = []
    
    vtk_data = vtk_reader.GetOutput()
    n_cells = vtk_data.GetNumberOfCells()

    cell_centers = np.zeros((n_cells, 3))
        
    for i in range(n_cells):
        cell = vtk_data.GetCell(i)

        # Calculate cell center manually 
        center = [0.0, 0.0, 0.0]
        points = cell.GetPoints()
        n_points = points.GetNumberOfPoints()
        
        # Average the coordinates of all points
        for j in range(n_points):
            point = points.GetPoint(j)
            center[0] += point[0]
            center[1] += point[1]
            center[2] += point[2]

        center[0] /= n_points
        center[1] /= n_points
        center[2] /= n_points

        cell_ids.append(i)        
        cell_centers[i,:] = center
    
    return cell_ids, cell_centers

def sample_vtk_in_domain(vtk_file, flow_variables, output_unit):
    """Sample VTK data within a domain
    
    Args:
        vtk_file: VTK file path
        flow_variables: List of flow variables to sample in the order of (h, u, v)
        output_unit: Output unit, SI or EN
        
    Returns:
        dict: Dictionary of sampled values
        numpy.ndarray: Numpy array of sampled values of all variables in the order of (h, u, v)
    """
    # Read VTK file
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.Update()

    vtk_data = reader.GetOutput()

    # Get cells inside domain
    cell_ids, cell_centers = get_cells_in_domain(reader)

    result_dict = {}
    result_array = np.zeros((len(cell_ids), len(flow_variables)+1))

    #make sure "Water_Depth" is in the first position of the flow_variables list
    if "Water_Depth" not in flow_variables[0]:
        print("Water_Depth is not in the first position of the flow_variables list")
        print("flow_variables = ", flow_variables)
        raise ValueError("Water_Depth is not in the first position of the flow_variables list")
    
    # Sample data for these cells   
    index_var = 0
    for var in flow_variables:       
        #print("sampling ", var)

        array = np.array(vtk_data.GetCellData().GetArray(var))

        #print("array.shape = ", array.shape)       

        if array is not None:       

            #check whether the variable is the velocity vector
            if "Velocity" in var:
                #extract the x and y components of the velocity vector
                u = array[:,0]
                v = array[:,1]

                #save the x and y components of the velocity vector to the result dictionary
                result_dict["Velocity_x"] = u
                result_dict["Velocity_y"] = v

                #save the x and y components of the velocity vector to the result array
                result_array[:, index_var] = u 

                index_var += 1

                result_array[:, index_var] = v

                index_var += 1


            else:
                result_dict[var] = array

                result_array[:, index_var] = array

                index_var += 1

        else:
            raise ValueError("variable = ", var, " is not found in the VTK data")    

    return cell_centers, result_dict, result_array



def extract_simulation_results(vtk_file, flow_variables, output_unit):
    """Extract simulation results from a given VTK file
    
    Args:
        vtk_file (str): VTK file path
        flow_variables (list): List of flow variables to sample in the order of (h, u, v)
        output_unit (str): Output unit, SI or EN
    """
   
    #cell centers are the same for all VTK files
    cell_centers = None    
        
    # Process sampling
    cell_centers, results_dict, results_array = sample_vtk_in_domain(vtk_file, flow_variables, output_unit)         

    return cell_centers, results_dict, results_array

def convert_numpy_to_list(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def split_indices(sample_indices, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split sample indices into training, validation, and test sets.
    
    Parameters:
    -----------
    sample_indices : list
        List of sample indices to split
    train_ratio : float, default=0.7
        Ratio of training samples
    val_ratio : float, default=0.2
        Ratio of validation samples
    test_ratio : float, default=0.1
        Ratio of test samples
        
    Returns:
    --------
    train_indices, val_indices, test_indices : tuple of lists
        The split indices for each set
    """
    # Verify that ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Convert to numpy array for easier handling
    indices = np.array(sample_indices)
    
    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=42
    )
    
    # Second split: separate train and validation sets
    # Calculate the validation ratio relative to the remaining data
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=relative_val_ratio, random_state=42
    )
    
    print(f"Total samples: {len(indices)}")
    print(f"Training samples: {len(train_indices)} ({len(train_indices)/len(indices):.1%})")
    print(f"Validation samples: {len(val_indices)} ({len(val_indices)/len(indices):.1%})")
    print(f"Test samples: {len(test_indices)} ({len(test_indices)/len(indices):.1%})")
    
    return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()

def postprocess_results(nSamples, sampled_parameters, nCells, flow_variables, output_unit, training_fraction, validation_fraction, test_fraction):
    """Postprocess the results of the simulations

    The split is done based on the successful simulations. Also, the split is based on cases, not samples (cell center data).

    Args:
        nSamples (int): Number of samples
        sampled_parameters (numpy array): Sampled parameters with shape (nSamples, n_features)
        nCells (int): Number of cells per case
        flow_variables (list): List of flow variables to sample in the order of (h, u, v)
        output_unit (str): Output unit, SI or EN
        training_fraction (float): Training fraction
        validation_fraction (float): Validation fraction
        test_fraction (float): Test fraction
    """

    # Create output directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/val', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    # Generate indices for splitting
    indices_allSamples = list(range(1, nSamples+1))  #1-based index
    train_indices, val_indices, test_indices = split_indices(indices_allSamples, training_fraction, validation_fraction, test_fraction)
    
    # Save split indices for reference
    split_indices_dict = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }
    with open('data/split_indices.json', 'w') as f:
        json.dump(split_indices_dict, f, indent=4)
    
    # Process each split
    for split_name, indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
        print("Processing split = ", split_name)

        # Create HDF5 file for this split
        h5_file = h5py.File(f'data/{split_name}/data.h5', 'w')
        
        # Calculate total number of data points for this split
        total_data_points = len(indices) * nCells
        
        # Initialize datasets with chunking
        chunk_size = min(10000, total_data_points)  # Adjust chunk size as needed
        n_features = sampled_parameters.shape[1]  # Number of branch input features
        n_coords = 2  # x, y coordinates for trunk inputs
        n_outputs = 3  # h, u, v for outputs
        
        # Create datasets with chunking
        branch_inputs = h5_file.create_dataset(
            'branch_inputs',
            shape=(total_data_points, n_features),
            chunks=(chunk_size, n_features),
            dtype='float32'
        )
        
        trunk_inputs = h5_file.create_dataset(
            'trunk_inputs',
            shape=(total_data_points, n_coords),
            chunks=(chunk_size, n_coords),
            dtype='float32'
        )
        
        outputs = h5_file.create_dataset(
            'outputs',
            shape=(total_data_points, n_outputs),
            chunks=(chunk_size, n_outputs),
            dtype='float32'
        )
        
        # Process data in chunks
        current_data_point = 0
        for case_idx in indices:
            print("    Processing case = ", case_idx)

            vtk_file = f"cases/vtks/case_{case_idx:06d}.vtk"
            cell_centers, results_dict, results_array = extract_simulation_results(vtk_file, flow_variables, output_unit)
            
            # Get the number of cells for this case
            n_cells_case = len(cell_centers)

            #make sure the number of cells is correct
            assert n_cells_case == nCells, "Number of cells is not correct"
            
            # Prepare data for this case
            case_branch_inputs = np.tile(sampled_parameters[case_idx-1, :], (n_cells_case, 1))  # -1 because indices are 1-based
            case_trunk_inputs = cell_centers[:, :2]  # Only need x,y coordinates
            case_outputs = results_array[:, :3]  # h, u, v
            
            # Write this case's data to the HDF5 file
            branch_inputs[current_data_point:current_data_point + n_cells_case] = case_branch_inputs
            trunk_inputs[current_data_point:current_data_point + n_cells_case] = case_trunk_inputs
            outputs[current_data_point:current_data_point + n_cells_case] = case_outputs
            
            current_data_point += n_cells_case
        
        # Add metadata
        h5_file.attrs['n_samples'] = total_data_points
        h5_file.attrs['n_features'] = n_features
        h5_file.attrs['n_coords'] = n_coords
        h5_file.attrs['n_outputs'] = n_outputs
        h5_file.attrs['output_unit'] = output_unit
        
        # Close the HDF5 file
        h5_file.close()
        
        print(f"Saved {total_data_points} data points to data/{split_name}/data.h5")
    
    print("Data postprocessing completed successfully!")


def plot_profile_results(case_index, variable_name, output_unit):
    """
    Plot some profile results for visual checking

    Args:
        case_index (int): The index of the case to plot (1-based index)
        variable_name (str): The name of the variable to plot
        output_unit (str): The output unit
    """

    print("Plotting the results for case index = ", case_index)

    #load the postprocessed results: center line results for the current case (h, u, v, elevation, length)
    postprocessed_results = np.load("data/center_line_results/case_"+str(case_index).zfill(6)+".npy")

    #water depth 
    h = postprocessed_results[:, 0]

    #velocity
    u = postprocessed_results[:, 1]
    v = postprocessed_results[:, 2]

    #elevation
    elevation = postprocessed_results[:, 3]

    #length
    length = postprocessed_results[:, 4]

    #compute the water surface elevation
    wse = h + elevation

    #create a single plot of the selected samples
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    #plot the water surface elevation
    axs.plot(length, wse, 'b--', label='Water Surface Elevation')

    #plot the bed elevation
    axs.plot(length, elevation, color='k', label='Bed')

    #add the legend
    axs.legend()

    #add the x-axis label
    axs.set_xlabel('Length (m)', fontsize=14)

    #add the y-axis label
    axs.set_ylabel('Elevation (m)', fontsize=14)

    #set y-axis limits
    #axs.set_ylim(0, 1.0)

    #set the fontsize of the tick labels
    axs.tick_params(axis='both', labelsize=12)  

    #set the fontsize of the title
    axs.set_title('Case '+str(case_index), fontsize=14)

    #save the plot
    plt.savefig("example_wse_profile_case_"+str(case_index).zfill(4)+".png", dpi=300, bbox_inches='tight')

    #show the plot
    plt.show()

    


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
    
    # Define the flow variables to be postprocessed based on the output unit. The variables are in the order of (h, u, v)
    if output_unit == "SI":
        flow_variables = ["Water_Depth_m", "Velocity_m_p_s"]
    elif output_unit == "EN":     #It seems that the name of the variables are slightly different for Windows and Linux. So depending on the system for which the code is run, the name of the variables are slightly different. Check the VTK files to see the actual names of the variables.
        if system_name == "Windows":
            flow_variables = ["Water_Depth_ft", "Velocity_ft_p_s"]
        elif system_name == "Linux":
            flow_variables = ["Water_Depth_ft", "Velocity_ft_p_s"]
    else:
        raise ValueError("Unsupported output unit: " + output_unit)


    #read the sampled parameters from the file 
    #get the sample file name from "simulations_config.json"
    sampled_parameters_file = run_specs['sampled_parameters_file']
    #load the sampled parameters from the file (the first row is the header)
    sampled_parameters = np.loadtxt(sampled_parameters_file, skiprows=1)

    #read the number of cells from the postprocessing specifications
    nCells = postprocessing_specs['nCells']

    #postprocess the results according to the postprocessing specifications
    print("Postprocessing the results ...")
    postprocess_results(nSamples, sampled_parameters, nCells, flow_variables, output_unit, training_fraction, validation_fraction, test_fraction)

    #plot some results for visual checking
    print("Plotting the results ...")
    case_indices = [77, 372, 821, 522]    #1-based index
    #case_indices = [77]
    #for case_index in case_indices:
    #    plot_profile_results(case_index, "wse", output_unit)

    #record the end time
    end_time = time.time()
    
    #print the total time taken in hours
    print("Total time taken: ", (end_time - start_time)/3600, " hours")


    print("All done!")


