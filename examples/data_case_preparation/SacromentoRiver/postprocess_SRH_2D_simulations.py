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

def sample_vtk_at_points(vtk_reader, point_names, point_coordinates, variables):
    """Sample VTK data at specific points    

    Args:
        vtk_reader: VTK reader object
        point_names (list): List of point names
        point_coordinates (list): List of [x,y] coordinates
        variables (list): List of variable names to sample
        


    Returns:
        dict: Dictionary of sampled values
        numpy.ndarray: Numpy array of sampled values of all variables
    """


    #print("points = ", points)
       
    # Create points object
    points_vtk = vtk.vtkPoints()
    for p in point_coordinates:
        points_vtk.InsertNextPoint(p[0], p[1], 0)

    #print("points_vtk = ", points_vtk)    

    # Get results
    result_dict = {}  #result as a dictionary
    result_array = np.zeros((len(variables), len(point_coordinates)))  #result as a numpy array

    for index_var, var in enumerate(variables):
        var_name = str(var)

        #print("var = ", var_name)

        #probe the variable at the probing point
        _, array, _ = probeUnstructuredGridVTKOnPoints(points_vtk, vtk_reader, var_name)

        #print("array = ", array)

        if array is not None:
            result_dict[var_name] = array      
            result_array[index_var,:] = array
        else:
            raise ValueError("point sampling array is empty for variable = ", var_name)

    
    #print("result_dict = ", result_dict)
    #print("result_array = ", result_array)

    return result_dict, result_array


def sample_vtk_along_line(vtk_reader, start_point, end_point, num_points, variables):

    """Sample VTK data along a line
    
    Args:
        vtk_reader: VTK reader object
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

    #create point names
    point_names = ["point_"+str(i) for i in range(num_points)]    

    return sample_vtk_at_points(vtk_reader, point_names, point_coordinates, variables)

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

def sample_vtk_in_domain(vtk_reader, flow_variables, output_unit):
    """Sample VTK data within a domain
    
    Args:
        vtk_data: VTK unstructured grid data
        flow_variables: List of flow variables to sample
        output_unit: Output unit, SI or EN
        
    Returns:
        dict: Dictionary of sampled values
    """
    vtk_data = vtk_reader.GetOutput()

    # Get cells inside domain
    cell_ids, cell_centers = get_cells_in_domain(vtk_reader)

    result_dict = {}
    result_array = np.zeros((len(cell_ids), len(flow_variables)+1))
    
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



def extract_all_simulation_results(vtk_files, flow_variables, output_unit):
    """Extract simulation results from all VTK files for a given case
    
    Args:
        vtk_files (list): List of VTK file paths
        flow_variables (list): List of flow variables to sample
        output_unit (str): Output unit, SI or EN
    """
    
    all_results_list = []
    all_results_array = []

    #cell centers are the same for all VTK files
    cell_centers = None
    
    #loop over all VTK files
    for i, vtk_file in enumerate(vtk_files):
    #for i, vtk_file in enumerate(vtk_files[0:2]):   #debugging
        #print("processing ", vtk_file)

        # Read VTK file
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(vtk_file)
        reader.Update()
        
        # Process sampling
        cell_centers, results_dict, results_array = sample_vtk_in_domain(reader, flow_variables, output_unit)

        all_results_list.append(results_dict)
        all_results_array.append(results_array)
        

    return cell_centers, all_results_list, all_results_array

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

def postprocess_results(nSamples, nSaveTimes, nHydrograph_points, flow_variables, output_unit, training_fraction, validation_fraction, test_fraction):
    """Postprocess the results of the simulations

    The split is done based on the successful simulations. Also, the split is based on cases, not samples (cell center data).

    Args:
        nSamples (int): Number of samples
        nSaveTimes (int): Number of save times in the simulation
        nHydrograph_points (int): Number of points in the hydrograph
        flow_variables (list): List of flow variables to sample
        output_unit (str): Output unit, SI or EN
        training_fraction (float): Training fraction
        validation_fraction (float): Validation fraction
        test_fraction (float): Test fraction
    """

    # Get list of subdirectories in "cases/vtks" directory
    case_dirs = [d for d in os.listdir("cases/vtks") if os.path.isdir(os.path.join("cases/vtks", d)) and d.startswith("case_")]

    # Extract indices using regular expression
    case_indices = []
    for case_dir in case_dirs:
        # Extract the number from "case_XXXXXX"
        match = re.search(r'case_(\d+)', case_dir)
        if match:
            case_indices.append(int(match.group(1)))  # Convert to integer

    # Sort the indices if needed
    case_indices.sort()

    print(f"Found {len(case_dirs)} cases out of {nSamples} simulations")

    #only do the first 10 cases for debugging
    #case_indices = case_indices[0:10]
    #case_dirs = case_dirs[0:10]
    
    #print("case_indices = ", case_indices)
    #print("case_dirs = ", case_dirs)

    #split the cases into training, validation, and test sets
    # Split the indices (1-based)
    train_indices, val_indices, test_indices = split_indices(case_indices, training_fraction, validation_fraction, test_fraction)

    print("train_indices[0:5] = ", train_indices[0:5])
    print("val_indices[0:5] = ", val_indices[0:5])

    print("test_indices[0:5] = ", test_indices[0:5])
    
    #list of store the results for all cases
    all_results_dict = []
    all_results_array = []

    #loop over all case directories
    for case_dir in case_dirs:
        print("case_dir = ", case_dir)

        #Get a list of all vtk files in "cases/vtks"
        files = glob.glob("cases/vtks/"+case_dir+"/SRH2D_oneD_channel_with_bump_C_*.vtk")

        #Extract indices using regex
        file_indices = []
        pattern = re.compile(r'SRH2D_oneD_channel_with_bump_C_(\d+)\.vtk')

        for file in files:
            match = pattern.search(file)
            if match:
                index = int(match.group(1))          #index is 1-based
                file_indices.append((file, index))

        #Sort files by index
        file_indices.sort(key=lambda x: x[1])   #file_indices is 1-based

        #make sure the file indices are from 1 to nSaveTimes
        if file_indices[0][1] != 1 or file_indices[-1][1] != nSaveTimes:
            raise ValueError("File indices are not from 1 to nSaveTimes")

        #Separate into two lists for the current case
        vtk_files = [f[0] for f in file_indices]
        sample_indices = [f[1] for f in file_indices]

        #print("vtk_files[0:5] = ", vtk_files[0:5])
        #print("sample_indices[0:5] = ", sample_indices[0:5])

        #extract the results from the vtk results files of current case
        #cell_centers: numpy array of shape (nCells, 3)
        #result_dict_current_case: list of dictionaries of results for the current case
        #result_array_current_case: list of numpy arrays of shape (nCells, nVariables)
        cell_centers, result_dict_current_case, result_array_current_case = extract_all_simulation_results(vtk_files, flow_variables, output_unit)

        #save the results for the current case
        all_results_dict.append(result_dict_current_case)
        all_results_array.append(result_array_current_case)


    #save the sampled results to a json file
    postprocessed_sampling_results_filename = "postprocessed_sampling_results"
    with open("data/"+postprocessed_sampling_results_filename+".json", 'w') as f:
        # Convert numpy types to Python native types before saving
        results_for_json = convert_numpy_to_list(all_results_dict)
        json.dump(results_for_json, f, indent=4)

    #combine train_indices, val_indices, test_indices and then save to one json file
    split_indices_dict = {"train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices, "sample_indices": case_indices}
    with open("data/split_indices.json", 'w') as f:
        json.dump(split_indices_dict, f, indent=4)

    #print("cell_centers.shape = ", cell_centers.shape)
    #print("len(all_results_list) = ", len(all_results_list))
    #print("all_results_list[0].keys() = ", all_results_list[0].keys())
    #print("len(all_results_array) = ", len(all_results_array))
    #print("all_results_array[0].shape = ", all_results_array[0].shape)
    
    #save the arrays of all results to a numpy file
    np.savez("data/"+postprocessed_sampling_results_filename+".npz", all_results_array=all_results_array, cell_centers=cell_centers, sample_indices=case_indices)

    print("All postprocessed results are saved to data/split_indices.json and data/", postprocessed_sampling_results_filename+".npz")

    #save the training, validation, and test sets to numpy files

    #if not exists, create the directories: data/train, data/val, data/test
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/train"):
        os.makedirs("data/train")
    if not os.path.exists("data/val"):
        os.makedirs("data/val")
    if not os.path.exists("data/test"):
        os.makedirs("data/test")

    #For each set, save the input for branch net, the trunk net, and the output for DeepONet
    #For the branch net, the input is the hydrograph (discharge Q values at nHydrograph_points)
    #For the trunk net, the input is x, y coordinates of the cell centers and time t (in hours)
    #For the output, it is the velocity vector and the water depth (u, v, h)
    #save the training set
    #number of training samples = len(train_indices)*number of cells 
    nCells = cell_centers.shape[0]
    nTrainingSamples = len(train_indices)*nCells*nSaveTimes

    #print("cell_centers.shape = ", cell_centers.shape)
    #print("samples.shape = ", samples.shape)

    #print("train_indices = ", train_indices)    
    #print("val_indices = ", val_indices)
    #print("test_indices = ", test_indices)

    train_input_branch = np.zeros((nTrainingSamples, nHydrograph_points))    #The only one feature for the branch net is the discharge Q hydrograph
    #loop over all simulations in the training set
    for i in range(len(train_indices)):
        #print("train_indices[i] = ", train_indices[i])
        train_index = train_indices[i] #1-based index 

        #read the hydrograph from "hydrographs/hydrograph_xxxx.xys" file
        hydrograph_filename = "hydrographs/hydrograph_"+str(train_index-1).zfill(4)+".xys"   #hydrograph_filename is 0-based index
        hydrograph = np.loadtxt(hydrograph_filename, comments="//", delimiter=" ")

        #loop over all cells in the current simulation
        for j in range(nCells):
            #loop over all save times
            for k in range(nSaveTimes):
                #print("i, j, k = ", i, j, k)                

                #extract the discharge Q values at nHydrograph_points
                train_input_branch[i*nCells*nSaveTimes+j*nSaveTimes+k, :] = hydrograph[:, 1]

    #The input for the trunk net is the x and y coordinates of the cell centers and time t (in hours)
    train_input_trunk = np.zeros((nTrainingSamples, 3))
    #loop over all simulations in the training set
    for i in range(len(train_indices)):
        #loop over all cells in the current simulation
        for j in range(nCells):
            #loop over all save times
            for k in range(nSaveTimes):
                train_input_trunk[i*nCells*nSaveTimes+j*nSaveTimes+k,0] = cell_centers[j, 0]
                train_input_trunk[i*nCells*nSaveTimes+j*nSaveTimes+k,1] = cell_centers[j, 1]
                train_input_trunk[i*nCells*nSaveTimes+j*nSaveTimes+k,2] = save_times[k]

    #The output for the DeepONet is the velocity vector u and v and the water depth h
    train_output = np.zeros((nTrainingSamples, 3))
    #loop over all simulations in the training set
    for i in range(len(train_indices)):
        #loop over all cells in the current simulation
        for j in range(nCells):
            #loop over all save times
            for k in range(nSaveTimes):
                train_output[i*nCells*nSaveTimes+j*nSaveTimes+k,0] = all_results_array[train_indices[i]-1][k][j,0]
                train_output[i*nCells*nSaveTimes+j*nSaveTimes+k,1] = all_results_array[train_indices[i]-1][k][j,1]
                train_output[i*nCells*nSaveTimes+j*nSaveTimes+k,2] = all_results_array[train_indices[i]-1][k][j,2]

    #save the training set
    np.save("data/train/branch_inputs.npy", train_input_branch)
    np.save("data/train/trunk_inputs.npy", train_input_trunk)
    np.save("data/train/outputs.npy", train_output)

    #save the validation set
    nValidationSamples = len(val_indices)*nCells*nSaveTimes
    val_input_branch = np.zeros((nValidationSamples, nHydrograph_points))
    #loop over all simulations in the validation set
    for i in range(len(val_indices)):

        val_index = val_indices[i] #1-based index 

        #read the hydrograph from "hydrographs/hydrograph_xxxx.xys" file
        hydrograph_filename = "hydrographs/hydrograph_"+str(val_index-1).zfill(4)+".xys"   #hydrograph_filename is 0-based index
        hydrograph = np.loadtxt(hydrograph_filename, comments="//", delimiter=" ")

        #loop over all cells in the current simulation
        for j in range(nCells):
            #loop over all save times
            for k in range(nSaveTimes):
                val_input_branch[i*nCells*nSaveTimes+j*nSaveTimes+k, :] = hydrograph[:, 1]

    #The input for the trunk net is the x and y coordinates of the cell centers and time t (in hours)
    val_input_trunk = np.zeros((nValidationSamples, 3))
    #loop over all simulations in the validation set
    for i in range(len(val_indices)):
        #loop over all cells in the current simulation
        for j in range(nCells):
            #loop over all save times
            for k in range(nSaveTimes):
                val_input_trunk[i*nCells*nSaveTimes+j*nSaveTimes+k,0] = cell_centers[j, 0]
                val_input_trunk[i*nCells*nSaveTimes+j*nSaveTimes+k,1] = cell_centers[j, 1]
                val_input_trunk[i*nCells*nSaveTimes+j*nSaveTimes+k,2] = save_times[k]

    #The output for the DeepONet is the velocity vector u and v and the water depth h
    val_output = np.zeros((nValidationSamples, 3))
    #loop over all simulations in the validation set
    for i in range(len(val_indices)):
        #loop over all cells in the current simulation
        for j in range(nCells):
            #loop over all save times
            for k in range(nSaveTimes):
                val_output[i*nCells*nSaveTimes+j*nSaveTimes+k,0] = all_results_array[val_indices[i]-1][k][j,0]
                val_output[i*nCells*nSaveTimes+j*nSaveTimes+k,1] = all_results_array[val_indices[i]-1][k][j,1]
                val_output[i*nCells*nSaveTimes+j*nSaveTimes+k,2] = all_results_array[val_indices[i]-1][k][j,2]

    #save the validation set
    np.save("data/val/branch_inputs.npy", val_input_branch)
    np.save("data/val/trunk_inputs.npy", val_input_trunk)
    np.save("data/val/outputs.npy", val_output)

    #save the test set
    nTestSamples = len(test_indices)*nCells*nSaveTimes
    test_input_branch = np.zeros((nTestSamples, nHydrograph_points))
    #loop over all simulations in the test set
    for i in range(len(test_indices)):
        test_index = test_indices[i] #1-based index 

        #read the hydrograph from "hydrographs/hydrograph_xxxx.xys" file
        hydrograph_filename = "hydrographs/hydrograph_"+str(test_index-1).zfill(4)+".xys"   #hydrograph_filename is 0-based index
        hydrograph = np.loadtxt(hydrograph_filename, comments="//", delimiter=" ")

        #loop over all cells in the current simulation
        for j in range(nCells):
            #loop over all save times
            for k in range(nSaveTimes):
                test_input_branch[i*nCells*nSaveTimes+j*nSaveTimes+k, :] = hydrograph[:, 1]

    #The input for the trunk net is the x and y coordinates of the cell centers and time t (in hours)
    test_input_trunk = np.zeros((nTestSamples, 3))
    #loop over all simulations in the test set
    for i in range(len(test_indices)):
        #loop over all cells in the current simulation
        for j in range(nCells):
            #loop over all save times
            for k in range(nSaveTimes):
                test_input_trunk[i*nCells*nSaveTimes+j*nSaveTimes+k,0] = cell_centers[j, 0]
                test_input_trunk[i*nCells*nSaveTimes+j*nSaveTimes+k,1] = cell_centers[j, 1]
                test_input_trunk[i*nCells*nSaveTimes+j*nSaveTimes+k,2] = save_times[k]

    #The output for the DeepONet is the velocity vector u and v and the water depth h
    test_output = np.zeros((nTestSamples, 3))
    #loop over all simulations in the test set
    for i in range(len(test_indices)):
        #loop over all cells in the current simulation
        for j in range(nCells):
            #loop over all save times
            for k in range(nSaveTimes):
                test_output[i*nCells*nSaveTimes+j*nSaveTimes+k,0] = all_results_array[test_indices[i]-1][k][j,0]
                test_output[i*nCells*nSaveTimes+j*nSaveTimes+k,1] = all_results_array[test_indices[i]-1][k][j,1]
                test_output[i*nCells*nSaveTimes+j*nSaveTimes+k,2] = all_results_array[test_indices[i]-1][k][j,2]

    #save the test set
    np.save("data/test/branch_inputs.npy", test_input_branch)
    np.save("data/test/trunk_inputs.npy", test_input_trunk)
    np.save("data/test/outputs.npy", test_output)      

    print("All postprocessed results are saved to data/train, data/val, and data/test")


def plot_profile_results_history(case_index, save_times, nSaveTimes, variable_name, output_unit):
    """
    Plot some results for visual checking

    Args:
        case_index (int): The index of the case to plot (1-based index)
        save_times (list): The save times
        nSaveTimes (int): The number of save times
        variable_name (str): The name of the variable to plot
        output_unit (str): The output unit
    """

    print("Plotting the results for case index = ", case_index)

    #load the postprocessed results
    postprocessed_results = np.load("data/postprocessed_sampling_results.npz")

    #all_results_array[nCases][nSaveTimes][nCells,nVariables]
    all_results_array = postprocessed_results["all_results_array"]
    
    cell_centers = postprocessed_results["cell_centers"]
    

    #get the results for the current case: results_array[nSaveTimes][nCells,nVariables]
    results_array = all_results_array[case_index-1]

    #loop over all save times
    for k in range(nSaveTimes):
        #get the results for the current save time: results[nCells,nVariables]
        results = results_array[k]

        #compute the water surface elevation
        wse = cell_centers[:, 2] + results[:, 2]

        #create a single plot of the selected samples
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        #plot the water surface elevation
        axs.plot(cell_centers[:, 0], wse, 'k-', label=f'Case {case_index}')

        #plot the bed elevation
        axs.plot(cell_centers[:, 0], cell_centers[:, 2], color='k', label='Bed Elevation')

        #add the legend
        axs.legend()

        #add the x-axis label
        axs.set_xlabel('x (m)', fontsize=14)

        #add the y-axis label
        axs.set_ylabel('Elevation (m)', fontsize=14)

        #set y-axis limits
        axs.set_ylim(0, 1.0)

        #set the fontsize of the tick labels
        axs.tick_params(axis='both', labelsize=12)  

        #set the fontsize of the title
        axs.set_title('Case '+str(case_index)+', Time = '+str(save_times[k])+' hours', fontsize=14)

        #save the plot
        plt.savefig("example_wse_profile_case_"+str(case_index).zfill(4)+"_time_"+str(int(save_times[k])).zfill(4)+".png", dpi=300, bbox_inches='tight')

        #show the plot
        #plt.show()

        plt.close()



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
    
    # Define the flow variables to be postprocessed based on the output unit
    if output_unit == "SI":
        flow_variables = ["Velocity_m_p_s", "Water_Depth_m"]
    elif output_unit == "EN":     #It seems that the name of the variables are slightly different for Windows and Linux. So depending on the system for which the code is run, the name of the variables are slightly different. Check the VTK files to see the actual names of the variables.
        if system_name == "Windows":
            flow_variables = ["Velocity_ft_p_s", "Water_Depth_ft"]
        elif system_name == "Linux":
            flow_variables = ["Velocity_ft_p_s", "Water_Depth_ft"]
    else:
        raise ValueError("Unsupported output unit: " + output_unit)


    #for unsteady simulations, the save times (in hours); This needs to be modified based on the actual simulation results
    nHydrograph_points = postprocessing_specs['nHydrograph_points']
    save_times_start = postprocessing_specs['save_times']['start']
    save_times_end = postprocessing_specs['save_times']['end']
    save_times_step = postprocessing_specs['save_times']['step']
    save_times = np.arange(save_times_start, save_times_end, save_times_step)   
    nSaveTimes = len(save_times)

    #postprocess the results according to the postprocessing specifications
    print("Postprocessing the results ...")
    #postprocess_results(nSamples, nSaveTimes, nHydrograph_points, flow_variables, output_unit, training_fraction, validation_fraction, test_fraction)

    #plot some results for visual checking
    print("Plotting the results ...")
    #case_indices = [77, 372, 821, 522]    #1-based index
    case_indices = [77]
    for case_index in case_indices:
        plot_profile_results_history(case_index, save_times, nSaveTimes, "wse", output_unit)

    #record the end time
    end_time = time.time()
    
    #print the total time taken in hours
    print("Total time taken: ", (end_time - start_time)/3600, " hours")


    print("All done!")


