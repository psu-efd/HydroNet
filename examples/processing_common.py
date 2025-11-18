#some common functions for processing data

from vtk.util import numpy_support as VN
import vtk
import numpy as np

from sklearn.model_selection import train_test_split

from pyHMT2D.Misc.SRH_to_PINN_points import srh_to_pinn_points

import json
import h5py

# Import utility function for JSON serialization
import sys
import os
# Add parent directory to path to import from HydroNet (one level up from examples/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from HydroNet.utils import convert_to_json_serializable

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
        list: List of cell bed slopes in the x-direction
        list: List of cell bed slopes in the y-direction
    """
    
    # Get cell centers
    cell_ids = []
    
    vtk_data = vtk_reader.GetOutput()
    n_cells = vtk_data.GetNumberOfCells()

    cell_centers = np.zeros((n_cells, 3))

    cell_Sx = np.zeros(n_cells)
    cell_Sy = np.zeros(n_cells)    
        
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

        #compute the bed slope based on the z-coordinates of the nodes. We use the Gauss theorem to compute the gradient of the bed elevation.
        Sx = 0.0
        Sy = 0.0
        area = 0.0

        #loop over all edges of the cell (n_points is equal to the number of edges)
        for j in range(n_points):
            j_next = (j + 1) % n_points
            x0, y0, z0 = points.GetPoint(j)       #coordinates of the first point of the edge
            x1, y1, z1 = points.GetPoint(j_next)  #coordinates of the second point of the edge

            # Edge vector
            dx = x1 - x0
            dy = y1 - y0
            length = np.hypot(dx, dy)
            nx = dy / length
            ny = -dx / length

            zb_avg = 0.5 * (z0 + z1)

            Sx += zb_avg * nx * length
            Sy += zb_avg * ny * length

            area += (x0 * y1 - x1 * y0)

        if area < 0: #points in the cell are clockwise
            bCCW = False
        else:
            bCCW = True

        area = 0.5 * abs(area)

        cell_ids.append(i)        
        cell_centers[i,:] = center

        if bCCW:
            cell_Sx[i] = -Sx / area
            cell_Sy[i] = -Sy / area
        else:     #if points are counterclockwise, the bed slope's sign is reversed
            cell_Sx[i] = Sx / area
            cell_Sy[i] = Sy / area
        
    
    return cell_ids, cell_centers, cell_Sx, cell_Sy

def sample_vtk_in_domain(vtk_file, flow_variables, output_unit):
    """Sample VTK data within a domain
    
    Args:
        vtk_file: VTK file path
        flow_variables: List of flow variables to sample in the order of (h, u, v), also Bed_Elev and ManningN. Need to add bed slope
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
    cell_ids, cell_centers, cell_Sx, cell_Sy = get_cells_in_domain(reader)

    #print("flow_variables = ", flow_variables)
    #print("length of flow_variables = ", len(flow_variables))

    result_dict = {}
    result_array = np.zeros((len(cell_ids), len(flow_variables)+1))

    #make sure "Water_Depth" is in the first position of the flow_variables list
    if "Water_Depth" not in flow_variables[0]:
        print("Water_Depth is not in the first position of the flow_variables list")
        print("flow_variables = ", flow_variables)
        raise ValueError("Water_Depth is not in the first position of the flow_variables list")
    
    #check if "Bed_Elev", "ManningN", "Sx", "Sy" is in the flow_variables list. If not, report an error
    if not any("Bed_Elev" in var for var in flow_variables):
        raise ValueError("Bed_Elev is not in the flow_variables list")
    if not any("ManningN" in var for var in flow_variables):
        raise ValueError("ManningN is not in the flow_variables list")
    if "Sx" not in flow_variables:
        raise ValueError("Sx is not in the flow_variables list")
    if "Sy" not in flow_variables:
        raise ValueError("Sy is not in the flow_variables list")

    # Sample data for these cells   
    index_var = 0
    for var in flow_variables:       
        #print("sampling ", var)

        #print("var = ", var)
        #print("index_var = ", index_var)
        #print("content of result_dict = ", result_dict.keys())

        # Sx and Sy are computed from the z-coordinates of the nodes
        if var == "Sx":
            result_dict["Sx"] = cell_Sx
            result_array[:, index_var] = cell_Sx
            index_var += 1
            continue
        elif var == "Sy":
            result_dict["Sy"] = cell_Sy
            result_array[:, index_var] = cell_Sy
            index_var += 1
            continue
        else:
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
        flow_variables (list): List of flow variables to sample in the order of (h, u, v), Bed_Elev, ManningN, Sx, Sy
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

def postprocess_results_for_DeepONet(nSamples, sampled_parameters, flow_variables, output_unit, postprocessing_specs):
    """Postprocess the results of the simulations

    The split is done based on the successful simulations. First, the split (train, val, test) is based on cases, not samples (cell center data). Then, the train and validation cases are combined and then split based on shuffled samples (cell center data).

    Args:
        nSamples (int): Number of samples
        sampled_parameters (numpy array): Sampled parameters with shape (nSamples, n_features)
        flow_variables (list): List of flow variables to sample in the order of (h, u, v), Bed_Elev, ManningN, Sx, Sy
        output_unit (str): Output unit, SI or EN
        postprocessing_specs (dict): Postprocessing specifications
    """

    # Unpack the postprocessing specifications
    nCells = postprocessing_specs['nCells']
    training_fraction = postprocessing_specs['split_fractions']['training']
    validation_fraction = postprocessing_specs['split_fractions']['validation']
    test_fraction = postprocessing_specs['split_fractions']['test']

    branch_inputs_normalization_method = postprocessing_specs['DeepONet_normalization_specs']['branch_inputs']
    trunk_inputs_normalization_method = postprocessing_specs['DeepONet_normalization_specs']['trunk_inputs']
    outputs_normalization_method = postprocessing_specs['DeepONet_normalization_specs']['outputs']
    static_fields_normalization_method = postprocessing_specs['DeepONet_normalization_specs']['static_fields']

    # Create output directories
    os.makedirs('data/DeepONet/train', exist_ok=True)
    os.makedirs('data/DeepONet/val', exist_ok=True)
    os.makedirs('data/DeepONet/test', exist_ok=True)
    
    # Generate indices for splitting
    indices_allSamples = list(range(1, nSamples+1))  #1-based index
    train_indices, val_indices, test_indices = split_indices(indices_allSamples, training_fraction, validation_fraction, test_fraction)
    
    # Save split indices for reference
    split_indices_dict = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }
    with open('data/DeepONet/split_indices.json', 'w') as f:
        json.dump(split_indices_dict, f, indent=4)
    
    # Process each split
    for split_name, indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
        print("Processing split = ", split_name)

        # Create the temporary HDF5 file for this split (to be used for shuffling and normalization)
        h5_file = h5py.File(f'data/DeepONet/{split_name}/data_temp.h5', 'w')
        
        # Calculate total number of data points for this split (= number of cases * number of cells per case)
        total_data_points = len(indices) * nCells
        
        # Initialize datasets with chunking
        chunk_size = min(10000, total_data_points)  # Adjust chunk size as needed
        n_features = sampled_parameters.shape[1]  # Number of branch input features
        n_coords = 2  # x, y coordinates for trunk inputs
        n_outputs = 3  # h, u, v for outputs
        n_pde_data = 4  # Zb, ManningN, Sx, Sy for pde data (at cell centers for now, which are also used for computing PDE loss)
        
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

        pde_data = h5_file.create_dataset(
            'pde_data',
            shape=(total_data_points, n_pde_data),
            chunks=(chunk_size, n_pde_data),
            dtype='float32'
        )
        
        # Process data 
        current_data_point = 0
        for index, case_idx in enumerate(indices):
            print("    Processing case = ", case_idx, " (", index+1, " out of ", len(indices), ")")

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
            case_pde_data = results_array[:, 3:]  # Zb, ManningN, Sx, Sy
            
            # Write this case's data to the HDF5 file
            branch_inputs[current_data_point:current_data_point + n_cells_case] = case_branch_inputs
            trunk_inputs[current_data_point:current_data_point + n_cells_case] = case_trunk_inputs
            outputs[current_data_point:current_data_point + n_cells_case] = case_outputs
            pde_data[current_data_point:current_data_point + n_cells_case] = case_pde_data

            current_data_point += n_cells_case
        
        # Add metadata
        h5_file.attrs['n_samples'] = total_data_points
        h5_file.attrs['n_features'] = n_features
        h5_file.attrs['n_coords'] = n_coords
        h5_file.attrs['n_outputs'] = n_outputs
        h5_file.attrs['output_unit'] = output_unit
        
        # Close the temporary HDF5 file (data within are not normalized yet)
        h5_file.close()
        
        print(f"Saved {total_data_points} data points to data/DeepONet/{split_name}")

    #read the train and validation data again and combine/shuffle them; also read in the test data so we can compute the mean and std of the whole dataset
    train_data = h5py.File('data/DeepONet/train/data_temp.h5', 'r')
    val_data = h5py.File('data/DeepONet/val/data_temp.h5', 'r')
    test_data = h5py.File('data/DeepONet/test/data_temp.h5', 'r')

    #combine the whole dataset (train, val, test) for computing the statistics
    train_val_test_data_branch_inputs = np.concatenate((train_data['branch_inputs'], val_data['branch_inputs'], test_data['branch_inputs']), axis=0).astype(np.float64)
    train_val_test_data_trunk_inputs = np.concatenate((train_data['trunk_inputs'], val_data['trunk_inputs'], test_data['trunk_inputs']), axis=0).astype(np.float64)
    train_val_test_data_outputs = np.concatenate((train_data['outputs'], val_data['outputs'], test_data['outputs']), axis=0).astype(np.float64)    

    #compute the min, max, mean and std of the whole dataset (train, val, test)
    #For this case, the branch inputs are the sampled parameters (discharges at three inflow boundaries), which are not normalized yet.
    train_val_test_min_branch_inputs = np.min(train_val_test_data_branch_inputs, axis=0)
    train_val_test_max_branch_inputs = np.max(train_val_test_data_branch_inputs, axis=0)
    train_val_test_mean_branch_inputs = np.mean(train_val_test_data_branch_inputs, axis=0)
    train_val_test_std_branch_inputs = np.std(train_val_test_data_branch_inputs, axis=0)

    train_val_test_min_trunk_inputs = np.min(train_val_test_data_trunk_inputs, axis=0)
    train_val_test_max_trunk_inputs = np.max(train_val_test_data_trunk_inputs, axis=0)
    train_val_test_mean_trunk_inputs = np.mean(train_val_test_data_trunk_inputs, axis=0)
    train_val_test_std_trunk_inputs = np.std(train_val_test_data_trunk_inputs, axis=0)

    train_val_test_min_outputs = np.min(train_val_test_data_outputs, axis=0)
    train_val_test_max_outputs = np.max(train_val_test_data_outputs, axis=0)
    train_val_test_mean_outputs = np.mean(train_val_test_data_outputs, axis=0)
    train_val_test_std_outputs = np.std(train_val_test_data_outputs, axis=0)

    #create a dictionary to store the min, max, mean and std of the branch inputs, trunk inputs, and outputs for the DeepONet
    all_DeepONet_branch_trunk_outputs_stats = {
        'normalization_method': {
            'branch_inputs': branch_inputs_normalization_method,
            'trunk_inputs': trunk_inputs_normalization_method,
            'outputs': outputs_normalization_method,
            'static_fields': static_fields_normalization_method
        },
        'branch_inputs': {
            'min': train_val_test_min_branch_inputs,
            'max': train_val_test_max_branch_inputs,
            'mean': train_val_test_mean_branch_inputs,
            'std': train_val_test_std_branch_inputs
        },
        'trunk_inputs': {
            'min': train_val_test_min_trunk_inputs,
            'max': train_val_test_max_trunk_inputs,
            'mean': train_val_test_mean_trunk_inputs,
            'std': train_val_test_std_trunk_inputs
        },
        'outputs': {
            'min': train_val_test_min_outputs,
            'max': train_val_test_max_outputs,
            'mean': train_val_test_mean_outputs,
            'std': train_val_test_std_outputs
        }
    }

    #compute the min, max, mean and std of the x, y
    x_min = np.min(train_val_test_data_trunk_inputs[:, 0])
    x_max = np.max(train_val_test_data_trunk_inputs[:, 0])
    x_mean = np.mean(train_val_test_data_trunk_inputs[:, 0])
    x_std = np.std(train_val_test_data_trunk_inputs[:, 0])

    y_min = np.min(train_val_test_data_trunk_inputs[:, 1])  
    y_max = np.max(train_val_test_data_trunk_inputs[:, 1])
    y_mean = np.mean(train_val_test_data_trunk_inputs[:, 1])
    y_std = np.std(train_val_test_data_trunk_inputs[:, 1])

    #this case is for steady case, thus time t is not relevant. But for completeness, we save it in all_points_stats
    t_min = 0.0
    t_max = 0.0
    t_mean = 0.0
    t_std = 0.0

    all_points_stats = {
        'x_min': float(x_min), 'x_max': float(x_max), 'x_mean': float(x_mean), 'x_std': float(x_std),
        'y_min': float(y_min), 'y_max': float(y_max), 'y_mean': float(y_mean), 'y_std': float(y_std),
        't_min': float(t_min), 't_max': float(t_max), 't_mean': float(t_mean), 't_std': float(t_std),
    }

    #compute the min, max, mean and std of h, u, v, and Umag
    h_min = np.min(train_val_test_data_outputs[:, 0])
    h_max = np.max(train_val_test_data_outputs[:, 0])
    h_mean = np.mean(train_val_test_data_outputs[:, 0])
    h_std = np.std(train_val_test_data_outputs[:, 0])

    u_min = np.min(train_val_test_data_outputs[:, 1])
    u_max = np.max(train_val_test_data_outputs[:, 1])
    u_mean = np.mean(train_val_test_data_outputs[:, 1])
    u_std = np.std(train_val_test_data_outputs[:, 1])

    v_min = np.min(train_val_test_data_outputs[:, 2])
    v_max = np.max(train_val_test_data_outputs[:, 2])
    v_mean = np.mean(train_val_test_data_outputs[:, 2])
    v_std = np.std(train_val_test_data_outputs[:, 2])

    #velocity magnitude
    Umag = np.sqrt(train_val_test_data_outputs[:, 1]**2 + train_val_test_data_outputs[:, 2]**2)
    Umag_min = np.min(Umag)
    Umag_max = np.max(Umag)
    Umag_mean = np.mean(Umag)
    Umag_std = np.std(Umag)

    all_data_stats = {
        'h_min': float(h_min), 'h_max': float(h_max), 'h_mean': float(h_mean), 'h_std': float(h_std),
        'u_min': float(u_min), 'u_max': float(u_max), 'u_mean': float(u_mean), 'u_std': float(u_std),
        'v_min': float(v_min), 'v_max': float(v_max), 'v_mean': float(v_mean), 'v_std': float(v_std),
        'Umag_min': float(Umag_min), 'Umag_max': float(Umag_max), 'Umag_mean': float(Umag_mean), 'Umag_std': float(Umag_std)
    }

    #combine the train and validation data for normalization and shuffling
    train_val_data_branch_inputs = np.concatenate((train_data['branch_inputs'], val_data['branch_inputs']), axis=0)
    train_val_data_trunk_inputs = np.concatenate((train_data['trunk_inputs'], val_data['trunk_inputs']), axis=0)
    train_val_data_outputs = np.concatenate((train_data['outputs'], val_data['outputs']), axis=0)

    #get the test data for normalization (no shuffling for test data because we need each simulation case to be together)
    test_data_branch_inputs = test_data['branch_inputs']
    test_data_trunk_inputs = test_data['trunk_inputs']
    test_data_outputs = test_data['outputs']

    #normalize the data (train, validation and test)
    if branch_inputs_normalization_method == 'z-score':
        train_val_data_branch_inputs = (train_val_data_branch_inputs - train_val_test_mean_branch_inputs) / train_val_test_std_branch_inputs
    elif branch_inputs_normalization_method == 'min-max':
        train_val_data_branch_inputs = (train_val_data_branch_inputs - train_val_test_min_branch_inputs) / (train_val_test_max_branch_inputs - train_val_test_min_branch_inputs)
    else:
        raise ValueError(f"Invalid branch inputs normalization method: {branch_inputs_normalization_method}")

    if trunk_inputs_normalization_method == 'z-score':
        train_val_data_trunk_inputs = (train_val_data_trunk_inputs - train_val_test_mean_trunk_inputs) / train_val_test_std_trunk_inputs
    elif trunk_inputs_normalization_method == 'min-max':
        train_val_data_trunk_inputs = (train_val_data_trunk_inputs - train_val_test_min_trunk_inputs) / (train_val_test_max_trunk_inputs - train_val_test_min_trunk_inputs)
    else:
        raise ValueError(f"Invalid trunk inputs normalization method: {trunk_inputs_normalization_method}")

    if outputs_normalization_method == 'z-score':
        train_val_data_outputs = (train_val_data_outputs - train_val_test_mean_outputs) / train_val_test_std_outputs
    elif outputs_normalization_method == 'min-max':
        train_val_data_outputs = (train_val_data_outputs - train_val_test_min_outputs) / (train_val_test_max_outputs - train_val_test_min_outputs)
    else:
        raise ValueError(f"Invalid outputs normalization method: {outputs_normalization_method}")

    #normalize the test data
    if branch_inputs_normalization_method == 'z-score':
        test_data_branch_inputs = (test_data_branch_inputs - train_val_test_mean_branch_inputs) / train_val_test_std_branch_inputs
    elif branch_inputs_normalization_method == 'min-max':
        test_data_branch_inputs = (test_data_branch_inputs - train_val_test_min_branch_inputs) / (train_val_test_max_branch_inputs - train_val_test_min_branch_inputs)
    else:
        raise ValueError(f"Invalid test data branch inputs normalization method: {branch_inputs_normalization_method}")

    if trunk_inputs_normalization_method == 'z-score':
        test_data_trunk_inputs = (test_data_trunk_inputs - train_val_test_mean_trunk_inputs) / train_val_test_std_trunk_inputs
    elif trunk_inputs_normalization_method == 'min-max':
        test_data_trunk_inputs = (test_data_trunk_inputs - train_val_test_min_trunk_inputs) / (train_val_test_max_trunk_inputs - train_val_test_min_trunk_inputs)
    else:
        raise ValueError(f"Invalid test data trunk inputs normalization method: {trunk_inputs_normalization_method}")

    if outputs_normalization_method == 'z-score':
        test_data_outputs = (test_data_outputs - train_val_test_mean_outputs) / train_val_test_std_outputs
    elif outputs_normalization_method == 'min-max':
        test_data_outputs = (test_data_outputs - train_val_test_min_outputs) / (train_val_test_max_outputs - train_val_test_min_outputs)
    else:
        raise ValueError(f"Invalid test data outputs normalization method: {outputs_normalization_method}")

    print("Combined data shapes:")
    print(f"Branch inputs: {train_val_data_branch_inputs.shape}")
    print(f"Trunk inputs: {train_val_data_trunk_inputs.shape}")
    print(f"Outputs: {train_val_data_outputs.shape}")

    #get the total number of train and validation data points
    n_train_val_data_points = train_val_data_branch_inputs.shape[0]

    print("n_train_val_data_points = ", n_train_val_data_points)

    #split the train and validation data into training and validation sets based on training_fraction and validation_fraction only
    # Calculate the new training fraction (relative to train+val)
    training_fraction_new = training_fraction/(training_fraction + validation_fraction)

    # Calculate split point
    n_train = int(n_train_val_data_points * training_fraction_new)

    # Randomly split into train and val
    train_indices = np.random.choice(n_train_val_data_points, size=n_train, replace=False)
    val_indices = np.setdiff1d(np.arange(n_train_val_data_points), train_indices)

    print("Split sizes:")
    print(f"Train: {len(train_indices)}")
    print(f"Val: {len(val_indices)}")

    #save the train data to the train HDF5 file
    train_data = h5py.File('data/DeepONet/train/data.h5', 'w')
    train_data['branch_inputs'] = train_val_data_branch_inputs[train_indices]
    train_data['trunk_inputs'] = train_val_data_trunk_inputs[train_indices]
    train_data['outputs'] = train_val_data_outputs[train_indices]

    #save the validation data to the validation HDF5 file
    val_data = h5py.File('data/DeepONet/val/data.h5', 'w')
    val_data['branch_inputs'] = train_val_data_branch_inputs[val_indices]
    val_data['trunk_inputs'] = train_val_data_trunk_inputs[val_indices]
    val_data['outputs'] = train_val_data_outputs[val_indices]

    #save the test data to the test HDF5 file
    test_data = h5py.File('data/DeepONet/test/data.h5', 'w')
    test_data['branch_inputs'] = test_data_branch_inputs
    test_data['trunk_inputs'] = test_data_trunk_inputs
    test_data['outputs'] = test_data_outputs
    
    #combine the stats dictionaries as sub-dictionaries and save them to a JSON file
    all_stats = {
        'all_DeepONet_branch_trunk_outputs_stats': all_DeepONet_branch_trunk_outputs_stats,
        'all_points_stats': all_points_stats, 
        'all_data_stats': all_data_stats
        }

    # Convert numpy arrays to lists for JSON serialization
    all_stats_serializable = convert_to_json_serializable(all_stats)

    #save DeepONet-relatred stats to a JSON file
    with open('data/DeepONet/all_DeepONet_stats.json', 'w') as f:
        json.dump(all_stats_serializable, f, indent=4)
    
    #close the train and validation HDF5 files
    train_data.close()
    val_data.close()
    test_data.close()

    #remove data_temp.h5 file
    os.remove('data/DeepONet/train/data_temp.h5')
    os.remove('data/DeepONet/val/data_temp.h5')
    os.remove('data/DeepONet/test/data_temp.h5')

    #verify the data
    #verify_data()
    
    print("Data postprocessing completed successfully!")

def verify_data_for_DeepONet():
    """
    Verify the data by reading the data from the HDF5 files and computing the mean and standard deviation of the data

    We assume the data is normalized based on the specified normalization method. 

    The data files are in the data/DeepONet/train, data/DeepONet/val and data/DeepONet/test directories.
    The all_DeepONet_stats.json file is in the data/DeepONet directory.

    Args:
        None
    """

    #read the train, validation and test data again and verify the mean and standard deviation 
    print("\nVerifying the mean and standard deviation of the data ...")
    train_data = h5py.File('data/DeepONet/train/data.h5', 'r')
    val_data = h5py.File('data/DeepONet/val/data.h5', 'r')
    test_data = h5py.File('data/DeepONet/test/data.h5', 'r')    

    #print the content of train_data
    print("content of train_data = ", train_data.keys())

    #compute the mean and standard deviation of the data
    #branch inputs (as float64 is required for the computation of statistics because the dataset is very large; otherwise, the computation will be incorrect due to overflow)
    train_mean_branch_inputs = np.mean(train_data['branch_inputs'].astype(np.float64), axis=0)
    train_std_branch_inputs = np.std(train_data['branch_inputs'].astype(np.float64), axis=0)
    val_mean_branch_inputs = np.mean(val_data['branch_inputs'].astype(np.float64), axis=0)
    val_std_branch_inputs = np.std(val_data['branch_inputs'].astype(np.float64), axis=0)
    test_mean_branch_inputs = np.mean(test_data['branch_inputs'].astype(np.float64), axis=0)
    test_std_branch_inputs = np.std(test_data['branch_inputs'].astype(np.float64), axis=0)

    print("train_mean_branch_inputs = ", train_mean_branch_inputs)
    print("train_std_branch_inputs = ", train_std_branch_inputs)
    print("val_mean_branch_inputs = ", val_mean_branch_inputs)
    print("val_std_branch_inputs = ", val_std_branch_inputs)
    print("test_mean_branch_inputs = ", test_mean_branch_inputs)
    print("test_std_branch_inputs = ", test_std_branch_inputs)

    #trunk inputs
    train_mean_trunk_inputs = np.mean(train_data['trunk_inputs'].astype(np.float64), axis=0)
    train_std_trunk_inputs = np.std(train_data['trunk_inputs'].astype(np.float64), axis=0)
    val_mean_trunk_inputs = np.mean(val_data['trunk_inputs'].astype(np.float64), axis=0)
    val_std_trunk_inputs = np.std(val_data['trunk_inputs'].astype(np.float64), axis=0)
    test_mean_trunk_inputs = np.mean(test_data['trunk_inputs'].astype(np.float64), axis=0)
    test_std_trunk_inputs = np.std(test_data['trunk_inputs'].astype(np.float64), axis=0)

    print("train_mean_trunk_inputs = ", train_mean_trunk_inputs)
    print("train_std_trunk_inputs = ", train_std_trunk_inputs)
    print("val_mean_trunk_inputs = ", val_mean_trunk_inputs)
    print("val_std_trunk_inputs = ", val_std_trunk_inputs)
    print("test_mean_trunk_inputs = ", test_mean_trunk_inputs)
    print("test_std_trunk_inputs = ", test_std_trunk_inputs)

    #outputs
    train_mean_outputs = np.mean(train_data['outputs'].astype(np.float64), axis=0)
    train_std_outputs = np.std(train_data['outputs'].astype(np.float64), axis=0)
    val_mean_outputs = np.mean(val_data['outputs'].astype(np.float64), axis=0)
    val_std_outputs = np.std(val_data['outputs'].astype(np.float64), axis=0)
    test_mean_outputs = np.mean(test_data['outputs'].astype(np.float64), axis=0)
    test_std_outputs = np.std(test_data['outputs'].astype(np.float64), axis=0)

    print("train_mean_outputs = ", train_mean_outputs)
    print("train_std_outputs = ", train_std_outputs)
    print("val_mean_outputs = ", val_mean_outputs)
    print("val_std_outputs = ", val_std_outputs)
    print("test_mean_outputs = ", test_mean_outputs)
    print("test_std_outputs = ", test_std_outputs)    

    #stats of the data    
    all_stats = json.load(open('data/DeepONet/all_DeepONet_stats.json', 'r'))
    all_points_stats = all_stats['all_points_stats']
    all_data_stats = all_stats['all_data_stats']
    print("all_points_stats = ", all_points_stats)    
    print("all_data_stats = ", all_data_stats)

    #close the train, validation and test data files
    train_data.close()
    val_data.close()
    test_data.close()

    print("Data verification completed successfully!")

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

def convert_mesh_points_for_PINN(postprocessing_specs):
    """
    Convert mesh points in a json file (mesh_points.json, derived from SRH-2D mesh file) to PINNDataset format.
    
    Parameters
    ----------
        postprocessing_specs (dict): The postprocessing specifications dictionary        

    Returns
    -------
    dict
        Dictionary containing shapes of generated arrays
    """

    json_file = "mesh_points.json"
    output_dir = "data/PINN"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the JSON file
    print(f"Reading points from {json_file}")
    try:
        with open(json_file, 'r') as f:
            points_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON file: {e}")
    
    # Extract training points
    if 'training_points' not in points_data:
        raise ValueError("JSON file must contain 'training_points' key")
    
    training_points = points_data['training_points']
    
    # Process PDE points (equation points, i.e., points where the PDE is enforced. These points should be within the domain of the problem.)
    if 'equation_points' not in training_points:
        raise ValueError("Training points must contain 'equation_points' key")
    
    equation_points = training_points['equation_points']
    spatial_points = []  # temporary storage for spatial x and y coordinates
    zb_points = []  # temporary storage for zb points
    Sx_points = []  # temporary storage for Sx points
    Sy_points = []  # temporary storage for Sy points
    ManningN_points = []  # temporary storage for ManningN points
    

    # Convert equation points dictionary to array (spatial coordinates only)
    for point_id, point_data in equation_points.items():
        spatial_points.append([
            point_data['x'],
            point_data['y']
        ])
        zb_points.append(point_data['z'])
        Sx_points.append(point_data['Sx'])
        Sy_points.append(point_data['Sy'])
        ManningN_points.append(point_data['ManningN'])

    spatial_points = np.array(spatial_points, dtype=np.float32)
    zb_points = np.array(zb_points, dtype=np.float32)
    Sx_points = np.array(Sx_points, dtype=np.float32)
    Sy_points = np.array(Sy_points, dtype=np.float32)
    ManningN_points = np.array(ManningN_points, dtype=np.float32)

    # assemble the pde points
    n_spatial_points = len(spatial_points)
    pde_points = np.zeros((n_spatial_points, 2), dtype=np.float32)   #rows of (x, y), no time for steady case
    pde_data = np.zeros((n_spatial_points, 4), dtype=np.float32)   #rows of (zb, Sx, Sy, ManningN)
    
    # Fill in the expanded points
    pde_points[:, :2] = spatial_points  # Copy x, y coordinates
    pde_data[:, 0] = zb_points
    pde_data[:, 1] = Sx_points
    pde_data[:, 2] = Sy_points
    pde_data[:, 3] = ManningN_points

    #we only want the statistics of the zb points, bed slope and ManningN at pde points
    zb_min = np.min(zb_points)
    zb_max = np.max(zb_points)
    zb_mean = np.mean(zb_points)
    zb_std = np.std(zb_points)
    Sx_min = np.min(Sx_points)
    Sx_max = np.max(Sx_points)
    Sx_mean = np.mean(Sx_points)
    Sx_std = np.std(Sx_points)
    Sy_min = np.min(Sy_points)
    Sy_max = np.max(Sy_points)
    Sy_mean = np.mean(Sy_points)
    Sy_std = np.std(Sy_points)
    ManningN_min = np.min(ManningN_points)
    ManningN_max = np.max(ManningN_points)
    ManningN_mean = np.mean(ManningN_points)
    ManningN_std = np.std(ManningN_points)
    

    # Process boundary points
    if 'boundary_points' not in training_points:
        raise ValueError("Training points must contain 'boundary_points' key")

    # First collect all boundary spatial points and their info
    all_boundary_spatial_points = []
    all_boundary_points_z = []
    all_boundary_normals = []
    all_boundary_represented_lengths = []
    all_boundary_ids = []
    all_boundary_ManningN = []

    # Loop over all boundaries and collect spatial points, normals and IDs
    for boundary_name, boundary_data in training_points['boundary_points'].items():
        print(f"Processing boundary: {boundary_name}")

        # boundary_name should be something like "boundary_1", "boundary_2", etc.
        # extract the number from the boundary_name
        boundary_name_parts = boundary_name.split('_')
        if len(boundary_name_parts) != 2 or boundary_name_parts[0] != 'boundary':
            raise ValueError(f"Invalid boundary name: {boundary_name}. It should be something like 'boundary_1', 'boundary_2', etc.")
        
        boundary_id = int(boundary_name_parts[-1])

        print(f"boundary_id: {boundary_id}")
        
        # Get spatial coordinates and normals for this boundary
        for point_id, point_data in boundary_data.items():
            all_boundary_spatial_points.append([
                point_data['x'],
                point_data['y']
            ])
            all_boundary_points_z.append(point_data['z'])
            all_boundary_normals.append([
                point_data['normal_x'],
                point_data['normal_y']
            ])
            all_boundary_represented_lengths.append(point_data['represented_length'])
            all_boundary_ids.append(boundary_id)
            all_boundary_ManningN.append(point_data['ManningN'])


    # Convert to numpy arrays
    all_boundary_spatial_points = np.array(all_boundary_spatial_points, dtype=np.float32)
    all_boundary_points_z = np.array(all_boundary_points_z, dtype=np.float32)
    all_boundary_normals = np.array(all_boundary_normals, dtype=np.float32)
    all_boundary_represented_lengths = np.array(all_boundary_represented_lengths, dtype=np.float32)
    all_boundary_ids = np.array(all_boundary_ids, dtype=np.float32)   #ID is integer, but we need to convert to float32 for compatibility
    all_boundary_ManningN = np.array(all_boundary_ManningN, dtype=np.float32)

    # Get total number of boundary points
    n_boundary_spatial = len(all_boundary_spatial_points)

    # Create arrays for all boundary points and info
    boundary_points = np.zeros((n_boundary_spatial, 2), dtype=np.float32)   #rows of (x, y)
    boundary_info = np.zeros((n_boundary_spatial, 6), dtype=np.float32)   #rows of (ID, z, nx, ny, represented_length, ManningN)

    # Copy spatial coordinates
    boundary_points[:, :2] = all_boundary_spatial_points
        
    # Copy boundary info (ID and normals)
    boundary_info[:, 0] = all_boundary_ids
    boundary_info[:, 1] = all_boundary_points_z
    boundary_info[:, 2:4] = all_boundary_normals
    boundary_info[:, 4] = all_boundary_represented_lengths
    boundary_info[:, 5] = all_boundary_ManningN

    #compute the statistics of all the points (pde_points and boundary_points)
    #min, max, mean, std, median, etc.
    #combine all points together and compute the statistics
    all_points = np.concatenate((pde_points, boundary_points), axis=0)
    print(f"All points shape: {all_points.shape}")
    print(f"All points: {all_points}")
    x_min = np.min(all_points[:, 0])
    x_max = np.max(all_points[:, 0])
    x_mean = np.mean(all_points[:, 0])
    x_std = np.std(all_points[:, 0])
    y_min = np.min(all_points[:, 1])
    y_max = np.max(all_points[:, 1])
    y_mean = np.mean(all_points[:, 1])
    y_std = np.std(all_points[:, 1])
    
    #this case is for steady case, thus time t is not relevant. But for completeness, we save it in all_points_stats
    t_min = 0.0
    t_max = 0.0
    t_mean = 0.0
    t_std = 0.0

    #get the normalization specs from the postprocessing_specs
    PINN_normalization_specs = postprocessing_specs['PINN_normalization_specs']
    x_normalization_method = PINN_normalization_specs['x']
    y_normalization_method = PINN_normalization_specs['y']
    t_normalization_method = PINN_normalization_specs['t']
    zb_normalization_method = PINN_normalization_specs['zb']
    Sx_normalization_method = PINN_normalization_specs['Sx']
    Sy_normalization_method = PINN_normalization_specs['Sy']
    ManningN_normalization_method = PINN_normalization_specs['ManningN']

    if x_normalization_method == "min-max":
        pde_points[:, 0] = (pde_points[:, 0] - x_min) / (x_max - x_min)
        boundary_points[:, 0] = (boundary_points[:, 0] - x_min) / (x_max - x_min)
    elif x_normalization_method == "z-score":
        pde_points[:, 0] = (pde_points[:, 0] - x_mean) / x_std
        boundary_points[:, 0] = (boundary_points[:, 0] - x_mean) / x_std
    elif x_normalization_method == "none":
        pass
    else:
        raise ValueError(f"Invalid x normalization method: {x_normalization_method}")

    if y_normalization_method == "min-max":
        pde_points[:, 1] = (pde_points[:, 1] - y_min) / (y_max - y_min)
        boundary_points[:, 1] = (boundary_points[:, 1] - y_min) / (y_max - y_min)
    elif y_normalization_method == "z-score":
        pde_points[:, 1] = (pde_points[:, 1] - y_mean) / y_std
        boundary_points[:, 1] = (boundary_points[:, 1] - y_mean) / y_std
    elif y_normalization_method == "none":
        pass
    else:
        raise ValueError(f"Invalid y normalization method: {y_normalization_method}")

    #only do this for unsteady case, i.e., pde_points has time t
    if pde_points.shape[1] == 3:
        if t_normalization_method == "min-max":
            pde_points[:, 2] = (pde_points[:, 2] - t_min) / (t_max - t_min)
        elif t_normalization_method == "z-score":
            pde_points[:, 2] = (pde_points[:, 2] - t_mean) / t_std
        elif t_normalization_method == "none":
            pass
        else:
            raise ValueError(f"Invalid t normalization method: {t_normalization_method}")

    if zb_normalization_method == "min-max":
        pde_data[:, 0] = (pde_data[:, 0] - zb_min) / (zb_max - zb_min)
    elif zb_normalization_method == "z-score":
        pde_data[:, 0] = (pde_data[:, 0] - zb_mean) / zb_std
    elif zb_normalization_method == "none":
        pass
    else:
        raise ValueError(f"Invalid zb normalization method: {zb_normalization_method}")

    if Sx_normalization_method == "min-max":
        pde_data[:, 1] = (pde_data[:, 1] - Sx_min) / (Sx_max - Sx_min)
    elif Sx_normalization_method == "z-score":
        pde_data[:, 1] = (pde_data[:, 1] - Sx_mean) / Sx_std
    elif Sx_normalization_method == "none":
        pass
    else:
        raise ValueError(f"Invalid Sx normalization method: {Sx_normalization_method}")

    if Sy_normalization_method == "min-max":
        pde_data[:, 2] = (pde_data[:, 2] - Sy_min) / (Sy_max - Sy_min)
    elif Sy_normalization_method == "z-score":
        pde_data[:, 2] = (pde_data[:, 2] - Sy_mean) / Sy_std
    elif Sy_normalization_method == "none":
        pass
    else:
        raise ValueError(f"Invalid Sy normalization method: {Sy_normalization_method}")

    if ManningN_normalization_method == "min-max":
        pde_data[:, 3] = (pde_data[:, 3] - ManningN_min) / (ManningN_max - ManningN_min)
    elif ManningN_normalization_method == "z-score":
        pde_data[:, 3] = (pde_data[:, 3] - ManningN_mean) / ManningN_std
    elif ManningN_normalization_method == "none":
        pass
    else:
        raise ValueError(f"Invalid ManningN normalization method: {ManningN_normalization_method}")

    #compute and print the mean and std of the normalized data
    print(f"Mean of normalized pde_points - x: {np.mean(pde_points[:, 0])}")
    print(f"Std of normalized pde_points - x: {np.std(pde_points[:, 0])}")
    print(f"Mean of normalized pde_points - y: {np.mean(pde_points[:, 1])}")
    print(f"Std of normalized pde_points - y: {np.std(pde_points[:, 1])}")

    if pde_points.shape[1] == 3:
        print(f"Mean of normalized pde_points - t: {np.mean(pde_points[:, 2])}")
        print(f"Std of normalized pde_points - t: {np.std(pde_points[:, 2])}")
    else:
        print("pde_points has no time t")

    print(f"Mean of normalized pde_data - zb: {np.mean(pde_data[:, 0])}")
    print(f"Std of normalized pde_data - zb: {np.std(pde_data[:, 0])}")
    print(f"Mean of normalized pde_data - Sx: {np.mean(pde_data[:, 1])}")
    print(f"Std of normalized pde_data - Sx: {np.std(pde_data[:, 1])}")
    print(f"Mean of normalized pde_data - Sy: {np.mean(pde_data[:, 2])}")
    print(f"Std of normalized pde_data - Sy: {np.std(pde_data[:, 2])}")
    print(f"Mean of normalized pde_data - ManningN: {np.mean(pde_data[:, 3])}")
    print(f"Std of normalized pde_data - ManningN: {np.std(pde_data[:, 3])}")

    all_points_stats = np.array([x_min, x_max, x_mean, x_std, y_min, y_max, y_mean, y_std, t_min, t_max, t_mean, t_std, zb_min, zb_max, zb_mean, zb_std, Sx_min, Sx_max, Sx_mean, Sx_std, Sy_min, Sy_max, Sy_mean, Sy_std, ManningN_min, ManningN_max, ManningN_mean, ManningN_std])
    
    all_points_stats_dict = {
        'x_min': float(x_min),
        'x_max': float(x_max),
        'x_mean': float(x_mean),
        'x_std': float(x_std),
        'y_min': float(y_min),
        'y_max': float(y_max),
        'y_mean': float(y_mean),
        'y_std': float(y_std),
        't_min': float(t_min),
        't_max': float(t_max),
        't_mean': float(t_mean),
        't_std': float(t_std),
        'zb_min': float(zb_min),
        'zb_max': float(zb_max),
        'zb_mean': float(zb_mean),
        'zb_std': float(zb_std),
        'Sx_min': float(Sx_min),
        'Sx_max': float(Sx_max),
        'Sx_mean': float(Sx_mean),
        'Sx_std': float(Sx_std),
        'Sy_min': float(Sy_min),
        'Sy_max': float(Sy_max),
        'Sy_mean': float(Sy_mean),
        'Sy_std': float(Sy_std),   
        'ManningN_min': float(ManningN_min),
        'ManningN_max': float(ManningN_max),
        'ManningN_mean': float(ManningN_mean),
        'ManningN_std': float(ManningN_std)
    }

    #added the PINN_normalization_specs to the all_points_stats_dict
    all_points_stats_dict = {
        "PINN_normalization_specs": PINN_normalization_specs,
        "all_points_stats": all_points_stats_dict
        }

    # Print summary
    print("\nPoints Summary:")
    print(f"Number of spatial points: {n_spatial_points}")
    print(f"Total PDE points: {len(pde_points)}")
    print("\nBoundary points by boundary:")
    for boundary_name in training_points['boundary_points'].keys():
        n_points = len(training_points['boundary_points'][boundary_name])
        print(f"{boundary_name}: {n_points} spatial points "
              f"(ID: {boundary_name})")
        
    print(f"x_min: {x_min}, x_max: {x_max}, x_std: {x_std}")
    print(f"y_min: {y_min}, y_max: {y_max}, y_std: {y_std}")
    print(f"t_min: {t_min}, t_max: {t_max}, t_std: {t_std}")
    print(f"zb_min: {zb_min}, zb_max: {zb_max}, zb_std: {zb_std}")
    print(f"Sx_min: {Sx_min}, Sx_max: {Sx_max}, Sx_std: {Sx_std}")
    print(f"Sy_min: {Sy_min}, Sy_max: {Sy_max}, Sy_std: {Sy_std}")

    #print the first 5 boundary points
    print(f"First 5 boundary points: {boundary_points[:5]}")
    print(f"First 5 boundary info: {boundary_info[:5]}")
    
    # Save the arrays
    print(f"\nSaving points to {output_dir}")
    print(f"PDE points shape: {pde_points.shape}")
    print(f"Boundary points shape: {boundary_points.shape}")
    print(f"Boundary info shape: {boundary_info.shape}")
    
    try:
        np.save(os.path.join(output_dir, 'pde_points.npy'), pde_points)
        np.save(os.path.join(output_dir, 'pde_data.npy'), pde_data)
        np.save(os.path.join(output_dir, 'boundary_points.npy'), boundary_points)
        np.save(os.path.join(output_dir, 'boundary_info.npy'), boundary_info)

        print("all_points_stats_dict = ", all_points_stats_dict)
        
        #save the statistics of all the points as a json file
        with open(os.path.join(output_dir, 'all_PINN_points_stats.json'), 'w') as f:
            json.dump(all_points_stats_dict, f, indent=4)
    except Exception as e:
        raise RuntimeError(f"Failed to save files: {e}")
    
    print("\nConversion completed successfully!")
    
    return {
        'pde_points': pde_points.shape,
        'boundary_points': boundary_points.shape,
        'boundary_info': boundary_info.shape
    }
