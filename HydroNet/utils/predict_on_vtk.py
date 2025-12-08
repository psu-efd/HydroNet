"""
Once a NN is trained, make prediction on a 2D unstructured mesh (read in from a vtk file). The prediction can be made on cell centers or points.

It is assumed: 
- the mesh (read in from a vtk file) is a 2D mesh
- the NN model has inputs: x, y, t for unsteady problems or x, y for steady problems
- the NN model outputs a list of variables, which are the prediction variables and depend on x, y, t
- the prediction variable names are provided in prediction_variable_names_list

"""

import numpy as np
import sys
import meshio
import os
import torch
import vtk
from vtk import vtkUnstructuredGridReader, vtkUnstructuredGrid
from vtk.util import numpy_support as VN

def predict_on_vtk2d_mesh(model, vtk2d_fileName, prediction_variable_names_list, vtkFileName, bNodal, 
                          mesh_stats=None, data_stats=None,
                          time_pred=None, device=None):
    """Make prediction on a 2D unstructured vtk mesh.

    Parameters
    ----------
    model : torch.nn.Module
        The trained NN model. Must be in evaluation mode (call model.eval() before calling)
    vtk2d_fileName : str
        Input VTK file containing the 2D unstructured mesh
    prediction_variable_names_list : list
        List of prediction variable names (must match model output order)
    vtkFileName : str
        Output VTK file name for saving predictions
    bNodal : bool
        If True, predict at nodes; if False, predict at cell centers
    mesh_stats : dict, optional
        Mesh statistics (x_mean, x_std, y_mean, y_std, t_mean, t_std, etc.)
    data_stats : dict, optional
        Data statistics (h_mean, h_std, u_mean, u_std, v_mean, v_std, etc.)
    time_pred : float, optional
        Time value for unsteady predictions. If None, assumes steady state
    device : torch.device, optional
        Device to run predictions on. If None, uses model's device

    Returns
    -------
    None
        Saves predictions to VTK file
    """
    print("Making predictions on the provided VTK 2D unstructured mesh...")

    # Set device
    if device is None:
        device = next(model.parameters()).device

    # Read mesh
    try:
        mesh = meshio.read(vtk2d_fileName)
    except Exception as e:
        raise RuntimeError(f"Failed to read VTK file: {e}")

    # Get prediction coordinates
    prediction_coordinate_x, prediction_coordinate_y, cells_list = get_prediction_coordinates(mesh, bNodal)

    # Convert coordinates to PyTorch tensors for normalization
    prediction_coordinate_x = torch.from_numpy(prediction_coordinate_x).float()
    prediction_coordinate_y = torch.from_numpy(prediction_coordinate_y).float()


    #normalize the prediction coordinates
    # Convert stats to tensors if they are not already
    def to_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        else:
            return torch.tensor(value, dtype=torch.float32)
    
    x_min = to_tensor(mesh_stats['x_min'])
    x_max = to_tensor(mesh_stats['x_max'])
    y_min = to_tensor(mesh_stats['y_min'])
    y_max = to_tensor(mesh_stats['y_max'])
    
    if time_pred is not None:
        t_min = to_tensor(mesh_stats['t_min'])
        t_max = to_tensor(mesh_stats['t_max'])

    # Create input tensor
    if time_pred is not None:
        prediction_coordinates = torch.stack([
            prediction_coordinate_x,
            prediction_coordinate_y,
            torch.full_like(prediction_coordinate_x, time_pred)
        ], dim=1)
    else:
        prediction_coordinates = torch.stack([
            prediction_coordinate_x,
            prediction_coordinate_y
        ], dim=1)

    print(f"prediction_coordinates before normalization: {prediction_coordinates}")

    # Move to device first, then normalize (ensures all tensors are on same device)
    prediction_coordinates = prediction_coordinates.to(device)
    x_min = x_min.to(device)
    x_max = x_max.to(device)
    y_min = y_min.to(device)
    y_max = y_max.to(device)
    if time_pred is not None:
        t_min = t_min.to(device)
        t_max = t_max.to(device)

    #normalize the prediction coordinates (min-max normalization)
    prediction_coordinates[:, 0] = (prediction_coordinates[:, 0] - x_min) / (x_max - x_min)
    prediction_coordinates[:, 1] = (prediction_coordinates[:, 1] - y_min) / (y_max - y_min)
    if time_pred is not None:
        prediction_coordinates[:, 2] = (prediction_coordinates[:, 2] - t_min) / (t_max - t_min)

    print(f"prediction_coordinates after normalization: {prediction_coordinates}")

    with torch.no_grad():
        predictions = model(prediction_coordinates)

    #denormalize the predictions
    h_mean = data_stats['h_mean']
    h_std = data_stats['h_std']
    u_mean = data_stats['u_mean']
    u_std = data_stats['u_std']
    v_mean = data_stats['v_mean']
    v_std = data_stats['v_std']
    
    predictions[:, 0] = predictions[:, 0] * h_std + h_mean
    predictions[:, 1] = predictions[:, 1] * u_std + u_mean
    predictions[:, 2] = predictions[:, 2] * v_std + v_mean

    # Convert predictions to numpy for VTK writing
    predictions = predictions.cpu().numpy()

    # Write results
    write_to_vtk(mesh, cells_list, prediction_variable_names_list, predictions, vtkFileName, bNodal)

    print(f"Finished making predictions. Results saved to {vtkFileName}")

def get_prediction_coordinates(mesh, bNodal):
    """Get coordinates for prediction points.

    Parameters
    ----------
    mesh : meshio.Mesh
        Input mesh object
    bNodal : bool
        If True, return node coordinates; if False, return cell center coordinates

    Returns
    -------
    tuple
        (x_coordinates, y_coordinates, cells_list)
    """
    prediction_coordinate_x = []
    prediction_coordinate_y = []
    cells_list = []

    if bNodal:
        # Use node coordinates
        prediction_coordinate_x = mesh.points[:, 0]
        prediction_coordinate_y = mesh.points[:, 1]
        print(f"Using {mesh.points.shape[0]} nodes for prediction")

        # Collect valid cell types
        for cell_block in mesh.cells:
            if cell_block.type in ['triangle', 'quad']:
                cells_list.append(cell_block)
            elif cell_block.type != 'line':
                raise ValueError(f"Unsupported cell type: {cell_block.type}")

    else:
        # Use cell center coordinates
        cell_counter = 0
        for cell_block in mesh.cells:
            if cell_block.type == 'line':
                continue
            elif cell_block.type == 'triangle':
                cells_list.append(cell_block)
                for tri in cell_block.data:
                    p0, p1, p2 = mesh.points[tri]
                    center = (p0 + p1 + p2) / 3.0
                    prediction_coordinate_x.append(center[0])
                    prediction_coordinate_y.append(center[1])
                    cell_counter += 1
            elif cell_block.type == 'quad':
                cells_list.append(cell_block)
                for quad in cell_block.data:
                    p0, p1, p2, p3 = mesh.points[quad]
                    center = (p0 + p1 + p2 + p3) / 4.0  # Fixed: was dividing by 3.0
                    prediction_coordinate_x.append(center[0])
                    prediction_coordinate_y.append(center[1])
                    cell_counter += 1
            else:
                raise ValueError(f"Unsupported cell type: {cell_block.type}")

        print(f"Using {cell_counter} cell centers for prediction")

    return (np.array(prediction_coordinate_x),
            np.array(prediction_coordinate_y),
            cells_list)

def write_to_vtk(mesh, cells_list, prediction_variable_names_list, predictions, vtkFileName, bNodal):
    """Write predictions to VTK file.

    Parameters
    ----------
    mesh : meshio.Mesh
        Original mesh object
    cells_list : list
        List of cell blocks
    prediction_variable_names_list : list
        List of variable names
    predictions : numpy.ndarray
        Array of predictions
    vtkFileName : str
        Output VTK file name
    bNodal : bool
        If True, write nodal data; if False, write cell data
    """
    # Initialize data dictionaries
    point_data = {}
    cell_data = {}

    # Get indices for variables
    h_idx = prediction_variable_names_list.index('h')
    u_idx = prediction_variable_names_list.index('u')
    v_idx = prediction_variable_names_list.index('v')

    if bNodal:
        # Write nodal data
        point_data['h'] = predictions[:, h_idx]
        velocity = np.zeros((predictions.shape[0], 3))
        velocity[:, 0] = predictions[:, u_idx]
        velocity[:, 1] = predictions[:, v_idx]
        point_data['Velocity'] = velocity
    else:
        # Write cell data
        cell_counter = 0
        h_cell_data = []
        velocity_cell_data = []

        for cell_block in cells_list:
            n_cells = cell_block.data.shape[0]
            current_h = []
            current_velocity = []

            for _ in range(n_cells):
                current_h.append(predictions[cell_counter, h_idx])
                vel = np.array([
                    predictions[cell_counter, u_idx],
                    predictions[cell_counter, v_idx],
                    0.0
                ])
                current_velocity.append(vel)
                cell_counter += 1

            h_cell_data.append(np.array(current_h))
            velocity_cell_data.append(np.array(current_velocity))

        cell_data['h'] = h_cell_data
        cell_data['Velocity'] = velocity_cell_data

    # Create output mesh
    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells=cells_list,
        point_data=point_data,
        cell_data=cell_data
    )

    # Write to VTK
    meshio.write(vtkFileName, out_mesh, binary=False)


def compute_bed_slope(vtk_2d_fileName):
    """Compute bed slope from a 2D unstructured mesh in vtk format.

    The bed slope of each cell is computed using the Gauss theorem, which is given by:
    Sx = -(1/A) * \int_{\partial A} z_b * ds
    Sy = -(1/A) * \int_{\partial A} z_b * ds
    where A is the area of the cell, z_b is the bed elevation, and ds is a vector whose direction is the outward normal vector and the length is the length of the boundary edge.

    Currently, the mesh can only have triangular and quadrilateral cells.

    Parameters
    ----------
    vtk_2d_fileName : str
        Input VTK file containing the 2D unstructured mesh

    Returns
    -------
    Sx : numpy.ndarray
        Bed slope in the x direction at cell centers
    Sy : numpy.ndarray
        Bed slope in the y direction at cell centers
    """    

    #check the file exists
    if not os.path.exists(vtk_2d_fileName):
        raise FileNotFoundError(f"The file {vtk_2d_fileName} does not exist.")

    # Read the VTK file
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(vtk_2d_fileName)
    reader.Update()
    mesh = reader.GetOutput()

    # Get points and cells
    points = VN.vtk_to_numpy(mesh.GetPoints().GetData())
    cells = VN.vtk_to_numpy(mesh.GetCells().GetData())
    cell_types = VN.vtk_to_numpy(mesh.GetCellTypesArray())
    
    # Get bed elevation
    zb = points[:, 2]
    
    # Initialize arrays for slopes
    n_cells = mesh.GetNumberOfCells()
    Sx = np.zeros(n_cells)
    Sy = np.zeros(n_cells)
    
    # Process each cell
    cell_id = 0
    i = 0
    while i < len(cells):
        n_points = cells[i]
        cell_type = cell_types[cell_id]
        
        if cell_type == 5:  # Triangle
            # Get point indices for this cell
            p1, p2, p3 = cells[i+1:i+4]
            
            # Get coordinates
            x1, y1 = points[p1][:2]
            x2, y2 = points[p2][:2]
            x3, y3 = points[p3][:2]
            
            # Get bed elevations
            zb1, zb2, zb3 = zb[p1], zb[p2], zb[p3]
            
            # Compute cell area
            area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
            
            # Compute edge lengths and normals
            # Edge 1-2
            dx12 = x2 - x1
            dy12 = y2 - y1
            length12 = np.sqrt(dx12**2 + dy12**2)
            nx12 = dy12 / length12
            ny12 = -dx12 / length12
            
            # Edge 2-3
            dx23 = x3 - x2
            dy23 = y3 - y2
            length23 = np.sqrt(dx23**2 + dy23**2)
            nx23 = dy23 / length23
            ny23 = -dx23 / length23
            
            # Edge 3-1
            dx31 = x1 - x3
            dy31 = y1 - y3
            length31 = np.sqrt(dx31**2 + dy31**2)
            nx31 = dy31 / length31
            ny31 = -dx31 / length31
            
            # Compute contributions to slopes
            Sx[cell_id] = -(1/area) * (
                (zb1 + zb2)/2 * nx12 * length12 +
                (zb2 + zb3)/2 * nx23 * length23 +
                (zb3 + zb1)/2 * nx31 * length31
            )
            
            Sy[cell_id] = -(1/area) * (
                (zb1 + zb2)/2 * ny12 * length12 +
                (zb2 + zb3)/2 * ny23 * length23 +
                (zb3 + zb1)/2 * ny31 * length31
            )
            
        elif cell_type == 9:  # Quadrilateral
            # Get point indices for this cell
            p1, p2, p3, p4 = cells[i+1:i+5]
            
            # Get coordinates
            x1, y1 = points[p1][:2]
            x2, y2 = points[p2][:2]
            x3, y3 = points[p3][:2]
            x4, y4 = points[p4][:2]
            
            # Get bed elevations
            zb1, zb2, zb3, zb4 = zb[p1], zb[p2], zb[p3], zb[p4]
            
            # Compute cell area using shoelace formula
            area = 0.5 * abs(
                x1*y2 + x2*y3 + x3*y4 + x4*y1 -
                (y1*x2 + y2*x3 + y3*x4 + y4*x1)
            )
            
            # Compute edge lengths and normals
            # Edge 1-2
            dx12 = x2 - x1
            dy12 = y2 - y1
            length12 = np.sqrt(dx12**2 + dy12**2)
            nx12 = dy12 / length12
            ny12 = -dx12 / length12
            
            # Edge 2-3
            dx23 = x3 - x2
            dy23 = y3 - y2
            length23 = np.sqrt(dx23**2 + dy23**2)
            nx23 = dy23 / length23
            ny23 = -dx23 / length23
            
            # Edge 3-4
            dx34 = x4 - x3
            dy34 = y4 - y3
            length34 = np.sqrt(dx34**2 + dy34**2)
            nx34 = dy34 / length34
            ny34 = -dx34 / length34
            
            # Edge 4-1
            dx41 = x1 - x4
            dy41 = y1 - y4
            length41 = np.sqrt(dx41**2 + dy41**2)
            nx41 = dy41 / length41
            ny41 = -dx41 / length41
            
            # Compute contributions to slopes
            Sx[cell_id] = -(1/area) * (
                (zb1 + zb2)/2 * nx12 * length12 +
                (zb2 + zb3)/2 * nx23 * length23 +
                (zb3 + zb4)/2 * nx34 * length34 +
                (zb4 + zb1)/2 * nx41 * length41
            )
            
            Sy[cell_id] = -(1/area) * (
                (zb1 + zb2)/2 * ny12 * length12 +
                (zb2 + zb3)/2 * ny23 * length23 +
                (zb3 + zb4)/2 * ny34 * length34 +
                (zb4 + zb1)/2 * ny41 * length41
            )
            
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")
        
        i += n_points + 1
        cell_id += 1
    
    return Sx, Sy





