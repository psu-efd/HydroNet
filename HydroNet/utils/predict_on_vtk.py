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
import torch
import vtk

def predict_on_vtk2d_mesh(model, vtk2d_fileName, prediction_variable_names_list, vtkFileName, bNodal, 
                          bNormalize=None, normalization_method=None,
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
    bNormalize : bool, optional
        Whether the model takes normalized data as input
    normalization_method : str, optional
        Normalization method (e.g., 'min-max', 'z-score')
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
    if bNormalize:
        # Convert stats to tensors if they are not already and move to device
        x_mean = mesh_stats['x_mean'] if isinstance(mesh_stats['x_mean'], torch.Tensor) else torch.tensor(mesh_stats['x_mean'], dtype=torch.float32)
        x_std = mesh_stats['x_std'] if isinstance(mesh_stats['x_std'], torch.Tensor) else torch.tensor(mesh_stats['x_std'], dtype=torch.float32)
        y_mean = mesh_stats['y_mean'] if isinstance(mesh_stats['y_mean'], torch.Tensor) else torch.tensor(mesh_stats['y_mean'], dtype=torch.float32)
        y_std = mesh_stats['y_std'] if isinstance(mesh_stats['y_std'], torch.Tensor) else torch.tensor(mesh_stats['y_std'], dtype=torch.float32)
        
        # Move all tensors to the same device
        x_mean = x_mean.to(device)
        x_std = x_std.to(device)
        y_mean = y_mean.to(device)
        y_std = y_std.to(device)
        prediction_coordinate_x = prediction_coordinate_x.to(device)
        prediction_coordinate_y = prediction_coordinate_y.to(device)
        
        prediction_coordinate_x = (prediction_coordinate_x - x_mean) / x_std
        prediction_coordinate_y = (prediction_coordinate_y - y_mean) / y_std
        
        if time_pred is not None:
            t_mean = mesh_stats['t_mean'] if isinstance(mesh_stats['t_mean'], torch.Tensor) else torch.tensor(mesh_stats['t_mean'], dtype=torch.float32)
            t_std = mesh_stats['t_std'] if isinstance(mesh_stats['t_std'], torch.Tensor) else torch.tensor(mesh_stats['t_std'], dtype=torch.float32)
            time_pred = torch.tensor(time_pred, dtype=torch.float32)
            
            # Move time-related tensors to device
            t_mean = t_mean.to(device)
            t_std = t_std.to(device)
            time_pred = time_pred.to(device)
            
            time_pred = (time_pred - t_mean) / t_std

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

    # Move to device and make predictions
    prediction_coordinates = prediction_coordinates.to(device)
    with torch.no_grad():
        predictions = model(prediction_coordinates)

    #denormalize the predictions
    if bNormalize:
        # Convert stats to tensors if they are not already and move to device
        h_mean = data_stats['h_mean'] if isinstance(data_stats['h_mean'], torch.Tensor) else torch.tensor(data_stats['h_mean'], dtype=torch.float32)
        h_std = data_stats['h_std'] if isinstance(data_stats['h_std'], torch.Tensor) else torch.tensor(data_stats['h_std'], dtype=torch.float32)
        u_mean = data_stats['u_mean'] if isinstance(data_stats['u_mean'], torch.Tensor) else torch.tensor(data_stats['u_mean'], dtype=torch.float32)
        u_std = data_stats['u_std'] if isinstance(data_stats['u_std'], torch.Tensor) else torch.tensor(data_stats['u_std'], dtype=torch.float32)
        v_mean = data_stats['v_mean'] if isinstance(data_stats['v_mean'], torch.Tensor) else torch.tensor(data_stats['v_mean'], dtype=torch.float32)
        v_std = data_stats['v_std'] if isinstance(data_stats['v_std'], torch.Tensor) else torch.tensor(data_stats['v_std'], dtype=torch.float32)

        # Move all tensors to device
        h_mean = h_mean.to(device)
        h_std = h_std.to(device)
        u_mean = u_mean.to(device)
        u_std = u_std.to(device)
        v_mean = v_mean.to(device)
        v_std = v_std.to(device)

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





