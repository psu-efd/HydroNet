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

from pyHMT2D.Misc.SRH_to_PINN_points import srh_to_pinn_points


plt.rc('text', usetex=False)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"


def convert_mesh_points_for_pinn():
    """
    Convert mesh points in a json file (mesh_points.json, derived from SRH-2D mesh file) to PINNDataset format.
    
    Parameters
    ----------
    
    Returns
    -------
    dict
        Dictionary containing shapes of generated arrays
    """

    json_file = "mesh_points.json"
    output_dir = "pinn_points"

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

    all_points_stats = np.array([x_min, x_max, x_mean, x_std, y_min, y_max, y_mean, y_std, t_min, t_max, t_mean, t_std, zb_min, zb_max, zb_mean, zb_std, Sx_min, Sx_max, Sx_mean, Sx_std, Sy_min, Sy_max, Sy_mean, Sy_std, ManningN_min, ManningN_max, ManningN_mean, ManningN_std])
    
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
        #save the statistics of all the points
        np.save(os.path.join(output_dir, 'all_mesh_points_stats.npy'), all_points_stats)   #these can be used for normalization
    except Exception as e:
        raise RuntimeError(f"Failed to save numpy files: {e}")
    
    print("\nConversion completed successfully!")
    
    return {
        'pde_points': pde_points.shape,
        'boundary_points': boundary_points.shape,
        'boundary_info': boundary_info.shape
    }

def create_data_files_from_SRH2D_vtk(vtk_file_name, units):
    """
    Create data file from SRH-2D simulation results in a vtk file. 

    The data in vtk is at cell centers. We need to extract the data at the cell centers and save it to a numpy file. 
    Specifically, we need to save data_points (x, y) and data_values (h, u, v) in numpy files in the "pinn_points" directory.
    We also save data_flags (h_flag, u_flag, v_flag) to indicate which variables are available for each point.

    Parameters
    ----------
    vtk_file_name : str
        Name of the vtk file containing the simulation results
    units : str
        Units of the simulation results: SI or EN
    """

    output_dir = "pinn_points"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the solution variable names in the vtk file
    # We only need the velocity and water depth. Also, the velocity is a vector field.
    if units == "SI":
        solution_variable_names = ["Velocity_m_p_s", "Water_Depth_m"]
    elif units == "EN":
        solution_variable_names = ["Velocity_ft_p_s", "Water_Depth_ft"]
    else:
        raise ValueError("Invalid units")
    
    # Read the VTK file
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(vtk_file_name)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    grid = reader.GetOutput()

    # Get cell centers using vtkCellCenters filter
    cell_centers_filter = vtkCellCenters()
    cell_centers_filter.SetInputData(grid)
    cell_centers_filter.Update()
    cell_centers = cell_centers_filter.GetOutput()

    # Convert cell centers to numpy array
    cell_centers_array = numpy_support.vtk_to_numpy(cell_centers.GetPoints().GetData())
    
    # Extract x, y coordinates
    data_points = cell_centers_array[:, :2]  # Only take x, y coordinates

    # Get velocity and water depth data
    velocity_data = numpy_support.vtk_to_numpy(grid.GetCellData().GetArray(solution_variable_names[0]))
    water_depth = numpy_support.vtk_to_numpy(grid.GetCellData().GetArray(solution_variable_names[1]))

    # Extract u, v components from velocity vector
    u = velocity_data[:, 0]
    v = velocity_data[:, 1]
    Umag = np.sqrt(u**2 + v**2)

    # Combine h, u, v into data_values
    data_values = np.column_stack((water_depth, u, v))

    # Create flags for each variable: should be customized based on what you really need. For example, if you don't want to use h or h is not available, you can set h_flag to 0.
    # (1 if value is not NaN, 0 if NaN)
    h_flag = ~np.isnan(water_depth)
    u_flag = ~np.isnan(u)
    v_flag = ~np.isnan(v)

    # Combine flags into a single array
    data_flags = np.column_stack((h_flag, u_flag, v_flag))

    #print the first 10 data points
    print(f"First 10 data points: {data_points[:10]}")
    print(f"First 10 data values: {data_values[:10]}")
    print(f"First 10 data flags: {data_flags[:10]}")

    #compute the statistics of the data points
    #min, max, mean, std, median, etc.
    x_min = np.min(data_points[:, 0])  #stats of x, y, and t are computed here, but not used in training. The normalization should use the statistics of the pde/boundary points from mesh_points.json.
    x_max = np.max(data_points[:, 0])
    x_mean = np.mean(data_points[:, 0])
    x_std = np.std(data_points[:, 0])
    y_min = np.min(data_points[:, 1])
    y_max = np.max(data_points[:, 1])
    y_mean = np.mean(data_points[:, 1])
    y_std = np.std(data_points[:, 1])
    t_min = 0.0
    t_max = 0.0
    t_mean = 0.0
    t_std = 0.0
    h_min = np.min(data_values[:, 0])
    h_max = np.max(data_values[:, 0])
    h_mean = np.mean(data_values[:, 0])
    h_std = np.std(data_values[:, 0])
    u_min = np.min(data_values[:, 1])
    u_max = np.max(data_values[:, 1])
    u_mean = np.mean(data_values[:, 1])
    u_std = np.std(data_values[:, 1])
    v_min = np.min(data_values[:, 2])
    v_max = np.max(data_values[:, 2])
    v_mean = np.mean(data_values[:, 2])
    v_std = np.std(data_values[:, 2])
    Umag_min = np.min(Umag)
    Umag_max = np.max(Umag)
    Umag_mean = np.mean(Umag)
    Umag_std = np.std(Umag)

    all_data_points_stats = np.array([x_min, x_max, x_mean, x_std, y_min, y_max, y_mean, y_std, t_min, t_max, t_mean, t_std, h_min, h_max, h_mean, h_std, u_min, u_max, u_mean, u_std, v_min, v_max, v_mean, v_std, Umag_min, Umag_max, Umag_mean, Umag_std])
    print(f"x_min: {x_min}, x_max: {x_max}, x_mean: {x_mean}, x_std: {x_std}")
    print(f"y_min: {y_min}, y_max: {y_max}, y_mean: {y_mean}, y_std: {y_std}")
    print(f"t_min: {t_min}, t_max: {t_max}, t_mean: {t_mean}, t_std: {t_std}")
    print(f"h_min: {h_min}, h_max: {h_max}, h_mean: {h_mean}, h_std: {h_std}")
    print(f"u_min: {u_min}, u_max: {u_max}, u_mean: {u_mean}, u_std: {u_std}")
    print(f"v_min: {v_min}, v_max: {v_max}, v_mean: {v_mean}, v_std: {v_std}")
    print(f"Umag_min: {Umag_min}, Umag_max: {Umag_max}, Umag_mean: {Umag_mean}, Umag_std: {Umag_std}")
    
    
    # Save data points (x, y coordinates)
    np.save(os.path.join(output_dir, 'data_points.npy'), data_points)
    
    # Save data values (h, u, v)
    np.save(os.path.join(output_dir, 'data_values.npy'), data_values)

    # Save data flags (h_flag, u_flag, v_flag)
    np.save(os.path.join(output_dir, 'data_flags.npy'), data_flags)

    #save the statistics of the data points
    np.save(os.path.join(output_dir, 'all_data_points_stats.npy'), all_data_points_stats)   #these can be used for normalization

    print(f"Saved data points shape: {data_points.shape}")
    print(f"Saved data values shape: {data_values.shape}")
    print(f"Saved data flags shape: {data_flags.shape}")
    print(f"Files saved in: {output_dir}")

if __name__ == '__main__':

    # Convert SRH-2D mesh to points (save to mesh_points.json file)
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the SRH-2D case file
    # You can use either .srhhydro or _SIF.dat file
    srhcontrol_file = os.path.join(current_dir, "block_in_channel.srhhydro")   #block_in_channel.srhhydro is the SRH-2D case file for the block_in_channel example
    
    # Check if the file exists
    if not os.path.exists(srhcontrol_file):
        print(f"Error: File {srhcontrol_file} does not exist.")
        print("Please make sure you have the SRH-2D case file in the correct location.")
        exit()
    
    print(f"Processing SRH-2D case: {srhcontrol_file}")
    
    # Convert SRH-2D mesh to points in mesh_points.json file
    # refinement=2 means generate 2 points per cell/edge
    srh_to_pinn_points(srhcontrol_file, refinement_pde=2, refinement_bc=4)
        
    print("\nConversion completed successfully!")
    print("Generated files:")
    print("1. mesh_points.json - Contains all points for PINN training")
    print("2. equation_points.vtk - Visualization of interior points")
    print("3. boundary_points.vtk - Visualization of boundary points with normal vectors")

    # Convert the points in mesh_points.json to npy files in "pinn_points" directory (to be loaded by PINNDataset)
    convert_mesh_points_for_pinn()

    # Create data files for SRH-2D simulation results
    vtk_file_name = "SRH2D_block_in_channel_C_0005.vtk"
    units = "SI"
    create_data_files_from_SRH2D_vtk(vtk_file_name, units)

    print("All done!")

