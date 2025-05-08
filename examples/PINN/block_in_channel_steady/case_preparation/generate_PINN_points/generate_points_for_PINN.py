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

from HydroNet.utils import gmsh2D_to_points


plt.rc('text', usetex=False)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def generate_meshes(mshFineName):
    """ Generate meshes using pygmsh

    A rectangular hole in a channel.

    pygmsh can also add circle, or any other irregular shape as a hole.

    Returns
    -------

    """

    print("Generate mesh with pygmsh")

    scale = 10.0

    # Characteristic length (resolution)
    resolution = 3e-2*scale

    # Coordinates of lower-left and upper-right vertices of the channel domain
    xmin = 0.0
    xmax = 1.5*scale
    ymin = 0.0
    ymax = 0.5*scale

    #center of the rectangular hole
    hole_center_x = 0.5 * scale
    hole_center_y = 0.21 * scale

    #side length of the rectangular hole
    hole_lx = 0.15 * scale
    hole_ly = 0.12 * scale

    # Initialize empty geometry using the builtin kernel in GMSH
    geometry = pygmsh.geo.Geometry()

    # Fetch model we would like to add data to
    model = geometry.__enter__()

    # Add the rectangular hole (elevation z = 0)
    rectanle = model.add_rectangle(hole_center_x - hole_lx / 2, hole_center_x + hole_lx / 2,
                                   hole_center_y - hole_ly / 2, hole_center_y + hole_ly / 2,
                                   z=0.0,
                                   mesh_size=resolution)

    # Add points for the channel
    points = [model.add_point((xmin, ymin, 0), mesh_size=resolution),
              model.add_point((xmax, ymin, 0), mesh_size=resolution),
              model.add_point((xmax, ymax, 0), mesh_size=resolution),
              model.add_point((xmin, ymax, 0), mesh_size=resolution)]

    # Add lines between all points creating the rectangle channel
    channel_lines = [model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)]

    # Create a line loop and plane surface for meshing
    channel_loop = model.add_curve_loop(channel_lines)
    plane_surface = model.add_plane_surface(channel_loop, holes=[rectanle.curve_loop])

    # Call gmsh kernel before add physical entities
    model.synchronize()

    # Add physcial boundaries
    # For SRH-2D, if we want to explicitly specify boundary conditions for wall,
    # we can not combine disconnected lines to one boundary. They have to be separated.
    model.add_physical([plane_surface], "channel")
    model.add_physical([channel_lines[0]], "inlet")
    model.add_physical([channel_lines[1],channel_lines[3]], "walls")
    model.add_physical([channel_lines[2]], "outlet")
    model.add_physical(rectanle.curve_loop.curves, "obstacle")

    # Generate the mesh using the pygmsh
    geometry.generate_mesh(dim=2)

    # Specify the mesh version as 2 (no need for latest version)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)

    #write the Gmsh MSH file
    gmsh.write(mshFineName)

    #clean up and exit
    gmsh.clear()
    geometry.__exit__()


def load_and_plot_points():

    all_points = points("points.json", "training_points")

    all_points.plot_all_points()

def convert_gmsh2d_points_for_pinn(boundary_id_map):
    """
    Convert mesh points in a json file (derived from Gmsh mesh file) to PINNDataset format.
    
    Parameters
    ----------
    boundary_id_map : dict
        Dictionary mapping boundary names to their IDs
    
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
    spatial_points = []  # temporary storage for spatial coordinates
    
    # Convert equation points dictionary to array (spatial coordinates only)
    for point_id, point_data in equation_points.items():
        spatial_points.append([
            point_data['x'],
            point_data['y']
        ])
    spatial_points = np.array(spatial_points, dtype=np.float32)
    
    # assemble the pde points
    n_spatial_points = len(spatial_points)
    pde_points = np.zeros((n_spatial_points, 2), dtype=np.float32)   #rows of (x, y), no time for steady case
    
    # Fill in the expanded points
    pde_points[:, :2] = spatial_points  # Copy x, y coordinates
    
    # Process boundary points
    if 'boundary_points' not in training_points:
        raise ValueError("Training points must contain 'boundary_points' key")

    # First collect all boundary spatial points and their info
    all_boundary_spatial_points = []
    all_boundary_normals = []
    all_boundary_represented_lengths = []
    all_boundary_ids = []

    # Loop over all boundaries and collect spatial points, normals and IDs
    for boundary_name, boundary_data in training_points['boundary_points'].items():
        print(f"Processing boundary: {boundary_name}")
        boundary_id = boundary_id_map.get(boundary_name, 0)

        print(f"boundary_id: {boundary_id}")
        
        # Get spatial coordinates and normals for this boundary
        for point_id, point_data in boundary_data.items():
            all_boundary_spatial_points.append([
                point_data['x'],
                point_data['y']
            ])
            all_boundary_normals.append([
                point_data['normal_x'],
                point_data['normal_y']
            ])
            all_boundary_represented_lengths.append(point_data['represented_length'])
            all_boundary_ids.append(boundary_id)

    # Convert to numpy arrays
    all_boundary_spatial_points = np.array(all_boundary_spatial_points, dtype=np.float32)
    all_boundary_normals = np.array(all_boundary_normals, dtype=np.float32)
    all_boundary_represented_lengths = np.array(all_boundary_represented_lengths, dtype=np.float32)
    all_boundary_ids = np.array(all_boundary_ids, dtype=np.float32)   #ID is integer, but we need to convert to float32 for compatibility

    # Get total number of boundary points
    n_boundary_spatial = len(all_boundary_spatial_points)

    # Create arrays for all boundary points and info
    boundary_points = np.zeros((n_boundary_spatial, 2), dtype=np.float32)   #rows of (x, y)
    boundary_info = np.zeros((n_boundary_spatial, 4), dtype=np.float32)   #rows of (ID, nx, ny, represented_length)

    # Copy spatial coordinates
    boundary_points[:, :2] = all_boundary_spatial_points
        
    # Copy boundary info (ID and normals)
    boundary_info[:, 0] = all_boundary_ids
    boundary_info[:, 1:3] = all_boundary_normals
    boundary_info[:, 3] = all_boundary_represented_lengths

    # Print summary
    print("\nPoints Summary:")
    print(f"Number of spatial points: {n_spatial_points}")
    print(f"Total PDE points: {len(pde_points)}")
    print("\nBoundary points by boundary:")
    for boundary_name in training_points['boundary_points'].keys():
        n_points = len(training_points['boundary_points'][boundary_name])
        print(f"{boundary_name}: {n_points} spatial points "
              f"(ID: {boundary_id_map.get(boundary_name, 0)})")
        
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
        np.save(os.path.join(output_dir, 'boundary_points.npy'), boundary_points)
        np.save(os.path.join(output_dir, 'boundary_info.npy'), boundary_info)
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

    # Combine h, u, v into data_values
    data_values = np.column_stack((water_depth, u, v))

    # Create flags for each variable (1 if value is not NaN, 0 if NaN)
    h_flag = ~np.isnan(water_depth)
    u_flag = ~np.isnan(u)
    v_flag = ~np.isnan(v)

    # Combine flags into a single array
    data_flags = np.column_stack((h_flag, u_flag, v_flag))

    #print the first 10 data points
    print(f"First 10 data points: {data_points[:10]}")
    print(f"First 10 data values: {data_values[:10]}")
    print(f"First 10 data flags: {data_flags[:10]}")
    
    # Save data points (x, y coordinates)
    np.save(os.path.join(output_dir, 'data_points.npy'), data_points)
    
    # Save data values (h, u, v)
    np.save(os.path.join(output_dir, 'data_values.npy'), data_values)

    # Save data flags (h_flag, u_flag, v_flag)
    np.save(os.path.join(output_dir, 'data_flags.npy'), data_flags)

    print(f"Saved data points shape: {data_points.shape}")
    print(f"Saved data values shape: {data_values.shape}")
    print(f"Saved data flags shape: {data_flags.shape}")
    print(f"Files saved in: {output_dir}")

if __name__ == '__main__':

    #generate mesh using pygmsh
    gmsh2d_fileName = "block_in_channel.msh"

    #generate_meshes(gmsh2d_fileName)

    # Convert Gmsh mesh to points (save to points.json file)
    gmsh2D_to_points(gmsh2d_fileName, refinement=1)

    exit()

    #load_and_plot_points()

    # Define boundary ID map (this should match the boundary IDs in the SRH-2D case)
    boundary_id_map = {
        'inlet': 1,
        'outlet': 2,
        'top': 3,
        'bottom': 4,
        'obstacle': 5
    }
    
    # Convert the points to npy files in "pinn_points" directory (to be loaded by PINNDataset)
    convert_gmsh2d_points_for_pinn(boundary_id_map)

    # Create data files for SRH-2D simulation results
    vtk_file_name = "SRH2D_block_in_channel_C_0005.vtk"
    units = "SI"
    create_data_files_from_SRH2D_vtk(vtk_file_name, units)

    print("All done!")

