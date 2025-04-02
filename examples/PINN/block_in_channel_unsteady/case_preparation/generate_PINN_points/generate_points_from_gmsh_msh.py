#test gmsh_to_points

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

import pygmsh
import gmsh

# Get the project root directory (assumes this script is in examples/DeepONet directory)
script_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
project_root = os.path.dirname(examples_dir)

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from HydroNet import gmsh2D_to_points


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

def convert_points_for_pinn(json_file, output_dir, t_min, t_max, n_time_steps):
    """
    Convert points from json file to PINNDataset format.
    
    Parameters
    ----------
    json_file : str
        Path to the points.json file
    output_dir : str
        Directory to save the numpy files
    t_min : float
        Start time
    t_max : float
        End time
    n_time_steps : int
        Number of time steps

    Returns
    -------
    dict
        Dictionary containing shapes of generated arrays
    """
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
    
    # Process PDE points (equation points)
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
    
    # Create time steps
    time_points = np.linspace(t_min, t_max, n_time_steps, dtype=np.float32)
    
    # Expand spatial points along time dimension
    n_spatial_points = len(spatial_points)
    pde_points = np.zeros((n_spatial_points * n_time_steps, 3), dtype=np.float32)   #rows of (x, y, t)
    
    # Fill in the expanded points
    for i, t in enumerate(time_points):
        start_idx = i * n_spatial_points
        end_idx = (i + 1) * n_spatial_points
        pde_points[start_idx:end_idx, :2] = spatial_points  # Copy x, y coordinates
        pde_points[start_idx:end_idx, 2] = t  # Set time coordinate
    
    # Process boundary points
    if 'boundary_points' not in training_points:
        raise ValueError("Training points must contain 'boundary_points' key")

    boundary_id_map = {
        'inlet': 1,
        'outlet': 2,
        'walls': 3,
        'obstacle': 4
    }

    # First collect all boundary spatial points and their info
    all_boundary_spatial_points = []
    all_boundary_normals = []
    all_boundary_ids = []

    # Loop over all boundaries and collect spatial points, normals and IDs
    for boundary_name, boundary_data in training_points['boundary_points'].items():
        print(f"Processing boundary: {boundary_name}")
        boundary_id = boundary_id_map.get(boundary_name, 0)
        
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
            all_boundary_ids.append(boundary_id)

    # Convert to numpy arrays
    all_boundary_spatial_points = np.array(all_boundary_spatial_points, dtype=np.float32)
    all_boundary_normals = np.array(all_boundary_normals, dtype=np.float32)
    all_boundary_ids = np.array(all_boundary_ids, dtype=np.float32)   #ID is integer, but we need to convert to float32 for compatibility

    # Get total number of boundary points
    n_boundary_spatial = len(all_boundary_spatial_points)

    # Create arrays for all boundary points and info
    boundary_points = np.zeros((n_boundary_spatial * n_time_steps, 3), dtype=np.float32)   #rows of (x, y, t)
    boundary_info = np.zeros((n_boundary_spatial * n_time_steps, 3), dtype=np.float32)   #rows of (ID, nx, ny)

    # Expand along time dimension
    for i, t in enumerate(time_points):
        start_idx = i * n_boundary_spatial
        end_idx = (i + 1) * n_boundary_spatial
        
        # Copy spatial coordinates and set time
        boundary_points[start_idx:end_idx, :2] = all_boundary_spatial_points
        boundary_points[start_idx:end_idx, 2] = t
        
        # Copy boundary info (ID and normals)
        boundary_info[start_idx:end_idx, 0] = all_boundary_ids
        boundary_info[start_idx:end_idx, 1:] = all_boundary_normals

    # Create initial points (just take the points at t=t_min)
    initial_points = pde_points[pde_points[:, 2] == t_min].copy()
    
    # Print summary
    print("\nPoints Summary:")
    print(f"Number of spatial points: {n_spatial_points}")
    print(f"Number of time steps: {n_time_steps}")
    print(f"Total PDE points: {len(pde_points)}")
    print(f"Initial points: {len(initial_points)}")
    print("\nBoundary points by boundary:")
    for boundary_name in training_points['boundary_points'].keys():
        n_points = len(training_points['boundary_points'][boundary_name])
        print(f"{boundary_name}: {n_points} spatial points, "
              f"{n_points * n_time_steps} space-time points "
              f"(ID: {boundary_id_map.get(boundary_name, 0)})")
    
    # Save the arrays
    print(f"\nSaving points to {output_dir}")
    print(f"PDE points shape: {pde_points.shape}")
    print(f"Boundary points shape: {boundary_points.shape}")
    print(f"Boundary info shape: {boundary_info.shape}")
    print(f"Initial points shape: {initial_points.shape}")
    
    try:
        np.save(os.path.join(output_dir, 'pde_points.npy'), pde_points)
        np.save(os.path.join(output_dir, 'boundary_points.npy'), boundary_points)
        np.save(os.path.join(output_dir, 'boundary_info.npy'), boundary_info)
        np.save(os.path.join(output_dir, 'initial_points.npy'), initial_points)
    except Exception as e:
        raise RuntimeError(f"Failed to save numpy files: {e}")
    
    print("\nConversion completed successfully!")
    
    return {
        'pde_points': pde_points.shape,
        'boundary_points': boundary_points.shape,
        'boundary_info': boundary_info.shape,
        'initial_points': initial_points.shape
    }

if __name__ == '__main__':

    #generate mesh using pygmsh
    gmsh2d_fileName = "block_in_channel.msh"

    #generate_meshes(gmsh2d_fileName)

    # Convert Gmsh mesh to points
    gmsh2D_to_points(gmsh2d_fileName, refinement=1)

    #load_and_plot_points()

    # Specify your files
    points_json_file = "points.json"
    output_directory = "pinn_points"

    # Specify the time range and number of time steps
    t_min = 0.0
    t_max = 10.0
    n_time_steps = 100
    
    # Convert the points with time expansion
    shapes = convert_points_for_pinn(
        points_json_file, 
        output_directory,
        t_min,
        t_max,
        n_time_steps
    )



    print("All done!")

