"""
This set of tools accomplish this task: convert a Gmsh 2D mesh to points needed for using NN to solve PDEs.
    - "points" include both equation points and boundary points
    - Save format: JSON


Note: currently only 2D mesh is tested. To solve 3D PDEs, modifications are needed.

"""

import meshio
import os
import numpy as np
import json
from .misc_tools import generate_random01_exclude_boundaries_with_center, point_on_triangle, point_on_line

def gmsh2D_to_points(mshFileName, refinement=1):
    """Convert Gmsh 2D mesh to points for PINN training. The saved "mesh_points.json" file can be used for PINN training. The file contains "equation_points" and "boundary_points".

    The saved points only have spatial information. The time information is not included.

    Parameters
    ----------
    mshFileName : str
        File name of the Gmsh 2D mesh
    refinement : int
        Number of points to generate per cell (>= 1)

    Returns
    -------
    None
        Saves points to mesh_points.json
        Saves domain and boundary meshes to domain_{filename}.xdmf and boundaries_{filename}.xdmf
        Saves equation and boundary points to equation_points.vtk and boundary_points.vtk (for visualization and inspection)
    """
    # Validate refinement parameter
    if not isinstance(refinement, int) or refinement < 1:
        raise ValueError("The parameter refinement must be an integer >= 1")

    # Get base filename
    filename, _ = os.path.splitext(mshFileName)

    # Read Gmsh mesh
    try:
        mesh_from_file = meshio.read(mshFileName)
    except Exception as e:
        raise RuntimeError(f"Failed to read Gmsh file: {e}")

    # Process domain (interior) points
    cell_mesh, cell_physical_ID_to_name = extract_gmsh_parts(mesh_from_file, ["triangle", "quad"], prune_z=False)
    meshio.write(f"domain_{filename}.xdmf", cell_mesh)
    equation_points_dict = process_domain_gmsh(cell_mesh, cell_physical_ID_to_name, refinement)

    # Process boundary points
    boundary_mesh, boundary_physical_ID_to_name = extract_gmsh_parts(mesh_from_file, ["line"], prune_z=False)
    meshio.write(f"boundaries_{filename}.xdmf", boundary_mesh)
    boundary_points_dict = process_boundary_gmsh(boundary_mesh, boundary_physical_ID_to_name, refinement)

    # Assemble all points
    all_points = {
        "training_points": {
            "equation_points": equation_points_dict,
            "boundary_points": boundary_points_dict
        }
    }

    # Save to JSON
    with open("mesh_points.json", 'w') as f:
        json.dump(all_points, f, indent=4, sort_keys=False)

def extract_gmsh_parts(mesh, desired_cell_types, prune_z=False):
    """Extract specified cell types from mesh.

    Parameters
    ----------
    mesh : meshio.Mesh
        Input mesh object
    desired_cell_types : list
        List of cell types to extract
    prune_z : bool
        Whether to set z-coordinates to zero

    Returns
    -------
    tuple
        (meshio.Mesh, dict) containing extracted mesh and physical ID mapping
    """
    cells_dict = {}
    cells_data_list = []
    cell_physical_ID_to_name = {}

    # Build physical ID to name mapping
    for physical_name, physical_data in mesh.field_data.items():
        cell_physical_ID_to_name[physical_data[0]] = physical_name

    # Extract cells and cell data
    for cell_type in desired_cell_types:
        cells = mesh.get_cells_type(cell_type)
        if cells.size != 0:
            cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
            cells_dict[cell_type] = cells
            cells_data_list.append(cell_data)

    # Combine cell data
    all_cells_data = np.concatenate(cells_data_list) if cells_data_list else np.array([])

    # Create output mesh
    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells=cells_dict,
        cell_data={"name_to_read": [all_cells_data]}
    )

    if prune_z:
        out_mesh.points[:, 2] = 0.0

    return out_mesh, cell_physical_ID_to_name

def process_boundary_gmsh(boundary_mesh, boundary_physical_ID_to_name, refinement):
    """Process boundary mesh to generate boundary points.

    Parameters
    ----------
    boundary_mesh : meshio.Mesh
        Boundary mesh object
    boundary_physical_ID_to_name : dict
        Mapping of physical IDs to names
    refinement : int
        Number of points per line segment

    Returns
    -------
    dict
        Dictionary containing boundary points and their properties
    """
    boundary_points_dict = {}

    all_bc_lines_bcID = boundary_mesh.cell_data['name_to_read'][0]
    all_bc_lines_nodes = boundary_mesh.cells_dict['line']
    all_bc_nodes = boundary_mesh.points

    n_points = len(all_bc_lines_nodes) * refinement
    bc_points = np.zeros((n_points, 3))
    bc_normal_vectors = np.zeros((n_points, 3))
    bc_IDs = np.zeros(n_points, dtype=np.int64)
    bc_represented_lengths = np.zeros(n_points, dtype=np.float64)

    pointID = 0
    for lineI, line in enumerate(all_bc_lines_nodes):
        p1, p2 = all_bc_nodes[line]
        line_length = np.sqrt(np.sum((p2 - p1)**2))
        represented_length = line_length / refinement

        # Generate sampling points
        if refinement == 1:
            s_all = np.array([0.5])
        elif refinement == 2:
            s_all = np.array([0.333, 0.666])
        else:
            s_all = np.linspace(0.05, 0.95, refinement)

        for s in s_all:
            point = point_on_line(p1, p2, s)
            bc_IDs[pointID] = all_bc_lines_bcID[lineI]
            bc_points[pointID, :2] = point

            # Calculate and normalize normal vector
            normal = np.array([p2[1] - p1[1], -(p2[0] - p1[0]), 0.0])
            normal /= np.linalg.norm(normal[:2])
            bc_normal_vectors[pointID] = normal

            bc_represented_lengths[pointID] = represented_length
            pointID += 1

    # Write boundary points for visualization
    point_data = {
        'bc_ID': bc_IDs,
        'normal_vector': bc_normal_vectors,
        'represented_length': bc_represented_lengths
    }
    meshio.write_points_cells("boundary_points.vtk", bc_points, {}, point_data, {}, binary=True)

    # Organize points by boundary ID
    unique_BC_IDs = np.unique(all_bc_lines_bcID)
    for bc_ID in unique_BC_IDs:
        mask = bc_IDs == bc_ID
        current_boundary_dict = {}
        
        for i, point in enumerate(bc_points[mask]):
            current_boundary_dict[str(i)] = {
                "x": point[0],
                "y": point[1],
                "z": 0.0,
                "normal_x": bc_normal_vectors[mask][i][0],
                "normal_y": bc_normal_vectors[mask][i][1],
                "normal_z": bc_normal_vectors[mask][i][2],
                "t": 0.0,
                "spatial_dimensionality": 2,
                "represented_length": bc_represented_lengths[mask][i]
            }

        boundary_points_dict[boundary_physical_ID_to_name[bc_ID]] = current_boundary_dict

    return boundary_points_dict

def process_domain_gmsh(cell_mesh, cell_physical_ID_to_name, refinement):
    """Process domain mesh to generate interior points.

    Note: currently only support quad and triangle cells.

    Parameters
    ----------
    cell_mesh : meshio.Mesh
        Domain mesh object
    cell_physical_ID_to_name : dict
        Mapping of physical IDs to names
    refinement : int
        Number of points per cell

    Returns
    -------
    dict
        Dictionary containing interior points and their properties
    """
    equation_points_dict = {}
    points = cell_mesh.points

    # Count total cells
    total_cells = sum(len(cells) for cells in cell_mesh.cells_dict.values())
    print(f"Total number of cells in the mesh: {total_cells}")

    # Initialize arrays for visualization
    total_points = total_cells * refinement
    equation_points = np.zeros((total_points, 3))
    pointID = 0

    # Process triangles
    if 'triangle' in cell_mesh.cells_dict:
        for triangle in cell_mesh.cells_dict['triangle']:
            p0, p1, p2 = points[triangle]
            
            # Generate sampling points using barycentric coordinates
            st_all = generate_random01_exclude_boundaries_with_center(
                centers=[1.0/3.0, 2.0/3.0],
                size=refinement
            )

            for st in st_all:
                point = point_on_triangle(p0, p1, p2, st[0], st[1])
                equation_points[pointID] = [point[0], point[1], point[2]]
                
                equation_points_dict[str(pointID)] = {
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "spatial_dimensionality": 2
                }
                pointID += 1

    # Process quads
    if 'quad' in cell_mesh.cells_dict:
        for quad in cell_mesh.cells_dict['quad']:
            p0, p1, p2, p3 = points[quad]
            
            # Generate sampling points using bilinear interpolation
            st_all = generate_random01_exclude_boundaries_with_center(
                centers=[0.5, 0.5],
                size=refinement
            )

            for st in st_all:
                s, t = st
                # Bilinear interpolation
                bottom_edge = (1-s)*p0 + s*p1
                top_edge = (1-s)*p3 + s*p2
                point = (1-t)*bottom_edge + t*top_edge
                
                equation_points[pointID] = [point[0], point[1], point[2]]
                
                equation_points_dict[str(pointID)] = {
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "spatial_dimensionality": 2
                }
                pointID += 1

    # Write points for visualization
    meshio.write_points_cells("equation_points.vtk", equation_points, {}, {}, {}, binary=True)

    return equation_points_dict


