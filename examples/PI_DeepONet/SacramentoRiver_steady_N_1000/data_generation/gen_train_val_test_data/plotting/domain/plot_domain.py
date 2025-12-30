# This script is used to plot the domain of the Sacramento River.

import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as tick
from matplotlib.collections import PolyCollection
import meshio

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def extract_boundary_outline(vtk_fileName):
    """
    This function extracts the boundary outline of the domain from a VTK file.
    """
    # Extract boundary outline from the triangulated VTK file using VTK
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_fileName)
    reader.Update()
    mesh = reader.GetOutput()
    
    # Convert unstructured grid to polydata (surface)
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(mesh)
    surface_filter.Update()
    surface_polydata = surface_filter.GetOutput()
    
    # Extract boundary edges from the surface
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(surface_polydata)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.Update()
    
    boundary_polydata = feature_edges.GetOutput()
    boundary_points = boundary_polydata.GetPoints()
    
    # Extract boundary points and reorder them to form a continuous loop
    if boundary_points and boundary_points.GetNumberOfPoints() > 0:
        # Get all points
        all_points = vtk_to_numpy(boundary_points.GetData())[:, :2]
        
        # Get the lines (edges) from the polydata
        num_cells = boundary_polydata.GetNumberOfCells()
        if num_cells > 0:
            # Build a graph of connected edges
            edges = []
            for i in range(num_cells):
                cell = boundary_polydata.GetCell(i)
                if cell.GetNumberOfPoints() == 2:  # Line cell
                    pt0 = cell.GetPointId(0)
                    pt1 = cell.GetPointId(1)
                    edges.append((pt0, pt1))
            
            if len(edges) > 0:
                # Reorder points to form a continuous path
                # Start with the first edge
                ordered_points = []
                used_edges = set()
                
                # Find a starting point (one that appears only once, or just use first edge)
                current_edge = edges[0]
                ordered_points.append(all_points[current_edge[0]])
                ordered_points.append(all_points[current_edge[1]])
                used_edges.add(0)
                current_point_id = current_edge[1]
                
                # Continue connecting edges
                while len(used_edges) < len(edges):
                    found_next = False
                    for i, edge in enumerate(edges):
                        if i in used_edges:
                            continue
                        # Check if this edge connects to the current point
                        if edge[0] == current_point_id:
                            ordered_points.append(all_points[edge[1]])
                            current_point_id = edge[1]
                            used_edges.add(i)
                            found_next = True
                            break
                        elif edge[1] == current_point_id:
                            ordered_points.append(all_points[edge[0]])
                            current_point_id = edge[0]
                            used_edges.add(i)
                            found_next = True
                            break
                    
                    if not found_next:
                        # No more connected edges, try to find another disconnected component
                        for i, edge in enumerate(edges):
                            if i not in used_edges:
                                # Start a new path segment
                                ordered_points.append(all_points[edge[0]])
                                ordered_points.append(all_points[edge[1]])
                                used_edges.add(i)
                                current_point_id = edge[1]
                                break
                        else:
                            break
                
                boundary_outline = np.array(ordered_points)
            else:
                # No edges found, just use all points
                boundary_outline = all_points
        else:
            # No cells, just use all points
            boundary_outline = all_points
    else:
        print("Warning: No boundary edges found.")
        boundary_outline = np.array([]).reshape(0, 2)

    return boundary_outline


def plot_domain():
    """
    Make plot for scheme:
    1. bathymetry contour
    2. Manning's n contour    

    :return:
    """

    fig, axs = plt.subplots(1, 2, figsize=(16, 9), sharex=False, sharey=False, facecolor='w', edgecolor='k')

    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    zb_min = -8.0
    zb_max = 14.0

    local_levels = np.linspace(zb_min, zb_max, 51)

    #1. plot bathymetry
    vtk_fileName = 'case_001000_triangulated.vtk'

    #read data from vtk file
    vtk_result = meshio.read(vtk_fileName)

    boundary_outline = extract_boundary_outline(vtk_fileName)
    
    
    # number of triangles
    nTri = vtk_result.cells[0].data.shape[0]

    tri = np.zeros((nTri, 3))
    for i in range(nTri):
        tri[i, 0] = vtk_result.cells[0].data[i, 0]
        tri[i, 1] = vtk_result.cells[0].data[i, 1]
        tri[i, 2] = vtk_result.cells[0].data[i, 2]

    xCoord = vtk_result.points[:, 0]
    yCoord = vtk_result.points[:, 1]

    xl = xCoord.min()
    xh = xCoord.max()
    yl = yCoord.min()
    yh = yCoord.max()

    zb = np.squeeze(vtk_result.point_data['Bed_Elev_m'])

    cf_zb = axs[0].tricontourf(xCoord, yCoord, tri, zb, local_levels,
                                        vmin=zb_min, vmax=zb_max, cmap=plt.cm.terrain, extend='neither')
    
    # Plot boundary outline
    if boundary_outline.shape[0] > 0:
        axs[0].plot(boundary_outline[:, 0], boundary_outline[:, 1], 'k-', linewidth=1.0, label='Domain boundary')
    
    axs[0].set_xlim([xl, xh])
    axs[0].set_ylim([yl, yh])
    axs[0].set_xlabel('$x$ (m)', fontsize=32)
    axs[0].set_ylabel('$y$ (m)', fontsize=32)
    axs[0].tick_params(axis='x', labelsize=24)
    axs[0].tick_params(axis='y', labelsize=24)
    axs[0].set_aspect('equal')
    axs[0].set_title("Bathymetry", fontsize=32)

    #add text to mark the three inlet-q and one exit-h boundaries. 
    #The position of the text uses the plot coordinates (fractional 0-1), not the data coordinates.
    #axs[0].text(0.1, 0.1, '1: Inlet-q', fontsize=24, ha='center', va='center', transform=axs[0].transAxes)
    #axs[0].text(0.24, 0.37, '2: Inlet-q', fontsize=24, ha='center', va='center', transform=axs[0].transAxes)
    #axs[0].text(0.15, 0.98, '3: Inlet-q', fontsize=24, ha='center', va='center', transform=axs[0].transAxes)
    #axs[0].text(0.86, 0.02, '4: Exit-h', fontsize=24, ha='center', va='center', transform=axs[0].transAxes)

    #draw two arrows to mark the flow direction
    #axs[0].arrow(0.36, 0.05, 0.13, 0.2, head_width=0.01, head_length=0.01, fc='k', ec='k', transform=axs[0].transAxes)
    #axs[0].arrow(0.35, 0.8, 0.1, -0.2, head_width=0.01, head_length=0.01, fc='k', ec='k', transform=axs[0].transAxes)

    divider_zb = make_axes_locatable(axs[0])
    cax_zb = divider_zb.append_axes("right", size="3%", pad=0.2)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(zb_min, zb_max, 7), cax=cax_zb)
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.0f'))
    clb_zb.ax.tick_params(labelsize=24)
    clb_zb.ax.set_title("(m)", loc='center', fontsize=24)

    # plot Manning's n (cell-centered data)
    Manning_n = np.squeeze(vtk_result.cell_data['ManningN'])
    Manning_n_min = Manning_n.min()
    Manning_n_max = Manning_n.max()
    local_levels = np.linspace(Manning_n_min, Manning_n_max, 5)

    # Create polygons for each triangle with their cell values
    polygons = []
    colors = []
    for i in range(nTri):
        # Get the three vertices of this triangle
        pt0 = int(tri[i, 0])
        pt1 = int(tri[i, 1])
        pt2 = int(tri[i, 2])
        
        # Create polygon from triangle vertices
        polygon = np.array([
            [xCoord[pt0], yCoord[pt0]],
            [xCoord[pt1], yCoord[pt1]],
            [xCoord[pt2], yCoord[pt2]]
        ])
        polygons.append(polygon)
        colors.append(Manning_n[i])
    
    # Create PolyCollection and set colors
    collection = PolyCollection(polygons, array=np.array(colors), 
                                cmap=plt.cm.jet, edgecolors='none')
    collection.set_clim(vmin=Manning_n_min, vmax=Manning_n_max)
    axs[1].add_collection(collection)
    axs[1].autoscale()
    
    # Plot boundary outline
    if boundary_outline.shape[0] > 0:
        axs[1].plot(boundary_outline[:, 0], boundary_outline[:, 1], 'k-', linewidth=1.0, label='Domain boundary')
    
    # Create colorbar
    cf_Manning_n = collection

    axs[1].set_xlim([xl, xh])
    axs[1].set_ylim([yl, yh])
    axs[1].set_xlabel('$x$ (m)', fontsize=32)
    axs[1].set_ylabel('$y$ (m)', fontsize=32)
    axs[1].set_aspect('equal')
    axs[1].set_title("Manning's $n$", fontsize=32)
    axs[1].tick_params(axis='x', labelsize=24)
    axs[1].tick_params(axis='y', labelsize=24)

    divider_Manning_n = make_axes_locatable(axs[1])
    cax_Manning_n = divider_Manning_n.append_axes("right", size="3%", pad=0.2)
    clb_Manning_n = fig.colorbar(cf_Manning_n, ticks=np.linspace(Manning_n_min, Manning_n_max, 5), cax=cax_Manning_n)
    clb_Manning_n.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_Manning_n.ax.tick_params(labelsize=24)
    clb_Manning_n.ax.set_title("(s/m$^{1/3}$)", loc='center', fontsize=24)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    
    plt.savefig("SacramentoRiver_domain.png", dpi=300, bbox_inches='tight', pad_inches=0)
    #plt.show()
    plt.close()



def triangle_vtk_unstructured_grid(vtk_file):
    """
    This function triangulates a VTK unstructured grid which contains triangles and quadrilaterals.
    It only triangulates the quads (splits them into two triangles), keeping triangles as they are.
    All point data and cell data arrays are preserved.
    Also extracts the boundary outline of the domain.
    
    Parameters
    ----------
    vtk_file : str
        Path to the input VTK unstructured grid file
        
    Returns
    -------
    str
        Path to the output triangulated VTK file
    numpy.ndarray
        Boundary outline points as (N, 2) array with x, y coordinates
    """
    # VTK cell type constants
    VTK_TRIANGLE = 5
    VTK_QUAD = 9
    
    # Read the input VTK file
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    input_mesh = reader.GetOutput()
    
    # Create a new unstructured grid for the output
    output_mesh = vtk.vtkUnstructuredGrid()
    
    # Copy all points (they remain unchanged)
    points = input_mesh.GetPoints()
    output_mesh.SetPoints(points)
    
    # Get number of cells
    num_cells = input_mesh.GetNumberOfCells()
    
    # Lists to store new cells and cell data mappings
    new_cells = []
    cell_data_mapping = []  # Maps new cell index to original cell index
    
    # Iterate through all cells
    for cell_id in range(num_cells):
        cell = input_mesh.GetCell(cell_id)
        cell_type = cell.GetCellType()
        
        # Get point IDs for this cell
        point_ids = cell.GetPointIds()
        num_points = point_ids.GetNumberOfIds()
        
        if cell_type == VTK_TRIANGLE:
            # Keep triangles as they are
            new_cell = vtk.vtkTriangle()
            new_cell.GetPointIds().SetId(0, point_ids.GetId(0))
            new_cell.GetPointIds().SetId(1, point_ids.GetId(1))
            new_cell.GetPointIds().SetId(2, point_ids.GetId(2))
            new_cells.append(new_cell)
            cell_data_mapping.append(cell_id)
            
        elif cell_type == VTK_QUAD:
            # Split quad into two triangles
            # Triangle 1: points 0, 1, 2
            tri1 = vtk.vtkTriangle()
            tri1.GetPointIds().SetId(0, point_ids.GetId(0))
            tri1.GetPointIds().SetId(1, point_ids.GetId(1))
            tri1.GetPointIds().SetId(2, point_ids.GetId(2))
            new_cells.append(tri1)
            cell_data_mapping.append(cell_id)  # Both triangles map to original quad
            
            # Triangle 2: points 0, 2, 3
            tri2 = vtk.vtkTriangle()
            tri2.GetPointIds().SetId(0, point_ids.GetId(0))
            tri2.GetPointIds().SetId(1, point_ids.GetId(2))
            tri2.GetPointIds().SetId(2, point_ids.GetId(3))
            new_cells.append(tri2)
            cell_data_mapping.append(cell_id)  # Both triangles map to original quad
            
        else:
            # For other cell types, keep as is (or raise an error)
            print(f"Warning: Cell {cell_id} has unsupported type {cell_type}. Skipping.")
            continue
    
    # Add all new cells to the output mesh
    for new_cell in new_cells:
        output_mesh.InsertNextCell(new_cell.GetCellType(), new_cell.GetPointIds())
    
    # Copy all point data arrays (unchanged)
    point_data = input_mesh.GetPointData()
    output_point_data = output_mesh.GetPointData()
    
    num_point_arrays = point_data.GetNumberOfArrays()
    for i in range(num_point_arrays):
        array = point_data.GetArray(i)
        output_point_data.AddArray(array)
    
    # Copy and map cell data arrays
    # When a quad becomes two triangles, we duplicate the cell data for both triangles
    cell_data = input_mesh.GetCellData()
    output_cell_data = output_mesh.GetCellData()
    
    num_cell_arrays = cell_data.GetNumberOfArrays()
    for i in range(num_cell_arrays):
        input_array = cell_data.GetArray(i)
        array_name = input_array.GetName()
        num_components = input_array.GetNumberOfComponents()
        num_tuples = input_array.GetNumberOfTuples()
        
        # Create output array with size matching the number of new cells
        output_array = vtk.vtkDataArray.CreateDataArray(input_array.GetDataType())
        output_array.SetName(array_name)
        output_array.SetNumberOfComponents(num_components)
        output_array.SetNumberOfTuples(len(new_cells))
        
        # Map cell data from original cells to new cells
        for new_cell_idx, original_cell_idx in enumerate(cell_data_mapping):
            # Get the tuple from the original array
            tuple_values = input_array.GetTuple(original_cell_idx)
            output_array.SetTuple(new_cell_idx, tuple_values)
        
        output_cell_data.AddArray(output_array)
    
    # Generate output filename
    base_name = os.path.splitext(vtk_file)[0]
    output_file = f"{base_name}_triangulated.vtk"
    
    # Write the output VTK file
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(output_mesh)
    writer.Write()
    
    print(f"Triangulated mesh saved to: {output_file}")
    print(f"Original cells: {num_cells}, New cells: {len(new_cells)}")
    
    # Extract boundary outline using vtkFeatureEdges
    # Convert unstructured grid to polydata (surface) first
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(output_mesh)
    surface_filter.Update()
    surface_polydata = surface_filter.GetOutput()
    
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(surface_polydata)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.Update()
    
    boundary_polydata = feature_edges.GetOutput()
    boundary_points = boundary_polydata.GetPoints()
    
    # Extract boundary points and reorder them to form a continuous loop
    if boundary_points and boundary_points.GetNumberOfPoints() > 0:
        # Get all points
        all_points = vtk_to_numpy(boundary_points.GetData())[:, :2]
        
        # Get the lines (edges) from the polydata
        num_cells = boundary_polydata.GetNumberOfCells()
        if num_cells > 0:
            # Build a graph of connected edges
            edges = []
            for i in range(num_cells):
                cell = boundary_polydata.GetCell(i)
                if cell.GetNumberOfPoints() == 2:  # Line cell
                    pt0 = cell.GetPointId(0)
                    pt1 = cell.GetPointId(1)
                    edges.append((pt0, pt1))
            
            if len(edges) > 0:
                # Reorder points to form a continuous path
                # Start with the first edge
                ordered_points = []
                used_edges = set()
                
                # Find a starting point (one that appears only once, or just use first edge)
                current_edge = edges[0]
                ordered_points.append(all_points[current_edge[0]])
                ordered_points.append(all_points[current_edge[1]])
                used_edges.add(0)
                current_point_id = current_edge[1]
                
                # Continue connecting edges
                while len(used_edges) < len(edges):
                    found_next = False
                    for i, edge in enumerate(edges):
                        if i in used_edges:
                            continue
                        # Check if this edge connects to the current point
                        if edge[0] == current_point_id:
                            ordered_points.append(all_points[edge[1]])
                            current_point_id = edge[1]
                            used_edges.add(i)
                            found_next = True
                            break
                        elif edge[1] == current_point_id:
                            ordered_points.append(all_points[edge[0]])
                            current_point_id = edge[0]
                            used_edges.add(i)
                            found_next = True
                            break
                    
                    if not found_next:
                        # No more connected edges, try to find another disconnected component
                        for i, edge in enumerate(edges):
                            if i not in used_edges:
                                # Start a new path segment
                                ordered_points.append(all_points[edge[0]])
                                ordered_points.append(all_points[edge[1]])
                                used_edges.add(i)
                                current_point_id = edge[1]
                                break
                        else:
                            break
                
                boundary_outline = np.array(ordered_points)
            else:
                # No edges found, just use all points
                boundary_outline = all_points
        else:
            # No cells, just use all points
            boundary_outline = all_points
    else:
        print("Warning: No boundary edges found.")
        boundary_outline = np.array([]).reshape(0, 2)
    
    print(f"Boundary outline extracted: {boundary_outline.shape[0]} points")
    
    return output_file, boundary_outline
    

if __name__ == "__main__":
    #triangle_vtk_unstructured_grid('case_001000.vtk')

    plot_domain()

    print('Done')