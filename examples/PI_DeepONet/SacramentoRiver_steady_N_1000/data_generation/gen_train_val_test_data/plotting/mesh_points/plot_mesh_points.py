# This script is used to plot the mesh and points (equation and boundary conditions) of the Sacramento River.

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


def plot_mesh_points():
    """
    Make plot for scheme:
    1. mesh
    2. points (equation and boundary conditions) on top of the mesh (zoom in to the confluence region)

    :return:
    """

    fig, axs = plt.subplots(1, 2, figsize=(16, 9), sharex=False, sharey=False, facecolor='w', edgecolor='k')

    # Don't use subplots_adjust if using tight_layout later
    # fig.subplots_adjust(hspace=0.2, wspace=0.3, left=0.05, right=0.98, top=0.95, bottom=0.1)   

    #1. plot mesh
    vtk_fileName = 'case_mesh.vtk'

    #read data from vtk file
    vtk_result = meshio.read(vtk_fileName)

    boundary_outline = extract_boundary_outline(vtk_fileName)    
    xl = boundary_outline[:, 0].min()
    xh = boundary_outline[:, 0].max()
    yl = boundary_outline[:, 1].min()
    yh = boundary_outline[:, 1].max()

    # Plot mesh (triangles and quads)
    if len(vtk_result.cells) > 0:
        points = vtk_result.points[:, :2]  # Get x, y coordinates
        
        # Iterate through all cell blocks
        for cell_block in vtk_result.cells:
            cell_type = cell_block.type
            cell_data = cell_block.data
            
            if cell_type == "triangle":
                # Plot triangles
                for tri in cell_data:
                    tri_points = points[tri]
                    # Close the triangle
                    tri_points_closed = np.vstack([tri_points, tri_points[0]])
                    axs[0].plot(tri_points_closed[:, 0], tri_points_closed[:, 1], 'k', linewidth=0.5, alpha=0.8)
            elif cell_type == "quad":
                # Plot quadrilaterals
                for quad in cell_data:
                    quad_points = points[quad]
                    # Close the quad
                    quad_points_closed = np.vstack([quad_points, quad_points[0]])
                    axs[0].plot(quad_points_closed[:, 0], quad_points_closed[:, 1], 'k', linewidth=0.5, alpha=0.8)
    
    # Plot mesh boundary outline
    if boundary_outline.shape[0] > 0:
        axs[0].plot(boundary_outline[:, 0], boundary_outline[:, 1], 'k-', linewidth=1.0, label='Domain boundary')
    
    axs[0].set_xlim([xl, xh])
    axs[0].set_ylim([yl, yh])
    axs[0].set_xlabel('$x$ (m)', fontsize=32)
    axs[0].set_ylabel('$y$ (m)', fontsize=32)
    axs[0].tick_params(axis='x', labelsize=24)
    axs[0].tick_params(axis='y', labelsize=24)
    axs[0].set_aspect('equal')
    axs[0].set_title("Mesh", fontsize=32)    

    # plot points (equation and boundary conditions) on top of the mesh (zoom in to the confluence region)
    # equation points in equation_points.vtk
    # boundary points in boundary_points.vtk
    # Read point cloud files using VTK (they may not have cells)
    
    def read_points_vtk(filename):
        """Read points from VTK unstructured grid file (points only, no cells)."""
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        data = reader.GetOutput()
        
        points_vtk = data.GetPoints()
        if points_vtk:
            points = vtk_to_numpy(points_vtk.GetData())[:, :2]
            return points, data
        else:
            return np.array([]).reshape(0, 2), None
    
    # Read equation points
    equation_points, eq_data = read_points_vtk('equation_points.vtk')
    if len(equation_points) == 0:
        print("Warning: No equation points found")
    
    # Read boundary points
    boundary_points, bd_data = read_points_vtk('boundary_points.vtk')
    if len(boundary_points) == 0:
        print("Warning: No boundary points found")
        bc_ID = np.array([])
    else:
        # Get bc_ID for boundary points
        if bd_data:
            point_data = bd_data.GetPointData()
            bc_ID_array = None
            for i in range(point_data.GetNumberOfArrays()):
                array = point_data.GetArray(i)
                if array and array.GetName() == 'bc_ID':
                    bc_ID_array = vtk_to_numpy(array)
                    break
            
            if bc_ID_array is not None:
                bc_ID = np.squeeze(bc_ID_array) if bc_ID_array.ndim > 1 else bc_ID_array
            else:
                print("Warning: bc_ID not found in boundary_points.vtk, using default color")
                bc_ID = np.zeros(len(boundary_points))
        else:
            bc_ID = np.zeros(len(boundary_points))

    # Determine zoom region (confluence area) - use the center of boundary points or a specific region
    # For now, use a region around the boundary points
    if len(boundary_points) > 0:
        # Find a region that contains most boundary points (confluence region)
        x_center = boundary_points[:, 0].mean()
        y_center = boundary_points[:, 1].mean()
        x_range = boundary_points[:, 0].max() - boundary_points[:, 0].min()
        y_range = boundary_points[:, 1].max() - boundary_points[:, 1].min()
        
        # Zoom to a region around the confluence (adjust these factors as needed)
        zoom_factor = 0.15  # Show 15% of the domain around confluence
        x_zoom_width = (xh - xl) * zoom_factor
        y_zoom_width = (yh - yl) * zoom_factor
        
        x_zoom_min = 18500
        x_zoom_max = 21240
        y_zoom_min = -7036
        y_zoom_max = -5358
    else:
        # Fallback to center of domain
        x_zoom_min = (xl + xh) / 2 - (xh - xl) * 0.1
        x_zoom_max = (xl + xh) / 2 + (xh - xl) * 0.1
        y_zoom_min = (yl + yh) / 2 - (yh - yl) * 0.1
        y_zoom_max = (yl + yh) / 2 + (yh - yl) * 0.1

    # Plot the mesh in zoomed region (triangles and quads)
    if len(vtk_result.cells) > 0:
        points = vtk_result.points[:, :2]
        
        # Iterate through all cell blocks
        for cell_block in vtk_result.cells:
            cell_type = cell_block.type
            cell_data = cell_block.data
            
            if cell_type == "triangle":
                # Plot triangles in zoom region
                for tri in cell_data:
                    tri_points = points[tri]
                    # Check if triangle is in zoom region
                    if (tri_points[:, 0].min() < x_zoom_max and tri_points[:, 0].max() > x_zoom_min and
                        tri_points[:, 1].min() < y_zoom_max and tri_points[:, 1].max() > y_zoom_min):
                        tri_points_closed = np.vstack([tri_points, tri_points[0]])
                        axs[1].plot(tri_points_closed[:, 0], tri_points_closed[:, 1], 'lightgray', linewidth=0.5, alpha=0.5)
            elif cell_type == "quad":
                # Plot quadrilaterals in zoom region
                for quad in cell_data:
                    quad_points = points[quad]
                    # Check if quad is in zoom region
                    if (quad_points[:, 0].min() < x_zoom_max and quad_points[:, 0].max() > x_zoom_min and
                        quad_points[:, 1].min() < y_zoom_max and quad_points[:, 1].max() > y_zoom_min):
                        quad_points_closed = np.vstack([quad_points, quad_points[0]])
                        axs[1].plot(quad_points_closed[:, 0], quad_points_closed[:, 1], 'lightgray', linewidth=0.5, alpha=0.5)

    # Plot boundary outline in zoom region
    if boundary_outline.shape[0] > 0:
        axs[1].plot(boundary_outline[:, 0], boundary_outline[:, 1], 'k-', linewidth=1.0)

    # Filter points to zoom region
    eq_mask = ((equation_points[:, 0] >= x_zoom_min) & (equation_points[:, 0] <= x_zoom_max) &
               (equation_points[:, 1] >= y_zoom_min) & (equation_points[:, 1] <= y_zoom_max))
    bd_mask = ((boundary_points[:, 0] >= x_zoom_min) & (boundary_points[:, 0] <= x_zoom_max) &
               (boundary_points[:, 1] >= y_zoom_min) & (boundary_points[:, 1] <= y_zoom_max))
    
    equation_points_zoom = equation_points[eq_mask]
    boundary_points_zoom = boundary_points[bd_mask]
    bc_ID_zoom = bc_ID[bd_mask] if len(bc_ID) > 0 else np.array([])

    # Plot the equation points on top of the mesh
    if len(equation_points_zoom) > 0:
        axs[1].scatter(equation_points_zoom[:, 0], equation_points_zoom[:, 1], 
                      color='red', marker='o', s=3, label='Equation points', zorder=5)

    # Plot the boundary points colored by bc_ID
    if len(boundary_points_zoom) > 0 and len(bc_ID_zoom) > 0:
        scatter = axs[1].scatter(boundary_points_zoom[:, 0], boundary_points_zoom[:, 1], 
                                 c='b', marker='o', s=5, 
                                 label='Boundary points', zorder=5)
        # Add colorbar for bc_ID
        #divider = make_axes_locatable(axs[1])
        #cax = divider.append_axes("right", size="3%", pad=0.2)
        #cbar = plt.colorbar(scatter, cax=cax)
        #cbar.set_label('bc\_ID', fontsize=14)
        #cbar.ax.tick_params(labelsize=12)
    
    axs[1].set_xlim([x_zoom_min, x_zoom_max])
    axs[1].set_ylim([y_zoom_min, y_zoom_max])
    axs[1].set_xlabel('$x$ (m)', fontsize=32)
    axs[1].set_ylabel('$y$ (m)', fontsize=32)
    axs[1].tick_params(axis='x', labelsize=24)
    axs[1].tick_params(axis='y', labelsize=24)
    axs[1].set_aspect('equal')
    axs[1].set_title("Points (Confluence Region)", fontsize=32)
    #axs[1].legend(fontsize=12, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    axs[1].legend(fontsize=18, loc='upper right', markerscale=3)
    
    # Use tight_layout to ensure nothing is cut off
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(hspace=0.5)

    plt.savefig("SacramentoRiver_mesh_points.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    #plt.show()
    plt.close()



if __name__ == "__main__":
    plot_mesh_points()

    print('Done')