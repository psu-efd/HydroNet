#This script plots the flow velocity vectors comparison of the results of the DeepONet and PI-DeepONet models and the simulation results for the test cases.
# All results are in vtk files. Each vtk file contains the results for a test case and it also has the SRH-2D simulation results for the same test case.
# It produces one figure. 
# It plots 3 rows of subplots. Each row is for SRH-2D simulation, SWE-DeepONet, and PI-SWE-DeepONet, respectively.Each subplot plots velocity vector plots with Umag contourf.

# To reduce clutter, all subfigures will not have axes labels and ticks.

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as tick
from matplotlib.collections import PolyCollection
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import meshio

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def extract_boundary_outline(vtk_fileName):
    """Extract boundary outline from VTK file."""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_fileName)
    reader.Update()
    mesh = reader.GetOutput()
    
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(mesh)
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
    
    if boundary_points and boundary_points.GetNumberOfPoints() > 0:
        all_points = vtk_to_numpy(boundary_points.GetData())[:, :2]
        num_cells = boundary_polydata.GetNumberOfCells()
        if num_cells > 0:
            edges = []
            for i in range(num_cells):
                cell = boundary_polydata.GetCell(i)
                if cell.GetNumberOfPoints() == 2:
                    pt0 = cell.GetPointId(0)
                    pt1 = cell.GetPointId(1)
                    edges.append((pt0, pt1))
            
            if len(edges) > 0:
                ordered_points = []
                used_edges = set()
                current_edge = edges[0]
                ordered_points.append(all_points[current_edge[0]])
                ordered_points.append(all_points[current_edge[1]])
                used_edges.add(0)
                current_point_id = current_edge[1]
                
                while len(used_edges) < len(edges):
                    found_next = False
                    for i, edge in enumerate(edges):
                        if i in used_edges:
                            continue
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
                        for i, edge in enumerate(edges):
                            if i not in used_edges:
                                ordered_points.append(all_points[edge[0]])
                                ordered_points.append(all_points[edge[1]])
                                used_edges.add(i)
                                current_point_id = edge[1]
                                break
                        else:
                            break
                
                return np.array(ordered_points)
            else:
                return all_points
        else:
            return all_points
    else:
        return np.array([]).reshape(0, 2)

def plot_compare_test_case_results(test_case_indices, deeponet_result_dir, pi_deeponet_result_dir):
    """
    Plots the comparison of the results of the DeepONet and PI-DeepONet models and the simulation results for the test cases.

    The figures ar saved as png files in the current directory. The figure names are 'compare_test_case_results_{case_id}.png' and 'compare_test_case_diff_2_{case_id}.png'.

    Args:
        test_case_indices: The indices of the test cases to plot.
        deeponet_result_dir: The directory of the DeepONet result files.
        pi_deeponet_result_dir: The directory of the PI-DeepONet result files.
    Returns:
        None
    """

    # Loop through the test case indices
    for test_case_index in test_case_indices:
        # Get the result files - check file naming convention
        deeponet_result_file = os.path.join(deeponet_result_dir, f'case_{test_case_index}_test_results.vtk')
        if not os.path.exists(deeponet_result_file):
            raise FileNotFoundError(f"DeepONet result file not found: {deeponet_result_file}")
        
        pi_deeponet_result_file = os.path.join(pi_deeponet_result_dir, f'case_{test_case_index}_test_results.vtk')
        if not os.path.exists(pi_deeponet_result_file):
            raise FileNotFoundError(f"PI-DeepONet result file not found: {pi_deeponet_result_file}")        

        # Read the result files using meshio
        deeponet_result = meshio.read(deeponet_result_file)
        pi_deeponet_result = meshio.read(pi_deeponet_result_file)
        
        # Extract boundary outline (use first file for boundary)
        boundary_outline = extract_boundary_outline(deeponet_result_file)
        
        # Get mesh points and cells
        points = deeponet_result.points[:, :2]  # x, y coordinates
        xl, xh = points[:, 0].min(), points[:, 0].max()
        yl, yh = points[:, 1].min(), points[:, 1].max()
        
        # Get all cells (triangles and quads) and build cell-to-point mapping
        all_cells = []
        cell_data_indices = []  # Track which cell data index corresponds to each cell
        cell_idx = 0
        
        for cell_block in deeponet_result.cells:
            if cell_block.type == "triangle":
                for cell in cell_block.data:
                    all_cells.append(cell)
                    cell_data_indices.append(cell_idx)
                    cell_idx += 1
            elif cell_block.type == "quad":
                # Split quads into two triangles for contour plotting
                for quad in cell_block.data:
                    # Triangle 1: points 0, 1, 2
                    all_cells.append([quad[0], quad[1], quad[2]])
                    cell_data_indices.append(cell_idx)
                    # Triangle 2: points 0, 2, 3
                    all_cells.append([quad[0], quad[2], quad[3]])
                    cell_data_indices.append(cell_idx)
                    cell_idx += 1
        
        triangles = np.array(all_cells) if all_cells else None
        
        # Helper function to interpolate cell data to points
        def cell_to_point_data(cell_data, all_cells_list, cell_data_indices_list, n_points):
            """Interpolate cell-centered data to point data by averaging values from all cells containing each point."""
            point_data = np.zeros(n_points)
            point_count = np.zeros(n_points)
            for cell, data_idx in zip(all_cells_list, cell_data_indices_list):
                cell_value = cell_data[data_idx]
                for pt_id in cell:
                    point_data[pt_id] += cell_value
                    point_count[pt_id] += 1
            point_data = point_data / np.maximum(point_count, 1)
            return point_data
        
        # Helper function to extract cell data (handles multiple cell blocks)
        def extract_cell_data_array(cell_data_dict, key):
            """Extract cell data array, concatenating across all cell blocks if needed."""
            if key not in cell_data_dict:
                raise KeyError(f"Key '{key}' not found in cell_data")
            data = cell_data_dict[key]
            # If it's a list (multiple cell blocks), concatenate
            if isinstance(data, list):
                return np.concatenate([np.array(d).flatten() for d in data])
            else:
                return np.array(data).flatten()
        
        def extract_cell_data_vector(cell_data_dict, key, n_components=3):
            """Extract vector cell data array, concatenating across all cell blocks if needed."""
            if key not in cell_data_dict:
                raise KeyError(f"Key '{key}' not found in cell_data")
            data = cell_data_dict[key]
            # If it's a list (multiple cell blocks), concatenate
            if isinstance(data, list):
                arrays = []
                for d in data:
                    d_array = np.array(d)
                    if d_array.ndim == 1:
                        # Reshape to (n_cells, n_components)
                        d_array = d_array.reshape(-1, n_components)
                    arrays.append(d_array)
                return np.vstack(arrays)
            else:
                d_array = np.array(data)
                if d_array.ndim == 1:
                    d_array = d_array.reshape(-1, n_components)
                return d_array
        
        # Extract data from DeepONet results
        deeponet_cell_data = deeponet_result.cell_data
        deeponet_h_pred = extract_cell_data_array(deeponet_cell_data, 'Water_Depth_Pred')
        deeponet_u_pred = extract_cell_data_array(deeponet_cell_data, 'X_Velocity_Pred')
        deeponet_v_pred = extract_cell_data_array(deeponet_cell_data, 'Y_Velocity_Pred')
        deeponet_vel_pred = extract_cell_data_vector(deeponet_cell_data, 'Velocity_Pred', 3)[:, :2]  # x, y components
        
        # Extract SRH-2D simulation data (from DeepONet file - same mesh)
        sim_h = extract_cell_data_array(deeponet_cell_data, 'Water_Depth_Target')
        sim_u = extract_cell_data_array(deeponet_cell_data, 'X_Velocity_Target')
        sim_v = extract_cell_data_array(deeponet_cell_data, 'Y_Velocity_Target')
        sim_vel = extract_cell_data_vector(deeponet_cell_data, 'Velocity_Target', 3)[:, :2]
        
        # Extract data from PI-DeepONet results
        pi_cell_data = pi_deeponet_result.cell_data
        pi_h_pred = extract_cell_data_array(pi_cell_data, 'Water_Depth_Pred')
        pi_u_pred = extract_cell_data_array(pi_cell_data, 'X_Velocity_Pred')
        pi_v_pred = extract_cell_data_array(pi_cell_data, 'Y_Velocity_Pred')
        pi_vel_pred = extract_cell_data_vector(pi_cell_data, 'Velocity_Pred', 3)[:, :2]
        
        # Build cell list for interpolation (need original cells, not triangulated)
        original_cells = []
        original_cell_idx = 0
        for cell_block in deeponet_result.cells:
            if cell_block.type == "triangle":
                for cell in cell_block.data:
                    original_cells.append(cell)
            elif cell_block.type == "quad":
                for quad in cell_block.data:
                    original_cells.append(quad)
            original_cell_idx += len(cell_block.data)
        
        # Interpolate cell data to points for contour plotting
        if triangles is not None and len(original_cells) > 0:
            n_points = len(points)
            # Create mapping from original cells to data indices
            orig_cell_data_indices = list(range(len(original_cells)))
            
            sim_h_pt = cell_to_point_data(sim_h, original_cells, orig_cell_data_indices, n_points)
            sim_u_pt = cell_to_point_data(sim_u, original_cells, orig_cell_data_indices, n_points)
            sim_v_pt = cell_to_point_data(sim_v, original_cells, orig_cell_data_indices, n_points)
            deeponet_h_pt = cell_to_point_data(deeponet_h_pred, original_cells, orig_cell_data_indices, n_points)
            deeponet_u_pt = cell_to_point_data(deeponet_u_pred, original_cells, orig_cell_data_indices, n_points)
            deeponet_v_pt = cell_to_point_data(deeponet_v_pred, original_cells, orig_cell_data_indices, n_points)
            pi_h_pt = cell_to_point_data(pi_h_pred, original_cells, orig_cell_data_indices, n_points)
            pi_u_pt = cell_to_point_data(pi_u_pred, original_cells, orig_cell_data_indices, n_points)
            pi_v_pt = cell_to_point_data(pi_v_pred, original_cells, orig_cell_data_indices, n_points)
        
        # Calculate differences
        diff_deeponet_h = deeponet_h_pred - sim_h
        diff_deeponet_u = deeponet_u_pred - sim_u
        diff_deeponet_v = deeponet_v_pred - sim_v
        diff_deeponet_vel_mag = np.sqrt(deeponet_u_pred**2 + deeponet_v_pred**2) - np.sqrt(sim_u**2 + sim_v**2)
        
        diff_pi_h = pi_h_pred - sim_h
        diff_pi_u = pi_u_pred - sim_u
        diff_pi_v = pi_v_pred - sim_v
        diff_pi_vel_mag = np.sqrt(pi_u_pred**2 + pi_v_pred**2) - np.sqrt(sim_u**2 + sim_v**2)
        
        # Interpolate differences to points
        if triangles is not None and len(original_cells) > 0:
            diff_deeponet_h_pt = cell_to_point_data(diff_deeponet_h, original_cells, orig_cell_data_indices, n_points)
            diff_deeponet_u_pt = cell_to_point_data(diff_deeponet_u, original_cells, orig_cell_data_indices, n_points)
            diff_deeponet_v_pt = cell_to_point_data(diff_deeponet_v, original_cells, orig_cell_data_indices, n_points)
            diff_deeponet_vel_mag_pt = cell_to_point_data(diff_deeponet_vel_mag, original_cells, orig_cell_data_indices, n_points)
            diff_pi_h_pt = cell_to_point_data(diff_pi_h, original_cells, orig_cell_data_indices, n_points)
            diff_pi_u_pt = cell_to_point_data(diff_pi_u, original_cells, orig_cell_data_indices, n_points)
            diff_pi_v_pt = cell_to_point_data(diff_pi_v, original_cells, orig_cell_data_indices, n_points)
            diff_pi_vel_mag_pt = cell_to_point_data(diff_pi_vel_mag, original_cells, orig_cell_data_indices, n_points)

        # Compute the maximum and minimum of the velocity magnitude
        vel_mag_max = max(np.sqrt(sim_u**2 + sim_v**2).max(), np.sqrt(deeponet_u_pred**2 + deeponet_v_pred**2).max(), np.sqrt(pi_u_pred**2 + pi_v_pred**2).max())
        vel_mag_min = min(np.sqrt(sim_u**2 + sim_v**2).min(), np.sqrt(deeponet_u_pred**2 + deeponet_v_pred**2).min(), np.sqrt(pi_u_pred**2 + pi_v_pred**2).min())

        vel_mag_levels = np.linspace(vel_mag_min, vel_mag_max, 20)
        
        # Figure 1: Results comparison (3 columns)
        fig1, axs1 = plt.subplots(1, 3, figsize=(20, 6), facecolor='w')
        
        # Row 0: SRH-2D simulation
        if triangles is not None:            
            # Velocity vector (quiver plot on contour of magnitude)
            vel_mag_sim = np.sqrt(sim_u_pt**2 + sim_v_pt**2)
            cf = axs1[0].tricontourf(points[:, 0], points[:, 1], triangles, vel_mag_sim, levels=vel_mag_levels, cmap=plt.cm.RdBu_r)
            # Subsample for quiver
            #step = max(1, len(points) // 500)
            step = 2
            axs1[0].quiver(points[::step, 0], points[::step, 1], 
                             sim_u_pt[::step], sim_v_pt[::step], 
                             scale=10, width=0.002, color='white', alpha=0.6)
            axs1[0].plot(boundary_outline[:, 0], boundary_outline[:, 1], 'k-', linewidth=1.0)
            axs1[0].set_title('SRH-2D', fontsize=28)
            axs1[0].set_xticks([])
            axs1[0].set_yticks([])
            axs1[0].set_aspect('equal')
            axs1[0].set_xlim([18500, 21240])
            axs1[0].set_ylim([-7036, -5358])
            cbar = plt.colorbar(cf, ax=axs1[0], fraction=0.046, pad=0.04, shrink=0.5)
            cbar.ax.tick_params(labelsize=24)
            cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
            cbar.ax.text(0.5, 1.05, '(m/s)', fontsize=24, transform=cbar.ax.transAxes, horizontalalignment='center', verticalalignment='bottom')
        
        # Row 1: SWE-DeepONet
        if triangles is not None:
            
            # Velocity vector
            vel_mag_deeponet = np.sqrt(deeponet_u_pt**2 + deeponet_v_pt**2)
            cf = axs1[1].tricontourf(points[:, 0], points[:, 1], triangles, vel_mag_deeponet, levels=vel_mag_levels, cmap=plt.cm.RdBu_r)
            axs1[1].quiver(points[::step, 0], points[::step, 1], 
                             deeponet_u_pt[::step], deeponet_v_pt[::step], 
                             scale=10, width=0.002, color='white', alpha=0.6)
            axs1[1].plot(boundary_outline[:, 0], boundary_outline[:, 1], 'k-', linewidth=1.0)
            axs1[1].set_title('SWE-DeepONet', fontsize=28)
            axs1[1].set_xticks([])
            axs1[1].set_yticks([])
            axs1[1].set_aspect('equal')
            axs1[1].set_xlim([18500, 21240])
            axs1[1].set_ylim([-7036, -5358])
            cbar = plt.colorbar(cf, ax=axs1[1], fraction=0.046, pad=0.04, shrink=0.5)
            cbar.ax.tick_params(labelsize=26)
            cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
            cbar.ax.text(0.5, 1.05, '(m/s)', fontsize=26, transform=cbar.ax.transAxes, horizontalalignment='center', verticalalignment='bottom')
        
        # Row 2: PI-SWE-DeepONet
        if triangles is not None:
            
            # Velocity vector
            vel_mag_pi = np.sqrt(pi_u_pt**2 + pi_v_pt**2)
            cf = axs1[2].tricontourf(points[:, 0], points[:, 1], triangles, vel_mag_pi, levels=vel_mag_levels, cmap=plt.cm.RdBu_r)
            axs1[2].quiver(points[::step, 0], points[::step, 1], 
                             pi_u_pt[::step], pi_v_pt[::step], 
                             scale=10, width=0.002, color='white', alpha=0.6)
            axs1[2].plot(boundary_outline[:, 0], boundary_outline[:, 1], 'k-', linewidth=1.0)
            axs1[2].set_title('PI-SWE-DeepONet', fontsize=28)
            axs1[2].set_xticks([])
            axs1[2].set_yticks([])
            axs1[2].set_aspect('equal')
            axs1[2].set_xlim([18500, 21240])
            axs1[2].set_ylim([-7036, -5358])
            cbar = plt.colorbar(cf, ax=axs1[2], fraction=0.046, pad=0.04, shrink=0.5)
            cbar.ax.tick_params(labelsize=26)
            cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
            cbar.ax.text(0.5, 1.05, '(m/s)', fontsize=26, transform=cbar.ax.transAxes, horizontalalignment='center', verticalalignment='bottom')
        
        plt.tight_layout()
        #adjust the horizontal spacing between the subplots
        plt.subplots_adjust(wspace=0.2)
        plt.savefig(f'compare_test_case_results_{test_case_index}_vector.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

       

if __name__ == "__main__":

    # Test case indices
    test_case_indices = [11]

    # Test case result directories
    deeponet_result_dir = '../../SacramentoRiver_steady_DeepONet/data/DeepONet/test'
    pi_deeponet_result_dir = '../../SacramentoRiver_steady_PI_DeepONet/data/DeepONet/test'    

    plot_compare_test_case_results(test_case_indices, deeponet_result_dir, pi_deeponet_result_dir)

    print("Plotting comparison of test case results completed.")