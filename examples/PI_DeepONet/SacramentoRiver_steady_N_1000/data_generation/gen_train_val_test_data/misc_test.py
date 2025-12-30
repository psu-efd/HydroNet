#Misc. tests for the PI-SWE-DeepONet model data 


import numpy as np
import json
import vtk
from vtk import vtkPoints, vtkPolyData, vtkPolyDataWriter

def test_PINN_data():
    """
    Read the PINN point and data files, and print the statistics for verification.
    """

    #read the PINN point and data files
    pde_points = np.load('data/PINN/pde_points.npy')
    pde_data = np.load('data/PINN/pde_data.npy')

    #read the stat file
    stats_file = 'data/PINN/all_PINN_stats.json'
    with open(stats_file, 'r') as f:
        pinn_stats = json.load(f)
        print(f"Stats: {pinn_stats}")

    x_min = pinn_stats['all_points_stats']['x_min']
    x_max = pinn_stats['all_points_stats']['x_max']    
    y_min = pinn_stats['all_points_stats']['y_min']
    y_max = pinn_stats['all_points_stats']['y_max']

    #denormalize the points
    pde_points[:, 0] = x_min + (x_max - x_min) * pde_points[:, 0]
    pde_points[:, 1] = y_min + (y_max - y_min) * pde_points[:, 1]

    #print the statistics for verification
    print(f"Number of PDE points: {len(pde_points)}")
    print(f"Number of PDE data: {len(pde_data)}")
    print(f"PDE points shape: {pde_points.shape}")
    print(f"PDE data shape: {pde_data.shape}")
    print(f"PDE points-x-min: {np.min(pde_points[:, 0])}")
    print(f"PDE points-x-max: {np.max(pde_points[:, 0])}")
    print(f"PDE points-y-min: {np.min(pde_points[:, 1])}")
    print(f"PDE points-y-max: {np.max(pde_points[:, 1])}")
    print(f"PDE data-zb-min: {np.min(pde_data[:, 0])}")
    print(f"PDE data-zb-max: {np.max(pde_data[:, 0])}")
    print(f"PDE data-Sx-min: {np.min(pde_data[:, 1])}")
    print(f"PDE data-Sx-max: {np.max(pde_data[:, 1])}")
    print(f"PDE data-Sy-min: {np.min(pde_data[:, 2])}")
    print(f"PDE data-Sy-max: {np.max(pde_data[:, 2])}")
    print(f"PDE data-n-min: {np.min(pde_data[:, 3])}")
    print(f"PDE data-n-max: {np.max(pde_data[:, 3])}")

    #Save the denormalized points to a vtk file so they can be visualized
    vtk_file = 'pde_points_denormalized_verification.vtk'
    points = np.concatenate([pde_points, np.zeros((len(pde_points), 1))], axis=1)

    # Create VTK points object
    vtk_points = vtkPoints()
    for i in range(len(points)):
        vtk_points.InsertNextPoint(points[i][0], points[i][1], points[i][2])
    
    # Create polydata and add points
    polydata = vtkPolyData()
    polydata.SetPoints(vtk_points)
    
    # Write to file
    writer = vtkPolyDataWriter()
    writer.SetFileName(vtk_file)
    writer.SetInputData(polydata)
    writer.Write()

    print(f"Saved the denormalized points to {vtk_file}")

def test_application_data():
    """
    Read the application point and data h5 file, and print the statistics for verification.
    """

    with h5py.File('application/data.h5', 'r') as f:
        application_data = f['application_data'][:]
        print(f"Application data shape: {application_data.shape}")
        print(f"Application data: {application_data}")
    

if __name__ == "__main__":
    test_PINN_data()

    print(f"All tests passed.")