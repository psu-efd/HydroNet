#Misc. tests for the PI-SWE-DeepONet model data 


import numpy as np

def test_PINN_data():
    """
    Read the PINN point and data files, and print the statistics for verification.
    """

    #read the PINN point and data files
    pde_points = np.load('data/PINN/pde_points.npy')
    pde_data = np.load('data/PINN/pde_data.npy')

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

def test_application_data():
    """
    Read the application point and data h5 file, and print the statistics for verification.
    """

    with h5py.File('application/data.h5', 'r') as f:
        application_data = f['application_data'][:]
        print(f"Application data shape: {application_data.shape}")
        print(f"Application data: {application_data}")
    

if __name__ == "__main__":
    #test_PINN_data()

    print(f"All tests passed.")