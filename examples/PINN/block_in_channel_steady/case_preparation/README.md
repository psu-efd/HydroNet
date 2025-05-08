# Instructions

In this folder, we prepare the files needed for PINN model training. The resulted files are in "generate_PINN_points/pinn_points", which can be copied to the PINN training case directory.

- Run the SRH-2D case in the folder "SRH-2D" and convert the results to VTK format using the Python scripts within. 
- Copy the SRH-2D VTK result (last time step is sufficient) to the folder "generate_PINN_points". In this case, the VTK file is "SRH2D_block_in_channel_C_0005.vtk".
- Within "generate_PINN_points" folder, run the "generate_points_for_PINN.py" script to generate the following:
    - PDE points (internal points) pde_points: pde_points.npy. Each row is the (x, y) coordinates of each point.
    - boundary points: boundary_points.npy and boundary_info.npy. The info file has boundary point's boundary ID, nx, ny, and represented length.
    - initial points (for unsteady problem only)
    - data points: data_points.npy, data_values.npy (h, u, v), data_flags.npy (flags for whether each variable should be used in loss calculation). In this case, we use SRH-2D simulation result as our data. An alternative is to use measurement data. If so, the creation of the data point files should be done with a new python function.