import numpy as np
import pandas as pd
from vtk import vtkUnstructuredGrid, vtkPoints, vtkTriangle, vtkQuad, vtkUnstructuredGridWriter, vtkPolyData, vtkFloatArray, vtkPolyDataWriter
import pyproj

def transform_coordinates(x, y, from_epsg, to_epsg):
    """
    Transform coordinates from one coordinate system to another using pyproj
    """
    transformer = pyproj.Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    x_new, y_new = transformer.transform(x, y)
    return x_new, y_new

def read_2dm(filename):
    nodes = []
    elements = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            if parts[0] == 'ND':  # Node definition
                node_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                nodes.append([node_id, x, y, z])
                
            elif parts[0] == 'E3T':  # Triangle element
                elem_id = int(parts[1])
                n1 = int(parts[2])
                n2 = int(parts[3])
                n3 = int(parts[4])
                mat = int(parts[5])
                elements.append(['triangle', elem_id, [n1, n2, n3], mat])
                
            elif parts[0] == 'E4Q':  # Quadrilateral element
                elem_id = int(parts[1])
                n1 = int(parts[2])
                n2 = int(parts[3])
                n3 = int(parts[4])
                n4 = int(parts[5])
                mat = int(parts[6])
                elements.append(['quad', elem_id, [n1, n2, n3, n4], mat])
    
    return nodes, elements

def create_vtk_mesh(nodes, elements, output_file):
    # Create points
    points = vtkPoints()
    node_dict = {node[0]: i for i, node in enumerate(nodes)}
    
    for node in nodes:
        points.InsertNextPoint(node[1], node[2], node[3])
    
    # Create unstructured grid
    grid = vtkUnstructuredGrid()
    grid.SetPoints(points)
    
    # Add elements
    for elem in elements:
        if elem[0] == 'triangle':
            triangle = vtkTriangle()
            for i, node_id in enumerate(elem[2]):
                triangle.GetPointIds().SetId(i, node_dict[node_id])
            grid.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())
            
        elif elem[0] == 'quad':
            quad = vtkQuad()
            for i, node_id in enumerate(elem[2]):
                quad.GetPointIds().SetId(i, node_dict[node_id])
            grid.InsertNextCell(quad.GetCellType(), quad.GetPointIds())
    
    # Write to file
    writer = vtkUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(grid)
    writer.Write()

def create_vtk_points(x, y, z, u, v, w, output_file):
    # Create points
    points = vtkPoints()
    for i in range(len(x)):
        points.InsertNextPoint(x[i], y[i], z[i])
    
    # Create velocity vector array (3 components)
    velocity_array = vtkFloatArray()
    velocity_array.SetName("Velocity")
    velocity_array.SetNumberOfComponents(3)  # Set as 3D vector
    
    # Combine velocity components into vector
    for i in range(len(u)):
        velocity_array.InsertNextTuple3(u[i], v[i], w[i])
    
    # Create polydata
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(velocity_array)
    
    # Write to file
    writer = vtkPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.Write()

def process_adcp_data(filename, scale_factor=1.0):
    """
    Process ADCP data and transform coordinates to match 2DM mesh scale
    """
    # Read the data file
    df = pd.read_csv(filename, delimiter='\t', skiprows=0)
    
    # Clean column names by removing any whitespace
    df.columns = df.columns.str.strip()
    
    # Print original coordinate range
    print("Original coordinate range:")
    print(f"X: {df['X(m)'].min():.2f} to {df['X(m)'].max():.2f}")
    print(f"Y: {df['Y(m)'].min():.2f} to {df['Y(m)'].max():.2f}")
    
    # Transform coordinates to match 2DM mesh scale
    x_new = (df['X(m)'].values - 613000) / scale_factor
    y_new = (df['Y(m)'].values - 4290000) / scale_factor
    
    # Update coordinates in dataframe
    df['X(m)'] = x_new
    df['Y(m)'] = y_new
    
    print("\nTransformed coordinate range:")
    print(f"X: {df['X(m)'].min():.2f} to {df['X(m)'].max():.2f}")
    print(f"Y: {df['Y(m)'].min():.2f} to {df['Y(m)'].max():.2f}")
    
    # Find indices where new cross sections start (D_along_Xs = 0)
    section_starts = df[df['D_along_Xs(m)'] == 0].index.tolist()
    section_starts.append(len(df))  # Add the end of the last section
    
    print(f"\nFound {len(section_starts)-1} cross sections")
    
    # Process each cross section
    for i in range(len(section_starts)-1):
        start_idx = section_starts[i]
        end_idx = section_starts[i+1]
        
        # Extract data for this cross section
        section_data = df.iloc[start_idx:end_idx]
        
        # Create output filename
        output_file = f'cross_section_{i:03d}.vtk'
        
        # Create VTK file
        create_vtk_points(
            section_data['X(m)'].values,
            section_data['Y(m)'].values,
            section_data['D_along_Xs(m)'].values,
            section_data['U(m/s)'].values,
            section_data['V(m/s)'].values,
            np.zeros(len(section_data)),  # W component is zero for 2D data
            output_file
        )
        
        print(f'Created {output_file}')

def convert_2dm_to_vtk(input_file, output_file):
    # Read 2DM file
    print(f"Reading {input_file}...")
    nodes, elements = read_2dm(input_file)
    
    print(f"Found {len(nodes)} nodes and {len(elements)} elements")
    
    # Create VTK mesh
    print(f"Creating VTK mesh...")
    create_vtk_mesh(nodes, elements, output_file)
    
    print(f"Successfully created {output_file}")

def main():
    # Convert 2DM mesh to VTK
    print("Converting 2DM mesh to VTK...")
    convert_2dm_to_vtk("sac_dsbtc6_hyd.2dm", "output_mesh.vtk")
    
    # Process ADCP data
    print("\nProcessing ADCP data...")
    process_adcp_data("ADCP_Data_all_42XS.dat")

if __name__ == "__main__":
    main()