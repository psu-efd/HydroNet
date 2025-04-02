import numpy as np
from vtk import vtkUnstructuredGrid, vtkPoints, vtkTriangle, vtkQuad, vtkUnstructuredGridWriter, vtkIntArray

def read_2dm(filename):
    nodes = []
    elements = []
    materials = []
    
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
    
    # Create array for material IDs
    material_ids = vtkIntArray()
    material_ids.SetName("MaterialID")
    material_ids.SetNumberOfComponents(1)
    
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
        
        # Add material ID for this element
        material_ids.InsertNextValue(elem[3])
    
    # Add material IDs to the grid
    grid.GetCellData().AddArray(material_ids)
    
    # Write to file
    writer = vtkUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(grid)
    writer.Write()

def convert_2dm_to_vtk(input_file, output_file):
    # Read 2DM file
    print(f"Reading {input_file}...")
    nodes, elements = read_2dm(input_file)
    
    print(f"Found {len(nodes)} nodes and {len(elements)} elements")
    
    # Create VTK mesh
    print(f"Creating VTK mesh...")
    create_vtk_mesh(nodes, elements, output_file)
    
    print(f"Successfully created {output_file}")

def convert_2dm_to_srh(input_2dm, output_geom, output_mat):
    # Read 2DM file
    nodes = []
    elements = []
    nodestrings = []  # Store nodestrings
    material_ids = set()  # Track unique material IDs
    all_ns_values = []  # Store all NS line values
    
    with open(input_2dm, 'r') as f:
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
                material_ids.add(mat)
                
            elif parts[0] == 'E4Q':  # Quadrilateral element
                elem_id = int(parts[1])
                n1 = int(parts[2])
                n2 = int(parts[3])
                n3 = int(parts[4])
                n4 = int(parts[5])
                mat = int(parts[6])
                elements.append(['quad', elem_id, [n1, n2, n3, n4], mat])
                material_ids.add(mat)
                
            elif parts[0] == 'NS':  # Nodestring definition
                # Skip the 'NS' part and convert remaining parts to integers
                node_values = [int(p) for p in parts[1:]]
                # Add all numbers to a single array
                all_ns_values.extend(node_values)

    print("all_ns_values: ", all_ns_values)

    # Process the 1D array to extract nodestrings
    i = 0
    while i < len(all_ns_values):
        # Look ahead to find the next negative number
        j = i
        while j < len(all_ns_values) and all_ns_values[j] >= 0:
            j += 1
        
        if j < len(all_ns_values):  # Found a negative number
            # The negative number is the last node ID
            # The number after it is the nodestring ID
            node_ids = all_ns_values[i:j]  # All nodes up to but not including the negative number
            node_ids.append(abs(all_ns_values[j]))  # Add the positive version of the negative node ID
            nodestring_id = all_ns_values[j + 1]  # The nodestring ID
            
            nodestrings.append([nodestring_id, node_ids])
            i = j + 2  # Move to the start of the next nodestring
        else:
            break  # No more negative numbers found

    print(f"Found {len(nodestrings)} nodestrings")
    print(f"nodestrings: {nodestrings}")
    
    # Write SRHGEOM file
    with open(output_geom, 'w') as f:
        # Write header
        f.write("SRHGEOM 30\n")
        f.write("Name \"SacRiver\"\n")
        f.write("\n")
        f.write("GridUnit \"Meters\" \n")

        # Write Elem section
        for elem in elements:
            if elem[0] == 'triangle':
                f.write(f"Elem {elem[1]} {' '.join(map(str, elem[2]))}\n")
            elif elem[0] == 'quad':
                f.write(f"Elem {elem[1]} {' '.join(map(str, elem[2]))}\n")
        
        # Write Node section        
        for node in nodes:
            f.write(f"Node {node[0]} {node[1]:.8e} {node[2]:.8e} {node[3]:.8e}\n")       
        
        
        # Write NodeString section
        for ns_id, node_ids in nodestrings:
            # Write first line with NodeString keyword
            f.write(f"NodeString {ns_id} {' '.join(map(str, node_ids[:10]))}\n")
            # Write continuation lines if needed (indented)
            for i in range(10, len(node_ids), 10):
                f.write(f" {' '.join(map(str, node_ids[i:i+10]))}\n")
    
    # Write SRHMAT file
    with open(output_mat, 'w') as f:
        f.write("SRHMAT\n")
        f.write(f"NMaterials {len(material_ids)}\n")
        
        # Write material names and IDs
        for mat_id in sorted(material_ids):
            f.write(f'MatName {mat_id} "material_{mat_id}"\n')
        
        # Write element IDs for each material
        for mat_id in sorted(material_ids):
            f.write(f"Material {mat_id} ")
            # Get all element IDs for this material
            elem_ids = [elem[1] for elem in elements if elem[3] == mat_id]
            # Write 10 elements per line
            for i in range(0, len(elem_ids), 10):
                if i > 0:
                    f.write("\n ")  # Indent continuation lines
                f.write(" ".join(map(str, elem_ids[i:i+10])))
            f.write("\n")

def main():
    input_2dm = "sac_dsbtc6_hyd.2dm"
    output_geom = "SacromentoRiver.srhgeom"
    output_mat = "SacromentoRiver.srhmat"

    convert_2dm_to_vtk(input_2dm, "output.vtk")
    
    print(f"Converting {input_2dm} to SRH format...")
    convert_2dm_to_srh(input_2dm, output_geom, output_mat)
    print(f"Created {output_geom} and {output_mat}")

if __name__ == "__main__":
    main()