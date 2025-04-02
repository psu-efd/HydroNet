# Create rating curve for a given nodestring, for example the exit of the Sacramento River
# This script reads the SRHGEOM file and computes the rating curve for a given nodestring
# The rating curve is computed using Manning's equation
# The results are saved to a file and plotted

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rc('text', usetex=False)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def read_srhgeom(filename):
    """Read SRHGEOM file and return nodes and nodestrings."""
    nodes = {}  # Dictionary to store node coordinates
    nodestrings = {}  # Dictionary to store nodestrings

    #check if the file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
        return None, None
    
    current_ns_id = None
    
    with open(filename, 'r') as f:
        current_section = None
        for line in f:
            if not line:
                continue

            line = line.strip()
            parts = line.split()

            if len(parts) == 0:
                continue

            #print("parts: ", parts)
            #print("parts[0]: ", parts[0])
                
            if parts[0] == "Node":
                current_section = "Node"                
            elif parts[0] == "NodeString":
                current_section = "NodeString"
            elif parts[0] == "Elem":
                current_section = "Elem"
                
            if current_section == "Node":
                # Node format: Node ID x y z
                
                #print("parts: ", parts)
                node_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                nodes[node_id] = (x, y, z)

                #print("node_id: ", node_id, nodes[node_id])
                
            elif current_section == "NodeString":
                # First line: NodeString ID node1 node2 ...
                if line.startswith("NodeString"):
                    #parts = line.split()
                    current_ns_id = int(parts[1])
                    node_ids = [int(p) for p in parts[2:]]
                    nodestrings[current_ns_id] = node_ids
                else:
                    # Continuation line: node1 node2 ...
                    #parts = line.split()
                    node_ids = [int(p) for p in parts]

                    if current_ns_id is not None:
                        nodestrings[current_ns_id].extend(node_ids)
    
    return nodes, nodestrings

def compute_rating_curve(nodes, nodestring_id, nodestrings, 
                        wse_range, manning_n, slope,
                        output_file=None):
    """
    Compute rating curve for a given nodestring using Manning's equation.
    
    Parameters:
    -----------
    nodes : dict
        Dictionary mapping node IDs to (x, y, z) coordinates
    nodestring_id : int
        ID of the nodestring to analyze
    nodestrings : dict
        Dictionary mapping nodestring IDs to lists of node IDs
    wse_range : array-like
        Array of water surface elevations to compute discharge for
    manning_n : float
        Manning's roughness coefficient
    slope : float
        Channel slope (dimensionless)
    output_file : str, optional
        If provided, save results to this file
    
    Returns:
    --------
    wse : array
        Water surface elevations
    discharge : array
        Corresponding discharges
    XS_distances : array
        Distance along the nodestring
    XS_elevations : array
        Elevations along the nodestring
    """
    import numpy as np
    
    # Get node IDs for the specified nodestring
    node_ids = nodestrings[nodestring_id]
    
    # Extract coordinates and bed elevations
    x_coords = []
    y_coords = []
    bed_elevations = []
    for node_id in node_ids:
        x, y, z = nodes[node_id]
        x_coords.append(x)
        y_coords.append(y)
        bed_elevations.append(z)
    
    # Convert to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    bed_elevations = np.array(bed_elevations)
    
    # Compute distances between consecutive nodes
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    node_distances = np.sqrt(dx**2 + dy**2)
    cumulative_distances = np.concatenate(([0], np.cumsum(node_distances)))
    
    # Convert wse_range to numpy array if it isn't already
    wse = np.array(wse_range)
    
    # Initialize discharge array
    discharge = np.zeros_like(wse)
    
    # Compute discharge for each water surface elevation
    for i, water_level in enumerate(wse):
        # Compute flow depth at each node
        depths = water_level - bed_elevations
        depths = np.maximum(depths, 0)  # Only consider positive depths
        
        # Compute cross-sectional area using trapezoidal rule
        # Area between consecutive nodes = (d1 + d2) * L / 2
        # where d1, d2 are depths and L is distance between nodes
        areas = 0.5 * (depths[:-1] + depths[1:]) * node_distances
        area = np.sum(areas)
        
        # Compute wetted perimeter
        # For each segment: P = sqrt(L^2 + (d2-d1)^2)
        # where L is horizontal distance and d2-d1 is vertical difference
        wetted_lengths = np.sqrt(node_distances**2 + np.diff(depths)**2)
        wetted_perimeter = np.sum(wetted_lengths)
        
        # Compute hydraulic radius
        hydraulic_radius = area / wetted_perimeter if wetted_perimeter > 0 else 0
        
        # Manning's equation: Q = (1/n) * A * R^(2/3) * S^(1/2)
        discharge[i] = (1/manning_n) * area * (hydraulic_radius)**(2/3) * np.sqrt(slope)
    
    # Save results if output file is specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write("Rating_Curve\n")
            f.write("//For nodestring {}\n".format(nodestring_id))
            f.write("// Discharge (m3/s)   WSE (m)\n")
            for q, w in zip(discharge, wse):
                f.write(f"{q:.3f} {w:.3f}\n")
    
    return wse, discharge, cumulative_distances, bed_elevations

def plot_rating_curve(wse, discharge, example_discharge, example_wse, title=None):
    """Plot the rating curve."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(discharge, wse, 'b-', linewidth=2, label='Rating Curve')
    plt.plot(example_discharge, example_wse, 'ro', label='Example Discharge')
    plt.grid(True)
    plt.xlabel('Discharge (m³/s)')
    plt.ylabel('Water Surface Elevation (m)')
    plt.legend()
    if title:
        plt.title(title)

    # save the plot
    plt.savefig("exit_rating_curve.png", dpi=300, bbox_inches='tight')
    #plt.show()

def plot_cross_section(distances, bed_elevations, water_level=None, title=None):
    """Plot the cross-section with optional water level."""
    plt.figure(figsize=(10, 6))
    plt.plot(distances, bed_elevations, 'k-', linewidth=2, label='Bed')
    if water_level is not None:
        plt.axhline(y=water_level, color='b', linestyle='--', label='Water Level')
    plt.grid(True)
    plt.xlabel('Distance (m)', fontsize=14)
    plt.ylabel('Elevation (m)', fontsize=14)
    if title:
        plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig("exit_cross_section.png", dpi=300, bbox_inches='tight')
    #plt.show()

# Example usage:
if __name__ == "__main__":
    # Read the SRHGEOM file
    nodes, nodestrings = read_srhgeom("SacromentoRiver.srhgeom")

    #print("nodes: ", nodes)
    #print("nodestrings: ", nodestrings)
    
    # Parameters for rating curve computation
    nodestring_id = 2  # Specify the nodestring to analyze
    wse_range = np.linspace(-8.9, 10, 100)  # Water surface elevations from 0 to 10m
    manning_n = 0.026  # Manning's roughness coefficient
    slope = 0.0000015  # Channel slope

    # One discharge value (from the original case setup from USBR which has 7 inlets and field data)
    all_inlet_discharges = [267.56, 30.3, 128.35, 74.84, 7.76, 36.83, 9.45]
    example_discharge = sum(all_inlet_discharges)
    example_wse = 2.201
    
    # Compute rating curve
    wse, discharge, cumulative_distances, bed_elevations = compute_rating_curve(nodes, nodestring_id, nodestrings,
                                        wse_range, manning_n, slope,
                                        output_file="exit_rating_curve.xys")
    
    # Plot the rating curve
    plot_rating_curve(wse, discharge, example_discharge, example_wse,
                     title=f"Rating Curve for Nodestring {nodestring_id}")
    
    # Plot the cross-section
    plot_cross_section(cumulative_distances, bed_elevations,
                      title=f"Cross-Section for Nodestring {nodestring_id}")
    
    print("Done!")
