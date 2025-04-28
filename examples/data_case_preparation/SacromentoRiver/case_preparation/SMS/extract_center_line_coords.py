# This script extracts the center line coordinates from a shapefile and saves them as a CSV file
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

# Load the shapefile
shp_file_path = "main_channel_thalweg_line.shp"
gdf = gpd.read_file(shp_file_path)

print(gdf.head())

# Extract coordinates from the LineString geometry
# Get the first (and only) LineString geometry
line = gdf.geometry.iloc[0]

# Extract coordinates from the LineString
coords = np.array(line.coords)[:, :2]  # Only use x,y coordinates

# Create a DataFrame with the coordinates (x,y), not z (the shp file has no elevation data)
df = pd.DataFrame(coords, columns=['x', 'y'])

# Compute cumulative length along the line
lengths = [0.0]  # Start with 0 for the first point
for i in range(1, len(coords)):
    # Calculate distance between consecutive points
    dist = euclidean(coords[i-1][:2], coords[i][:2])  # Only use x,y coordinates for distance
    lengths.append(lengths[-1] + dist)

# Add length column to DataFrame
df['cumulative_length'] = lengths

# Save to CSV
output_file = "center_line_coordinates.csv"
df.to_csv(output_file, index=False)

print(f"Coordinates saved to {output_file}")
print(f"Number of points: {len(df)}")
print(f"Total line length: {lengths[-1]:.2f} units")
print("\nFirst few coordinates with lengths:")
print(df.head())
