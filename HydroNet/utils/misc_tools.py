"""
This set of tools is used to for example generate random points on a triangle or quadrilateral.

"""

import numpy as np

def generate_random01_exclude_boundaries_with_center(centers, size=1):
    """Generate random numbers in (0, 1) which exclude the boundaries 0 and 1 and include the center (0.5)

    Parameters
    ----------
    centers : numpy.ndarray
        Array of center points to include in the output
    size : int
        Number of random number sets to generate

    Returns
    -------
    numpy.ndarray
        Array of shape (size, len(centers)) containing random numbers
    """

    # Small number to nudge the random point away from boundary
    epsilon = 0.05

    results = np.zeros((size, len(centers)), dtype=np.float32)
    results[0, :] = centers  # First set is the centers

    # Generate remaining sets
    for i in range(1, size):
        while True:
            current_set = np.random.random(len(centers))
            
            # Check if points are too close to boundaries or centers
            if len(centers) == 1:
                if (min(abs(current_set)) < epsilon or 
                    min(abs(current_set - centers)) < epsilon or 
                    min(abs(current_set - 1.0)) < epsilon):
                    continue
            elif len(centers) == 2:
                if (min(abs(current_set)) < epsilon or 
                    min(abs(current_set - centers)) < epsilon or 
                    min(abs(current_set - 1.0)) < epsilon or 
                    abs(current_set[0] - current_set[1]) < epsilon):
                    continue
            else:
                raise ValueError(f"Unsupported number of random numbers in a set: {len(centers)}")
            
            results[i, :] = current_set
            break

    return results

def point_on_triangle(pt1, pt2, pt3, s, t):
    """Calculate a point on a triangle using barycentric coordinates.

    Parameters
    ----------
    pt1, pt2, pt3 : numpy.ndarray
        Triangle vertices
    s, t : float
        Barycentric coordinates

    Returns
    -------
    tuple
        (x, y, z) coordinates of the point
    """
    s, t = sorted([s, t])  # Ensure s <= t
    return (s * pt1[0] + (t - s) * pt2[0] + (1 - t) * pt3[0],
            s * pt1[1] + (t - s) * pt2[1] + (1 - t) * pt3[1],
            s * pt1[2] + (t - s) * pt2[2] + (1 - t) * pt3[2])

def point_on_line(pt1, pt2, s):
    """Calculate a point on a line segment using linear interpolation.

    Parameters
    ----------
    pt1, pt2 : numpy.ndarray
        Line segment endpoints
    s : float
        Interpolation parameter in [0,1]

    Returns
    -------
    tuple
        (x, y, z) coordinates of the point
    """
    return (s * pt1[0] + (1 - s) * pt2[0],
            s * pt1[1] + (1 - s) * pt2[1],
            s * pt1[2] + (1 - s) * pt2[2])

