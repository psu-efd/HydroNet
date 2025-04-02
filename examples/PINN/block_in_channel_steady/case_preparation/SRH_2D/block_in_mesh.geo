// Parameters
L = 15.0;   // Channel length
W = 5.0;    // Channel width

lh = 2.0;   // Hole length
wh = 1.0;   // Hole width
xc = 5.5;   // Hole center x
yc = 2.5;   // Hole center y

// Mesh size
meshSize = 0.4;

// Channel outer boundary points (counter-clockwise)
Point(1) = {0, 0, 0, meshSize};
Point(2) = {L, 0, 0, meshSize};
Point(3) = {L, W, 0, meshSize};
Point(4) = {0, W, 0, meshSize};

// Hole boundary points (clockwise to define hole)
Point(5) = {xc - lh/2, yc - wh/2, 0, meshSize};  // Bottom-left
Point(6) = {xc + lh/2, yc - wh/2, 0, meshSize};  // Bottom-right
Point(7) = {xc + lh/2, yc + wh/2, 0, meshSize};  // Top-right
Point(8) = {xc - lh/2, yc + wh/2, 0, meshSize};  // Top-left

// Lines for outer boundary
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Lines for hole boundary
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

// Curve loops
Curve Loop(1) = {1, 2, 3, 4};            // Outer boundary
Curve Loop(2) = {5, 6, 7, 8};            // Hole

// Plane surface with hole
Plane Surface(1) = {1, 2};

// Mesh generation
Physical Surface("ChannelWithHole") = {1};
Physical Line("inlet") = {4};
Physical Line("outlet") = {2};
Physical Line("top") = {3};
Physical Line("bottom") = {1};
Physical Line("obstacle") = {5, 6, 7, 8};
