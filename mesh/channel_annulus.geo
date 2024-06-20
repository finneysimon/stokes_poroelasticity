///////////////////////////////////////////////////////////////////
// Gmsh file for creating a finite element mesh
// In this case, we consider a sphere of radius R in a channel
// For a great tutorial on using Gmsh, see
// https://www.youtube.com/watch?v=aFc6Wpm69xo
///////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

// element size for the fluid
es = 8e-2;

// element size at the fluid-solid interface
esc = 10e-3;

// element size for the solid
esa = 4e-2;

// length and half-width of the channel
L = 20;
H = 1;

// radius of circle (outer & inner)
R1 = 0.60;
R0 = 0.30;

////////////////////////////////////////////////////////////

// Create all of the points

// Points for the circle (outer)
Point(1) = {-R1, 0, 0, esc};
Point(2) = {0, 0, 0, esa};
Point(3) = {R1, 0, 0, esc};

// Points for the domain corners
Point(4) = {L/2, 0, 0, es};
Point(5) = {L/2, H, 0, es};
Point(6) = {0, H, 0, esa};
Point(7) = {-L/2, H, 0, es};
Point(8) = {-L/2, 0, 0, es};

// Points for the circle (inner)
Point(9) = {-R0, 0, 0, esc};
Point(10) = {R0, 0, 0, esc};

// Create circle and lines
Circle(1) = {3, 2, 1};
Circle(12) = {10, 2, 9};

Line(2) = {1, 8};
Line(3) = {8, 7};
Line(4) = {7, 6};
Line(5) = {6, 5};
Line(6) = {5, 4};
Line(7) = {4, 3};

Line(8) = {9, 1};
Line(9) = {3, 10};
Line(10) = {10, 2};
Line(11) = {2, 9};

Curve Loop(1) = {1:7};  // Stokes flow
Curve Loop(2) = {1, -8, -12, -9};  // Annulus
Curve Loop(3) = {12, -11, -10};  // Cell

Plane Surface(1) = {1};  // Stokes flow
Plane Surface(2) = {2};  // Annulus
Plane Surface(3) = {3};  // Cell

// create physical lines (for Fenics)

// circle outer
Physical Curve(1) = {1};

// axis fluid
Physical Curve(2) = {2, 7};

// inlet
Physical Curve(3) = {3};

// outlet
Physical Curve(4) = {6};

// channel wall
Physical Curve(5) = {4, 5};

// axis solid
Physical Curve(6) = {8, 9};

// circle inner
Physical Curve(7) = {12};

// Axis Cell
Physical Curve(8) = {10, 11};

// bulk fluid
Physical Surface(10) = {1};

// bulk porous
Physical Surface(11) = {2};

// Bulk Cell
Physical Surface(12) = {3};