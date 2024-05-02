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

// radius of circle
R = 0.6;

////////////////////////////////////////////////////////////

// Create all of the points

// Points for the circle
Point(1) = {-R, 0, 0, esc};
Point(2) = {0, 0, 0, esa};
Point(3) = {R, 0, 0, esc};

// Points for the domain corners
Point(4) = {L/2, 0, 0, es};
Point(5) = {L/2, H, 0, es};
Point(6) = {0, H, 0, esa};
Point(7) = {-L/2, H, 0, es};
Point(8) = {-L/2, 0, 0, es};

// Create circle and lines
Circle(1) = {3, 2, 1};

Line(2) = {1, 8};
Line(3) = {8, 7};
Line(4) = {7, 6};
Line(5) = {6, 5};
Line(6) = {5, 4};
Line(7) = {4, 3};

Curve Loop(1) = {1:7};
Plane Surface(1) = {1};

// now let's add the circle to the domain and mesh it

Line(8) = {1, 2};
Line(9) = {2, 3};

Curve Loop(2) = {8, 9, 1};
Plane Surface(2) = {2};

// create physical lines (for Fenics)

// circle
Physical Curve(1) = {1};

// axis for fluid
Physical Curve(2) = {2, 7};

// inlet
Physical Curve(3) = {3};

// output
Physical Curve(4) = {6};

// channel wall
Physical Curve(5) = {4, 5};

// axis for solid
Physical Curve(6) = {8, 9};


// bulk (fluid)
Physical Surface(10) = {1};

// bulk (solid)
Physical Surface(11) = {2};
