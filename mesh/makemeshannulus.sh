#!/bin/bash

base_name="channel_annulus"
fname="channel_annulus_6_3"

gmsh $base_name.geo -format msh2 -2 -o $fname.msh
dolfin-convert $fname.msh $fname.xml
