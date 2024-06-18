#!/bin/bash
# Execute this file to recompile locally
c++ -Wall -shared -fPIC -std=c++11 -O2 -I/home/simon/anaconda3/envs/fenics_old/include -I/home/simon/anaconda3/envs/fenics_old/include/eigen3 -I/home/simon/anaconda3/envs/fenics_old/.cache/dijitso/include ffc_form_f5c58d5ed09c5f47cdbcd21502d3bd27f46bfb05.cpp -L/home/simon/anaconda3/envs/fenics_old/.cache/dijitso/lib -Wl,-rpath,/home/simon/anaconda3/envs/fenics_old/.cache/dijitso/lib -ldijitso-ffc_element_af47295a80519833ea33236567c095afa8490943 -ldijitso-ffc_element_de8439ade25e15ca5629b7c8b5e041c4eaec6481 -ldijitso-ffc_element_d75e45a35a43f90e04727899b6a1ccaba687e9d5 -ldijitso-ffc_element_d813efd86d1269ffed6166a5f2febcbe484faa4d -ldijitso-ffc_element_b6056e9c39d9d0154897eac8d86c6d3a5d1f55b9 -ldijitso-ffc_coordinate_mapping_fdf65ad1b4ee585aa7f358c9d8ed7cd18fb0ebf8 -olibdijitso-ffc_form_f5c58d5ed09c5f47cdbcd21502d3bd27f46bfb05.so