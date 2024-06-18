#!/bin/bash
# Execute this file to recompile locally
c++ -Wall -shared -fPIC -std=c++11 -O3 -fno-math-errno -fno-trapping-math -ffinite-math-only -I/home/simon/anaconda3/envs/fenics_old/include -I/home/simon/anaconda3/envs/fenics_old/include/eigen3 -I/home/simon/anaconda3/envs/fenics_old/.cache/dijitso/include dolfin_expression_c35a94dbbdb724cf115f2512abefb019.cpp -L/home/simon/anaconda3/envs/fenics_old/lib -L/home/conda/feedstock_root/build_artifacts/fenics-pkgs_1696906530109/_build_env/x86_64-conda-linux-gnu/sysroot/usr/lib -L/home/simon/anaconda3/envs/fenics_old/home/simon/anaconda3/envs/fenics_old/lib -L/home/simon/anaconda3/envs/fenics_old/.cache/dijitso/lib -Wl,-rpath,/home/simon/anaconda3/envs/fenics_old/.cache/dijitso/lib -lmpi -lpetsc -lslepc -lm -ldl -lz -lsz -lpthread -lcurl -lcrypto -lhdf5 -lboost_timer -ldolfin -olibdijitso-dolfin_expression_c35a94dbbdb724cf115f2512abefb019.so