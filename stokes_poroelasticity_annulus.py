"""
This is a monolithic solver for pressure-driven flow around a
poroelastic particle.

The problem is formulated using the ALE
method, which maps the deformed geometry to the initial
geometry.  The problem is solved using the
initial geometry; the deformed geometry can by using the
WarpByVector filter in Paraview using the displacement
computed in the fluid domain.

The problem uses Lagrange multipliers to ensure zero mean fluid
pressure and zero net z-displacement

The code works by initially solving the problem with a small
value of epsilon (ratio of fluid stress to elastic stiffness)
and then gradually ramping up epsilon.  If convergence is
not obtained then the code tries again using a smaller value
of epsilon.

This code is 2-D (not axisymmetric)

"""

from dolfin import *
from multiphenics import *
from helpers import *
import matplotlib.pyplot as plt
from csv import writer
import numpy as np


# writing to csvfile
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


def write_list_as_row(file_name, list_of_elem):
    with open(file_name, 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


# ---------------------------------------------------------------------
# Setting up file names and paramerers
# ---------------------------------------------------------------------
# directory for file output
dir = '/home/simon/data/fenics/stokes_poroelasticity/'


def generate_output_files(rad_out, rad_in):
    output_s = XDMFFile(dir + "poro_2d_solid_0-"
                        + str(float('%.2g' % rad_out))[2:] + "_" + str(float('%.2g' % rad_in))[2:] + ".xdmf")
    output_s.parameters['rewrite_function_mesh'] = False
    output_s.parameters["functions_share_mesh"] = True
    output_s.parameters["flush_output"] = True

    output_f = XDMFFile(dir + "poro_2d_fluid_0-"
                        + str(float('%.2g' % rad_out))[2:] + "_" + str(float('%.2g' % rad_in))[2:] + ".xdmf")
    output_f.parameters['rewrite_function_mesh'] = False
    output_f.parameters["functions_share_mesh"] = True
    output_f.parameters["flush_output"] = True

    output_c = XDMFFile(dir + "poro_2d_cell_0-"
                        + str(float('%.2g' % rad_out))[2:] + "_" + str(float('%.2g' % rad_in))[2:] + ".xdmf")
    output_c.parameters['rewrite_function_mesh'] = False
    output_c.parameters["functions_share_mesh"] = True
    output_c.parameters["flush_output"] = True

    return output_s, output_f, output_c


def get_mesh(rad_out, rad_in):
    meshname = 'channel_annulus_' + str(float('%.1g' % rad_out))[2:] + "_" + str(float('%.1g' % rad_in))[2:]
    print(meshname)
    mesh = Mesh('mesh/' + meshname + '.xml')
    subdomains = MeshFunction("size_t", mesh, 'mesh/' + meshname + '_physical_region.xml')
    bdry = MeshFunction("size_t", mesh, 'mesh/'
                        + meshname + '_facet_region.xml')
    return mesh, subdomains, bdry


"""
Solver parameters
"""
snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "absolute_tolerance": 1e-8,
                                          "error_on_nonconvergence": False}}

parameters["ghost_mode"] = "shared_facet"

for ii in range(1):
    rad_out = (ii + 6) / 10
    rad_in = 3 / 10
    print("doing radii = ", rad_out, " ", rad_in)

    output_s, output_f, output_c = generate_output_files(rad_out, rad_in)

    # mesh has been created with Gmsh
    mesh, subdomains, bdry = get_mesh(rad_out, rad_in)

    """
        Physical parameters
    """

    # the initial value of epsilon to try solving the problem with
    eps_try = 0.005
    eps = Constant(eps_try)

    # the max value of epsilon
    eps_max = 0.15

    # the incremental increase in epsilon
    de = 0.005

    # the min and max values of the increments to make.
    de_min = 1e-3
    de_max = 1e-1

    # physical parameters
    lambda_s = 1  # lambda = 2*nu/(1-2*nu)
    phi_f_0 = 1 - 0.5
    k0_try = 1.e-2
    kappa_0 = Constant(k0_try)
    g0_try = 1.
    gamma = Constant(g0_try)

    # parameter continuation in kappa
    dk0 = 1.e-3
    k0_conv = 0
    dk0_min = 1.e-5
    k0_max = 1.e-2

    # parameter continuation in gamma
    dg0 = 1.
    g0_conv = 0
    dg0_min = 1.e-1
    g0_max = 100

    # define the boundaries (values from the gmsh file)
    circle_outer = 1
    fluid_axis = 2
    inlet = 3
    outlet = 4
    wall = 5
    solid_axis = 6
    circle_inner = 7
    cell_axis = 8


    # define the domains
    fluid = 10
    solid = 11
    cell = 12

    Of = generate_subdomain_restriction(mesh, subdomains, fluid)
    Os = generate_subdomain_restriction(mesh, subdomains, solid)
    Oc = generate_subdomain_restriction(mesh, subdomains, cell)
    Sig_1 = generate_interface_restriction(mesh, subdomains, {fluid, solid})
    Sig_0 = generate_interface_restriction(mesh, subdomains, {solid, cell})

    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
    ds = Measure("ds", domain=mesh, subdomain_data=bdry)
    dS = Measure("dS", domain=mesh, subdomain_data=bdry)
    dS1 = dS(circle_outer)
    dS0 = dS(circle_inner)

    # normal and tangent vectors
    nn = FacetNormal(mesh);
    tt = as_vector((-nn[1], nn[0]))

    ez = as_vector([1, 0])
    ex = as_vector([0, 1])

    # define the surface differential on the circle
    x = SpatialCoordinate(mesh)
    r = x[1]

    # ---------------------------------------------------------------------
    # elements, function spaces, and test/trial functions
    # ---------------------------------------------------------------------
    V2 = VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    DGT = VectorElement("CG", mesh.ufl_cell(), 1)
    DGT_FE = FiniteElement("CG", mesh.ufl_cell(), 1)
    P0 = FiniteElement("R", mesh.ufl_cell(), 0)

    """
    Setting up the elements and solution: here is the notation:

    u_f: fluid velocity from Stokes equations
    p_f: fluid pressure from Stokes equations

    u_s: solid displacement from compressible nonlinear elasticity
    p_p: Darcy pressure

    f_0: Lagrange multiplier corresponding to the force needed to pin the
    solid in place (should end up being zero since particle is free)

    U_0: the translational velocity of the solid

    lam: Lagrange multiplier corresponding to the fluid traction 
    acting on the solid

    lam_2: Lagrange multiplier corresponding to the normal fluid flux

    lam_p: Lagrange multiplier to ensure the mean fluid pressure is zero

    u_a: fluid "displacement" from the ALE method (e.g. how to deform
    the fluid geometry)

    lam_a: Lagrange multiplier to ensure continuity of fluid and solid
    displacement (ensures compatibility between fluid/solid domains)

    u_c: ALE displacement for cell
    
    p_c: cell pressure - used as lagrange multiplier to fix cell volume
    
    lam_c: displacement continuity on cell surface
    
    lam_3: lagrange multiplier corresponding to solid traction on inner surface
    
    lam_4: lagrange multiplier for normal velocity - should equal zero 
    
    """
    mixed_element = BlockElement(V2, P1, V2, P1, P0, P0, DGT, DGT_FE, P0, V2, DGT, V2, P0, DGT, DGT, DGT_FE)
    V = BlockFunctionSpace(mesh, mixed_element,
                           restrict=[Of, Of, Os, Os, Os, Os, Sig_1, Sig_1, Of, Of, Sig_1, Oc, Oc, Sig_0, Sig_0, Sig_0])

    X = BlockFunction(V)
    (u_f, p_f, u_s, p_p, f_0, U_0, lam, lam_2, lam_p, u_a, lam_a, u_c, p_c, lam_c, lam_3, lam_4) = block_split(X)

    # unknowns and test functions
    Y = BlockTestFunction(V)
    (v_f, q_f, v_s, q_p, g_0, V_0, eta, eta_2, eta_p, v_a, eta_a, v_c, q_c, eta_c, eta_3, eta_4) = block_split(Y)

    Xt = BlockTrialFunction(V)

    # Placeholder for the last converged solution
    X_old = BlockFunction(V)

    # ---------------------------------------------------------------------
    # boundary conditions
    # ---------------------------------------------------------------------

    """
    Physical boundary conditions
    """
    # Far-field fluid velocity
    far_field = Expression(('(1 - x[1] * x[1]) * t', '0'), degree=0, t=1)  # / 0.25

    # impose the far-field fluid velocity upstream and downstream
    bc_inlet = DirichletBC(V.sub(0), far_field, bdry, inlet)
    bc_outlet = DirichletBC(V.sub(0), far_field, bdry, outlet)

    # impose no vertical fluid flow at the centreline axis
    bc_fluid_axis = DirichletBC(V.sub(0).sub(1), Constant(0), bdry, fluid_axis)

    # impose no-slip and no-penetration at the channel wall
    bc_wall = DirichletBC(V.sub(0), Constant((0, 0)), bdry, wall)

    # impose zero vertical solid displacement at the centreline axis
    bc_solid_axis = DirichletBC(V.sub(2).sub(1), Constant(0), bdry, solid_axis)

    """
    Boundary conditions for the ALE problem for fluid 
    displacement.  These are no normal displacements
    """
    ac_inlet = DirichletBC(V.sub(9).sub(0), Constant(0), bdry, inlet)
    ac_outlet = DirichletBC(V.sub(9).sub(0), Constant(0), bdry, outlet)
    ac_fluid_axis = DirichletBC(V.sub(9).sub(1), Constant(0), bdry, fluid_axis)
    ac_wall = DirichletBC(V.sub(9).sub(1), Constant(0), bdry, wall)
    ac_cell = DirichletBC(V.sub(11).sub(1), Constant(0), bdry, cell_axis)

    # Combine all BCs together
    bcs = BlockDirichletBC(
        [bc_inlet, bc_outlet, bc_fluid_axis, bc_wall, bc_solid_axis,
         ac_inlet, ac_outlet, ac_fluid_axis, ac_wall, ac_cell])

    # ---------------------------------------------------------------------
    # Define the model
    # ---------------------------------------------------------------------

    I = Identity(2)

    """
    Solids problem
    """
    # deformation gradient tensor
    F = I + grad(u_s)
    H = inv(F.T)

    # (non-dim) compressible PK1 stress tensor with Darcy pressure
    lambda_s = 1  # nu_s = lambda_s/(2(lambda_s + mu_s))
    J_s = det(F)
    Sigma_s = 1 / eps * (F - H) + lambda_s / eps * (J_s - 1) * J_s * H - p_p * J_s * H

    # linear elasticity
    # Sigma_s = lambda_s / eps * div(u_s) * I + (grad(u_s) + grad(u_s).T) / eps - p_p * J_s * H

    # Permeability tensor
    K = kappa_0 * J_s * inv(F.T * F)

    # eul normal and tangent
    nn_eul = H * nn / sqrt(inner(H * nn, H * nn))
    TT_eul = I - outer(nn_eul, nn_eul)
    tt_eul = F * tt / sqrt(inner(F * tt, F * tt))


    # functions for post-processing projections
    def F_func(u_s):
        return I + grad(u_s)


    def J_s_func(u_s):
        return det(F_func(u_s))


    def H_func(u_s):
        return inv(F_func(u_s).T)


    # neo-Hookean - no darcy pressure
    def Sigma_s_func(u_s, p_p, eps):
        return (1 / eps * (F_func(u_s) - H_func(u_s))
                + lambda_s / eps * (J_s_func(u_s) - 1) * J_s_func(u_s) * H_func(u_s))


    # linear - no darcy pressure
    # def Sigma_s_func(u_s, p_p, eps):
    #     return (lambda_s / eps * div(u_s) * I + (grad(u_s) + grad(u_s).T) / eps)

    def darcy_flux(p_p):
        return -kappa_0 * H_func(u_s) * grad(p_p)


    """
    External Fluids problem: mapping the current configuration to the 
    initial configuration leads a different form of the
    incompressible Stokes equations
    """

    # Deformation gradient for the fluid
    F_a = I + grad(u_a)
    H_a = inv(F_a.T)

    # Jacobian for the fluid
    J_a = det(F_a)

    # PK1 stress tensor and incompressibility condition for the fluid
    Sigma_f = J_a * (-p_f * I + grad(u_f) * H_a.T + H_a * grad(u_f).T) * H_a
    ic_f = div(J_a * inv(F_a) * u_f)


    def Sigma_f_func(u_f, p_f, u_a):
        return (J_s_func(u_a) * (-p_f * I + grad(u_f) * H_func(u_a).T + H_func(u_a) * grad(u_f).T) * H_func(u_a))


    """
    ALE problem: there are three different versions below
    """

    # Laplace
    # sigma_a = grad(u_a)

    # linear elasticity
    nu_a = Constant(0.1)
    E_a = 0.5 * (grad(u_a) + grad(u_a).T)
    sigma_a = nu_a / (1 + nu_a) / (1 - 2 * nu_a) * div(u_a) * I + 1 / (1 + nu_a) * E_a

    # nonlinear elasticity
    # nu_a = Constant(0.48)
    # E_a = 0.5 * (F_a.T * F_a - I)
    # sigma_a = F_a * (nu_a / (1 + nu_a) / (1 - 2 * nu_a) * tr(E_a) * I + 1 / (1 + nu_a) * E_a)

    ############### ALE method for the cell
    # Deformation gradient for the cell
    F_c = I + grad(u_c)
    H_c = inv(F_c.T)
    J_c = det(F_c)
    # ic_c = J_c - 1

    # linear elasticity
    nu_c = Constant(0.1)
    E_c = 0.5 * (grad(u_c) + grad(u_c).T)
    sigma_c = nu_c / (1 + nu_c) / (1 - 2 * nu_c) * div(u_c) * I + 1 / (1 + nu_c) * E_c

    # ---------------------------------------------------------------------
    # build equations
    # ---------------------------------------------------------------------

    # '-' solid, '+' fluid on dS1
    # '-' cell, '+' solid on dS0
    # Stokes equations for the fluid
    FUN1 = (-inner(Sigma_f, grad(v_f)) * dx(fluid)
            + inner(lam("+"), v_f("+")) * dS1)

    # Incompressibility for the fluid
    FUN2 = ic_f * q_f * dx(fluid) + lam_p * q_f * dx(fluid)

    f_body = 0

    # Nonlinear elasticity for the solid balancing with the Darcy pressure
    # compressible with Darcy pressure
    FUN3 = (-inner(Sigma_s, grad(v_s)) * dx(solid)
            + inner(as_vector([f_0, 0]), v_s) * dx(solid)  # equilibrium condition
            - inner(lam("-"), v_s("-")) * dS1  # lam = Sig.N on dS1
            + inner(lam_3('+'), v_s('+')) * dS0)  # lam_3 = Sig.N on dS0

    # div(Q + J F^-1 v_s) = 0, Q = -k J F^-1 F^-T grad(p_p)
    # conservation of mass for fluid and solid phases in particle
    FUN4 = (inner(K * grad(p_p), grad(q_p)) * dx(solid)
            - inner(J_s * H.T * as_vector([U_0, 0]), grad(q_p)) * dx(solid)
            + lam_2('+') * q_p('-') * dS1   # lam_2 = u.N on dS1
            - lam_4('-') * q_p('+') * dS0)  # lam_3 = u.N on dS0

    # stress balance - beavers and joseph (try gamma small for now)
    FUN5 = (inner(avg(eta), lam('+') - J_s('-') * p_p('-') * dot(H('-'), nn('-'))
                  + gamma * J_s('-') * sqrt(inner(H('-') * nn('-'), H('-') * nn('-')))
                  * (u_f('+') + kappa_0 * H('-') * grad(p_p('-')) - as_vector([U_0('-'), 0]))) * dS1)

    # Darcy stress balances with normal Stokes stress n.lam = -p
    # FUN11 = inner(avg(eta_2), -J_s('-') * sqrt(inner(H('-') * nn('-'), H('-') * nn('-'))) * p_p('-')
    #               + dot(nn_eul('-'), lam('+'))) * dS

    # Vel continuity
    FUN11 = inner(avg(eta_2),
                  dot(u_f('+') + kappa_0 * H('-') * grad(p_p('-')) - as_vector([U_0('-'), 0]), nn('-'))) * dS1

    # No total axial traction on the solid (ez . sigma_s . n = int_V f dV)
    FUN6 = dot(ez, lam('+')) * V_0("-") * dS1 - f_body * V_0 * dx(solid)

    # ALE bulk equation
    FUN7 = (-inner(sigma_a, grad(v_a)) * dx(fluid)
            + inner(lam_a("+"), v_a("+")) * dS1)

    # Continuity of fluid and solid displacement - outer
    FUN8 = inner(avg(eta_a), u_a("+") - u_s("-")) * dS1

    # mean axial solid displacement is zero
    FUN9 = dot(ez, u_s) * g_0 * dx(solid)

    # mean fluid pressure is zero
    FUN10 = p_f * eta_p * dx(fluid)

    # ALE bulk equation - cell
    FUN12 = (-inner(sigma_c, grad(v_c)) * dx(cell)
             + inner(lam_c("-"), v_c("-")) * dS0)

    # Continuity of fluid and solid displacement - inner
    FUN13 = inner(avg(eta_c), u_c("-") - u_s("+")) * dS0

    # FUN14 = ic_c * q_c * dx(cell)
    FUN14 = dot(u_s('+'), nn('+')) * q_c('-') * dS0

    # stress balance - inner
    FUN15 = inner(avg(eta_3), lam_3('+') - J_c('-') * p_c('-') * dot(H_c('-'), nn('-'))) * dS0

    # no normal flux
    FUN16 = inner(avg(eta_4),
                  dot(kappa_0 * H('+') * grad(p_p('+')), nn('-'))) * dS0

    # Combine equations and compute Jacobian
    FUN = [FUN1, FUN2, FUN3, FUN4, FUN5, FUN6, FUN7, FUN8, FUN9, FUN10, FUN11, FUN12, FUN13, FUN14, FUN15, FUN16]
    JAC = block_derivative(FUN, X, Xt)

    # ---------------------------------------------------------------------
    # set up the solver
    # ---------------------------------------------------------------------

    # Initialize solver
    problem = BlockNonlinearProblem(FUN, X, bcs, JAC)
    solver = BlockPETScSNESSolver(problem)
    solver.parameters.update(snes_solver_parameters["snes_solver"])

    # extract solution components
    (u_f, p_f, u_s, p_p, f_0, U_0, lam, lam_2, lam_p, u_a, lam_a, u_c, p_c, lam_c, lam_3, lam_4) = X.block_split()

    # ---------------------------------------------------------------------
    # Set up code to save solid quanntities only on the solid domain and
    # fluid quantities only on the fluid domain
    # ---------------------------------------------------------------------

    """
        Separate the meshes
    """
    mesh_f = SubMesh(mesh, subdomains, fluid)
    mesh_s = SubMesh(mesh, subdomains, solid)
    mesh_c = SubMesh(mesh, subdomains, cell)

    # Create function spaces for the velocity and displacement
    Vf = VectorFunctionSpace(mesh_f, "CG", 1)
    Pf = FunctionSpace(mesh_f, "CG", 1)
    Vs = VectorFunctionSpace(mesh_s, "CG", 1)
    Pp = FunctionSpace(mesh_s, "CG", 1)
    Vc = VectorFunctionSpace(mesh_c, "CG", 1)
    Pc = FunctionSpace(mesh_c, "DG", 0)
    P1v = VectorFunctionSpace(mesh, "DG", 1)

    # oonvert to polar coordinates
    cs = x[0] / sqrt(x[0] ** 2 + x[1] ** 2)
    sn = x[1] / sqrt(x[0] ** 2 + x[1] ** 2)
    A = as_tensor([[cs, sn], [-sn, cs]])

    u_f_only = Function(Vf)
    u_a_only = Function(Vf)
    u_s_only = Function(Vs)
    p_f_only = Function(Pf)
    p_p_only = Function(Pp)
    u_c_only = Function(Vc)
    p_c_only = Function(Pc)


    # Python function to save solution for a given value
    # of epsilon
    def save(eps):
        u_f_only = project(u_f - as_vector([U_0, 0]), Vf)
        u_a_only = project(u_a, Vf)
        u_s_only = project(u_s, Vs)
        p_f_only = project(p_f, Pf)
        p_p_only = project(p_p, Pp)
        u_c_only = project(u_c, Vc)
        p_c_only = project(p_c, Pc)

        MhP = BlockFunctionSpace([P1v], restrict=[Os])
        sig_s = project((A * Sigma_s_func(u_s, p_p, eps) * A.T) * as_vector([1, 0]), MhP.sub(0))

        VhSD = VectorFunctionSpace(mesh_s, 'DG', 0)
        sigma_s_int = interpolate(sig_s, VhSD)

        # total flux
        Q = project(darcy_flux(p_p), MhP.sub(0))
        VhQ = VectorFunctionSpace(mesh_s, 'DG', 0)
        Q_int = interpolate(Q, VhQ)

        # fluid stress
        MhF = BlockFunctionSpace([P1v], restrict=[Of])
        sig_f = project((A * Sigma_f_func(u_f, p_f, u_a) * A.T) * as_vector([1, 0]), MhF.sub(0))

        VhFD = VectorFunctionSpace(mesh_f, 'DG', 0)
        sigma_f_int = interpolate(sig_f, VhFD)

        u_f_only.rename("u_f", "u_f")
        p_f_only.rename("p_f", "p_f")
        u_a_only.rename("u_a", "u_a")
        u_s_only.rename("u_s", "u_s")
        sigma_s_int.rename("sigma", "sigma")
        sigma_f_int.rename("sigma", "sigma")
        p_p_only.rename("p_p", "p_p")
        Q_int.rename("Q", "Q")
        u_c_only.rename("u_c", "u_c")
        p_c_only.rename("p_c", "p_c")

        output_f.write(u_f_only, eps)
        output_f.write(u_a_only, eps)
        output_f.write(p_f_only, eps)
        output_f.write(sigma_f_int, eps)
        output_s.write(u_s_only, eps)
        output_s.write(sigma_s_int, eps)
        output_s.write(p_p_only, eps)
        output_s.write(Q_int, eps)
        output_c.write(u_c_only, eps)
        output_c.write(p_c_only, eps)


    # ---------------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------------

    n = 0

    # last converged value of epsilon
    eps_conv = 0

    # increment epsilon and solve
    while eps_conv < eps_max:

        print('-------------------------------------------------')
        print(f'attempting to solve problem with eps = {float(eps):.4e}')

        # make a prediction of the next solution based on how the solution
        # changed over the last two increments, e.g. using a simple
        # extrapolation formula
        if n > 1:
            X.block_vector()[:] = X_old.block_vector()[:] + (eps_try - eps_conv) * dX_deps

        (its, conv) = solver.solve()

        # if the solver converged...
        if conv:

            n += 1
            # update value of eps_conv and save
            eps_conv = float(eps)
            save(eps_conv)

            # update the value of epsilon to try to solve the problem with
            eps_try += de

            # copy the converged solution into old solution
            block_assign(X_old, X)

            # print some info to the screen
            print('Axial force on the particle: f_0 =', f_0.vector()[0])
            print('Translational speed of the particle: U_0 =', U_0.vector()[0])
            print('Disp1 = ', u_s(0, rad_out)[1])
            print('Disp2 = ', u_s(rad_out, 0)[0])
            print('Disp2 = ', u_s(-rad_out, 0)[0])
            print('pressure ', p_p(-rad_out, 0))
            print('cell pressure ', p_c.vector()[0])
            print('volume change', assemble((J_c - 1) * dx(cell)))

            # approximate the derivative of the solution wrt epsilon
            if n > 0:
                dX_deps = (X.block_vector()[:] - X_old.block_vector()[:]) / de

        # if the solver diverged...
        if not (conv):
            # halve the increment in epsilon if not at smallest value
            # and use the previously converged solution as the initial
            # guess
            if de > de_min:
                de /= 2
                eps_try = eps_conv + de
                block_assign(X, X_old)
            else:
                print('min increment reached...aborting')
                save(eps_try)
                break

        # update the value of epsilon
        eps.assign(eps_try)

    print('-----------------------------------------------------')
    print(f'Maximum eps reached {float(eps):.4e}, now increasing gamma')
    g0_conv = 0
    ng0 = 0
    # now increment kappa and solve
    while g0_conv < g0_max:

        print('-------------------------------------------------')
        print(f'attempting to solve problem with gamma = {float(gamma):.4e}')

        # make a prediction of the next solution based on how the solution
        # changed over the last two increments, e.g. using a simple
        # extrapolation formula
        if ng0 > 1:
            X.block_vector()[:] = X_old.block_vector()[:] + (g0_try - g0_conv) * dX_dg0

        (its, conv) = solver.solve()

        # if the solver converged...
        if conv:

            ng0 += 1
            # update value of eps_conv and save
            g0_conv = float(gamma)
            save(g0_conv)

            # update the value of epsilon to try to solve the problem with
            g0_try += dg0

            # copy the converged solution into old solution
            block_assign(X_old, X)

            # print some info to the screen
            print('Axial force on the particle: f_0 =', f_0.vector()[0])
            print('Translational speed of the particle: U_0 =', U_0.vector()[0])
            print('Disp1 = ', u_s(0, rad_out)[1])
            print('Disp2 = ', u_s(rad_out, 0)[0])
            print('Disp2 = ', u_s(-rad_out, 0)[0])
            print('pressure ', p_p(-rad_out, 0))
            print('cell pressure ', p_c.vector()[0])

            # approximate the derivative of the solution wrt epsilon
            if ng0 > 0:
                dX_dg0 = (X.block_vector()[:] - X_old.block_vector()[:]) / dg0

        # if the solver diverged...
        if not (conv):
            # halve the increment in epsilon if not at smallest value
            # and use the previously converged solution as the initial
            # guess
            if dg0 > dg0_min:
                dg0 /= 2
                g0_try = g0_conv + dg0
                block_assign(X, X_old)
            else:
                print('min increment reached for kappa ...aborting')
                save(g0_try)
                break

        # update the value of gamma
        gamma.assign(g0_try)
