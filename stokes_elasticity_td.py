"""
This is a monolithic solver for pressure-driven flow around an elastic particle
with quasi time dependence.

The problem is formulated using the ALE
method, which maps the deformed geometry to the initial
geometry.  The problem is solved using the
initial geometry; the deformed geometry can by using the
WarpByVector filter in Paraview using the displacement
computed in the fluid domain.

The problem uses Lagrange multipliers to ensure zero mean fluid
pressure

The code works by initially solving the problem with a small
value of epsilon (ratio of fluid stress to elastic stiffness)
and then gradually ramping up epsilon.  If convergence is
not obtained then the code tries again using a smaller value
of epsilon.

The code is 2-D (not axisymmetric)

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


def generate_output_files(rad):
    output_s = XDMFFile(dir + "poro_2d_solid_0-"
                        + str(float('%.2g' % rad))[2:] + ".xdmf")
    output_s.parameters['rewrite_function_mesh'] = False
    output_s.parameters["functions_share_mesh"] = True
    output_s.parameters["flush_output"] = True

    output_f = XDMFFile(dir + "poro_2d_fluid_0-"
                        + str(float('%.2g' % rad))[2:] + ".xdmf")
    output_f.parameters['rewrite_function_mesh'] = False
    output_f.parameters["functions_share_mesh"] = True
    output_f.parameters["flush_output"] = True

    return output_s, output_f


def get_mesh(rad):
    meshname = 'channel_sphere_' + str(float('%.1g' % rad))[2:]
    # meshname = 'channel_sphere'
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
    rad = (ii + 6) / 10
    print("doing rad = ", rad)

    output_s, output_f = generate_output_files(rad)

    # mesh has been created with Gmsh
    mesh, subdomains, bdry = get_mesh(rad)

    """
        Physical parameters
    """

    # physical parameters
    lambda_s = 1  # lambda = 2*nu/(1-2*nu)
    m = 1  # artificial mass
    eps = 0.4  # softness of particle

    # computational parameters
    dt = Constant(5e-2)
    Nt = 100
    t_vec = np.zeros(Nt + 1)
    t_d = Constant(0)  # for now

    # define the boundaries (values from the gmsh file)
    circle = 1
    fluid_axis = 2
    inlet = 3
    outlet = 4
    wall = 5
    solid_axis = 6

    # define the domains
    fluid = 10
    solid = 11

    Of = generate_subdomain_restriction(mesh, subdomains, fluid)
    Os = generate_subdomain_restriction(mesh, subdomains, solid)
    Sig = generate_interface_restriction(mesh, subdomains, {fluid, solid})

    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
    ds = Measure("ds", domain=mesh, subdomain_data=bdry)
    dS = Measure("dS", domain=mesh, subdomain_data=bdry)
    dS = dS(circle)

    # normal and tangent vectors
    nn = FacetNormal(mesh)
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

    """
    # mixed_element = BlockElement(V2, P1, V2, V2, DGT, DGT, P0, V2, DGT)
    # V = BlockFunctionSpace(mesh, mixed_element,
    #                        restrict=[Of, Of, Os, Os, Sig, Os, Of, Of, Sig])
    #
    # X = BlockFunction(V)
    # (u_f, p_f, u_s, u_vel, lam, lam2, lam_p, u_a, lam_a) = block_split(X)
    #
    # # unknowns and test functions
    # Y = BlockTestFunction(V)
    # (v_f, q_f, v_s, v_sol, eta, eta2, eta_p, v_a, eta_a) = block_split(Y)
    #
    # Xt = BlockTrialFunction(V)
    #
    # # Placeholder for the last converged solution
    # X_old = BlockFunction(V)
    # (u_f_old, p_f_old, u_s_old, u_vel_old, lam_old, lam2_old, lam_p_old, u_a_old, lam_a_old) = block_split(X_old)

    # mixed_element = BlockElement(V2, P1, V2, V2, P0, P0, DGT, P0, V2, DGT)
    mixed_element = BlockElement(V2, P1, V2, P0, P0, P0, DGT, P0, V2, DGT)
    V = BlockFunctionSpace(mesh, mixed_element,
                           restrict=[Of, Of, Os, Os, Os, Os, Sig, Of, Of, Sig])

    # u_f, p_f, u_s as expected
    # u_vel - solid velocity
    # f_0 - body force should go to zero
    # U_0 - mean axial solid velocity
    # u_com - mean z displacement
    X = BlockFunction(V)
    # (u_f, p_f, u_s, u_vel, f_0, U_0, lam, lam_p, u_a, lam_a) = block_split(X)
    (u_f, p_f, u_s, u_com, f_0, U_0, lam, lam_p, u_a, lam_a) = block_split(X)

    # unknowns and test functions
    Y = BlockTestFunction(V)
    # (v_f, q_f, v_s, v_sol, g_0, V_0, eta, eta_p, v_a, eta_a) = block_split(Y)
    (v_f, q_f, v_s, v_com, g_0, V_0, eta, eta_p, v_a, eta_a) = block_split(Y)

    Xt = BlockTrialFunction(V)

    # Placeholder for the last converged solution
    X_old = BlockFunction(V)
    # (u_f_old, p_f_old, u_s_old, u_vel_old, f_0_old, U_0_old, lam_old,
    #  lam_p_old, u_a_old, lam_a_old) = block_split(X_old)
    (u_f_old, p_f_old, u_s_old, u_com_old, f_0_old, U_0_old, lam_old,
     lam_p_old, u_a_old, lam_a_old) = block_split(X_old)

    # ---------------------------------------------------------------------
    # boundary conditions
    # ---------------------------------------------------------------------

    """
    Physical boundary conditions
    """
    # Far-field fluid velocity
    far_field = Expression(('(1 - x[1] * x[1]) * t_d', '0'), degree=0, t_d=t_d)  # * t_d / (1 + t_d) / 0.25

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
    # incompressible
    # ac_inlet = DirichletBC(V.sub(7).sub(0), Constant((0)), bdry, inlet)
    # ac_outlet = DirichletBC(V.sub(7).sub(0), Constant((0)), bdry, outlet)
    # ac_fluid_axis = DirichletBC(V.sub(7).sub(1), Constant((0)), bdry, fluid_axis)
    # ac_wall = DirichletBC(V.sub(7).sub(1), Constant((0)), bdry, wall)
    ac_inlet = DirichletBC(V.sub(8).sub(0), Constant((0)), bdry, inlet)
    ac_outlet = DirichletBC(V.sub(8).sub(0), Constant((0)), bdry, outlet)
    ac_fluid_axis = DirichletBC(V.sub(8).sub(1), Constant((0)), bdry, fluid_axis)
    ac_wall = DirichletBC(V.sub(8).sub(1), Constant((0)), bdry, wall)

    # Combine all BCs together
    bcs = BlockDirichletBC(
        [bc_inlet, bc_outlet, bc_fluid_axis, bc_wall, bc_solid_axis,
         ac_inlet, ac_outlet, ac_fluid_axis, ac_wall])

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
    Sigma_s = 1 / eps * (F - H) + lambda_s / eps * (J_s - 1) * J_s * H

    # linear elasticity
    # Sigma_s = lambda_s / eps * div(u_s) * I + (grad(u_s) + grad(u_s).T) / eps

    # functions for post-processing projections
    def F_func(u_s):
        return I + grad(u_s)


    def J_s_func(u_s):
        return det(F_func(u_s))


    def H_func(u_s):
        return inv(F_func(u_s).T)


    def Sigma_s_func(u_s, eps):
        return (1 / eps * (F_func(u_s) - H_func(u_s))
                + lambda_s / eps * (J_s_func(u_s) - 1) * J_s_func(u_s) * H_func(u_s))


    # linear - no darcy pressure
    # def Sigma_s_func(u_s, p_p, eps):
    #     return (lambda_s / eps * div(u_s) * I + (grad(u_s) + grad(u_s).T) / eps)


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

    # ---------------------------------------------------------------------
    # build equations
    # ---------------------------------------------------------------------

    # "-"" solid, "+"" fluid
    # Stokes equations for the fluid
    FUN1 = (-inner(Sigma_f, grad(v_f)) * dx(fluid)
            + inner(lam("+"), v_f("+")) * dS)

    # Incompressibility for the fluid
    FUN2 = ic_f * q_f * dx(fluid) + lam_p * q_f * dx(fluid)

    f_body = 0

    # Nonlinear elasticity for the solid balancing with the Darcy pressure
    # compressible with Darcy pressure
    FUN3 = (-inner(Sigma_s, grad(v_s)) * dx(solid)
            + inner(as_vector([f_0, 0]), v_s) * dx(solid)
            - inner(lam("-"), v_s("-")) * dS)  # fluid traction balances with solid stress

    # # Incompressibility condition
    # FUN4 = ic_s * q_p * dx(solid)

    # No total axial traction on the solid (ez . sigma_s . n = 0)
    FUN4 = m * (U_0 - U_0_old) / dt * V_0 * dx(solid) - dot(ez, lam("+")) * V_0("-") * dS

    # no slip u_f = u_vel
    FUN5 = inner(avg(eta), u_f("+") - as_vector([U_0("-"), 0])) * dS

    delta = 1.e7
    # du_s/dt-u_vel=0
    # FUN6 = delta * (inner((u_s - u_s_old) / dt, v_sol) * dx(solid)
    #         - inner(u_vel - u_vel_old, v_sol) * dx(solid))

    FUN6 = delta * (inner((u_com - u_com_old) / dt, v_com) * dx(solid)
            - inner(U_0 - U_0_old, v_com) * dx(solid))

    # ALE bulk equation
    FUN7 = (-inner(sigma_a, grad(v_a)) * dx(fluid)
            + inner(lam_a("+"), v_a("+")) * dS)

    # Continuity of fluid and solid displacement
    FUN8 = inner(avg(eta_a), u_a("+") - u_s("-")) * dS

    # define mean axial velocity
    # FUN9 = (dot(u_vel, ez) - U_0) * g_0 * dx(solid)

    # link mean axial displacement to the U_0
    FUN9 = (dot(ez, u_s) - u_com) * g_0 * dx(solid)

    # mean fluid pressure is zero
    FUN10 = p_f * eta_p * dx(fluid)

    # Combine equations and compute Jacobian
    # FUN = [FUN1, FUN2, FUN3, FUN4, FUN5, FUN6, FUN7, FUN8, FUN10]
    FUN = [FUN1, FUN2, FUN3, FUN4, FUN5, FUN6, FUN7, FUN8, FUN9, FUN10] # FUN4, FUN44, FUN9
    JAC = block_derivative(FUN, X, Xt)

    # ---------------------------------------------------------------------
    # set up the solver
    # ---------------------------------------------------------------------

    # Initialize solver
    problem = BlockNonlinearProblem(FUN, X, bcs, JAC)
    solver = BlockPETScSNESSolver(problem)
    solver.parameters.update(snes_solver_parameters["snes_solver"])

    # extract solution components
    # (u_f, p_f, u_s, u_vel, f_0, U_0, lam, lam_p, u_a, lam_a) = X.block_split()
    (u_f, p_f, u_s, u_com, f_0, U_0, lam, lam_p, u_a, lam_a) = X.block_split()

    # ---------------------------------------------------------------------
    # Set up code to save solid quanntities only on the solid domain and
    # fluid quantities only on the fluid domain
    # ---------------------------------------------------------------------

    """
        Separate the meshes
    """
    mesh_f = SubMesh(mesh, subdomains, fluid)
    mesh_s = SubMesh(mesh, subdomains, solid)

    # Create function spaces for the velocity and displacement
    Vf = VectorFunctionSpace(mesh_f, "CG", 1)
    Pf = FunctionSpace(mesh_f, "CG", 1)
    Vs = VectorFunctionSpace(mesh_s, "CG", 1)
    Pp = FunctionSpace(mesh_s, "CG", 1)
    P1v = VectorFunctionSpace(mesh, "DG", 1)

    # oonvert to polar coordinates
    cs = x[0] / sqrt(x[0] ** 2 + x[1] ** 2)
    sn = x[1] / sqrt(x[0] ** 2 + x[1] ** 2)
    A = as_tensor([[cs, sn], [-sn, cs]])

    u_f_only = Function(Vf)
    u_a_only = Function(Vf)
    u_s_only = Function(Vs)
    u_vel_only = Function(Vs)
    p_f_only = Function(Pf)


    # Python function to save solution for a given value
    # of epsilon
    def save(t):
        u_f_only = project(u_f, Vf)
        u_a_only = project(u_a, Vf)
        u_s_only = project(u_s, Vs)
        # u_vel_only = project(u_vel, Vs)
        u_vel_only = project(u_com, Pp)
        p_f_only = project(p_f, Pf)

        MhP = BlockFunctionSpace([P1v], restrict=[Os])
        sig_s = project((A * Sigma_s_func(u_s, eps) * A.T) * as_vector([1, 0]), MhP.sub(0))

        VhSD = VectorFunctionSpace(mesh_s, 'DG', 0)
        sigma_s_int = interpolate(sig_s, VhSD)

        # fluid stress
        MhF = BlockFunctionSpace([P1v], restrict=[Of])
        sig_f = project((A * Sigma_f_func(u_f, p_f, u_a) * A.T) * as_vector([1, 0]), MhF.sub(0))

        VhFD = VectorFunctionSpace(mesh_f, 'DG', 0)
        sigma_f_int = interpolate(sig_f, VhFD)

        u_f_only.rename("u_f", "u_f")
        u_a_only.rename("u_a", "u_a")
        u_s_only.rename("u_s", "u_s")
        u_vel_only.rename("u_vel", "u_vel")
        sigma_s_int.rename("sigma", "sigma")
        sigma_f_int.rename("sigma", "sigma")

        output_f.write(u_f_only, t)
        output_f.write(u_a_only, t)
        output_f.write(p_f_only, t)
        output_f.write(sigma_f_int, t)
        output_s.write(u_s_only, t)
        output_s.write(u_vel_only, t)
        output_s.write(sigma_s_int, t)


    # ---------------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------------

    for i in range(Nt):

        print('-------------------------------------------')
        print('iteration', i + 1, 'of', Nt)

        (its, conv) = solver.solve()

        if conv:
            save(t_vec[i])

            # update solid disp and vel
            u_s_old.assign(u_s)
            # u_vel_old.assign(u_vel)
            u_com_old.assign(u_com)
            U_0_old.assign(U_0)

            # print some info
            print('Disp1 = ', u_s(0, rad)[1])
            print('Disp2 = ', u_s(rad, 0)[0])
            print('Solid vel = ', U_0.vector()[0])
            print('Body force = ', f_0.vector()[0])



            t_vec[i + 1] = t_vec[i] + dt
            t_d.assign(float(np.tanh(t_vec[i+1])))
            i += 1

        else:
            print('NO CONVERGENCE')
            break

