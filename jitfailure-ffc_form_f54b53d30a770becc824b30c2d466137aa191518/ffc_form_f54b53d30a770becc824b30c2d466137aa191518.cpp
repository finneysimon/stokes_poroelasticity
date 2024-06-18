// This code conforms with the UFC specification version 2018.1.0
// and was automatically generated by FFC version 2019.1.0.
//
// This code was generated with the following parameters:
//

//  add_tabulate_tensor_timing:     False
//  convert_exceptions_to_warnings: False
//  cpp_optimize:                   True
//  cpp_optimize_flags:             '-O2'
//  epsilon:                        1e-14
//  error_control:                  False
//  external_include_dirs:          ['/home/simon/anaconda3/envs/fenics_old/include', 
//                                  '/home/simon/anaconda3/envs/fenics_old/include/eig
//                                  en3', '/home/simon/anaconda3/envs/fenics_old/inclu
//                                  de']
//  external_includes:              ''
//  external_libraries:             ''
//  external_library_dirs:          ''
//  form_postfix:                   False
//  format:                         'ufc'
//  generate_dummy_tabulate_tensor: False
//  max_signature_length:           0
//  no-evaluate_basis_derivatives:  True
//  optimize:                       True
//  precision:                      None
//  quadrature_degree:              None
//  quadrature_rule:                None
//  representation:                 'auto'
//  split:                          False

#include "ffc_form_f54b53d30a770becc824b30c2d466137aa191518.h"

// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

ffc_form_f54b53d30a770becc824b30c2d466137aa191518_cell_integral_main_11::ffc_form_f54b53d30a770becc824b30c2d466137aa191518_cell_integral_main_11() : ufc::cell_integral()
{

}

ffc_form_f54b53d30a770becc824b30c2d466137aa191518_cell_integral_main_11::~ffc_form_f54b53d30a770becc824b30c2d466137aa191518_cell_integral_main_11()
{

}

const std::vector<bool> & ffc_form_f54b53d30a770becc824b30c2d466137aa191518_cell_integral_main_11::enabled_coefficients() const
{
static const std::vector<bool> enabled({true});
return enabled;
}

void ffc_form_f54b53d30a770becc824b30c2d466137aa191518_cell_integral_main_11::tabulate_tensor(double * A,
                                    const double * const * w,
                                    const double * coordinate_dofs,
                                    int cell_orientation) const
{
    // This function was generated using 'uflacs' representation
    // with the following integrals metadata:
    // 
    // num_cells:         None
    // optimize:          True
    // precision:         16
    // quadrature_degree: 3
    // quadrature_rule:   'default'
    // representation:    'uflacs'
    // 
    // and the following integral 0 metadata:
    // 
    // estimated_polynomial_degree: 3
    // optimize:                    True
    // precision:                   16
    // quadrature_degree:           3
    // quadrature_rule:             'default'
    // representation:              'uflacs'
    // Quadrature rules
    alignas(32) static const double weights6[6] = { 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333 };
    // Precomputed values of basis functions and precomputations
    // FE* dimensions: [entities][points][dofs]
    // PI* dimensions: [entities][dofs][dofs] or [entities][dofs]
    // PM* dimensions: [entities][dofs][dofs]
    alignas(32) static const double FE0_C0_Q6[1][6][6] =
        { { { -0.08525999807368699, 0.2096071917300056, -0.1243471936563187, 0.611401985706872, 0.1011591387118274, 0.2874388755813006 },
            { -0.1243471936563187, 0.2096071917300056, -0.08525999807368706, 0.2874388755813007, 0.1011591387118274, 0.611401985706872 },
            { -0.085259998073687, -0.1243471936563188, 0.2096071917300056, 0.611401985706872, 0.2874388755813005, 0.1011591387118276 },
            { 0.2096071917300056, -0.1243471936563188, -0.08525999807368706, 0.1011591387118275, 0.2874388755813005, 0.611401985706872 },
            { -0.1243471936563187, -0.0852599980736871, 0.2096071917300056, 0.2874388755813006, 0.6114019857068719, 0.1011591387118276 },
            { 0.2096071917300056, -0.08525999807368712, -0.1243471936563187, 0.1011591387118275, 0.6114019857068719, 0.2874388755813006 } } };
    alignas(32) static const double FE6_C0_D01_Q6[1][1][2] = { { { -1.0, 1.0 } } };
    alignas(32) static const double FE6_C0_Q6[1][6][3] =
        { { { 0.1090390090728769, 0.659027622374092, 0.231933368553031 },
            { 0.231933368553031, 0.659027622374092, 0.109039009072877 },
            { 0.109039009072877, 0.231933368553031, 0.6590276223740918 },
            { 0.659027622374092, 0.231933368553031, 0.109039009072877 },
            { 0.231933368553031, 0.109039009072877, 0.6590276223740918 },
            { 0.659027622374092, 0.1090390090728769, 0.231933368553031 } } };
    // Unstructured piecewise computations
    const double J_c0 = coordinate_dofs[0] * FE6_C0_D01_Q6[0][0][0] + coordinate_dofs[2] * FE6_C0_D01_Q6[0][0][1];
    const double J_c3 = coordinate_dofs[1] * FE6_C0_D01_Q6[0][0][0] + coordinate_dofs[5] * FE6_C0_D01_Q6[0][0][1];
    const double J_c1 = coordinate_dofs[0] * FE6_C0_D01_Q6[0][0][0] + coordinate_dofs[4] * FE6_C0_D01_Q6[0][0][1];
    const double J_c2 = coordinate_dofs[1] * FE6_C0_D01_Q6[0][0][0] + coordinate_dofs[3] * FE6_C0_D01_Q6[0][0][1];
    alignas(32) double sp[4];
    sp[0] = J_c0 * J_c3;
    sp[1] = J_c1 * J_c2;
    sp[2] = sp[0] + -1 * sp[1];
    sp[3] = std::abs(sp[2]);
    alignas(32) double BF0[1] = {};
    for (int iq = 0; iq < 6; ++iq)
    {
        // Quadrature loop body setup (num_points=6)
        // Unstructured varying computations for num_points=6
        const double x_r1_c1 = coordinate_dofs_1[1] * FE6_C0_Q6[0][iq][0] + coordinate_dofs_1[3] * FE6_C0_Q6[0][iq][1] + coordinate_dofs_1[5] * FE6_C0_Q6[0][iq][2];
        double w0_c0 = 0.0;
        for (int ic = 0; ic < 6; ++ic)
            w0_c0 += w[0][ic] * FE0_C0_Q6[0][iq][ic];
        alignas(32) double sv6[2];
        sv6[0] = w0_c0 * x_r1_c1;
        sv6[1] = sv6[0] * sp[3];
        const double fw0 = sv6[1] * weights6[iq];
        for (int i = 0; i < 1; ++i)
            BF0[i] += fw0;
    }
    A[0] = 0.0;
    for (int i = 0; i < 1; ++i)
        A[i] += BF0[i];
}

extern "C" DLL_EXPORT ufc::cell_integral * create_ffc_form_f54b53d30a770becc824b30c2d466137aa191518_cell_integral_main_11()
{
  return new ffc_form_f54b53d30a770becc824b30c2d466137aa191518_cell_integral_main_11();
}


ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main() : ufc::form()
{
    // Do nothing
}

ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::~ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main()
{
    // Do nothing
}

const char * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::signature() const
{
    return "eca6e05b016ba6ebc5986f7417dbfa26a3b603a33340859dcc9acf0b5b34d81a7a235918f4ad7a0ca56ece54d51456fe0e960a3a9a9b6477ae9b8c3167ec0110";
}

std::size_t ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::rank() const
{
    return 1;
}

std::size_t ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::num_coefficients() const
{
    return 1;
}

std::size_t ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::original_coefficient_position(std::size_t i) const
{
    if (i >= 1)
    {
        throw std::runtime_error("Invalid original coefficient index.");
    }
    static const std::vector<std::size_t> position = {0};
    return position[i];
}

ufc::finite_element * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_coordinate_finite_element() const
{
    return create_ffc_element_d813efd86d1269ffed6166a5f2febcbe484faa4d_finite_element_main();
}

ufc::dofmap * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_coordinate_dofmap() const
{
    return create_ffc_element_d813efd86d1269ffed6166a5f2febcbe484faa4d_dofmap_main();
}

ufc::coordinate_mapping * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_coordinate_mapping() const
{
    return create_ffc_coordinate_mapping_fdf65ad1b4ee585aa7f358c9d8ed7cd18fb0ebf8_coordinate_mapping_main();
}

ufc::finite_element * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_finite_element(std::size_t i) const
{
    switch (i)
    {
    case 0:
        return create_ffc_element_b6056e9c39d9d0154897eac8d86c6d3a5d1f55b9_finite_element_main();
    case 1:
        return create_ffc_element_de8439ade25e15ca5629b7c8b5e041c4eaec6481_finite_element_main();
    default:
        return nullptr;
    }
}

ufc::dofmap * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_dofmap(std::size_t i) const
{
    switch (i)
    {
    case 0:
        return create_ffc_element_b6056e9c39d9d0154897eac8d86c6d3a5d1f55b9_dofmap_main();
    case 1:
        return create_ffc_element_de8439ade25e15ca5629b7c8b5e041c4eaec6481_dofmap_main();
    default:
        return nullptr;
    }
}

std::size_t ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::max_cell_subdomain_id() const
{
    return 12;
}

std::size_t ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::max_exterior_facet_subdomain_id() const
{
    return 0;
}

std::size_t ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::max_interior_facet_subdomain_id() const
{
    return 0;
}

std::size_t ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::max_vertex_subdomain_id() const
{
    return 0;
}

std::size_t ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::max_custom_subdomain_id() const
{
    return 0;
}

std::size_t ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::max_cutcell_subdomain_id() const
{
    return 0;
}

std::size_t ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::max_interface_subdomain_id() const
{
    return 0;
}

std::size_t ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::max_overlap_subdomain_id() const
{
    return 0;
}

bool ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::has_cell_integrals() const
{
    return true;
}

bool ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::has_exterior_facet_integrals() const
{
    return false;
}

bool ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::has_interior_facet_integrals() const
{
    return false;
}

bool ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::has_vertex_integrals() const
{
    return false;
}

bool ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::has_custom_integrals() const
{
    return false;
}

bool ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::has_cutcell_integrals() const
{
    return false;
}

bool ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::has_interface_integrals() const
{
    return false;
}

bool ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::has_overlap_integrals() const
{
    return false;
}

ufc::cell_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_cell_integral(std::size_t subdomain_id) const
{
    switch (subdomain_id)
    {
    case 11:
        return create_ffc_form_f54b53d30a770becc824b30c2d466137aa191518_cell_integral_main_11();
    default:
        return nullptr;
    }
}

ufc::exterior_facet_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_exterior_facet_integral(std::size_t subdomain_id) const
{
    return nullptr;
}

ufc::interior_facet_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_interior_facet_integral(std::size_t subdomain_id) const
{
    return nullptr;
}

ufc::vertex_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_vertex_integral(std::size_t subdomain_id) const
{
    return nullptr;
}

ufc::custom_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_custom_integral(std::size_t subdomain_id) const
{
    return nullptr;
}

ufc::cutcell_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_cutcell_integral(std::size_t subdomain_id) const
{
    return nullptr;
}

ufc::interface_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_interface_integral(std::size_t subdomain_id) const
{
    return nullptr;
}

ufc::overlap_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_overlap_integral(std::size_t subdomain_id) const
{
    return nullptr;
}

ufc::cell_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_default_cell_integral() const
{
    return nullptr;
}

ufc::exterior_facet_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_default_exterior_facet_integral() const
{
    return nullptr;
}

ufc::interior_facet_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_default_interior_facet_integral() const
{
    return nullptr;
}

ufc::vertex_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_default_vertex_integral() const
{
    return nullptr;
}

ufc::custom_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_default_custom_integral() const
{
    return nullptr;
}

ufc::cutcell_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_default_cutcell_integral() const
{
    return nullptr;
}

ufc::interface_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_default_interface_integral() const
{
    return nullptr;
}

ufc::overlap_integral * ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main::create_default_overlap_integral() const
{
    return nullptr;
}

extern "C" DLL_EXPORT ufc::form * create_ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main()
{
  return new ffc_form_f54b53d30a770becc824b30c2d466137aa191518_form_main();
}
