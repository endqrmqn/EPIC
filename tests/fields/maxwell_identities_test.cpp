#include <cmath>
#include <iostream>

#include "include/types.hpp"
#include "include/constants.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "include/kernels/math/algs/vectorops.hpp"

using types::real;
using kernels::math::structures::vec3;
using kernels::physics::structures::_3Mesh;
using kernels::physics::structures::_3Field;

int main() {
    auto check = [](const char* name, bool cond) {
        if (cond) {
            std::cout << "[PASS] " << name << "\n";
        } else {
            std::cout << "[FAIL] " << name << "\n";
        }
        return cond;
    };

    std::cout << "=== Maxwell identity convergence tests (3D) ===\n";

    auto compute_errors = [](int nx, int ny, int nz,
                             real& err_curl_grad,
                             real& err_div_curl) {
        _3Mesh m;
        m.x_init = 0.0; m.x_finl = 1.0;
        m.y_init = 0.0; m.y_finl = 1.0;
        m.z_init = 0.0; m.z_finl = 1.0;
        m.n_x = nx; m.n_y = ny; m.n_z = nz;

        const real two_pi = static_cast<real>(2.0 * constants::pi);

        // --- curl(grad(phi)) ---
        _3Field<real> phi;
        phi.n_x = nx; phi.n_y = ny; phi.n_z = nz;
        phi.data.resize(static_cast<std::size_t>(nx) * ny * nz);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    vec3 c = m.cell_center(i, j, k);
                    phi.valueAt(i, j, k) =
                        std::sin(two_pi * c.x) *
                        std::cos(two_pi * c.y) *
                        std::sin(two_pi * c.z);
                }
            }
        }

        _3Field<vec3> grad_phi;
        grad_phi.n_x = nx; grad_phi.n_y = ny; grad_phi.n_z = nz;
        grad_phi.data.resize(static_cast<std::size_t>(nx) * ny * nz);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    grad_phi.valueAt(i, j, k) =
                        kernels::math::algs::gradPoint(m, phi, i, j, k);
                }
            }
        }

        _3Field<vec3> curl_grad;
        curl_grad.n_x = nx; curl_grad.n_y = ny; curl_grad.n_z = nz;
        curl_grad.data.resize(static_cast<std::size_t>(nx) * ny * nz);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    curl_grad.valueAt(i, j, k) =
                        kernels::math::algs::curlPoint(m, grad_phi, i, j, k);
                }
            }
        }

        real max_norm = 0.0;
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    vec3 v = curl_grad.valueAt(i, j, k);
                    real n = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
                    if (n > max_norm) max_norm = n;
                }
            }
        }
        err_curl_grad = max_norm;

        // --- div(curl(A)) ---
        _3Field<vec3> A;
        A.n_x = nx; A.n_y = ny; A.n_z = nz;
        A.data.resize(static_cast<std::size_t>(nx) * ny * nz);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    vec3 c = m.cell_center(i, j, k);
                    real Az = std::sin(two_pi * c.x) *
                              std::sin(two_pi * c.y) *
                              std::sin(two_pi * c.z);
                    A.valueAt(i, j, k) = vec3{0.0, 0.0, Az};
                }
            }
        }

        _3Field<vec3> curlA;
        curlA.n_x = nx; curlA.n_y = ny; curlA.n_z = nz;
        curlA.data.resize(static_cast<std::size_t>(nx) * ny * nz);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    curlA.valueAt(i, j, k) =
                        kernels::math::algs::curlPoint(m, A, i, j, k);
                }
            }
        }

        real max_div = 0.0;
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    real divB = kernels::math::algs::divPoint(m, curlA, i, j, k);
                    max_div = std::max(max_div, std::abs(divB));
                }
            }
        }
        err_div_curl = max_div;
    };

    // Run at two resolutions and estimate convergence order.
    const int nx1 = 16, ny1 = 16, nz1 = 16;
    const int nx2 = 32, ny2 = 32, nz2 = 32;

    real e1_curl_grad = 0.0, e1_div_curl = 0.0;
    real e2_curl_grad = 0.0, e2_div_curl = 0.0;

    compute_errors(nx1, ny1, nz1, e1_curl_grad, e1_div_curl);
    compute_errors(nx2, ny2, nz2, e2_curl_grad, e2_div_curl);

    auto order = [](real e_coarse, real e_fine) -> real {
        return std::log(e_coarse / e_fine) / std::log(static_cast<real>(2.0));
    };

    real p_curl_grad = order(e1_curl_grad, e2_curl_grad);
    real p_div_curl  = order(e1_div_curl,  e2_div_curl);

    std::cout << "Resolution " << nx1 << "^3: "
              << "curl(grad) err = " << e1_curl_grad
              << ", div(curl) err = " << e1_div_curl << "\n";
    std::cout << "Resolution " << nx2 << "^3: "
              << "curl(grad) err = " << e2_curl_grad
              << ", div(curl) err = " << e2_div_curl << "\n";

    std::cout << "Observed order curl(grad): " << p_curl_grad << "\n";
    std::cout << "Observed order div(curl):  " << p_div_curl  << "\n";

    std::cout << "Maxwell identity convergence tests done\n";
    return 0;
}
