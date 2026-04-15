#include <cmath>
#include <iostream>

#include "include/types.hpp"
#include "include/constants.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/math/algs/vectorops.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "src/fields/fieldManager.cpp"
#include "src/fields/solvers/poissonSolver.cpp"

using types::real;
using kernels::math::structures::vec2;
using kernels::math::structures::vec3;
using kernels::physics::structures::_2Mesh;
using kernels::physics::structures::_3Mesh;
using kernels::physics::structures::_2Field;
using kernels::physics::structures::_3Field;
using src::fields::_2FieldManager;
using src::fields::_3FieldManager;

int main() {
    auto check = [](const char* name, bool cond) {
        if (cond) {
            std::cout << "[PASS] " << name << "\n";
        } else {
            std::cout << "[FAIL] " << name << "\n";
        }
        return cond;
    };

    std::cout << "=== Poisson solver tests ===\n";

    // 2D test: Poisson PDE residual check plus Gauss-law diagnostics.
    {
        const int nx = 512;
        const int ny = 512;

        _2Mesh m(0.0, 1.0, 0.0, 1.0, nx, ny);
        _2FieldManager fm(m);
        auto& rho = fm.rho();

        const real two_pi = static_cast<real>(2.0 * constants::pi);

        // Smooth manufactured charge density.
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                vec2 c = m.cell_center(i, j);
                rho.valueAt(i, j) =
                    std::sin(two_pi * c.x) * std::cos(two_pi * c.y);
            }
        }

        const real eps0 = static_cast<real>(constants::epsilon_0);

        // Solve directly for potential phi using the same wide Laplacian
        // (div∘grad with centred differences) as the production solver.
        _2Field<real> phi;
        src::fields::solvers::solve_poisson_phi_2d(m, rho, phi);

        real max_res_pde = 0.0;

        const real h  = m.dx(); // dx = dy
        const real h2 = h * h;

        // Wide 5-point Laplacian consistent with div(grad):
        // (φ_{i+2,j} + φ_{i-2,j} + φ_{i,j+2} + φ_{i,j-2} - 4 φ_{i,j}) / (4 h^2)
        for (int j = 2; j < ny - 2; ++j) {
            for (int i = 2; i < nx - 2; ++i) {
                real phi_ip2 = phi.valueAt(i + 2, j);
                real phi_im2 = phi.valueAt(i - 2, j);
                real phi_jp2 = phi.valueAt(i,     j + 2);
                real phi_jm2 = phi.valueAt(i,     j - 2);
                real phi_c   = phi.valueAt(i,     j);

                real lap = (phi_ip2 + phi_im2 + phi_jp2 + phi_jm2
                            - static_cast<real>(4.0) * phi_c)
                           / (static_cast<real>(4.0) * h2);

                real res = lap * eps0 + rho.valueAt(i, j);
                max_res_pde = std::max(max_res_pde, std::abs(res));
            }
        }

        std::cout << "2D max |Lap(phi)*eps0 + rho| = "
                  << max_res_pde << "\n";

        // Also compute Gauss-law residuals via the wrapper E field as
        // diagnostics (not for pass/fail).
        _2Field<vec2> E = src::fields::solvers::poisson_solver(fm);

        real max_res_plus  = 0.0;
        real max_res_minus = 0.0;

        for (int j = 2; j < ny - 2; ++j) {
            for (int i = 2; i < nx - 2; ++i) {
                real divE = kernels::math::algs::divPoint(m, E, i, j);
                real res_plus  = divE * eps0 + rho.valueAt(i, j);
                real res_minus = divE * eps0 - rho.valueAt(i, j);
                max_res_plus  = std::max(max_res_plus,  std::abs(res_plus));
                max_res_minus = std::max(max_res_minus, std::abs(res_minus));
            }
        }

        std::cout << "2D max |divE*eps0 + rho| = " << max_res_plus  << "\n";
        std::cout << "2D max |divE*eps0 - rho| = " << max_res_minus << "\n";

        check("2D Poisson: max |Lap(phi)*eps0 + rho| < 1e-2",
              max_res_pde < static_cast<real>(1e-2));
        check("2D Gauss: max |divE*eps0 - rho| < 1e-2",
              max_res_minus < static_cast<real>(1e-2));
    }

    // 3D test: Poisson PDE residual check plus Gauss-law diagnostics.
    {
        const int nx = 32;
        const int ny = 32;
        const int nz = 32;

        _3Mesh m;
        m.x_init = 0.0; m.x_finl = 1.0;
        m.y_init = 0.0; m.y_finl = 1.0;
        m.z_init = 0.0; m.z_finl = 1.0;
        m.n_x = nx; m.n_y = ny; m.n_z = nz;

        _3FieldManager fm(m);
        auto& rho = fm.rho();

        const real two_pi = static_cast<real>(2.0 * constants::pi);

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    vec3 c = m.cell_center(i, j, k);
                    rho.valueAt(i, j, k) =
                        std::sin(two_pi * c.x) *
                        std::cos(two_pi * c.y) *
                        std::sin(two_pi * c.z);
                }
            }
        }

        const real eps0 = static_cast<real>(constants::epsilon_0);

        // Solve for phi with the same wide 7-point Laplacian as the solver.
        _3Field<real> phi;
        src::fields::solvers::solve_poisson_phi_3d(m, rho, phi);

        real max_res_pde = 0.0;

        const real h  = m.dx();
        const real h2 = h * h;

        // Wide 7-point Laplacian from div(grad):
        // (sum_nb(±2) - 6 φ_ijk) / (4 h^2)
        for (int k = 2; k < nz - 2; ++k) {
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 2; i < nx - 2; ++i) {
                    real phi_ip2 = phi.valueAt(i + 2, j, k);
                    real phi_im2 = phi.valueAt(i - 2, j, k);
                    real phi_jp2 = phi.valueAt(i,     j + 2, k);
                    real phi_jm2 = phi.valueAt(i,     j - 2, k);
                    real phi_kp2 = phi.valueAt(i,     j,     k + 2);
                    real phi_km2 = phi.valueAt(i,     j,     k - 2);
                    real phi_c   = phi.valueAt(i,     j,     k);

                    real sum_nb = phi_ip2 + phi_im2 + phi_jp2 + phi_jm2
                                  + phi_kp2 + phi_km2;

                    real lap = (sum_nb
                                - static_cast<real>(6.0) * phi_c)
                               / (static_cast<real>(4.0) * h2);

                    real res = lap * eps0 + rho.valueAt(i, j, k);
                    max_res_pde = std::max(max_res_pde, std::abs(res));
                }
            }
        }

        std::cout << "3D max |Lap(phi)*eps0 + rho| = "
                  << max_res_pde << "\n";

        // Gauss-law diagnostics using the wrapper E field.
        _3Field<vec3> E = src::fields::solvers::poisson_solver(fm);

        real max_res_plus  = 0.0;
        real max_res_minus = 0.0;

        for (int k = 2; k < nz - 2; ++k) {
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 2; i < nx - 2; ++i) {
                    real divE = kernels::math::algs::divPoint(m, E, i, j, k);
                    real res_plus  = divE * eps0 + rho.valueAt(i, j, k);
                    real res_minus = divE * eps0 - rho.valueAt(i, j, k);
                    max_res_plus  = std::max(max_res_plus,  std::abs(res_plus));
                    max_res_minus = std::max(max_res_minus, std::abs(res_minus));
                }
            }
        }

        std::cout << "3D max |divE*eps0 + rho| = " << max_res_plus  << "\n";
        std::cout << "3D max |divE*eps0 - rho| = " << max_res_minus << "\n";

        check("3D Poisson: max |Lap(phi)*eps0 + rho| < 5e-2",
              max_res_pde < static_cast<real>(5e-2));
        check("3D Gauss: max |divE*eps0 - rho| < 5e-2",
              max_res_minus < static_cast<real>(5e-2));
    }

    std::cout << "Poisson solver tests done\n";
    return 0;
}
