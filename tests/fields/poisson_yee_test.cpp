#include <cmath>
#include <iostream>

#include "include/types.hpp"
#include "include/constants.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
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

// Yee divergence of E on a 2D Yee grid at a single cell.
static real div_yee_E_2d(const _2Mesh& m,
                         const _2FieldManager& fm,
                         int i, int j){
    const auto& Ex = fm.Ex();
    const auto& Ey = fm.Ey();

    const real dx = m.dx();
    const real dy = m.dy();

    // Ex: i = 0..nx, j = 0..ny-1
    // Ey: i = 0..nx-1, j = 0..ny
    const real dExdx =
        (Ex.valueAt(i + 1, j) - Ex.valueAt(i, j)) / dx;
    const real dEydy =
        (Ey.valueAt(i, j + 1) - Ey.valueAt(i, j)) / dy;

    return dExdx + dEydy;
}

// Yee divergence of E on a 3D Yee grid at a single cell.
static real div_yee_E_3d(const _3Mesh& m,
                         const _3FieldManager& fm,
                         int i, int j, int k){
    const auto& Ex = fm.Ex();
    const auto& Ey = fm.Ey();
    const auto& Ez = fm.Ez();

    const real dx = m.dx();
    const real dy = m.dy();
    const real dz = m.dz();

    // Ex: i = 0..nx,   j = 0..ny-1, k = 0..nz-1
    // Ey: i = 0..nx-1, j = 0..ny,   k = 0..nz-1
    // Ez: i = 0..nx-1, j = 0..ny-1, k = 0..nz
    const real dExdx =
        (Ex.valueAt(i + 1, j, k) - Ex.valueAt(i, j, k)) / dx;
    const real dEydy =
        (Ey.valueAt(i, j + 1, k) - Ey.valueAt(i, j, k)) / dy;
    const real dEzdz =
        (Ez.valueAt(i, j, k + 1) - Ez.valueAt(i, j, k)) / dz;

    return dExdx + dEydy + dEzdz;
}

static real compute_yee_gauss_error_2d(int nx, int ny){
    _2Mesh m(0.0, 1.0, 0.0, 1.0, nx, ny);
    _2FieldManager fm(m);

    auto& rho = fm.rho();

    const real two_pi = static_cast<real>(2.0 * constants::pi);

    // Smooth manufactured charge density at cell centres.
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            vec2 c = m.cell_center(i, j);
            rho.valueAt(i, j) =
                std::sin(two_pi * c.x) * std::cos(two_pi * c.y);
        }
    }

    const real eps0 = static_cast<real>(constants::epsilon_0);

    // Solve Poisson on the Yee layout: fills fm.Ex()/Ey().
    src::fields::solvers::poisson_solve_yee(fm);

    real max_res = 0.0;

    // Check Gauss's law on interior cells (avoid physical boundaries).
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            real divE = div_yee_E_2d(m, fm, i, j);
            real res  = divE * eps0 - rho.valueAt(i, j);
            max_res = std::max(max_res, std::abs(res));
        }
    }

    return max_res;
}

static real compute_yee_gauss_error_3d(int nx, int ny, int nz){
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

    src::fields::solvers::poisson_solve_yee(fm);

    real max_res = 0.0;

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                real divE = div_yee_E_3d(m, fm, i, j, k);
                real res  = divE * eps0 - rho.valueAt(i, j, k);
                max_res = std::max(max_res, std::abs(res));
            }
        }
    }

    return max_res;
}

int main(){
    auto check = [](const char* name, bool cond) {
        if (cond) {
            std::cout << "[PASS] " << name << "\n";
        } else {
            std::cout << "[FAIL] " << name << "\n";
        }
        return cond;
    };

    std::cout << "=== Yee Poisson Gauss-law convergence tests ===\n";

    // 2D convergence: run at two resolutions and estimate order.
    {
        const int nx1 = 64, ny1 = 64;
        const int nx2 = 128, ny2 = 128;

        real e1 = compute_yee_gauss_error_2d(nx1, ny1);
        real e2 = compute_yee_gauss_error_2d(nx2, ny2);

        auto order = [](real ec, real ef) -> real {
            return std::log(ec / ef) / std::log(static_cast<real>(2.0));
        };

        real p = order(e1, e2);

        std::cout << "2D: nx=" << nx1 << " -> " << nx2
                  << ", error coarse=" << e1
                  << ", error fine=" << e2
                  << ", observed order=" << p << "\n";

        check("2D Yee Poisson Gauss-law: error decreases and order ~2",
              (e2 < e1) && (p > static_cast<real>(1.5)));
    }

    // 3D convergence.
    {
        const int nx1 = 16, ny1 = 16, nz1 = 16;
        const int nx2 = 32, ny2 = 32, nz2 = 32;

        real e1 = compute_yee_gauss_error_3d(nx1, ny1, nz1);
        real e2 = compute_yee_gauss_error_3d(nx2, ny2, nz2);

        auto order = [](real ec, real ef) -> real {
            return std::log(ec / ef) / std::log(static_cast<real>(2.0));
        };

        real p = order(e1, e2);

        std::cout << "3D: nx=" << nx1 << " -> " << nx2
                  << ", error coarse=" << e1
                  << ", error fine=" << e2
                  << ", observed order=" << p << "\n";

        check("3D Yee Poisson Gauss-law: error decreases and order ~2",
              (e2 < e1) && (p > static_cast<real>(1.5)));
    }

    std::cout << "Yee Poisson Gauss-law convergence tests done\n";
    return 0;
}

