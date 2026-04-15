#include <cmath>
#include <iostream>

#include "include/types.hpp"
#include "include/constants.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "src/fields/fieldManager.cpp"
#include "src/fields/solvers/maxwell_fdtd_2d.cpp"
#include "src/fields/solvers/maxwell_fdtd_3d.cpp"
#include "src/coupling/yee_coupling.hpp"

using types::real;
using kernels::math::structures::vec2;
using kernels::math::structures::vec3;
using kernels::physics::structures::_2Mesh;
using kernels::physics::structures::_3Mesh;
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

    std::cout << "=== Maxwell FDTD tests ===\n";

    // 2D: uniform E, zero B should remain unchanged.
    {
        const int nx = 16;
        const int ny = 16;

        _2Mesh m(0.0, 1.0, 0.0, 1.0, nx, ny);
        _2FieldManager fm(m);

        auto& Ex = fm.Ex();
        auto& Ey = fm.Ey();
        auto& Bz = fm.Bz();

        const real E0 = static_cast<real>(0.5);

        // Initialize uniform Ex, zero Ey, zero Bz
        for (int j = 0; j < Ex.n_y; ++j) {
            for (int i = 0; i < Ex.n_x; ++i) {
                Ex.valueAt(i, j) = E0;
            }
        }
        for (int j = 0; j < Ey.n_y; ++j) {
            for (int i = 0; i < Ey.n_x; ++i) {
                Ey.valueAt(i, j) = 0.0;
            }
        }
        for (int j = 0; j < Bz.n_y; ++j) {
            for (int i = 0; i < Bz.n_x; ++i) {
                Bz.valueAt(i, j) = 0.0;
            }
        }

        const real dx = m.dx();
        const real dy = m.dy();
        const real c  = static_cast<real>(1.0);
        const real dt = static_cast<real>(0.2) *
                        std::min(dx, dy) / c;

        for (int n = 0; n < 20; ++n) {
            src::fields::solvers::advance_maxwell_2d(fm, dt);
        }

        real max_delta_Ex = 0.0;
        real max_Bz       = 0.0;

        for (int j = 0; j < Ex.n_y; ++j) {
            for (int i = 0; i < Ex.n_x; ++i) {
                max_delta_Ex = std::max(
                    max_delta_Ex,
                    std::abs(Ex.valueAt(i, j) - E0)
                );
            }
        }
        for (int j = 0; j < Bz.n_y; ++j) {
            for (int i = 0; i < Bz.n_x; ++i) {
                max_Bz = std::max(
                    max_Bz,
                    std::abs(Bz.valueAt(i, j))
                );
            }
        }

        check("2D Maxwell: uniform Ex stays uniform",
              max_delta_Ex < static_cast<real>(1e-6));
        check("2D Maxwell: Bz remains ~0",
              max_Bz < static_cast<real>(1e-6));
    }

    // 3D: uniform E and B should remain unchanged in cell-centred view.
    {
        const int nx = 8;
        const int ny = 8;
        const int nz = 8;

        _3Mesh m;
        m.x_init = 0.0; m.x_finl = 1.0;
        m.y_init = 0.0; m.y_finl = 1.0;
        m.z_init = 0.0; m.z_finl = 1.0;
        m.n_x = nx; m.n_y = ny; m.n_z = nz;

        _3FieldManager fm(m);

        auto& Ex = fm.Ex();
        auto& Ey = fm.Ey();
        auto& Ez = fm.Ez();
        auto& Bx = fm.Bx();
        auto& By = fm.By();
        auto& Bz = fm.Bz();

        const real E0 = static_cast<real>(0.3);
        const real B0 = static_cast<real>(0.1);

        // Initialize uniform Yee fields
        for (int k = 0; k < Ex.n_z; ++k)
            for (int j = 0; j < Ex.n_y; ++j)
                for (int i = 0; i < Ex.n_x; ++i)
                    Ex.valueAt(i, j, k) = E0;

        for (int k = 0; k < Ey.n_z; ++k)
            for (int j = 0; j < Ey.n_y; ++j)
                for (int i = 0; i < Ey.n_x; ++i)
                    Ey.valueAt(i, j, k) = E0;

        for (int k = 0; k < Ez.n_z; ++k)
            for (int j = 0; j < Ez.n_y; ++j)
                for (int i = 0; i < Ez.n_x; ++i)
                    Ez.valueAt(i, j, k) = E0;

        for (int k = 0; k < Bx.n_z; ++k)
            for (int j = 0; j < Bx.n_y; ++j)
                for (int i = 0; i < Bx.n_x; ++i)
                    Bx.valueAt(i, j, k) = B0;

        for (int k = 0; k < By.n_z; ++k)
            for (int j = 0; j < By.n_y; ++j)
                for (int i = 0; i < By.n_x; ++i)
                    By.valueAt(i, j, k) = B0;

        for (int k = 0; k < Bz.n_z; ++k)
            for (int j = 0; j < Bz.n_y; ++j)
                for (int i = 0; i < Bz.n_x; ++i)
                    Bz.valueAt(i, j, k) = B0;

        auto E_cc_initial = src::coupling::make_cell_center_E(fm);
        auto B_cc_initial = src::coupling::make_cell_center_B(fm);

        const real dx = m.dx();
        const real dy = m.dy();
        const real dz = m.dz();
        const real c  = static_cast<real>(1.0);
        const real dt = static_cast<real>(0.2) *
                        std::min({dx, dy, dz}) / c;

        for (int n = 0; n < 10; ++n) {
            src::fields::solvers::advance_maxwell_3d(fm, dt);
        }

        auto E_cc_final = src::coupling::make_cell_center_E(fm);
        auto B_cc_final = src::coupling::make_cell_center_B(fm);

        real max_delta_E = 0.0;
        real max_delta_B = 0.0;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    vec3 Ei = E_cc_initial.valueAt(i, j, k);
                    vec3 Ef = E_cc_final.valueAt(i, j, k);
                    vec3 Bi = B_cc_initial.valueAt(i, j, k);
                    vec3 Bf = B_cc_final.valueAt(i, j, k);

                    max_delta_E = std::max(
                        max_delta_E,
                        static_cast<real>(
                            std::sqrt((Ef.x - Ei.x)*(Ef.x - Ei.x) +
                                      (Ef.y - Ei.y)*(Ef.y - Ei.y) +
                                      (Ef.z - Ei.z)*(Ef.z - Ei.z))
                        )
                    );

                    max_delta_B = std::max(
                        max_delta_B,
                        static_cast<real>(
                            std::sqrt((Bf.x - Bi.x)*(Bf.x - Bi.x) +
                                      (Bf.y - Bi.y)*(Bf.y - Bi.y) +
                                      (Bf.z - Bi.z)*(Bf.z - Bi.z))
                        )
                    );
                }
            }
        }

        check("3D Maxwell: uniform E cell-centred invariant",
              max_delta_E < static_cast<real>(1e-5));
        check("3D Maxwell: uniform B cell-centred invariant",
              max_delta_B < static_cast<real>(1e-5));
    }

    std::cout << "Maxwell FDTD tests done\n";
    return 0;
}

