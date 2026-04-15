#include <cmath>
#include <iostream>

#include "../../include/types.hpp"
#include "../../include/kernels/math/structures/vec2.hpp"
#include "../../include/kernels/math/structures/vec3.hpp"
#include "../../include/kernels/physics/structures/mesh.hpp"
#include "../../include/kernels/physics/structures/field.hpp"
#include "../../include/constants.hpp"

using types::real;
using kernels::math::structures::vec2;
using kernels::math::structures::vec3;
using kernels::physics::structures::_2Mesh;
using kernels::physics::structures::_3Mesh;
using kernels::physics::structures::_2Field;
using kernels::physics::structures::_3Field;
using constants::pi;

int main() {
    const real tol = 1e-6;

    auto check = [](const char* name, bool cond) {
        if (cond) {
            std::cout << "[PASS] " << name << "\n";
        } else {
            std::cout << "[FAIL] " << name << "\n";
        }
        return cond;
    };

    std::cout << "=== field interpolation tests ===\n";

    // 2D scalar field: f(x,y) = x + 2y on a 4x4 grid
    {
        _2Mesh m(0.0, 1.0, 0.0, 1.0, 4, 4);

        _2Field<real> f;
        f.n_x = m.nx_cells();
        f.n_y = m.ny_cells();
        f.data.resize(f.n_x * f.n_y);

        for (int j = 0; j < f.n_y; ++j) {
            for (int i = 0; i < f.n_x; ++i) {
                vec2 c = m.cell_center(i, j);
                f.valueAt(i, j) = c.x + 2.0 * c.y;
            }
        }

        // Check integer indexing at a few points
        {
            int i = 1, j = 2;
            vec2 c = m.cell_center(i, j);
            real val = f.valueAt(i, j);
            real exact = c.x + 2.0 * c.y;
            check("2D int index valueAt", std::abs(val - exact) < tol);
        }

        // Check interpolation at a fractional index (should match linear f)
        {
            int i = 2, j = 1;
            // Continuous index space: u,v (0..nx-1, 0..ny-1)
            real u = static_cast<real>(i) + static_cast<real>(0.25);
            real v = static_cast<real>(j) + static_cast<real>(0.25);

            real x = m.x_init + (u + static_cast<real>(0.5)) * m.dx();
            real y = m.y_init + (v + static_cast<real>(0.5)) * m.dy();

            real val_interp = static_cast<const _2Field<real>&>(f).valueAt(u, v);
            real exact = x + 2.0 * y;
            check("2D real index valueAt(fractional)",
                  std::abs(val_interp - exact) < 1e-4);
        }
    }

    // 2D vector field: F(x,y) = (x^2, y^2) on a 4x4 grid
    {
        _2Mesh m(0.0, 1.0, 0.0, 1.0, 4, 4);

        _2Field<vec2> F;
        F.n_x = m.nx_cells();
        F.n_y = m.ny_cells();
        F.data.resize(F.n_x * F.n_y);

        for (int j = 0; j < F.n_y; ++j) {
            for (int i = 0; i < F.n_x; ++i) {
                vec2 c = m.cell_center(i, j);
                F.valueAt(i, j) = vec2(c.x * c.x, c.y * c.y);
            }
        }

        // Integer index check
        {
            int i = 1, j = 2;
            vec2 c = m.cell_center(i, j);
            vec2 val = F.valueAt(i, j);
            vec2 exact(c.x * c.x, c.y * c.y);
            bool ok = std::abs(val.x - exact.x) < tol &&
                      std::abs(val.y - exact.y) < tol;
            check("2D vector int index valueAt", ok);
        }

        // Fractional index check (expect small interpolation error)
        {
            int i = 1, j = 1;
            real u = static_cast<real>(i) + static_cast<real>(0.3);
            real v = static_cast<real>(j) + static_cast<real>(0.7);

            real x = m.x_init + (u + static_cast<real>(0.5)) * m.dx();
            real y = m.y_init + (v + static_cast<real>(0.5)) * m.dy();

            vec2 val = static_cast<const _2Field<vec2>&>(F).valueAt(u, v);
            vec2 exact(x * x, y * y);
            bool ok = std::abs(val.x - exact.x) < 5e-2 &&
                      std::abs(val.y - exact.y) < 5e-2;
            check("2D vector real index valueAt(fractional)", ok);
        }
    }

    // 2D "complex" scalar field: f(x,y) = sin(2πx) * cos(2πy)
    {
        _2Mesh m(0.0, 1.0, 0.0, 1.0, 256, 256);

        _2Field<real> f;
        f.n_x = m.nx_cells();
        f.n_y = m.ny_cells();
        f.data.resize(f.n_x * f.n_y);

        const real two_pi = static_cast<real>(2.0 * pi);

        for (int j = 0; j < f.n_y; ++j) {
            for (int i = 0; i < f.n_x; ++i) {
                vec2 c = m.cell_center(i, j);
                f.valueAt(i, j) =
                    std::sin(two_pi * c.x) * std::cos(two_pi * c.y);
            }
        }

        // Sample a few fractional index points across the domain
        bool all_ok = true;
        for (int jj = 3; jj <= 12; jj += 3) {
            for (int ii = 3; ii <= 12; ii += 3) {
                real u = static_cast<real>(ii) + static_cast<real>(0.37);
                real v = static_cast<real>(jj) + static_cast<real>(0.61);

                real x = m.x_init + (u + static_cast<real>(0.5)) * m.dx();
                real y = m.y_init + (v + static_cast<real>(0.5)) * m.dy();

                real val = static_cast<const _2Field<real>&>(f).valueAt(u, v);
                real exact = std::sin(two_pi * x) * std::cos(two_pi * y);

                if (std::abs(val - exact) > 5e-2) {
                    all_ok = false;
                }
            }
        }
        check("2D complex scalar field interpolation", all_ok);
    }

    // 3D "complex" scalar field: f(x,y,z) = sin(2πx) + cos(2πy) + sin(2πz)
    {
        _3Mesh m;
        m.x_init = 0.0; m.x_finl = 1.0;
        m.y_init = 0.0; m.y_finl = 1.0;
        m.z_init = 0.0; m.z_finl = 1.0;
        m.n_x = 16; m.n_y = 16; m.n_z = 16;

        _3Field<real> f;
        f.n_x = m.nx_cells();
        f.n_y = m.ny_cells();
        f.n_z = m.nz_cells();
        f.data.resize(static_cast<std::size_t>(f.n_x) * f.n_y * f.n_z);

        const real two_pi = static_cast<real>(2.0 * pi);

        for (int k = 0; k < f.n_z; ++k) {
            for (int j = 0; j < f.n_y; ++j) {
                for (int i = 0; i < f.n_x; ++i) {
                    vec3 c = m.cell_center(i, j, k);
                    f.valueAt(i, j, k) =
                        std::sin(two_pi * c.x) +
                        std::cos(two_pi * c.y) +
                        std::sin(two_pi * c.z);
                }
            }
        }

        bool all_ok = true;
        for (int kk = 2; kk <= 5; kk += 2) {
            for (int jj = 2; jj <= 5; jj += 2) {
                for (int ii = 2; ii <= 5; ii += 2) {
                    real u = static_cast<real>(ii) + static_cast<real>(0.29);
                    real v = static_cast<real>(jj) + static_cast<real>(0.41);
                    real w = static_cast<real>(kk) + static_cast<real>(0.73);

                    real x = m.x_init + (u + static_cast<real>(0.5)) * m.dx();
                    real y = m.y_init + (v + static_cast<real>(0.5)) * m.dy();
                    real z = m.z_init + (w + static_cast<real>(0.5)) * m.dz();

                    real val =
                        static_cast<const _3Field<real>&>(f).valueAt(u, v, w);
                    real exact =
                        std::sin(two_pi * x) +
                        std::cos(two_pi * y) +
                        std::sin(two_pi * z);

                    if (std::abs(val - exact) > 7e-2) {
                        all_ok = false;
                    }
                }
            }
        }
        check("3D complex scalar field interpolation", all_ok);
    }

    // 3D scalar field: f(x,y,z) = x + 2y + 3z on a 4x4x4 grid
    {
        _3Mesh m;
        m.x_init = 0.0; m.x_finl = 1.0;
        m.y_init = 0.0; m.y_finl = 1.0;
        m.z_init = 0.0; m.z_finl = 1.0;
        m.n_x = 4; m.n_y = 4; m.n_z = 4;

        _3Field<real> f;
        f.n_x = m.nx_cells();
        f.n_y = m.ny_cells();
        f.n_z = m.nz_cells();
        f.data.resize(static_cast<std::size_t>(f.n_x) * f.n_y * f.n_z);

        for (int k = 0; k < f.n_z; ++k) {
            for (int j = 0; j < f.n_y; ++j) {
                for (int i = 0; i < f.n_x; ++i) {
                    vec3 c = m.cell_center(i, j, k);
                    f.valueAt(i, j, k) = c.x + 2.0 * c.y + 3.0 * c.z;
                }
            }
        }

        // Check integer indexing at a point
        {
            int i = 1, j = 1, k = 2;
            vec3 c = m.cell_center(i, j, k);
            real val = f.valueAt(i, j, k);
            real exact = c.x + 2.0 * c.y + 3.0 * c.z;
            check("3D int index valueAt", std::abs(val - exact) < tol);
        }

        // Check interpolation at a fractional index (should match linear f)
        {
            int i = 2, j = 1, k = 1;
            real u = static_cast<real>(i) + static_cast<real>(0.25);
            real v = static_cast<real>(j) + static_cast<real>(0.25);
            real w = static_cast<real>(k) + static_cast<real>(0.25);

            real x = m.x_init + (u + static_cast<real>(0.5)) * m.dx();
            real y = m.y_init + (v + static_cast<real>(0.5)) * m.dy();
            real z = m.z_init + (w + static_cast<real>(0.5)) * m.dz();

            real val_interp = static_cast<const _3Field<real>&>(f).valueAt(u, v, w);
            real exact = x + 2.0 * y + 3.0 * z;
            check("3D real index valueAt(fractional)",
                  std::abs(val_interp - exact) < 1e-3);
        }
    }

    std::cout << "field_interp_test done\n";
    return 0;
}
