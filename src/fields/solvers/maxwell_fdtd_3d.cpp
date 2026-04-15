#include <algorithm>

#include "include/types.hpp"
#include "include/constants.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "src/fields/fieldManager.cpp"
#include "src/coupling/yee_coupling.hpp"

namespace src::fields::solvers {
    using real = types::real;
    using vec3 = kernels::math::structures::vec3;
    using _3FieldVec = kernels::physics::structures::_3Field<vec3>;

    // Cell-centred Maxwell update in 3D, using temporary cell-centred fields
    // built from Yee face-centred components. This keeps the implementation
    // simple while remaining compatible with the Yee layout.
    void advance_maxwell_3d(src::fields::_3FieldManager& fm, real dt) {
        const auto& mesh = fm.mesh();

        const int nx = mesh.nx_cells();
        const int ny = mesh.ny_cells();
        const int nz = mesh.nz_cells();

        const real dx = mesh.dx();
        const real dy = mesh.dy();
        const real dz = mesh.dz();

        const real inv_2dx = static_cast<real>(0.5) / dx;
        const real inv_2dy = static_cast<real>(0.5) / dy;
        const real inv_2dz = static_cast<real>(0.5) / dz;

        // Speed of light in code units; treat c = 1 for now.
        const real c2 = static_cast<real>(1.0);

        // Build cell-centred fields from Yee arrays.
        _3FieldVec E = src::coupling::make_cell_center_E(fm);
        _3FieldVec B = src::coupling::make_cell_center_B(fm);

        // Cell-centred current density from the field manager.
        auto& Jx = fm.Jx();
        auto& Jy = fm.Jy();
        auto& Jz = fm.Jz();

        _3FieldVec B_new = B;

        // --- 1) Update B from curl(E) ---
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    const vec3 E_ip = E.valueAt(i + 1, j,     k);
                    const vec3 E_im = E.valueAt(i - 1, j,     k);
                    const vec3 E_jp = E.valueAt(i,     j + 1, k);
                    const vec3 E_jm = E.valueAt(i,     j - 1, k);
                    const vec3 E_kp = E.valueAt(i,     j,     k + 1);
                    const vec3 E_km = E.valueAt(i,     j,     k - 1);

                    const real dEz_dy = (E_jp.z - E_jm.z) * inv_2dy;
                    const real dEy_dz = (E_kp.y - E_km.y) * inv_2dz;

                    const real dEx_dz = (E_kp.x - E_km.x) * inv_2dz;
                    const real dEz_dx = (E_ip.z - E_im.z) * inv_2dx;

                    const real dEy_dx = (E_ip.y - E_im.y) * inv_2dx;
                    const real dEx_dy = (E_jp.x - E_jm.x) * inv_2dy;

                    vec3 b = B.valueAt(i, j, k);

                    // curl(E) at cell centre
                    const real curlEx = dEz_dy - dEy_dz;
                    const real curlEy = dEx_dz - dEz_dx;
                    const real curlEz = dEy_dx - dEx_dy;

                    b.x -= dt * curlEx;
                    b.y -= dt * curlEy;
                    b.z -= dt * curlEz;

                    B_new.valueAt(i, j, k) = b;
                }
            }
        }

        B.data.swap(B_new.data);

        _3FieldVec E_new = E;

        const real eps0 =
            static_cast<real>(constants::epsilon_0);

        // --- 2) Update E from curl(B) (vacuum, no J) ---
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    const vec3 B_ip = B.valueAt(i + 1, j,     k);
                    const vec3 B_im = B.valueAt(i - 1, j,     k);
                    const vec3 B_jp = B.valueAt(i,     j + 1, k);
                    const vec3 B_jm = B.valueAt(i,     j - 1, k);
                    const vec3 B_kp = B.valueAt(i,     j,     k + 1);
                    const vec3 B_km = B.valueAt(i,     j,     k - 1);

                    const real dBz_dy = (B_jp.z - B_jm.z) * inv_2dy;
                    const real dBy_dz = (B_kp.y - B_km.y) * inv_2dz;

                    const real dBx_dz = (B_kp.x - B_km.x) * inv_2dz;
                    const real dBz_dx = (B_ip.z - B_im.z) * inv_2dx;

                    const real dBy_dx = (B_ip.y - B_im.y) * inv_2dx;
                    const real dBx_dy = (B_jp.x - B_jm.x) * inv_2dy;

                    vec3 e = E.valueAt(i, j, k);

                    const real curlBx = dBz_dy - dBy_dz;
                    const real curlBy = dBx_dz - dBz_dx;
                    const real curlBz = dBy_dx - dBx_dy;

                    // Ampere-Maxwell with sources:
                    //   ∂E/∂t = c^2 curl(B) - J/eps0
                    const real Jx_c = Jx.valueAt(i, j, k);
                    const real Jy_c = Jy.valueAt(i, j, k);
                    const real Jz_c = Jz.valueAt(i, j, k);

                    e.x += dt * (c2 * curlBx - Jx_c / eps0);
                    e.y += dt * (c2 * curlBy - Jy_c / eps0);
                    e.z += dt * (c2 * curlBz - Jz_c / eps0);

                    E_new.valueAt(i, j, k) = e;
                }
            }
        }

        E.data.swap(E_new.data);

        // --- 3) Map updated cell-centred E,B back to Yee face grids ---
        auto& Ex = fm.Ex();
        auto& Ey = fm.Ey();
        auto& Ez = fm.Ez();

        auto& Bx = fm.Bx();
        auto& By = fm.By();
        auto& Bz = fm.Bz();

        const int nx_ex = Ex.n_x; // nx+1
        const int ny_ex = Ex.n_y; // ny
        const int nz_ex = Ex.n_z; // nz

        const int nx_ey = Ey.n_x; // nx
        const int ny_ey = Ey.n_y; // ny+1
        const int nz_ey = Ey.n_z; // nz

        const int nx_ez = Ez.n_x; // nx
        const int ny_ez = Ez.n_y; // ny
        const int nz_ez = Ez.n_z; // nz+1

        // Ex/Bx on x-faces
        for (int k = 0; k < nz_ex; ++k) {
            for (int j = 0; j < ny_ex; ++j) {
                for (int i = 0; i < nx_ex; ++i) {
                    if (i == 0) {
                        Ex.valueAt(i, j, k) = E.valueAt(0, j, k).x;
                        Bx.valueAt(i, j, k) = B.valueAt(0, j, k).x;
                    } else if (i == nx_ex - 1) {
                        Ex.valueAt(i, j, k) = E.valueAt(nx - 1, j, k).x;
                        Bx.valueAt(i, j, k) = B.valueAt(nx - 1, j, k).x;
                    } else {
                        const int ic = i - 1;
                        const real Ex_avg =
                            static_cast<real>(0.5) *
                            (E.valueAt(ic,     j, k).x +
                             E.valueAt(ic + 1, j, k).x);
                        const real Bx_avg =
                            static_cast<real>(0.5) *
                            (B.valueAt(ic,     j, k).x +
                             B.valueAt(ic + 1, j, k).x);

                        Ex.valueAt(i, j, k) = Ex_avg;
                        Bx.valueAt(i, j, k) = Bx_avg;
                    }
                }
            }
        }

        // Ey/By on y-faces
        for (int k = 0; k < nz_ey; ++k) {
            for (int j = 0; j < ny_ey; ++j) {
                for (int i = 0; i < nx_ey; ++i) {
                    if (j == 0) {
                        Ey.valueAt(i, j, k) = E.valueAt(i, 0, k).y;
                        By.valueAt(i, j, k) = B.valueAt(i, 0, k).y;
                    } else if (j == ny_ey - 1) {
                        Ey.valueAt(i, j, k) = E.valueAt(i, ny - 1, k).y;
                        By.valueAt(i, j, k) = B.valueAt(i, ny - 1, k).y;
                    } else {
                        const int jc = j - 1;
                        const real Ey_avg =
                            static_cast<real>(0.5) *
                            (E.valueAt(i, jc,     k).y +
                             E.valueAt(i, jc + 1, k).y);
                        const real By_avg =
                            static_cast<real>(0.5) *
                            (B.valueAt(i, jc,     k).y +
                             B.valueAt(i, jc + 1, k).y);

                        Ey.valueAt(i, j, k) = Ey_avg;
                        By.valueAt(i, j, k) = By_avg;
                    }
                }
            }
        }

        // Ez/Bz on z-faces
        for (int k = 0; k < nz_ez; ++k) {
            for (int j = 0; j < ny_ez; ++j) {
                for (int i = 0; i < nx_ez; ++i) {
                    if (k == 0) {
                        Ez.valueAt(i, j, k) = E.valueAt(i, j, 0).z;
                        Bz.valueAt(i, j, k) = B.valueAt(i, j, 0).z;
                    } else if (k == nz_ez - 1) {
                        Ez.valueAt(i, j, k) = E.valueAt(i, j, nz - 1).z;
                        Bz.valueAt(i, j, k) = B.valueAt(i, j, nz - 1).z;
                    } else {
                        const int kc = k - 1;
                        const real Ez_avg =
                            static_cast<real>(0.5) *
                            (E.valueAt(i, j, kc    ).z +
                             E.valueAt(i, j, kc + 1).z);
                        const real Bz_avg =
                            static_cast<real>(0.5) *
                            (B.valueAt(i, j, kc    ).z +
                             B.valueAt(i, j, kc + 1).z);

                        Ez.valueAt(i, j, k) = Ez_avg;
                        Bz.valueAt(i, j, k) = Bz_avg;
                    }
                }
            }
        }
    }
}
