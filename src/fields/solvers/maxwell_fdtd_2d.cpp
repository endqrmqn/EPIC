#include <chrono>
#include <iostream>

#include "include/types.hpp"
#include "include/constants.hpp"
#include "src/fields/fieldManager.cpp"

namespace src::fields::solvers {
    using real = types::real;
    using _2sf = kernels::physics::structures::_2Field<real>;

    // 2D Yee FDTD stepper for TM^z-style fields:
    //
    // Fields in _2FieldManager:
    //   Ex: (nx_xfaces, ny_xfaces)   ~ Ex(i+1/2, j)
    //   Ey: (nx_yfaces, ny_yfaces)   ~ Ey(i, j+1/2)
    //   Bz: (nx_cells,  ny_cells)    ~ Bz(i+1/2, j+1/2)
    //
    // Simple update with optional current sources Jx, Jy stored on the
    // cell-centred grid in the field manager. When J is zero this
    // reduces to the vacuum Yee scheme used in the tests.
    //
    //   Bz^{n+1} = Bz^{n}
    //              + dt * [ (Ex(i,j+1) - Ex(i,j))/dy
    //                       - (Ey(i+1,j) - Ey(i,j))/dx ]
    //
    //   Ex^{n+1} = Ex^{n}
    //              + dt * ( Bz(i, j) - Bz(i, j-1) ) / dy
    //
    //   Ey^{n+1} = Ey^{n}
    //              - dt * ( Bz(i, j) - Bz(i-1, j) ) / dx
    //
    void advance_maxwell_2d(src::fields::_2FieldManager& fm,
                            real dt,
                            const _2sf* Jx_face_ptr = nullptr,
                            const _2sf* Jy_face_ptr = nullptr) {
        using clock    = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<double>;

        auto t_start = clock::now();

        auto& mesh = fm.mesh();

        const int nx = mesh.nx_cells();
        const int ny = mesh.ny_cells();

        auto& Ex = fm.Ex();
        auto& Ey = fm.Ey();
        auto& Bz = fm.Bz();

        auto& Jx = fm.Jx();
        auto& Jy = fm.Jy();

        const real dx = mesh.dx();
        const real dy = mesh.dy();
        const real inv_dx = static_cast<real>(1.0) / dx;
        const real inv_dy = static_cast<real>(1.0) / dy;

        const real eps0 =
            static_cast<real>(constants::epsilon_0);

        auto t_bz_start = clock::now();

        // --- 1) Update Bz using old E ---
        for (int j = 0; j < ny - 1; ++j) {
            for (int i = 0; i < nx - 1; ++i) {
                real dEx_dy = (Ex.valueAt(i, j + 1) - Ex.valueAt(i, j)) * inv_dy;
                real dEy_dx = (Ey.valueAt(i + 1, j) - Ey.valueAt(i, j)) * inv_dx;

                Bz.valueAt(i, j) += dt * (dEx_dy - dEy_dx);
            }
        }

        auto t_bz_end = clock::now();

        auto t_ex_start = clock::now();

        // --- 2) Update Ex using new Bz and Jx ---
        for (int j = 1; j < ny; ++j) {
            for (int i = 0; i < nx + 1; ++i) {
                const int ib = (i == nx) ? nx - 1 : i;
                real dBz_dy =
                    (Bz.valueAt(ib, j) - Bz.valueAt(ib, j - 1)) * inv_dy;

                real Jx_face = real{0};
                if (Jx_face_ptr != nullptr) {
                    Jx_face = Jx_face_ptr->valueAt(i, j);
                } else {
                    // Fallback for legacy callers: approximate from
                    // cell-centred Jx values.
                    if (i == 0) {
                        Jx_face = Jx.valueAt(0, j);
                    } else if (i == nx) {
                        Jx_face = Jx.valueAt(nx - 1, j);
                    } else {
                        Jx_face = static_cast<real>(0.5) *
                                  (Jx.valueAt(i - 1, j) + Jx.valueAt(i, j));
                    }
                }

                Ex.valueAt(i, j) +=
                    dt * (dBz_dy - Jx_face / eps0);
            }
        }

        auto t_ex_end = clock::now();

        auto t_ey_start = clock::now();

        // --- 3) Update Ey using new Bz and Jy ---
        for (int j = 0; j < ny + 1; ++j) {
            for (int i = 1; i < nx; ++i) {
                const int jb = (j == ny) ? ny - 1 : j;
                real dBz_dx =
                    (Bz.valueAt(i, jb) - Bz.valueAt(i - 1, jb)) * inv_dx;

                real Jy_face = real{0};
                if (Jy_face_ptr != nullptr) {
                    Jy_face = Jy_face_ptr->valueAt(i, j);
                } else {
                    // Fallback for legacy callers: approximate from
                    // cell-centred Jy values.
                    if (j == 0) {
                        Jy_face = Jy.valueAt(i, 0);
                    } else if (j == ny) {
                        Jy_face = Jy.valueAt(i, ny - 1);
                    } else {
                        Jy_face = static_cast<real>(0.5) *
                                  (Jy.valueAt(i, j - 1) + Jy.valueAt(i, j));
                    }
                }

                Ey.valueAt(i, j) -=
                    dt * (dBz_dx + Jy_face / eps0);
            }
        }

        auto t_ey_end = clock::now();
        auto t_end    = clock::now();

        duration bz_time  = t_bz_end  - t_bz_start;
        duration ex_time  = t_ex_end  - t_ex_start;
        duration ey_time  = t_ey_end  - t_ey_start;
        duration total    = t_end     - t_start;

#ifdef EPIC_ENABLE_FIELD_TIMING
        std::cout << "[Maxwell2D diagnostics] "
                  << "Bz_update=" << bz_time.count() << "s, "
                  << "Ex_update=" << ex_time.count() << "s, "
                  << "Ey_update=" << ey_time.count() << "s, "
                  << "total="    << total.count()    << "s"
                  << std::endl;
#endif
    }
}
