#pragma once

#include <vector>

#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"
#include "src/fields/fieldManager.cpp"

namespace src::coupling {
    using kernels::physics::structures::_2Mesh;
    using kernels::physics::structures::_3Mesh;
    using kernels::physics::structures::_2Field;
    using kernels::physics::structures::_3Field;
    using kernels::physics::structures::_2ParticleGroup;
    using kernels::physics::structures::_3ParticleGroup;
    using kernels::physics::structures::accFromLorentzField;
    using kernels::math::structures::vec2;
    using kernels::math::structures::vec3;
    using real = types::real;

    // Build a cell-centred E field from Yee Ex/Ey, for use with
    // accFromLorentzField. This is a simple face-average; it is not the only
    // possible choice but is good enough for early tests.
    inline _2Field<vec2> make_cell_center_E(const src::fields::_2FieldManager& fm) {
        const auto& mesh = fm.mesh();
        const int nx = mesh.nx_cells();
        const int ny = mesh.ny_cells();

        _2Field<vec2> Ecc;
        Ecc.n_x = nx;
        Ecc.n_y = ny;
        Ecc.data.resize(static_cast<std::size_t>(nx) * ny);

        const auto& Ex = fm.Ex();
        const auto& Ey = fm.Ey();

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                // Ex is defined on x-faces: i = 0..nx, j = 0..ny-1
                // Average left/right faces to the cell centre.
                const real Ex_left  = Ex.valueAt(i,     j);
                const real Ex_right = Ex.valueAt(i + 1, j);
                const real Ex_cc    = static_cast<real>(0.5) * (Ex_left + Ex_right);

                // Ey is defined on y-faces: i = 0..nx-1, j = 0..ny
                // Average bottom/top faces to the cell centre.
                const real Ey_bottom = Ey.valueAt(i, j);
                const real Ey_top    = Ey.valueAt(i, j + 1);
                const real Ey_cc     = static_cast<real>(0.5) * (Ey_bottom + Ey_top);

                Ecc.valueAt(i, j) = vec2{Ex_cc, Ey_cc};
            }
        }

        return Ecc;
    }

    // Build a cell-centred B field for 2D TM^z: only Bz is non-zero.
    inline _2Field<vec3> make_cell_center_B(const src::fields::_2FieldManager& fm) {
        const auto& mesh = fm.mesh();
        const int nx = mesh.nx_cells();
        const int ny = mesh.ny_cells();

        _2Field<vec3> Bcc;
        Bcc.n_x = nx;
        Bcc.n_y = ny;
        Bcc.data.resize(static_cast<std::size_t>(nx) * ny);

        const auto& Bz = fm.Bz();

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const real bz = Bz.valueAt(i, j);
                Bcc.valueAt(i, j) = vec3{0.0, 0.0, bz};
            }
        }

        return Bcc;
    }

    // Yee-aware 2D Lorentz acceleration: converts Yee fields to a cell-centred
    // vector field representation and then reuses the generic kernel.
    inline std::vector<vec2>
    accFromLorentzYee(_2ParticleGroup& p, const src::fields::_2FieldManager& fm) {
        const auto& mesh = fm.mesh();
        auto Ecc = make_cell_center_E(fm);
        auto Bcc = make_cell_center_B(fm);
        return accFromLorentzField(p, mesh, Ecc, Bcc);
    }

    // 3D versions: build cell-centred E/B from Yee face-centred components.
    inline _3Field<vec3> make_cell_center_E(const src::fields::_3FieldManager& fm) {
        const auto& mesh = fm.mesh();
        const int nx = mesh.nx_cells();
        const int ny = mesh.ny_cells();
        const int nz = mesh.nz_cells();

        _3Field<vec3> Ecc;
        Ecc.n_x = nx;
        Ecc.n_y = ny;
        Ecc.n_z = nz;
        Ecc.data.resize(static_cast<std::size_t>(nx) * ny * nz);

        const auto& Ex = fm.Ex();
        const auto& Ey = fm.Ey();
        const auto& Ez = fm.Ez();

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const real Ex_left  = Ex.valueAt(i,     j, k);
                    const real Ex_right = Ex.valueAt(i + 1, j, k);
                    const real Ex_cc    = static_cast<real>(0.5) * (Ex_left + Ex_right);

                    const real Ey_bottom = Ey.valueAt(i, j,     k);
                    const real Ey_top    = Ey.valueAt(i, j + 1, k);
                    const real Ey_cc     = static_cast<real>(0.5) * (Ey_bottom + Ey_top);

                    const real Ez_back = Ez.valueAt(i, j, k);
                    const real Ez_front = Ez.valueAt(i, j, k + 1);
                    const real Ez_cc = static_cast<real>(0.5) * (Ez_back + Ez_front);

                    Ecc.valueAt(i, j, k) = vec3{Ex_cc, Ey_cc, Ez_cc};
                }
            }
        }

        return Ecc;
    }

    inline _3Field<vec3> make_cell_center_B(const src::fields::_3FieldManager& fm) {
        const auto& mesh = fm.mesh();
        const int nx = mesh.nx_cells();
        const int ny = mesh.ny_cells();
        const int nz = mesh.nz_cells();

        _3Field<vec3> Bcc;
        Bcc.n_x = nx;
        Bcc.n_y = ny;
        Bcc.n_z = nz;
        Bcc.data.resize(static_cast<std::size_t>(nx) * ny * nz);

        const auto& Bx = fm.Bx();
        const auto& By = fm.By();
        const auto& Bz = fm.Bz();

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const real Bx_left  = Bx.valueAt(i,     j, k);
                    const real Bx_right = Bx.valueAt(i + 1, j, k);
                    const real Bx_cc    = static_cast<real>(0.5) * (Bx_left + Bx_right);

                    const real By_bottom = By.valueAt(i, j,     k);
                    const real By_top    = By.valueAt(i, j + 1, k);
                    const real By_cc     = static_cast<real>(0.5) * (By_bottom + By_top);

                    const real Bz_back = Bz.valueAt(i, j, k);
                    const real Bz_front = Bz.valueAt(i, j, k + 1);
                    const real Bz_cc = static_cast<real>(0.5) * (Bz_back + Bz_front);

                    Bcc.valueAt(i, j, k) = vec3{Bx_cc, By_cc, Bz_cc};
                }
            }
        }

        return Bcc;
    }

    inline std::vector<vec3>
    accFromLorentzYee(_3ParticleGroup& p, const src::fields::_3FieldManager& fm) {
        const auto& mesh = fm.mesh();
        auto Ecc = make_cell_center_E(fm);
        auto Bcc = make_cell_center_B(fm);
        return accFromLorentzField(p, mesh, Ecc, Bcc);
    }
}

