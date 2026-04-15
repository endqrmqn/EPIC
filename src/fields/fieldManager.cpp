#pragma once

#include <algorithm>
#include <cstddef>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"

namespace src::fields {
    using vec2   = kernels::math::structures::vec2;
    using vec3   = kernels::math::structures::vec3;
    using _2Mesh = kernels::physics::structures::_2Mesh;
    using _3Mesh = kernels::physics::structures::_3Mesh;
    using real   = types::real;

    using _2sf   = kernels::physics::structures::_2Field<real>;
    using _3sf   = kernels::physics::structures::_3Field<real>;

    // 2D Yee-style field manager (TM^z-like: Ex/Ey on faces, Bz at cell centres)
    class _2FieldManager {
    public:
        using Mesh        = _2Mesh;
        using ScalarField = _2sf;

        explicit _2FieldManager(const Mesh& mesh)
            : mesh_(mesh) {
            const int nx_cells = mesh_.nx_cells();
            const int ny_cells = mesh_.ny_cells();

            const int nx_ex = mesh_.nx_xfaces();
            const int ny_ex = mesh_.ny_xfaces();
            const int nx_ey = mesh_.nx_yfaces();
            const int ny_ey = mesh_.ny_yfaces();

            // Ex on x-faces
            Ex_.n_x = nx_ex;
            Ex_.n_y = ny_ex;
            Ex_.data.assign(static_cast<std::size_t>(nx_ex) * ny_ex, real{0});

            // Ey on y-faces
            Ey_.n_x = nx_ey;
            Ey_.n_y = ny_ey;
            Ey_.data.assign(static_cast<std::size_t>(nx_ey) * ny_ey, real{0});

            // Bz at cell centres
            Bz_.n_x = nx_cells;
            Bz_.n_y = ny_cells;
            Bz_.data.assign(static_cast<std::size_t>(nx_cells) * ny_cells, real{0});

            // cell-centred charge/current densities
            rho_.n_x = nx_cells;
            rho_.n_y = ny_cells;
            Jx_.n_x  = nx_cells;
            Jx_.n_y  = ny_cells;
            Jy_.n_x  = nx_cells;
            Jy_.n_y  = ny_cells;

            const std::size_t N_cells =
                static_cast<std::size_t>(nx_cells) * ny_cells;

            rho_.data.assign(N_cells, real{0});
            Jx_.data.assign(N_cells, real{0});
            Jy_.data.assign(N_cells, real{0});
        }

        const Mesh& mesh() const noexcept { return mesh_; }

        // Yee E-field components
        ScalarField&       Ex()       noexcept { return Ex_; }
        const ScalarField& Ex() const noexcept { return Ex_; }

        ScalarField&       Ey()       noexcept { return Ey_; }
        const ScalarField& Ey() const noexcept { return Ey_; }

        // Yee B-field component (out-of-plane)
        ScalarField&       Bz()       noexcept { return Bz_; }
        const ScalarField& Bz() const noexcept { return Bz_; }

        // Sources (cell-centred)
        ScalarField&       rho()       noexcept { return rho_; }
        const ScalarField& rho() const noexcept { return rho_; }

        ScalarField&       Jx()        noexcept { return Jx_; }
        const ScalarField& Jx()  const noexcept { return Jx_; }

        ScalarField&       Jy()        noexcept { return Jy_; }
        const ScalarField& Jy()  const noexcept { return Jy_; }

        void reset_sources() {
            std::fill(rho_.data.begin(), rho_.data.end(), real{0});
            std::fill(Jx_.data.begin(),  Jx_.data.end(),  real{0});
            std::fill(Jy_.data.begin(),  Jy_.data.end(),  real{0});
        }

    private:
        Mesh        mesh_;
        ScalarField Ex_;
        ScalarField Ey_;
        ScalarField Bz_;
        ScalarField rho_;
        ScalarField Jx_;
        ScalarField Jy_;
    };

    // 3D Yee-style field manager: E and B components on face-centred grids,
    // sources cell-centred.
    class _3FieldManager {
    public:
        using Mesh        = _3Mesh;
        using ScalarField = _3sf;

        explicit _3FieldManager(const Mesh& mesh)
            : mesh_(mesh) {
            const int nx_cells = mesh_.nx_cells();
            const int ny_cells = mesh_.ny_cells();
            const int nz_cells = mesh_.nz_cells();

            const int nx_ex = mesh_.nx_xfaces();
            const int ny_ex = mesh_.ny_xfaces();
            const int nz_ex = mesh_.nz_xfaces();

            const int nx_ey = mesh_.nx_yfaces();
            const int ny_ey = mesh_.ny_yfaces();
            const int nz_ey = mesh_.nz_yfaces();

            const int nx_ez = mesh_.nx_zfaces();
            const int ny_ez = mesh_.ny_zfaces();
            const int nz_ez = mesh_.nz_zfaces();

            // E-field components on faces
            Ex_.n_x = nx_ex;
            Ex_.n_y = ny_ex;
            Ex_.n_z = nz_ex;
            Ex_.data.assign(static_cast<std::size_t>(nx_ex) * ny_ex * nz_ex,
                            real{0});

            Ey_.n_x = nx_ey;
            Ey_.n_y = ny_ey;
            Ey_.n_z = nz_ey;
            Ey_.data.assign(static_cast<std::size_t>(nx_ey) * ny_ey * nz_ey,
                            real{0});

            Ez_.n_x = nx_ez;
            Ez_.n_y = ny_ez;
            Ez_.n_z = nz_ez;
            Ez_.data.assign(static_cast<std::size_t>(nx_ez) * ny_ez * nz_ez,
                            real{0});

            // B-field components, same face grids for now
            Bx_.n_x = nx_ex;
            Bx_.n_y = ny_ex;
            Bx_.n_z = nz_ex;
            Bx_.data.assign(static_cast<std::size_t>(nx_ex) * ny_ex * nz_ex,
                            real{0});

            By_.n_x = nx_ey;
            By_.n_y = ny_ey;
            By_.n_z = nz_ey;
            By_.data.assign(static_cast<std::size_t>(nx_ey) * ny_ey * nz_ey,
                            real{0});

            Bz_.n_x = nx_ez;
            Bz_.n_y = ny_ez;
            Bz_.n_z = nz_ez;
            Bz_.data.assign(static_cast<std::size_t>(nx_ez) * ny_ez * nz_ez,
                            real{0});

            // cell-centred sources
            rho_.n_x = nx_cells;
            rho_.n_y = ny_cells;
            rho_.n_z = nz_cells;
            Jx_.n_x  = nx_cells;
            Jx_.n_y  = ny_cells;
            Jx_.n_z  = nz_cells;
            Jy_.n_x  = nx_cells;
            Jy_.n_y  = ny_cells;
            Jy_.n_z  = nz_cells;
            Jz_.n_x  = nx_cells;
            Jz_.n_y  = ny_cells;
            Jz_.n_z  = nz_cells;

            const std::size_t N_cells =
                static_cast<std::size_t>(nx_cells) * ny_cells * nz_cells;

            rho_.data.assign(N_cells, real{0});
            Jx_.data.assign(N_cells, real{0});
            Jy_.data.assign(N_cells, real{0});
            Jz_.data.assign(N_cells, real{0});
        }

        const Mesh& mesh() const noexcept { return mesh_; }

        // Yee E-field components
        ScalarField&       Ex()       noexcept { return Ex_; }
        const ScalarField& Ex() const noexcept { return Ex_; }

        ScalarField&       Ey()       noexcept { return Ey_; }
        const ScalarField& Ey() const noexcept { return Ey_; }

        ScalarField&       Ez()       noexcept { return Ez_; }
        const ScalarField& Ez() const noexcept { return Ez_; }

        // Yee B-field components
        ScalarField&       Bx()       noexcept { return Bx_; }
        const ScalarField& Bx() const noexcept { return Bx_; }

        ScalarField&       By()       noexcept { return By_; }
        const ScalarField& By() const noexcept { return By_; }

        ScalarField&       Bz()       noexcept { return Bz_; }
        const ScalarField& Bz() const noexcept { return Bz_; }

        // Sources (cell-centred)
        ScalarField&       rho()       noexcept { return rho_; }
        const ScalarField& rho() const noexcept { return rho_; }

        ScalarField&       Jx()        noexcept { return Jx_; }
        const ScalarField& Jx()  const noexcept { return Jx_; }

        ScalarField&       Jy()        noexcept { return Jy_; }
        const ScalarField& Jy()  const noexcept { return Jy_; }

        ScalarField&       Jz()        noexcept { return Jz_; }
        const ScalarField& Jz()  const noexcept { return Jz_; }

        void reset_sources() {
            std::fill(rho_.data.begin(), rho_.data.end(), real{0});
            std::fill(Jx_.data.begin(),  Jx_.data.end(),  real{0});
            std::fill(Jy_.data.begin(),  Jy_.data.end(),  real{0});
            std::fill(Jz_.data.begin(),  Jz_.data.end(),  real{0});
        }

    private:
        Mesh        mesh_;

        // E components
        ScalarField Ex_;
        ScalarField Ey_;
        ScalarField Ez_;

        // B components
        ScalarField Bx_;
        ScalarField By_;
        ScalarField Bz_;

        // Sources
        ScalarField rho_;
        ScalarField Jx_;
        ScalarField Jy_;
        ScalarField Jz_;
    };
}
