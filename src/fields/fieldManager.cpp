#pragma once

#include <algorithm>
#include <cstddef>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"

namespace src::fields {
    using vec2  = kernels::math::structures::vec2;
    using vec3  = kernels::math::structures::vec3;
    using _2Mesh = kernels::physics::structures::_2Mesh;
    using _3Mesh = kernels::physics::structures::_3Mesh;
    using real  = types::real;

    using _2sf  = kernels::physics::structures::_2Field<real>;
    using _3sf  = kernels::physics::structures::_3Field<real>;

    using _2vf2 = kernels::physics::structures::_2Field<vec2>;
    using _2vf3 = kernels::physics::structures::_2Field<vec3>;
    using _3vf3 = kernels::physics::structures::_3Field<vec3>;

    class _2FieldManager {
    public:
        using Mesh        = _2Mesh;
        using EField      = _2vf2;
        using BField      = _2vf3;
        using ScalarField = _2sf;

        explicit _2FieldManager(const Mesh& mesh)
            : mesh_(mesh) {
            const int nx = mesh_.nx_cells();
            const int ny = mesh_.ny_cells();
            const std::size_t N = static_cast<std::size_t>(nx) * ny;

            e_.n_x = nx;
            e_.n_y = ny;
            e_.data.assign(N, vec2{});

            b_.n_x = nx;
            b_.n_y = ny;
            b_.data.assign(N, vec3{});

            q_dens_.n_x = nx;
            q_dens_.n_y = ny;
            J_dens_.n_x = nx;
            J_dens_.n_y = ny;
            m_dens_.n_x = nx;
            m_dens_.n_y = ny;

            q_dens_.data.assign(N, real{0});
            J_dens_.data.assign(N, real{0});
            m_dens_.data.assign(N, real{0});
        }

        void reset_sources() {
            std::fill(q_dens_.data.begin(), q_dens_.data.end(), real{0});
            std::fill(J_dens_.data.begin(), J_dens_.data.end(), real{0});
            std::fill(m_dens_.data.begin(), m_dens_.data.end(), real{0});
        }

        const Mesh& mesh() const noexcept { return mesh_; }

        EField&       E()       noexcept { return e_; }
        const EField& E() const noexcept { return e_; }

        BField&       B()       noexcept { return b_; }
        const BField& B() const noexcept { return b_; }

        ScalarField&       rho()       noexcept { return q_dens_; }
        const ScalarField& rho() const noexcept { return q_dens_; }

        ScalarField&       J()         noexcept { return J_dens_; }
        const ScalarField& J()   const noexcept { return J_dens_; }

        ScalarField&       m()         noexcept { return m_dens_; }
        const ScalarField& m()   const noexcept { return m_dens_; }

    private:
        Mesh        mesh_;
        EField      e_;
        BField      b_;
        ScalarField q_dens_;
        ScalarField J_dens_;
        ScalarField m_dens_;
    };

    class _3FieldManager {
    public:
        using Mesh        = _3Mesh;
        using EField      = _3vf3;
        using BField      = _3vf3;
        using ScalarField = _3sf;

        explicit _3FieldManager(const Mesh& mesh)
            : mesh_(mesh) {
            const int nx = mesh_.nx_cells();
            const int ny = mesh_.ny_cells();
            const int nz = mesh_.nz_cells();
            const std::size_t N = static_cast<std::size_t>(nx) * ny * nz;

            e_.n_x = nx;
            e_.n_y = ny;
            e_.n_z = nz;
            e_.data.assign(N, vec3{});

            b_.n_x = nx;
            b_.n_y = ny;
            b_.n_z = nz;
            b_.data.assign(N, vec3{});

            q_dens_.n_x = nx;
            q_dens_.n_y = ny;
            q_dens_.n_z = nz;
            J_dens_.n_x = nx;
            J_dens_.n_y = ny;
            J_dens_.n_z = nz;
            m_dens_.n_x = nx;
            m_dens_.n_y = ny;
            m_dens_.n_z = nz;

            q_dens_.data.assign(N, real{0});
            J_dens_.data.assign(N, real{0});
            m_dens_.data.assign(N, real{0});
        }

        void reset_sources() {
            std::fill(q_dens_.data.begin(), q_dens_.data.end(), real{0});
            std::fill(J_dens_.data.begin(), J_dens_.data.end(), real{0});
            std::fill(m_dens_.data.begin(), m_dens_.data.end(), real{0});
        }

        const Mesh& mesh() const noexcept { return mesh_; }

        EField&       E()       noexcept { return e_; }
        const EField& E() const noexcept { return e_; }

        BField&       B()       noexcept { return b_; }
        const BField& B() const noexcept { return b_; }

        ScalarField&       rho()       noexcept { return q_dens_; }
        const ScalarField& rho() const noexcept { return q_dens_; }

        ScalarField&       J()         noexcept { return J_dens_; }
        const ScalarField& J()   const noexcept { return J_dens_; }

        ScalarField&       m()         noexcept { return m_dens_; }
        const ScalarField& m()   const noexcept { return m_dens_; }

    private:
        Mesh        mesh_;
        EField      e_;
        BField      b_;
        ScalarField q_dens_;
        ScalarField J_dens_;
        ScalarField m_dens_;
    };
}

