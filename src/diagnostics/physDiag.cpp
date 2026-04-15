#pragma once

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "include/kernels/physics/structures/mesh.hpp"

namespace src::diagnostics{
    using real = types::real;
    using vec2 = kernels::math::structures::vec2;
    using vec3 = kernels::math::structures::vec3;

    using _2ParticleGroup = kernels::physics::structures::_2ParticleGroup;
    using _3ParticleGroup = kernels::physics::structures::_3ParticleGroup;

    template<typename T>
    using _2Field = kernels::physics::structures::_2Field<T>;
    template<typename T>
    using _3Field = kernels::physics::structures::_3Field<T>;

    using _2Mesh = kernels::physics::structures::_2Mesh;
    using _3Mesh = kernels::physics::structures::_3Mesh;

    // ---------------------------------------------------------------------
    // Particle momentum diagnostics
    // ---------------------------------------------------------------------

    inline vec2 total_momentum(const _2ParticleGroup& p){
        vec2 P{0.0, 0.0};
        const std::size_t n = kernels::physics::structures::size(p);
        for (std::size_t i = 0; i < n; ++i){
            P.x += p.mass[i] * p.vel[i].x;
            P.y += p.mass[i] * p.vel[i].y;
        }
        return P;
    }

    inline vec3 total_momentum(const _3ParticleGroup& p){
        vec3 P{0.0, 0.0, 0.0};
        const std::size_t n = kernels::physics::structures::size(p);
        for (std::size_t i = 0; i < n; ++i){
            P.x += p.mass[i] * p.vel[i].x;
            P.y += p.mass[i] * p.vel[i].y;
            P.z += p.mass[i] * p.vel[i].z;
        }
        return P;
    }

    // ---------------------------------------------------------------------
    // Field momentum diagnostics (optional).
    //
    // For TM^z 2D (E = (Ex,Ey,0), B = (0,0,Bz)), the electromagnetic
    // momentum density is g = eps0 * E x B:
    //   g_x = eps0 * Ey * Bz
    //   g_y = -eps0 * Ex * Bz
    //
    // We provide helpers that assume cell-centred E and B.
    // ---------------------------------------------------------------------

    inline vec2 total_field_momentum_2d(const _2Mesh& mesh,
                                        const _2Field<vec2>& Ecc,
                                        const _2Field<vec3>& Bcc){
        const int nx = Ecc.n_x;
        const int ny = Ecc.n_y;

        const real dx = mesh.dx();
        const real dy = mesh.dy();
        const real dA = dx * dy;

        const real eps0 =
            static_cast<real>(constants::epsilon_0);

        vec2 P{0.0, 0.0};
        for (int j = 0; j < ny; ++j){
            for (int i = 0; i < nx; ++i){
                const vec2 E = Ecc.valueAt(i, j);
                const vec3 B = Bcc.valueAt(i, j);
                const real gx = eps0 * E.y * B.z;
                const real gy = -eps0 * E.x * B.z;
                P.x += gx * dA;
                P.y += gy * dA;
            }
        }
        return P;
    }

    inline vec3 total_field_momentum_3d(const _3Mesh& mesh,
                                        const _3Field<vec3>& Ecc,
                                        const _3Field<vec3>& Bcc){
        const int nx = Ecc.n_x;
        const int ny = Ecc.n_y;
        const int nz = Ecc.n_z;

        const real dx = mesh.dx();
        const real dy = mesh.dy();
        const real dz = mesh.dz();
        const real dV = dx * dy * dz;

        const real eps0 =
            static_cast<real>(constants::epsilon_0);

        vec3 P{0.0, 0.0, 0.0};
        for (int k = 0; k < nz; ++k){
            for (int j = 0; j < ny; ++j){
                for (int i = 0; i < nx; ++i){
                    const vec3 E = Ecc.valueAt(i, j, k);
                    const vec3 B = Bcc.valueAt(i, j, k);
                    const vec3 g = eps0 * kernels::math::structures::cross(E, B);
                    P.x += g.x * dV;
                    P.y += g.y * dV;
                    P.z += g.z * dV;
                }
            }
        }
        return P;
    }
}

