#pragma once

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"
#include "src/fields/fieldManager.cpp"

namespace src::particles {
    using real = types::real;
    using vec2 = kernels::math::structures::vec2;
    using vec3 = kernels::math::structures::vec3;

    using _2Mesh = kernels::physics::structures::_2Mesh;
    using _3Mesh = kernels::physics::structures::_3Mesh;
    using _2FieldManager = src::fields::_2FieldManager;

    template<typename T>
    using _2Field = kernels::physics::structures::_2Field<T>;
    template<typename T>
    using _3Field = kernels::physics::structures::_3Field<T>;

    using _2ParticleGroup = kernels::physics::structures::_2ParticleGroup;
    using _3ParticleGroup = kernels::physics::structures::_3ParticleGroup;

    namespace {
        inline vec2 gather_E_from_phi_cell_2d(const _2Mesh& mesh,
                                              const _2Field<real>& phi,
                                              const vec2& pos){
            const real fx =
                (pos.x - mesh.x_init) / mesh.dx() - static_cast<real>(0.5);
            const real fy =
                (pos.y - mesh.y_init) / mesh.dy() - static_cast<real>(0.5);

            const int i0 = static_cast<int>(std::floor(fx));
            const int j0 = static_cast<int>(std::floor(fy));
            const real tx = fx - static_cast<real>(i0);
            const real ty = fy - static_cast<real>(j0);

            vec2 E{};
            for (int dj = 0; dj <= 1; ++dj){
                const int j = j0 + dj;
                if (j < 0 || j >= phi.n_y) continue;

                const real wy = (dj == 0)
                    ? (static_cast<real>(1.0) - ty) : ty;
                const real dwy = (dj == 0)
                    ? -static_cast<real>(1.0) / mesh.dy()
                    :  static_cast<real>(1.0) / mesh.dy();

                for (int di = 0; di <= 1; ++di){
                    const int i = i0 + di;
                    if (i < 0 || i >= phi.n_x) continue;

                    const real wx = (di == 0)
                        ? (static_cast<real>(1.0) - tx) : tx;
                    const real dwx = (di == 0)
                        ? -static_cast<real>(1.0) / mesh.dx()
                        :  static_cast<real>(1.0) / mesh.dx();

                    const real phi_ij = phi.valueAt(i, j);
                    E.x += phi_ij * dwx * wy;
                    E.y += phi_ij * wx * dwy;
                }
            }
            return E;
        }

        inline vec2 boris_velocity_update_2d(const vec2& v,
                                             const vec2& Eval,
                                             const vec3& Bval,
                                             real q,
                                             real m,
                                             real dt){
            const real qmdt2 = (q / m) * (dt * static_cast<real>(0.5));

            vec2 v_minus = v + qmdt2 * Eval;

            const real tz = qmdt2 * Bval.z;
            const real t2 = tz * tz;
            const real sz = (t2 > real{0})
                ? static_cast<real>(2.0) * tz / (static_cast<real>(1.0) + t2)
                : real{0};

            vec2 v_prime = v_minus + kernels::math::structures::cross(v_minus, tz);
            vec2 v_plus  = v_minus + kernels::math::structures::cross(v_prime, sz);
            return v_plus + qmdt2 * Eval;
        }

        inline vec3 boris_velocity_update_3d(const vec3& v,
                                             const vec3& Eval,
                                             const vec3& Bval,
                                             real q,
                                             real m,
                                             real dt){
            const real qmdt2 = (q / m) * (dt * static_cast<real>(0.5));

            vec3 v_minus = v + qmdt2 * Eval;
            vec3 t = qmdt2 * Bval;
            const real t2 = kernels::math::structures::dot(t, t);
            const real s_fac = static_cast<real>(2.0) /
                               (static_cast<real>(1.0) + t2);
            vec3 s = s_fac * t;

            vec3 v_prime = v_minus + kernels::math::structures::cross(v_minus, t);
            vec3 v_plus  = v_minus + kernels::math::structures::cross(v_prime, s);
            return v_plus + qmdt2 * Eval;
        }

        inline real sample_bilinear(const _2Field<real>& field,
                                    real fx,
                                    real fy){
            int i0 = static_cast<int>(std::floor(fx));
            int j0 = static_cast<int>(std::floor(fy));

            if (i0 < 0) i0 = 0;
            if (j0 < 0) j0 = 0;
            if (i0 > field.n_x - 2) i0 = field.n_x - 2;
            if (j0 > field.n_y - 2) j0 = field.n_y - 2;

            const real sx = fx - static_cast<real>(i0);
            const real sy = fy - static_cast<real>(j0);

            const real w00 = (static_cast<real>(1.0) - sx) *
                             (static_cast<real>(1.0) - sy);
            const real w10 = sx * (static_cast<real>(1.0) - sy);
            const real w01 = (static_cast<real>(1.0) - sx) * sy;
            const real w11 = sx * sy;

            return field.valueAt(i0,     j0    ) * w00 +
                   field.valueAt(i0 + 1, j0    ) * w10 +
                   field.valueAt(i0,     j0 + 1) * w01 +
                   field.valueAt(i0 + 1, j0 + 1) * w11;
        }

        inline vec2 sample_yee_E_2d(const _2FieldManager& fm,
                                    const vec2& pos){
            const auto& mesh = fm.mesh();
            const real dx = mesh.dx();
            const real dy = mesh.dy();

            const real ex_fx = (pos.x - mesh.x_init) / dx;
            const real ex_fy =
                (pos.y - mesh.y_init) / dy - static_cast<real>(0.5);

            const real ey_fx =
                (pos.x - mesh.x_init) / dx - static_cast<real>(0.5);
            const real ey_fy = (pos.y - mesh.y_init) / dy;

            return vec2{
                sample_bilinear(fm.Ex(), ex_fx, ex_fy),
                sample_bilinear(fm.Ey(), ey_fx, ey_fy)
            };
        }

        inline vec3 sample_yee_B_2d(const _2FieldManager& fm,
                                    const vec2& pos){
            const auto& mesh = fm.mesh();
            const real dx = mesh.dx();
            const real dy = mesh.dy();

            const real bz_fx =
                (pos.x - mesh.x_init) / dx - static_cast<real>(0.5);
            const real bz_fy =
                (pos.y - mesh.y_init) / dy - static_cast<real>(0.5);

            return vec3{0.0, 0.0, sample_bilinear(fm.Bz(), bz_fx, bz_fy)};
        }

        inline bool pos_to_cell_center_coords(const _2Mesh& mesh,
                                              const vec2& pos,
                                              real& i,
                                              real& j){
            if (pos.x < mesh.x_init || pos.x > mesh.x_finl ||
                pos.y < mesh.y_init || pos.y > mesh.y_finl){
                return false;
            }

            i = (pos.x - mesh.x_init) / mesh.dx() - static_cast<real>(0.5);
            j = (pos.y - mesh.y_init) / mesh.dy() - static_cast<real>(0.5);
            return true;
        }

        inline bool pos_to_cell_center_coords(const _3Mesh& mesh,
                                              const vec3& pos,
                                              real& i,
                                              real& j,
                                              real& k){
            if (pos.x < mesh.x_init || pos.x > mesh.x_finl ||
                pos.y < mesh.y_init || pos.y > mesh.y_finl ||
                pos.z < mesh.z_init || pos.z > mesh.z_finl){
                return false;
            }

            i = (pos.x - mesh.x_init) / mesh.dx() - static_cast<real>(0.5);
            j = (pos.y - mesh.y_init) / mesh.dy() - static_cast<real>(0.5);
            k = (pos.z - mesh.z_init) / mesh.dz() - static_cast<real>(0.5);
            return true;
        }
    }

    // ---------------------------------------------------------------------
    // Boris pusher: full Lorentz force in 2D (v in-plane, B out-of-plane).
    //
    // Positions live in real space; fields are provided as cell-centred
    // interpolants (E: vec2, B: vec3 with only z used).
    // ---------------------------------------------------------------------
    inline void boris_push_2d(_2ParticleGroup& p,
                              const _2Mesh& mesh,
                              const _2Field<vec2>& E,
                              const _2Field<vec3>& B,
                              real dt){
        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t i = 0; i < n; ++i){
            vec2 x = p.pos[i];
            vec2 v = p.vel[i];

            // Map position to continuous cell indices.
            real ix, iy;
            bool inside = pos_to_cell_center_coords(mesh, x, ix, iy);

            vec2 Eval{};
            vec3 Bval{};
            if (inside){
                Eval = E.valueAt(ix, iy);
                Bval = B.valueAt(ix, iy);
            }

            const real q = p.charge[i];
            const real m = p.mass[i];

            if (m == real{0}) {
                // Avoid division by zero; leave particle unchanged.
                continue;
            }

            vec2 v_new = boris_velocity_update_2d(v, Eval, Bval, q, m, dt);

            // Position update with midpoint drift.
            vec2 x_new = x + static_cast<real>(0.5) * (v + v_new) * dt;

            p.vel[i] = v_new;
            p.pos[i] = x_new;

            // Optional: store updated acceleration for diagnostics.
            vec2 a_new = (q / m) *
                         (Eval + kernels::math::structures::cross(v_new, Bval.z));
            p.acc[i] = a_new;
        }
    }

    inline void boris_push_yee_2d(_2ParticleGroup& p,
                                  const _2FieldManager& fm,
                                  real dt){
        const auto& mesh = fm.mesh();
        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t i = 0; i < n; ++i){
            const vec2 x = p.pos[i];
            vec2 v = p.vel[i];

            if (x.x < mesh.x_init || x.x > mesh.x_finl ||
                x.y < mesh.y_init || x.y > mesh.y_finl){
                continue;
            }

            const vec2 Eval = sample_yee_E_2d(fm, x);
            const vec3 Bval = sample_yee_B_2d(fm, x);

            const real q = p.charge[i];
            const real m = p.mass[i];
            if (m == real{0}) {
                continue;
            }

            vec2 v_new = boris_velocity_update_2d(v, Eval, Bval, q, m, dt);
            vec2 x_new = x + static_cast<real>(0.5) * (v + v_new) * dt;

            p.vel[i] = v_new;
            p.pos[i] = x_new;
            p.acc[i] = (q / m) *
                       (Eval + kernels::math::structures::cross(v_new, Bval.z));
        }
    }

    // ---------------------------------------------------------------------
    // Boris pusher in 3D: full Lorentz force.
    // ---------------------------------------------------------------------
    inline void boris_push_3d(_3ParticleGroup& p,
                              const _3Mesh& mesh,
                              const _3Field<vec3>& E,
                              const _3Field<vec3>& B,
                              real dt){
        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t i = 0; i < n; ++i){
            vec3 x = p.pos[i];
            vec3 v = p.vel[i];

            real ix, iy, iz;
            bool inside = pos_to_cell_center_coords(mesh, x, ix, iy, iz);

            vec3 Eval{};
            vec3 Bval{};
            if (inside){
                Eval = E.valueAt(ix, iy, iz);
                Bval = B.valueAt(ix, iy, iz);
            }

            const real q = p.charge[i];
            const real m = p.mass[i];
            if (m == real{0}) {
                continue;
            }

            vec3 v_new = boris_velocity_update_3d(v, Eval, Bval, q, m, dt);
            vec3 x_new = x + static_cast<real>(0.5) * (v + v_new) * dt;

            p.vel[i] = v_new;
            p.pos[i] = x_new;

            vec3 a_new =
                (q / m) * (Eval + kernels::math::structures::cross(v_new, Bval));
            p.acc[i] = a_new;
        }
    }

    inline void boris_kick_2d(_2ParticleGroup& p,
                              const _2Mesh& mesh,
                              const _2Field<vec2>& E,
                              const _2Field<vec3>& B,
                              real dt){
        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t i = 0; i < n; ++i){
            const vec2 x = p.pos[i];
            const vec2 v = p.vel[i];

            real ix, iy;
            bool inside = pos_to_cell_center_coords(mesh, x, ix, iy);

            vec2 Eval{};
            vec3 Bval{};
            if (inside){
                Eval = E.valueAt(ix, iy);
                Bval = B.valueAt(ix, iy);
            }

            const real q = p.charge[i];
            const real m = p.mass[i];
            if (m == real{0}) {
                continue;
            }

            vec2 v_new = boris_velocity_update_2d(v, Eval, Bval, q, m, dt);
            p.vel[i] = v_new;
            p.acc[i] = (q / m) *
                       (Eval + kernels::math::structures::cross(v_new, Bval.z));
        }
    }

    inline void boris_kick_2d_with_phi(_2ParticleGroup& p,
                                       const _2Mesh& mesh,
                                       const _2Field<vec2>& E,
                                       const _2Field<vec3>& B,
                                       const _2Field<real>& phi,
                                       real dt){
        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t i = 0; i < n; ++i){
            const vec2 x = p.pos[i];
            const vec2 v = p.vel[i];

            real ix, iy;
            bool inside = pos_to_cell_center_coords(mesh, x, ix, iy);

            vec2 Eval = gather_E_from_phi_cell_2d(mesh, phi, x);
            vec3 Bval{};
            if (inside){
                Eval += E.valueAt(ix, iy);
                Bval = B.valueAt(ix, iy);
            }

            const real q = p.charge[i];
            const real m = p.mass[i];
            if (m == real{0}) {
                continue;
            }

            vec2 v_new = boris_velocity_update_2d(v, Eval, Bval, q, m, dt);
            p.vel[i] = v_new;
            p.acc[i] = (q / m) *
                       (Eval + kernels::math::structures::cross(v_new, Bval.z));
        }
    }

    inline void boris_push_2d_staggered(_2ParticleGroup& p,
                                        const _2Mesh& mesh,
                                        const _2Field<vec2>& E,
                                        const _2Field<vec3>& B,
                                        real dt){
        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t i = 0; i < n; ++i){
            const vec2 x = p.pos[i];
            const vec2 v = p.vel[i];

            real ix, iy;
            bool inside = pos_to_cell_center_coords(mesh, x, ix, iy);

            vec2 Eval{};
            vec3 Bval{};
            if (inside){
                Eval = E.valueAt(ix, iy);
                Bval = B.valueAt(ix, iy);
            }

            const real q = p.charge[i];
            const real m = p.mass[i];
            if (m == real{0}) {
                continue;
            }

            vec2 v_new = boris_velocity_update_2d(v, Eval, Bval, q, m, dt);
            vec2 x_new = x + v_new * dt;

            p.vel[i] = v_new;
            p.pos[i] = x_new;
            p.acc[i] = (q / m) *
                       (Eval + kernels::math::structures::cross(v_new, Bval.z));
        }
    }

    inline void boris_push_2d_staggered_with_phi(_2ParticleGroup& p,
                                                 const _2Mesh& mesh,
                                                 const _2Field<vec2>& E,
                                                 const _2Field<vec3>& B,
                                                 const _2Field<real>& phi,
                                                 real dt){
        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t i = 0; i < n; ++i){
            const vec2 x = p.pos[i];
            const vec2 v = p.vel[i];

            real ix, iy;
            bool inside = pos_to_cell_center_coords(mesh, x, ix, iy);

            vec2 Eval = gather_E_from_phi_cell_2d(mesh, phi, x);
            vec3 Bval{};
            if (inside){
                Eval += E.valueAt(ix, iy);
                Bval = B.valueAt(ix, iy);
            }

            const real q = p.charge[i];
            const real m = p.mass[i];
            if (m == real{0}) {
                continue;
            }

            vec2 v_new = boris_velocity_update_2d(v, Eval, Bval, q, m, dt);
            vec2 x_new = x + v_new * dt;

            p.vel[i] = v_new;
            p.pos[i] = x_new;
            p.acc[i] = (q / m) *
                       (Eval + kernels::math::structures::cross(v_new, Bval.z));
        }
    }

    // ---------------------------------------------------------------------
    // Velocity Verlet for electrostatic fields (E-only, B ignored).
    // This is useful as a second-order integrator in purely electric
    // problems. For full EM with significant B, prefer Boris.
    // ---------------------------------------------------------------------
    inline void verlet_push_2d(_2ParticleGroup& p,
                               const _2Mesh& mesh,
                               const _2Field<vec2>& E,
                               real dt){
        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t i = 0; i < n; ++i){
            vec2 x = p.pos[i];
            vec2 v = p.vel[i];

            real ix, iy;
            bool inside = pos_to_cell_center_coords(mesh, x, ix, iy);

            vec2 E0{};
            if (inside){
                E0 = E.valueAt(ix, iy);
            }

            const real q = p.charge[i];
            const real m = p.mass[i];
            if (m == real{0}) {
                continue;
            }

            vec2 a0 = (q / m) * E0;

            // Half-kick
            vec2 v_half = v + static_cast<real>(0.5) * dt * a0;
            // Drift
            vec2 x_new  = x + dt * v_half;

            // Recompute field at new position.
            real ix1, iy1;
            bool inside1 = pos_to_cell_center_coords(mesh, x_new, ix1, iy1);
            vec2 E1{};
            if (inside1){
                E1 = E.valueAt(ix1, iy1);
            }

            vec2 a1 = (q / m) * E1;

            // Second half-kick
            vec2 v_new = v_half + static_cast<real>(0.5) * dt * a1;

            p.pos[i] = x_new;
            p.vel[i] = v_new;
            p.acc[i] = a1;
        }
    }

    inline void verlet_push_yee_2d(_2ParticleGroup& p,
                                   const _2FieldManager& fm,
                                   real dt){
        const auto& mesh = fm.mesh();
        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t i = 0; i < n; ++i){
            const vec2 x = p.pos[i];
            vec2 v = p.vel[i];

            if (x.x < mesh.x_init || x.x > mesh.x_finl ||
                x.y < mesh.y_init || x.y > mesh.y_finl){
                continue;
            }

            const vec2 E0 = sample_yee_E_2d(fm, x);

            const real q = p.charge[i];
            const real m = p.mass[i];
            if (m == real{0}) {
                continue;
            }

            const vec2 a0 = (q / m) * E0;
            const vec2 v_half = v + static_cast<real>(0.5) * dt * a0;
            const vec2 x_new = x + dt * v_half;

            vec2 E1{};
            if (x_new.x >= mesh.x_init && x_new.x <= mesh.x_finl &&
                x_new.y >= mesh.y_init && x_new.y <= mesh.y_finl){
                E1 = sample_yee_E_2d(fm, x_new);
            }

            const vec2 a1 = (q / m) * E1;
            const vec2 v_new = v_half + static_cast<real>(0.5) * dt * a1;

            p.pos[i] = x_new;
            p.vel[i] = v_new;
            p.acc[i] = a1;
        }
    }

    inline void verlet_push_3d(_3ParticleGroup& p,
                               const _3Mesh& mesh,
                               const _3Field<vec3>& E,
                               real dt){
        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t i = 0; i < n; ++i){
            vec3 x = p.pos[i];
            vec3 v = p.vel[i];

            real ix, iy, iz;
            bool inside = pos_to_cell_center_coords(mesh, x, ix, iy, iz);

            vec3 E0{};
            if (inside){
                E0 = E.valueAt(ix, iy, iz);
            }

            const real q = p.charge[i];
            const real m = p.mass[i];
            if (m == real{0}) {
                continue;
            }

            vec3 a0 = (q / m) * E0;

            vec3 v_half = v + static_cast<real>(0.5) * dt * a0;
            vec3 x_new  = x + dt * v_half;

            real ix1, iy1, iz1;
            bool inside1 = pos_to_cell_center_coords(mesh, x_new, ix1, iy1, iz1);
            vec3 E1{};
            if (inside1){
                E1 = E.valueAt(ix1, iy1, iz1);
            }

            vec3 a1 = (q / m) * E1;
            vec3 v_new = v_half + static_cast<real>(0.5) * dt * a1;

            p.pos[i] = x_new;
            p.vel[i] = v_new;
            p.acc[i] = a1;
        }
    }
}
