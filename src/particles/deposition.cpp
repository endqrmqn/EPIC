#pragma once

#include <cmath>
#include <vector>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"

namespace src::particles{
    using real = types::real;
    using vec2 = kernels::math::structures::vec2;
    using vec3 = kernels::math::structures::vec3;

    using _2Mesh = kernels::physics::structures::_2Mesh;
    using _3Mesh = kernels::physics::structures::_3Mesh;

    template<typename T>
    using _2Field = kernels::physics::structures::_2Field<T>;
    template<typename T>
    using _3Field = kernels::physics::structures::_3Field<T>;

    using _2ParticleGroup = kernels::physics::structures::_2ParticleGroup;
    using _3ParticleGroup = kernels::physics::structures::_3ParticleGroup;

    // ---------------------------------------------------------------------
    // Charge/current deposition with first-order (CIC) shape functions.
    // This deposits charge density rho and current density J onto
    // cell-centred grids, consistent with the interpolation used to
    // sample fields back to the particles.
    // ---------------------------------------------------------------------

    namespace {
        struct CellCenterWeights1D {
            int i0{0};
            real w0{0};
            real w1{0};
        };

        inline CellCenterWeights1D cell_center_weights_1d(real x,
                                                          real origin,
                                                          real h,
                                                          int n_cells){
            CellCenterWeights1D weights;

            real s = (x - origin) / h - static_cast<real>(0.5);
            int i0 = static_cast<int>(std::floor(s));

            if (i0 < 0) {
                i0 = 0;
                s = static_cast<real>(0);
            }
            if (i0 > n_cells - 2) {
                i0 = n_cells - 2;
                s = static_cast<real>(n_cells - 1);
            }

            const real frac = s - static_cast<real>(i0);
            weights.i0 = i0;
            weights.w0 = static_cast<real>(1.0) - frac;
            weights.w1 = frac;
            return weights;
        }

        inline real weight_at(const CellCenterWeights1D& weights, int idx){
            if (idx == weights.i0) {
                return weights.w0;
            }
            if (idx == weights.i0 + 1) {
                return weights.w1;
            }
            return real{0};
        }

        inline void deposit_x_segment_2d(const _2Mesh& mesh,
                                         real x0,
                                         real x1,
                                         real y_fixed,
                                         real q,
                                         real dt,
                                         real scale,
                                         _2Field<real>& Jx_face){
            const int nx = mesh.nx_cells();
            const int ny = mesh.ny_cells();
            const real dx = mesh.dx();
            const real dy = mesh.dy();

            if (dt == real{0}) {
                return;
            }

            const CellCenterWeights1D wx0 =
                cell_center_weights_1d(x0, mesh.x_init, dx, nx);
            const CellCenterWeights1D wx1 =
                cell_center_weights_1d(x1, mesh.x_init, dx, nx);
            const CellCenterWeights1D wy =
                cell_center_weights_1d(y_fixed, mesh.y_init, dy, ny);

            const int kmin = std::min(wx0.i0, wx1.i0);
            const int kmax = std::max(wx0.i0 + 1, wx1.i0 + 1);

            real cumulative = real{0};
            for (int k = kmin; k <= kmax; ++k){
                cumulative += weight_at(wx1, k) - weight_at(wx0, k);
                const int face_i = k + 1;
                if (face_i < 0 || face_i >= Jx_face.n_x) {
                    continue;
                }

                for (int j = wy.i0; j <= wy.i0 + 1; ++j){
                    if (j < 0 || j >= Jx_face.n_y) {
                        continue;
                    }
                    const real transverse = weight_at(wy, j);
                    Jx_face.valueAt(face_i, j) +=
                        -scale * q * cumulative * transverse / (dt * dy);
                }
            }
        }

        inline void deposit_y_segment_2d(const _2Mesh& mesh,
                                         real x_fixed,
                                         real y0,
                                         real y1,
                                         real q,
                                         real dt,
                                         real scale,
                                         _2Field<real>& Jy_face){
            const int nx = mesh.nx_cells();
            const int ny = mesh.ny_cells();
            const real dx = mesh.dx();
            const real dy = mesh.dy();

            if (dt == real{0}) {
                return;
            }

            const CellCenterWeights1D wx =
                cell_center_weights_1d(x_fixed, mesh.x_init, dx, nx);
            const CellCenterWeights1D wy0 =
                cell_center_weights_1d(y0, mesh.y_init, dy, ny);
            const CellCenterWeights1D wy1 =
                cell_center_weights_1d(y1, mesh.y_init, dy, ny);

            const int kmin = std::min(wy0.i0, wy1.i0);
            const int kmax = std::max(wy0.i0 + 1, wy1.i0 + 1);

            real cumulative = real{0};
            for (int k = kmin; k <= kmax; ++k){
                cumulative += weight_at(wy1, k) - weight_at(wy0, k);
                const int face_j = k + 1;
                if (face_j < 0 || face_j >= Jy_face.n_y) {
                    continue;
                }

                for (int i = wx.i0; i <= wx.i0 + 1; ++i){
                    if (i < 0 || i >= Jy_face.n_x) {
                        continue;
                    }
                    const real transverse = weight_at(wx, i);
                    Jy_face.valueAt(i, face_j) +=
                        -scale * q * cumulative * transverse / (dt * dx);
                }
            }
        }

        inline bool pos_to_cell_center_coords_deposition(const _2Mesh& mesh,
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

        inline bool pos_to_cell_center_coords_deposition(const _3Mesh& mesh,
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

    inline void deposit_rho_J_2d(const _2Mesh& mesh,
                                 const _2ParticleGroup& p,
                                 _2Field<real>& rho,
                                 _2Field<real>& Jx,
                                 _2Field<real>& Jy){
        const int nx = mesh.nx_cells();
        const int ny = mesh.ny_cells();

        const real dx = mesh.dx();
        const real dy = mesh.dy();
        const real cell_area = dx * dy;

        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t ip = 0; ip < n; ++ip){
            const vec2 x = p.pos[ip];
            const vec2 v = p.vel[ip];
            const real q = p.charge[ip];

            real ix, iy;
            if (!pos_to_cell_center_coords_deposition(mesh, x, ix, iy)){
                continue;
            }

            int i0 = static_cast<int>(std::floor(ix));
            int j0 = static_cast<int>(std::floor(iy));

            if (i0 < 0)      i0 = 0;
            if (j0 < 0)      j0 = 0;
            if (i0 > nx - 2) i0 = nx - 2;
            if (j0 > ny - 2) j0 = ny - 2;

            const real sx = ix - static_cast<real>(i0);
            const real sy = iy - static_cast<real>(j0);

            const real w00 = (static_cast<real>(1.0) - sx) *
                             (static_cast<real>(1.0) - sy);
            const real w10 = sx * (static_cast<real>(1.0) - sy);
            const real w01 = (static_cast<real>(1.0) - sx) * sy;
            const real w11 = sx * sy;

            const real q_over_V = q / cell_area;
            const real qvx_over_V = q * v.x / cell_area;
            const real qvy_over_V = q * v.y / cell_area;

            rho.valueAt(i0,     j0    ) += q_over_V * w00;
            rho.valueAt(i0 + 1, j0    ) += q_over_V * w10;
            rho.valueAt(i0,     j0 + 1) += q_over_V * w01;
            rho.valueAt(i0 + 1, j0 + 1) += q_over_V * w11;

            Jx.valueAt(i0,     j0    ) += qvx_over_V * w00;
            Jx.valueAt(i0 + 1, j0    ) += qvx_over_V * w10;
            Jx.valueAt(i0,     j0 + 1) += qvx_over_V * w01;
            Jx.valueAt(i0 + 1, j0 + 1) += qvx_over_V * w11;

            Jy.valueAt(i0,     j0    ) += qvy_over_V * w00;
            Jy.valueAt(i0 + 1, j0    ) += qvy_over_V * w10;
            Jy.valueAt(i0,     j0 + 1) += qvy_over_V * w01;
            Jy.valueAt(i0 + 1, j0 + 1) += qvy_over_V * w11;
        }
    }

    inline void deposit_J_yee_2d(const _2Mesh& mesh,
                                 const _2ParticleGroup& p,
                                 _2Field<real>& Jx_face,
                                 _2Field<real>& Jy_face){
        const int nx_xfaces = mesh.nx_xfaces();
        const int ny_xfaces = mesh.ny_xfaces();
        const int nx_yfaces = mesh.nx_yfaces();
        const int ny_yfaces = mesh.ny_yfaces();

        Jx_face.n_x = nx_xfaces;
        Jx_face.n_y = ny_xfaces;
        Jy_face.n_x = nx_yfaces;
        Jy_face.n_y = ny_yfaces;
        Jx_face.data.assign(static_cast<std::size_t>(nx_xfaces) * ny_xfaces, real{0});
        Jy_face.data.assign(static_cast<std::size_t>(nx_yfaces) * ny_yfaces, real{0});

        const real dx = mesh.dx();
        const real dy = mesh.dy();
        const real cell_area = dx * dy;

        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t ip = 0; ip < n; ++ip){
            const vec2 x = p.pos[ip];
            const vec2 v = p.vel[ip];
            const real q = p.charge[ip];

            // Deposit Jx on Ex faces at (i*dx, (j+1/2)dy).
            {
                real fx = (x.x - mesh.x_init) / dx;
                real fy = (x.y - mesh.y_init) / dy - static_cast<real>(0.5);

                int i0 = static_cast<int>(std::floor(fx));
                int j0 = static_cast<int>(std::floor(fy));

                if (i0 < 0) i0 = 0;
                if (j0 < 0) j0 = 0;
                if (i0 > nx_xfaces - 2) i0 = nx_xfaces - 2;
                if (j0 > ny_xfaces - 2) j0 = ny_xfaces - 2;

                const real sx = fx - static_cast<real>(i0);
                const real sy = fy - static_cast<real>(j0);

                const real w00 = (static_cast<real>(1.0) - sx) * (static_cast<real>(1.0) - sy);
                const real w10 = sx * (static_cast<real>(1.0) - sy);
                const real w01 = (static_cast<real>(1.0) - sx) * sy;
                const real w11 = sx * sy;

                const real qvx_over_V = q * v.x / cell_area;

                Jx_face.valueAt(i0,     j0    ) += qvx_over_V * w00;
                Jx_face.valueAt(i0 + 1, j0    ) += qvx_over_V * w10;
                Jx_face.valueAt(i0,     j0 + 1) += qvx_over_V * w01;
                Jx_face.valueAt(i0 + 1, j0 + 1) += qvx_over_V * w11;
            }

            // Deposit Jy on Ey faces at ((i+1/2)dx, j*dy).
            {
                real fx = (x.x - mesh.x_init) / dx - static_cast<real>(0.5);
                real fy = (x.y - mesh.y_init) / dy;

                int i0 = static_cast<int>(std::floor(fx));
                int j0 = static_cast<int>(std::floor(fy));

                if (i0 < 0) i0 = 0;
                if (j0 < 0) j0 = 0;
                if (i0 > nx_yfaces - 2) i0 = nx_yfaces - 2;
                if (j0 > ny_yfaces - 2) j0 = ny_yfaces - 2;

                const real sx = fx - static_cast<real>(i0);
                const real sy = fy - static_cast<real>(j0);

                const real w00 = (static_cast<real>(1.0) - sx) * (static_cast<real>(1.0) - sy);
                const real w10 = sx * (static_cast<real>(1.0) - sy);
                const real w01 = (static_cast<real>(1.0) - sx) * sy;
                const real w11 = sx * sy;

                const real qvy_over_V = q * v.y / cell_area;

                Jy_face.valueAt(i0,     j0    ) += qvy_over_V * w00;
                Jy_face.valueAt(i0 + 1, j0    ) += qvy_over_V * w10;
                Jy_face.valueAt(i0,     j0 + 1) += qvy_over_V * w01;
                Jy_face.valueAt(i0 + 1, j0 + 1) += qvy_over_V * w11;
            }
        }
    }

    inline void deposit_J_yee_2d_charge_conserving(
        const _2Mesh& mesh,
        const std::vector<vec2>& old_positions,
        const _2ParticleGroup& p_new,
        real dt,
        _2Field<real>& Jx_face,
        _2Field<real>& Jy_face){

        const int nx_xfaces = mesh.nx_xfaces();
        const int ny_xfaces = mesh.ny_xfaces();
        const int nx_yfaces = mesh.nx_yfaces();
        const int ny_yfaces = mesh.ny_yfaces();

        Jx_face.n_x = nx_xfaces;
        Jx_face.n_y = ny_xfaces;
        Jy_face.n_x = nx_yfaces;
        Jy_face.n_y = ny_yfaces;
        Jx_face.data.assign(static_cast<std::size_t>(nx_xfaces) * ny_xfaces, real{0});
        Jy_face.data.assign(static_cast<std::size_t>(nx_yfaces) * ny_yfaces, real{0});

        const std::size_t n = kernels::physics::structures::size(p_new);
        if (old_positions.size() != n) {
            return;
        }

        for (std::size_t ip = 0; ip < n; ++ip){
            const vec2& x_old = old_positions[ip];
            const vec2& x_new = p_new.pos[ip];
            const real q = p_new.charge[ip];

            deposit_x_segment_2d(mesh, x_old.x, x_new.x, x_old.y,
                                 q, dt, static_cast<real>(0.5), Jx_face);
            deposit_y_segment_2d(mesh, x_new.x, x_old.y, x_new.y,
                                 q, dt, static_cast<real>(0.5), Jy_face);

            deposit_y_segment_2d(mesh, x_old.x, x_old.y, x_new.y,
                                 q, dt, static_cast<real>(0.5), Jy_face);
            deposit_x_segment_2d(mesh, x_old.x, x_new.x, x_new.y,
                                 q, dt, static_cast<real>(0.5), Jx_face);
        }
    }

    inline void deposit_rho_J_3d(const _3Mesh& mesh,
                                 const _3ParticleGroup& p,
                                 _3Field<real>& rho,
                                 _3Field<real>& Jx,
                                 _3Field<real>& Jy,
                                 _3Field<real>& Jz){
        const int nx = mesh.nx_cells();
        const int ny = mesh.ny_cells();
        const int nz = mesh.nz_cells();

        const real dx = mesh.dx();
        const real dy = mesh.dy();
        const real dz = mesh.dz();
        const real cell_vol = dx * dy * dz;

        const std::size_t n = kernels::physics::structures::size(p);

        for (std::size_t ip = 0; ip < n; ++ip){
            const vec3 x = p.pos[ip];
            const vec3 v = p.vel[ip];
            const real q = p.charge[ip];

            real ix, iy, iz;
            if (!pos_to_cell_center_coords_deposition(mesh, x, ix, iy, iz)){
                continue;
            }

            int i0 = static_cast<int>(std::floor(ix));
            int j0 = static_cast<int>(std::floor(iy));
            int k0 = static_cast<int>(std::floor(iz));

            if (i0 < 0)      i0 = 0;
            if (j0 < 0)      j0 = 0;
            if (k0 < 0)      k0 = 0;
            if (i0 > nx - 2) i0 = nx - 2;
            if (j0 > ny - 2) j0 = ny - 2;
            if (k0 > nz - 2) k0 = nz - 2;

            const real sx = ix - static_cast<real>(i0);
            const real sy = iy - static_cast<real>(j0);
            const real sz = iz - static_cast<real>(k0);

            const real wx0 = static_cast<real>(1.0) - sx;
            const real wx1 = sx;
            const real wy0 = static_cast<real>(1.0) - sy;
            const real wy1 = sy;
            const real wz0 = static_cast<real>(1.0) - sz;
            const real wz1 = sz;

            const real q_over_V  = q / cell_vol;
            const real qvx_over_V = q * v.x / cell_vol;
            const real qvy_over_V = q * v.y / cell_vol;
            const real qvz_over_V = q * v.z / cell_vol;

            for (int dk = 0; dk <= 1; ++dk){
                const int k = k0 + dk;
                const real wk = (dk == 0) ? wz0 : wz1;
                for (int dj = 0; dj <= 1; ++dj){
                    const int j = j0 + dj;
                    const real wjk = wk * ((dj == 0) ? wy0 : wy1);
                    for (int di = 0; di <= 1; ++di){
                        const int i = i0 + di;
                        const real w = wjk * ((di == 0) ? wx0 : wx1);

                        rho.valueAt(i, j, k) += q_over_V  * w;
                        Jx.valueAt(i, j, k)  += qvx_over_V * w;
                        Jy.valueAt(i, j, k)  += qvy_over_V * w;
                        Jz.valueAt(i, j, k)  += qvz_over_V * w;
                    }
                }
            }
        }
    }
}
