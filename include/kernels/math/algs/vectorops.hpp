#pragma once

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"

#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"

#include <array>


namespace kernels::math::algs{
    using real = types::real;
    using vec2 = kernels::math::structures::vec2;
    using vec3 = kernels::math::structures::vec3;

    using _2Mesh = kernels::physics::structures::_2Mesh;
    using _3Mesh = kernels::physics::structures::_3Mesh;

    template<typename T>
    using _2Field = kernels::physics::structures::_2Field<T>;
    template<typename T>
    using _3Field = kernels::physics::structures::_3Field<T>;

    // -----------------------------
    // Interior-only operators
    // -----------------------------

    // Gradient at a single interior cell of a 2D scalar field.
    // Precondition: 1 <= i < nx-1, 1 <= j < ny-1.
    //
    // Use standard centred second-order differences in the interior.
    inline vec2 gradPointInterior(const _2Mesh& m,
                                  const _2Field<real>& f,
                                  int i, int j){
        const real dx = m.dx();
        const real dy = m.dy();

        const real dfdx =
            (f.valueAt(i + 1, j) - f.valueAt(i - 1, j)) / (2.0 * dx);
        const real dfdy =
            (f.valueAt(i, j + 1) - f.valueAt(i, j - 1)) / (2.0 * dy);

        return vec2(dfdx, dfdy);
    }

    // Gradient at a single interior cell of a 3D scalar field.
    // Precondition: 1 <= i < nx-1, 1 <= j < ny-1, 1 <= k < nz-1.
    //
    // Centred second-order differences in all directions.
    inline vec3 gradPointInterior(const _3Mesh& m,
                                  const _3Field<real>& f,
                                  int i, int j, int k){
        const real dx = m.dx();
        const real dy = m.dy();
        const real dz = m.dz();

        const real dfdx =
            (f.valueAt(i + 1, j, k) - f.valueAt(i - 1, j, k)) / (2.0 * dx);
        const real dfdy =
            (f.valueAt(i, j + 1, k) - f.valueAt(i, j - 1, k)) / (2.0 * dy);
        const real dfdz =
            (f.valueAt(i, j, k + 1) - f.valueAt(i, j, k - 1)) / (2.0 * dz);

        return vec3(dfdx, dfdy, dfdz);
    }

    // Divergence at a single interior cell of a 2D vector field F = (Fx, Fy).
    // Precondition: 1 <= i < nx-1, 1 <= j < ny-1.
    //
    // Centred second-order differences on each component.
    inline real divPointInterior(const _2Mesh& m,
                                 const _2Field<vec2>& F,
                                 int i, int j){
        const real dx = m.dx();
        const real dy = m.dy();

        const real dFxdx =
            (F.valueAt(i + 1, j).x - F.valueAt(i - 1, j).x) / (2.0 * dx);
        const real dFydy =
            (F.valueAt(i, j + 1).y - F.valueAt(i, j - 1).y) / (2.0 * dy);

        return dFxdx + dFydy;
    }

    // Divergence at a single interior cell of a 3D vector field F = (Fx, Fy, Fz).
    // Precondition: 1 <= i < nx-1, 1 <= j < ny-1, 1 <= k < nz-1.
    inline real divPointInterior(const _3Mesh& m,
                                 const _3Field<vec3>& F,
                                 int i, int j, int k){
        const real dx = m.dx();
        const real dy = m.dy();
        const real dz = m.dz();

        const real dFxdx =
            (F.valueAt(i + 1, j, k).x - F.valueAt(i - 1, j, k).x) / (2.0 * dx);
        const real dFydy =
            (F.valueAt(i, j + 1, k).y - F.valueAt(i, j - 1, k).y) / (2.0 * dy);
        const real dFzdz =
            (F.valueAt(i, j, k + 1).z - F.valueAt(i, j, k - 1).z) / (2.0 * dz);

        return dFxdx + dFydy + dFzdz;
    }

    // Curl at a single interior cell of a 3D vector field F.
    // Precondition: 1 <= i < nx-1, 1 <= j < ny-1, 1 <= k < nz-1.
    inline vec3 curlPointInterior(const _3Mesh& m,
                                  const _3Field<vec3>& F,
                                  int i, int j, int k){
        const real dx = m.dx();
        const real dy = m.dy();
        const real dz = m.dz();

        const vec3 Fx_plus  = F.valueAt(i,   j,   k+1);
        const vec3 Fx_minus = F.valueAt(i,   j,   k-1);
        const vec3 Fy_plus  = F.valueAt(i+1, j,   k);
        const vec3 Fy_minus = F.valueAt(i-1, j,   k);
        const vec3 Fz_plus  = F.valueAt(i,   j+1, k);
        const vec3 Fz_minus = F.valueAt(i,   j-1, k);

        const real dFz_dy = (Fz_plus.z - Fz_minus.z) / (2.0 * dy);
        const real dFy_dz = (Fx_plus.y - Fx_minus.y) / (2.0 * dz);

        const real dFx_dz = (Fx_plus.x - Fx_minus.x) / (2.0 * dz);
        const real dFz_dx = (Fy_plus.z - Fy_minus.z) / (2.0 * dx);

        const real dFy_dx = (Fy_plus.y - Fy_minus.y) / (2.0 * dx);
        const real dFx_dy = (Fz_plus.x - Fz_minus.x) / (2.0 * dy);

        return vec3(
            dFz_dy - dFy_dz,
            dFx_dz - dFz_dx,
            dFy_dx - dFx_dy
        );
    }

    // -----------------------------
    // Boundary-aware wrappers
    // -----------------------------

    inline vec2 gradPoint(const _2Mesh& m,
                          const _2Field<real>& f,
                          int i, int j){
        const int nx = m.nx_cells();
        const int ny = m.ny_cells();

        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1){
            return gradPointInterior(m, f, i, j);
        }

        const real dx = m.dx();
        const real dy = m.dy();

        // d/dx
        real dfdx;
        if (i <= 0){
            dfdx = (f.valueAt(1, j) - f.valueAt(0, j)) / dx;
        } else if (i >= nx - 1){
            dfdx = (f.valueAt(nx - 1, j) - f.valueAt(nx - 2, j)) / dx;
        } else {
            dfdx = (f.valueAt(i + 1, j) - f.valueAt(i - 1, j)) / (2.0 * dx);
        }

        // d/dy
        real dfdy;
        if (j <= 0){
            dfdy = (f.valueAt(i, 1) - f.valueAt(i, 0)) / dy;
        } else if (j >= ny - 1){
            dfdy = (f.valueAt(i, ny - 1) - f.valueAt(i, ny - 2)) / dy;
        } else {
            dfdy = (f.valueAt(i, j + 1) - f.valueAt(i, j - 1)) / (2.0 * dy);
        }

        return vec2(dfdx, dfdy);
    }

    inline vec3 gradPoint(const _3Mesh& m,
                          const _3Field<real>& f,
                          int i, int j, int k){
        const int nx = m.nx_cells();
        const int ny = m.ny_cells();
        const int nz = m.nz_cells();

        if (i > 0 && i < nx - 1 &&
            j > 0 && j < ny - 1 &&
            k > 0 && k < nz - 1){
            return gradPointInterior(m, f, i, j, k);
        }

        const real dx = m.dx();
        const real dy = m.dy();
        const real dz = m.dz();

        real dfdx;
        if (i <= 0){
            dfdx = (f.valueAt(1, j, k) - f.valueAt(0, j, k)) / dx;
        } else if (i >= nx - 1){
            dfdx = (f.valueAt(nx - 1, j, k) - f.valueAt(nx - 2, j, k)) / dx;
        } else {
            dfdx = (f.valueAt(i + 1, j, k) - f.valueAt(i - 1, j, k)) / (2.0 * dx);
        }

        real dfdy;
        if (j <= 0){
            dfdy = (f.valueAt(i, 1, k) - f.valueAt(i, 0, k)) / dy;
        } else if (j >= ny - 1){
            dfdy = (f.valueAt(i, ny - 1, k) - f.valueAt(i, ny - 2, k)) / dy;
        } else {
            dfdy = (f.valueAt(i, j + 1, k) - f.valueAt(i, j - 1, k)) / (2.0 * dy);
        }

        real dfdz;
        if (k <= 0){
            dfdz = (f.valueAt(i, j, 1) - f.valueAt(i, j, 0)) / dz;
        } else if (k >= nz - 1){
            dfdz = (f.valueAt(i, j, nz - 1) - f.valueAt(i, j, nz - 2)) / dz;
        } else {
            dfdz = (f.valueAt(i, j, k + 1) - f.valueAt(i, j, k - 1)) / (2.0 * dz);
        }

        return vec3(dfdx, dfdy, dfdz);
    }

    inline real divPoint(const _2Mesh& m,
                         const _2Field<vec2>& F,
                         int i, int j){
        const int nx = m.nx_cells();
        const int ny = m.ny_cells();

        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1){
            return divPointInterior(m, F, i, j);
        }

        const real dx = m.dx();
        const real dy = m.dy();

        real dFxdx;
        if (i <= 0){
            dFxdx = (F.valueAt(1, j).x - F.valueAt(0, j).x) / dx;
        } else if (i >= nx - 1){
            dFxdx = (F.valueAt(nx - 1, j).x - F.valueAt(nx - 2, j).x) / dx;
        } else {
            dFxdx = (F.valueAt(i + 1, j).x - F.valueAt(i - 1, j).x) / (2.0 * dx);
        }

        real dFydy;
        if (j <= 0){
            dFydy = (F.valueAt(i, 1).y - F.valueAt(i, 0).y) / dy;
        } else if (j >= ny - 1){
            dFydy = (F.valueAt(i, ny - 1).y - F.valueAt(i, ny - 2).y) / dy;
        } else {
            dFydy = (F.valueAt(i, j + 1).y - F.valueAt(i, j - 1).y) / (2.0 * dy);
        }

        return dFxdx + dFydy;
    }

    inline real divPoint(const _3Mesh& m,
                         const _3Field<vec3>& F,
                         int i, int j, int k){
        const int nx = m.nx_cells();
        const int ny = m.ny_cells();
        const int nz = m.nz_cells();

        if (i > 0 && i < nx - 1 &&
            j > 0 && j < ny - 1 &&
            k > 0 && k < nz - 1){
            return divPointInterior(m, F, i, j, k);
        }

        const real dx = m.dx();
        const real dy = m.dy();
        const real dz = m.dz();

        real dFxdx;
        if (i <= 0){
            dFxdx = (F.valueAt(1, j, k).x - F.valueAt(0, j, k).x) / dx;
        } else if (i >= nx - 1){
            dFxdx = (F.valueAt(nx - 1, j, k).x - F.valueAt(nx - 2, j, k).x) / dx;
        } else {
            dFxdx = (F.valueAt(i + 1, j, k).x - F.valueAt(i - 1, j, k).x) / (2.0 * dx);
        }

        real dFydy;
        if (j <= 0){
            dFydy = (F.valueAt(i, 1, k).y - F.valueAt(i, 0, k).y) / dy;
        } else if (j >= ny - 1){
            dFydy = (F.valueAt(i, ny - 1, k).y - F.valueAt(i, ny - 2, k).y) / dy;
        } else {
            dFydy = (F.valueAt(i, j + 1, k).y - F.valueAt(i, j - 1, k).y) / (2.0 * dy);
        }

        real dFzdz;
        if (k <= 0){
            dFzdz = (F.valueAt(i, j, 1).z - F.valueAt(i, j, 0).z) / dz;
        } else if (k >= nz - 1){
            dFzdz = (F.valueAt(i, j, nz - 1).z - F.valueAt(i, j, nz - 2).z) / dz;
        } else {
            dFzdz = (F.valueAt(i, j, k + 1).z - F.valueAt(i, j, k - 1).z) / (2.0 * dz);
        }

        return dFxdx + dFydy + dFzdz;
    }

    inline vec3 curlPoint(const _3Mesh& m,
                          const _3Field<vec3>& F,
                          int i, int j, int k){
        const int nx = m.nx_cells();
        const int ny = m.ny_cells();
        const int nz = m.nz_cells();

        if (i > 0 && i < nx - 1 &&
            j > 0 && j < ny - 1 &&
            k > 0 && k < nz - 1){
            return curlPointInterior(m, F, i, j, k);
        }

        const real dx = m.dx();
        const real dy = m.dy();
        const real dz = m.dz();

        auto F_ijk = [&](int ii, int jj, int kk) -> vec3 {
            return F.valueAt(ii, jj, kk);
        };

        auto d_dy = [&](int ii, int jj, int kk, auto comp) -> real {
            if (jj <= 0){
                return (comp(F_ijk(ii, 1, kk)) - comp(F_ijk(ii, 0, kk))) / dy;
            } else if (jj >= ny - 1){
                return (comp(F_ijk(ii, ny - 1, kk)) - comp(F_ijk(ii, ny - 2, kk))) / dy;
            } else {
                return (comp(F_ijk(ii, jj + 1, kk)) - comp(F_ijk(ii, jj - 1, kk))) / (2.0 * dy);
            }
        };

        auto d_dz = [&](int ii, int jj, int kk, auto comp) -> real {
            if (kk <= 0){
                return (comp(F_ijk(ii, jj, 1)) - comp(F_ijk(ii, jj, 0))) / dz;
            } else if (kk >= nz - 1){
                return (comp(F_ijk(ii, jj, nz - 1)) - comp(F_ijk(ii, jj, nz - 2))) / dz;
            } else {
                return (comp(F_ijk(ii, jj, kk + 1)) - comp(F_ijk(ii, jj, kk - 1))) / (2.0 * dz);
            }
        };

        auto d_dx = [&](int ii, int jj, int kk, auto comp) -> real {
            if (ii <= 0){
                return (comp(F_ijk(1, jj, kk)) - comp(F_ijk(0, jj, kk))) / dx;
            } else if (ii >= nx - 1){
                return (comp(F_ijk(nx - 1, jj, kk)) - comp(F_ijk(nx - 2, jj, kk))) / dx;
            } else {
                return (comp(F_ijk(ii + 1, jj, kk)) - comp(F_ijk(ii - 1, jj, kk))) / (2.0 * dx);
            }
        };

        const real dFz_dy = d_dy(i, j, k, [](const vec3& v){ return v.z; });
        const real dFy_dz = d_dz(i, j, k, [](const vec3& v){ return v.y; });

        const real dFx_dz = d_dz(i, j, k, [](const vec3& v){ return v.x; });
        const real dFz_dx = d_dx(i, j, k, [](const vec3& v){ return v.z; });

        const real dFy_dx = d_dx(i, j, k, [](const vec3& v){ return v.y; });
        const real dFx_dy = d_dy(i, j, k, [](const vec3& v){ return v.x; });

        return vec3(
            dFz_dy - dFy_dz,
            dFx_dz - dFz_dx,
            dFy_dx - dFx_dy
        );
    }
}
