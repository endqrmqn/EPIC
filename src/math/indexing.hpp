#pragma once
#include <cstddef>
/**
 * @file indexing.hpp
 *
 * Core indexing utilities for EPIC's grid-based data structures.
 *
 * This file handles ONLY one responsibility:
 *      mapping 3D grid coordinates (i, j, k) -> 1D linear memory indices.
 *
 * Why this exists:
 *  - All field arrays (E, B, rho, J, etc.) are stored as flat 1D arrays
 *    for cache efficiency and GPU/CPU uniformity.
 *  - Simulation logic, however, naturally works in 3D grid coordinates.
 *  - These helpers provide the fast, correct transformation between the two.
 *
 * What this file provides:
 *  - GridShape: stores Nx, Ny, Nz and derived strides (sx, sy, sz).
 *  - idx(i, j, k, g): convert a 3D coordinate into a linear array index.
 *  - in_bounds(i, j, k, g): check if a location is inside the domain.
 *  - wrap(x, N): periodic wrap for boundary conditions.
 *  - Offset3 + neighbor(): helper for stencil-based neighbor lookups.
 *
 * What this file intentionally DOES NOT do:
 *  - coordinate system transforms (grid <-> physical space)
 *  - geometry, metrics, or field interpolation
 *  - particle-based indexing or cell searching
 *
 * Keep this file minimal and dependency-free. It is called constantly
 * throughout the simulation and must remain lightweight and inlinable.
 *
 * TODO:
 *   Nothing! This header is intentionally minimal and complete.
 */

namespace math 
{
    // ------------------------------------------
    // Grid description + internal strides
    // ------------------------------------------
    struct GridShape {
        size_t Nx, Ny, Nz;
        size_t sx, sy, sz;

        GridShape(size_t Nx_, size_t Ny_, size_t Nz_)
            : Nx(Nx_), Ny(Ny_), Nz(Nz_),
              sx(1),
              sy(Nx_),
              sz(Nx_ * Ny_) {}
    };

    // ------------------------------------------
    // Flatten (i, j, k) -> linear index
    // ------------------------------------------
    inline constexpr size_t idx(size_t i, size_t j, size_t k,
                                const GridShape& g)
    {
        return i + j * g.sy + k * g.sz;
    }

    // ------------------------------------------
    // Bounds check
    // ------------------------------------------
    inline constexpr bool in_bounds(int i, int j, int k,
                                    const GridShape& g)
    {
        return (i >= 0 && i < (int)g.Nx) &&
               (j >= 0 && j < (int)g.Ny) &&
               (k >= 0 && k < (int)g.Nz);
    }

    // ------------------------------------------
    // Periodic wrap helper
    // ------------------------------------------
    inline constexpr size_t wrap(int x, int N)
    {
        return (x % N + N) % N;
    }

    // ------------------------------------------
    // Neighbor offsets for stencils
    // ------------------------------------------
    struct Offset3 { int di, dj, dk; };

    inline constexpr size_t neighbor(size_t i, size_t j, size_t k,
                                     const Offset3& o,
                                     const GridShape& g)
    {
        return idx(i + o.di, j + o.dj, k + o.dk, g);
    }

    // optional: 6-axis stencil offsets
    static constexpr Offset3 axial6[6] = {
        {+1,0,0}, {-1,0,0},
        {0,+1,0}, {0,-1,0},
        {0,0,+1}, {0,0,-1}
    };
} //namespace math