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
 */

namespace math 
{
    struct GridShape {
        size_t Nx, Ny, Nz;
        size_t sx, sy, sz;

        GridShape(size_t Nx_, size_t Ny_, size_t Nz_,
                  size_t sx_ = 1, size_t sy_ = 1, size_t sz_ = 1)
            : Nx(Nx_), Ny(Ny_), Nz(Nz_), sx(sx_), sy(sy_), sz(sz_) {}
    };




} //namespace math