#include <cmath>
#pragma once

/**
 * @file vec3.hpp
 *
 * math/vec3.hpp
 * --------------
 *
 * Lightweight 3-component vector type used throughout EPIC for small,
 * fixed-size 3D math: positions, velocities, accelerations, forces,
 * and general geometric or physical quantities.
 *
 * Purpose:
 *   Provide a minimal, zero-overhead 3D vector for single-vector math
 *   without the baggage of dynamic memory or heavy linear algebra
 *   frameworks.
 *
 * Design:
 *   - trivially copyable
 *   - plain-old-data layout (x, y, z contiguous in memory)
 *   - no heap allocations
 *   - no virtual functions
 *   - inline arithmetic for zero call overhead
 *
 * API includes:
 *   - basic arithmetic (+, -, scalar multiply/divide)
 *   - geometric operations (dot, cross, length, norm)
 *   - vector projection
 *
 * Notes:
 *   EPIC uses SoA (structure-of-arrays) layouts for bulk particle
 *   storage and high-performance kernels; Vec3 is intended only for
 *   local, single-vector operations.
 *
 * TODO:
 *   - Add constexpr support to all operations once C++20 integrations stabilize.
 *   - Add in-place operators (+=, -=, *=, /=) to reduce temporaries.
 *   - Add fast-path normalize() variant for hotspots where zero-length
 *     vectors are impossible.
 *   - Add helpers: distance(a,b), angle(a,b), lerp(a,b,t).
 *   - Consider SIMD-accelerated variants for hot loops.
 *   - Add Vec3f (float) for memory-critical particle workloads.
 *   - Write unit tests for zero-length normalization and projection edge cases.
 */

namespace math
{
    struct Vec3 {
        double x, y, z;
    };

    inline Vec3 operator+(const Vec3& a, const Vec3& b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    inline Vec3 operator+=(const Vec3& a, const Vec3& b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    inline Vec3 operator-(const Vec3& a, const Vec3& b) {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    inline Vec3 operator*(const Vec3& a, double scalar) {
        return {a.x * scalar, a.y * scalar, a.z * scalar};
    }

    inline Vec3 operator*(double scalar, const Vec3& a) {
        return a * scalar;
    }

    inline Vec3 operator/(const Vec3& a, double scalar) {
        return {a.x / scalar, a.y / scalar, a.z / scalar};
    }

    inline Vec3 operator/(double scalar, const Vec3& a) {
        return a / scalar;
    }

    inline double dot(const Vec3& a, const Vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline Vec3 cross(const Vec3& a, const Vec3& b) {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }

    inline double length(const Vec3& a) {
        return std::sqrt(dot(a, a));
    }

    inline Vec3 normalize(const Vec3& a) {
        double len = length(a);
        if (len == 0) {
            return {0, 0, 0}; // Avoid division by zero
        }
        return a / len;
    }

    inline Vec3 proj(const Vec3& a, const Vec3& b) {
        double b_len_sq = dot(b, b);
        if (b_len_sq < 1e-14) return {0,0,0};
        double scalar = dot(a, b) / b_len_sq;
        return b * scalar;
    }
} // namespace math