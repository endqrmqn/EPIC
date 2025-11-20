#include <cmath>

/*
    math/vec3.hpp
    --------------

    Lightweight 3-component vector type used throughout EPIC for small,
    fixed-size 3D math: positions, velocities, accelerations, forces,
    and general geometric or physical quantities.

    This struct is intentionally minimal:
      - trivially copyable
      - no heap allocations
      - no virtual functions
      - plain-old-data layout (x, y, z contiguous in memory)

    The API provides:
      - basic arithmetic (+, -, scalar multiply/divide)
      - geometric operations (dot, cross, length, normalization)
      - projection of one vector onto another

    Vec3 is designed for clarity and low overhead. For bulk particle
    storage and high-performance kernels, EPIC uses SoA (structure-of-
    arrays) containers instead; Vec3 is for single-vector math only.

    All functions are `inline` for zero call overhead.

    TODO:
        Add constexpr support for all operations once C++20 integrations stabilize.
        Add in-place operators (+=, -=, *=, /=) to reduce temporaries.
        Add a fast-path normalize() variant for hotspots where zero-length vectors are impossible.
        Add helper functions: distance(a, b), angle(a, b), lerp(a, b, t).
        Consider SIMD-accelerated versions for inner-loop kernels.
        Add Vec3f (float) for memory-intensive particle workloads.
        Write unit tests for zero-length normalization, projection edge cases, and NaN behavior.
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