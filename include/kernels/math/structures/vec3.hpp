#pragma once
#include <cmath>
#include "include/types.hpp"

namespace kernels::math::structures {

    struct vec3 {
        types::real x, y, z;

        vec3() : x{0}, y{0}, z{0} {}
        vec3(types::real x, types::real y, types::real z) : x{x}, y{y}, z{z} {}

        vec3& operator+=(const vec3& v) {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        vec3& operator-=(const vec3& v) {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        vec3& operator*=(const types::real& s) {
            x *= s;
            y *= s;
            z *= s;
            return *this;
        }

        vec3& operator/=(const types::real& s) {
            x /= s;
            y /= s;
            z /= s;
            return *this;
        }
    };

    inline vec3 operator+(vec3 a, const vec3& b) {
        return a += b;
    }

    inline vec3 operator-(vec3 a, const vec3& b) {
        return a -= b;
    }

    inline vec3 operator*(vec3 a, const types::real& b) {
        return a *= b;
    }

    inline vec3 operator*(const types::real& b, vec3 a) {
        return a *= b;
    }

    inline vec3 operator/(vec3 a, const types::real& b) {
        return a /= b;
    }

    inline types::real dot(const vec3& a, const vec3& b) {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    inline vec3 cross(const vec3& a, const vec3& b) {
        return vec3(
            a.y*b.z - a.z*b.y,
            a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x
        );
    }

    inline types::real norm(const vec3& a) {
        return std::sqrt(dot(a, a));
    }

    inline vec3 unitv(const vec3& a) {
        types::real n = norm(a);
        return (n == 0) ? vec3{} : a / n;
    }

}
