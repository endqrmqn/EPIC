#pragma once
#include <cmath>
#include "include/types.hpp"
#include "include/kernels/math/structures/vec3.hpp"

namespace kernels::math::structures{
    struct vec2{
        types::real x, y;

        vec2() : x(0), y(0) {}
        vec2(types::real x_, types::real y_) :
        x(x_), y(y_) {}

        vec2& operator+=(const vec2& v){
            x += v.x;
            y += v.y;
            return *this;
        };
        vec2& operator-=(const vec2& v){
            x -= v.x;
            y -= v.y;
            return *this;
        };
        vec2& operator*=(const types::real &s){
            x *= s;
            y *= s;
            return *this;
        };
        vec2& operator/=(const types::real &s){
            x /= s;
            y /= s;
            return *this;
        };
    };

    inline vec2 operator+(vec2 a, const vec2& b){
        return a += b;
    }

    inline vec2 operator-(vec2 a, const vec2& b){
        return a -= b;
    }

    inline vec2 operator*(vec2 a, const types::real& b){
        return a *= b;
    }

    inline vec2 operator*(const types::real& b, vec2 a){
        return a *= b;
    }

    inline vec2 operator/(vec2 a, const types::real& b){
        return a /= b;
    }

    inline types::real dot(const vec2& a, const vec2& b){
        return a.x * b.x + a.y * b.y;
    }

    inline vec3 cross(const vec2& a, const vec2& b){
        return vec3(
            0,
            0,
            a.x * b.y - a.y * b.x
        );
    }

    inline vec2 cross(const vec2& a, types::real b){
        return vec2(-b * a.y, b * a.x);
    }

    inline types::real norm(const vec2& a){
        return std::sqrt(a.x * a.x + a.y * a.y);
    }

    inline vec2 unitv(const vec2& a){
        types::real n = norm(a);
        return (n == 0) ? vec2{} : a / n;
    }
}