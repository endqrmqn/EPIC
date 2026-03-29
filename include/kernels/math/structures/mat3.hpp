#pragma once

#include "include/types.hpp"
#include "include/kernels/math/structures/vec3.hpp"

namespace kernels::math::structures{
    struct mat3{
        kernels::math::structures::vec3 a, b, c;

        mat3() : a(), b(), c() {}
        mat3(const kernels::math::structures::vec3 &a_, 
            const kernels::math::structures::vec3 &b_, 
            const kernels::math::structures::vec3 &c_) :
            a(a_), b(b_), c(c_) {}

        mat3& operator+=(const mat3& m);
        mat3& operator-=(const mat3& m);
        mat3& operator*=(const types::real &s);
        mat3& operator/=(const types::real &s);
        
    };

    inline mat3 operator+(mat3 a, const mat3& b);

    inline mat3 operator-(mat3 a, const mat3& b);

    inline mat3 operator*(mat3 a, const types::real& b);

    inline mat3 operator/(mat3 a, const types::real& b);

    inline mat3& T(const mat3& m);

}