#pragma once

#include <cstdint>
#include <array>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"

namespace kernels::math::other{
    using real = types::real;
    using vec3 = kernels::math::structures::vec3;
    using vec2 = kernels::math::structures::vec2;

    //vals = {f0, f1}
    inline real lerp(real t, const std::array<real, 2>& vals){
        return (static_cast<real>(1) - t) * vals[0] + t * vals[1];
    }

    //vals = {f00, f10, f01, f11}
    inline real bilerp(real tx, real ty, const std::array<real, 4>& vals){
        real a = lerp(tx, std::array<real,2>{vals[0], vals[1]});
        real b = lerp(tx, std::array<real,2>{vals[2], vals[3]});
        return lerp(ty, std::array<real,2>{a, b});
    }

    //vals = {f000, f100, f010, f110, f001, f101, f011, f111}
    inline real trilerp(real tx, real ty, real tz, const std::array<real, 8>& vals){
        real a = bilerp(tx, ty, std::array<real, 4>{vals[0], vals[1], vals[2], vals[3]});
        real b = bilerp(tx, ty, std::array<real, 4>{vals[4], vals[5], vals[6], vals[7]});
        return lerp(tz, std::array<real,2>{a, b});
    }
}
