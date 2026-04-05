#pragma once

#include <cstdint>
#include <array>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"

namespace kernels::math::algs{
    using real = types::real;
    using vec3 = kernels::math::structures::vec3;
    using vec2 = kernels::math::structures::vec2;

    template<typename T>
    //vals = {f0, f1}
    inline T lerp(real t, const std::array<T, 2>& vals){
        return (static_cast<real>(1) - t) * vals[0] + t * vals[1];
    }

    template<typename T>
    //vals = {f00, f10, f01, f11}
    inline T bilerp(real tx, real ty, const std::array<T, 4>& vals){
        return lerp(ty, std::array<T,2>{
            lerp(tx, std::array<T,2>{vals[0], vals[1]}), 
            lerp(tx, std::array<T,2>{vals[2], vals[3]})});
    }

    template<typename T>
    //vals = {f000, f100, f010, f110, f001, f101, f011, f111}
    inline T trilerp(real tx, real ty, real tz, const std::array<T, 8>& vals){
        return lerp(tz, std::array<T,2>{
            bilerp(tx, ty, std::array<T, 4>{vals[0], vals[1], vals[2], vals[3]}),
            bilerp(tx, ty, std::array<T, 4>{vals[4], vals[5], vals[6], vals[7]})});
    }
}
