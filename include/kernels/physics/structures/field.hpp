#pragma once

#include <cstdint>
#include <vector>
#include <array>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/math/algs/interpolate.hpp"

namespace kernels::physics::structures{
    template<typename T>
    using vector = std::vector<T>;
    using real = types::real;
    using vec3 = kernels::math::structures::vec3;
    using vec2 = kernels::math::structures::vec2;

    template<typename T>
    struct _2Field{
        vector<T> data;

        int n_x, n_y;

        inline T& valueAt(int i, int j){
            return data[i + n_x * j];
        }

        const inline T& valueAt(int i, int j) const{
            return data[i + n_x * j];
        }

        inline T& valueAt(real i, real j){
            iR = static_cast<int>(i);
            jR = static_cast<int>(i);
            return kernels::math::algs::bilerp(i - iR, j - jR, 
                std::array<real>{
                data[iR + n_x * jR],
                data[iR + 1 + n_x * jR],
                data[iR + n_x * (jR+1)],
                data[iR + 1 + n_x * (jR+1)]});
        }
    };

    template<typename T>
    struct _3Field{
        vector<T> data;

        int n_x, n_y, n_z;

        inline T& valueAt(int i, int j, int k){
            return data[i + n_x*j + n_x*n_y*k];
        }
        
        const inline T& valueAt(int i, int j, int k) const{
            return data[i + n_x*j + n_x*n_y*k];
        }

        inline T& valueAt(real i, real j, real k){
            iR = static_cast<int>(i);
            jR = static_cast<int>(j);
            kR = static_cast<int>(k);
            return kernels::math::algs::trilerp(i - iR, j - jR, k - kR
                std::array<real>{
                data[iR + n_x * jR + n_x*n_y*kR],
                data[iR + 1 + n_x * jR + n_x*n_y*kR],
                data[iR + n_x * (jR+1) + n_x*n_y*kR],
                data[iR + 1 + n_x * (jR+1) + n_x*n_y*kR],
                data[iR + n_x * jR + n_x*n_y*(kR+1)],
                data[iR + 1 + n_x * jR + n_x*n_y*(kR+1)],
                data[iR + n_x * (jR+1) + n_x*n_y*(kR+1)],
                data[iR + 1 + n_x * (jR+1) + n_x*n_y*(kR+1)]
                });
        }
    };
}