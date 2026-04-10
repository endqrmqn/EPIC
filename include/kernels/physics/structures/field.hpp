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

        inline T valueAt(real i, real j) const{
            // Continuous index (i,j) -> surrounding cell corners for bilerp
            int iR = static_cast<int>(i);
            int jR = static_cast<int>(j);

            // Clamp to valid range so iR+1, jR+1 are in-bounds
            if (iR < 0) iR = 0;
            if (jR < 0) jR = 0;
            if (iR > n_x - 2) iR = n_x - 2;
            if (jR > n_y - 2) jR = n_y - 2;

            const real tx = i - static_cast<real>(iR);
            const real ty = j - static_cast<real>(jR);

            std::array<T, 4> vals{
                data[iR +     n_x * jR],
                data[iR + 1 + n_x * jR],
                data[iR +     n_x * (jR + 1)],
                data[iR + 1 + n_x * (jR + 1)]
            };

            return kernels::math::algs::bilerp(tx, ty, vals);
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

        inline T valueAt(real i, real j, real k) const{
            int iR = static_cast<int>(i);
            int jR = static_cast<int>(j);
            int kR = static_cast<int>(k);

            if (iR < 0) iR = 0;
            if (jR < 0) jR = 0;
            if (kR < 0) kR = 0;
            if (iR > n_x - 2) iR = n_x - 2;
            if (jR > n_y - 2) jR = n_y - 2;
            if (kR > n_z - 2) kR = n_z - 2;

            const real tx = i - static_cast<real>(iR);
            const real ty = j - static_cast<real>(jR);
            const real tz = k - static_cast<real>(kR);

            std::array<T, 8> vals{
                data[iR     + n_x * jR     + n_x*n_y * kR],
                data[iR + 1 + n_x * jR     + n_x*n_y * kR],
                data[iR     + n_x * (jR+1) + n_x*n_y * kR],
                data[iR + 1 + n_x * (jR+1) + n_x*n_y * kR],
                data[iR     + n_x * jR     + n_x*n_y * (kR+1)],
                data[iR + 1 + n_x * jR     + n_x*n_y * (kR+1)],
                data[iR     + n_x * (jR+1) + n_x*n_y * (kR+1)],
                data[iR + 1 + n_x * (jR+1) + n_x*n_y * (kR+1)]
            };

            return kernels::math::algs::trilerp(tx, ty, tz, vals);
        }
    };
}
