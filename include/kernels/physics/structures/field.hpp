#pragma once

#include <cstdint>
#include <vector>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"

namespace kernels::physics::structures{
    template<typename T>
    using vector = std::vector<T>;
    using real = types::real;
    using vec3 = kernels::math::structures::vec3;
    using vec2 = kernels::math::structures::vec2;
    struct _2scalarField{
        vector<real> data;

        int n_x, n_y;

        inline real& valueAt(int i, int j){
            return data[i + n_x * j];
        }
    };

    struct _2vectorField{
        vector<vec2> data;

        int n_x, n_y;

        inline vec2& valueAt(int i, int j){
            return data[i + n_x * j];
        }
    }

    struct _3scalarField{
        vector<real> data;

        int n_x, n_y, n_z;

        inline real& valueAt(int i, int j, int k){
            return data[i + n_x*j + n_x*n_y*k];
        }
    };

    struct _3vectorField{
        vector<vec3> data;

        int n_x, n_y, n_z;

        inline vec3& valueAt(int i, int j, int k){
                return data[i + n_x*j + n_x*n_y*k];
        }
    };
}