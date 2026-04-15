#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "../../include/types.hpp"
#include "../../include/kernels/math/structures/vec2.hpp"
#include "../../include/kernels/math/structures/vec3.hpp"
#include "../../include/kernels/math/algs/vectorops.hpp"
#include "../../include/kernels/physics/structures/mesh.hpp"
#include "../../include/kernels/physics/structures/field.hpp"

using kernels::physics::structures::_2Mesh;
using kernels::physics::structures::_3Mesh;
using kernels::physics::structures::_2Field;
using kernels::physics::structures::_3Field;
using kernels::math::structures::vec2;
using kernels::math::structures::vec3;
using kernels::math::algs::gradPointInterior;
using kernels::math::algs::divPointInterior;
using kernels::math::algs::curlPointInterior;

using types::real;

namespace {

template <typename F>
double time_ms(F&& f) {
    auto start = std::chrono::steady_clock::now();
    f();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    return diff.count();
}

void bench_grad2d(int nx, int ny, int repeats) {
    _2Mesh m(0.0, 1.0, 0.0, 1.0, nx, ny);

    _2Field<real> phi;
    phi.n_x = nx;
    phi.n_y = ny;
    phi.data.resize(nx * ny);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            vec2 c = m.cell_center(i, j);
            phi.valueAt(i, j) = c.x * c.x + c.y * c.y;
        }
    }

    volatile double sink = 0.0;
    double ms = time_ms([&] {
        for (int r = 0; r < repeats; ++r) {
            // interior: branch-free, matches gradPointInterior preconditions
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    vec2 g = gradPointInterior(m, phi, i, j);
                    sink += g.x + g.y;
                }
            }
        }
    });

    const std::size_t total_cells =
        static_cast<std::size_t>(nx > 2 && ny > 2 ? (nx - 2) * (ny - 2) : 0) * repeats;
    std::cout << "[grad2d] nx=ny=" << nx
              << ", repeats=" << repeats
              << ", total cells=" << total_cells
              << ", time=" << ms << " ms"
              << ", time/cell=" << (ms * 1e6 / total_cells) << " ns"
              << "\n";
}

void bench_div2d(int nx, int ny, int repeats) {
    _2Mesh m(0.0, 1.0, 0.0, 1.0, nx, ny);

    _2Field<vec2> F;
    F.n_x = nx;
    F.n_y = ny;
    F.data.resize(nx * ny);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            vec2 c = m.cell_center(i, j);
            F.valueAt(i, j) = vec2(c.x, c.y);
        }
    }

    volatile double sink = 0.0;
    double ms = time_ms([&] {
        for (int r = 0; r < repeats; ++r) {
            // interior only, consistent with divPointInterior preconditions
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    real div = divPointInterior(m, F, i, j);
                    sink += div;
                }
            }
        }
    });

    const std::size_t total_cells =
        static_cast<std::size_t>(nx > 2 && ny > 2 ? (nx - 2) * (ny - 2) : 0) * repeats;
    std::cout << "[div2d]  nx=ny=" << nx
              << ", repeats=" << repeats
              << ", total cells=" << total_cells
              << ", time=" << ms << " ms"
              << ", time/cell=" << (ms * 1e6 / total_cells) << " ns"
              << "\n";
}

void bench_curl3d(int n, int repeats) {
    int nx = n, ny = n, nz = n;
    _3Mesh m;
    m.x_init = 0.0;
    m.x_finl = 1.0;
    m.y_init = 0.0;
    m.y_finl = 1.0;
    m.z_init = 0.0;
    m.z_finl = 1.0;
    m.n_x = nx;
    m.n_y = ny;
    m.n_z = nz;

    _3Field<vec3> F;
    F.n_x = nx;
    F.n_y = ny;
    F.n_z = nz;
    F.data.resize(static_cast<std::size_t>(nx) * ny * nz);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                vec3 c = m.cell_center(i, j, k);
                F.valueAt(i, j, k) = vec3(-c.y, c.x, c.z);
            }
        }
    }

    volatile double sink = 0.0;
    double ms = time_ms([&] {
        for (int r = 0; r < repeats; ++r) {
            // interior only, consistent with curlPointInterior preconditions
            for (int k = 1; k < nz - 1; ++k) {
                for (int j = 1; j < ny - 1; ++j) {
                    for (int i = 1; i < nx - 1; ++i) {
                        vec3 c = curlPointInterior(m, F, i, j, k);
                        sink += c.x + c.y + c.z;
                    }
                }
            }
        }
    });

    const std::size_t total_cells =
        static_cast<std::size_t>(
            (nx > 2 && ny > 2 && nz > 2) ? (nx - 2) * (ny - 2) * (nz - 2) : 0
        ) * repeats;
    std::cout << "[curl3d] n=" << n
              << ", repeats=" << repeats
              << ", total cells=" << total_cells
              << ", time=" << ms << " ms"
              << ", time/cell=" << (ms * 1e6 / total_cells) << " ns"
              << "\n";
}

} // namespace

int main() {
    std::cout << "=== vectorops benchmark ===\n";
    {
        const std::size_t target_total = static_cast<std::size_t>(1024) * 1024 * 8;
        for (int n : {64, 128, 256, 512, 1024}) {
            std::size_t cells_per_repeat = static_cast<std::size_t>(n) * n;
            int repeats = static_cast<int>(target_total / cells_per_repeat);
            if (repeats < 1) repeats = 1;
            bench_grad2d(n, n, repeats);
            bench_div2d(n, n, repeats);
        }
    }

    {
        const std::size_t target_total =
            static_cast<std::size_t>(32) * 1024 * 1024 * 4;
        for (int n : {64, 128, 256, 512, 1024}) {
            std::size_t cells_per_repeat =
                static_cast<std::size_t>(n) * n * n;
            int repeats = static_cast<int>(target_total / cells_per_repeat);
            if (repeats < 1) repeats = 1;
            bench_curl3d(n, repeats);
        }
    }

    return 0;
}
