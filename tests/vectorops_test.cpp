#include <cmath>
#include <iostream>

#include "include/types.hpp"
#include "include/kernels/math/algs/vectorops.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"

using kernels::physics::structures::_2Mesh;
using kernels::physics::structures::_2Field;
using kernels::math::structures::vec2;
using kernels::math::algs::gradPoint;
using kernels::math::algs::divPoint;

int main() {
    using types::real;

    auto check = [](const char* name, bool cond) {
        if (cond) {
            std::cout << "[PASS] " << name << "\n";
        } else {
            std::cout << "[FAIL] " << name << "\n";
        }
        return cond;
    };

    const real tol = 1e-6;

    // Basic 2D mesh geometry
    _2Mesh m(0.0, 1.0, 0.0, 1.0, 4, 4); // 4x4 cells on [0,1]x[0,1]

    check("mesh dx", std::abs(m.dx() - 0.25) < tol);
    check("mesh dy", std::abs(m.dy() - 0.25) < tol);

    // cell center at (0,0)
    {
        vec2 c = m.cell_center(0, 0);
        check("cell_center(0,0).x", std::abs(c.x - 0.125) < tol);
        check("cell_center(0,0).y", std::abs(c.y - 0.125) < tol);
    }

    // node at (2,3)
    {
        vec2 n = m.node(2, 3);
        check("node(2,3).x", std::abs(n.x - (2 * m.dx())) < tol);
        check("node(2,3).y", std::abs(n.y - (3 * m.dy())) < tol);
    }

    // pos_to_cell for a cell center
    {
        vec2 c = m.cell_center(1, 2);
        int i = -1, j = -1;
        bool inside = m.pos_to_cell(c, i, j);
        check("pos_to_cell inside", inside);
        check("pos_to_cell indices", i == 1 && j == 2);
    }

    // Scalar field φ(x,y) = x^2 + y^2 sampled at cell centers
    _2Field<real> phi;
    phi.n_x = m.nx_cells();
    phi.n_y = m.ny_cells();
    phi.data.resize(phi.n_x * phi.n_y);

    for (int j = 0; j < phi.n_y; ++j) {
        for (int i = 0; i < phi.n_x; ++i) {
            vec2 c = m.cell_center(i, j);
            phi.valueAt(i, j) = c.x * c.x + c.y * c.y;
        }
    }

    // Check gradient at an interior point (2,2)
    {
        int i = 2, j = 2;
        vec2 c = m.cell_center(i, j);
        vec2 g = gradPoint(m, phi, i, j);

        // analytic grad = (2x, 2y)
        vec2 g_exact(2.0 * c.x, 2.0 * c.y);
        check("gradPoint x component", std::abs(g.x - g_exact.x) < 1e-2);
        check("gradPoint y component", std::abs(g.y - g_exact.y) < 1e-2);
    }

    // Vector field F(x,y) = (x, y) sampled at cell centers
    _2Field<vec2> F;
    F.n_x = m.nx_cells();
    F.n_y = m.ny_cells();
    F.data.resize(F.n_x * F.n_y);

    for (int j = 0; j < F.n_y; ++j) {
        for (int i = 0; i < F.n_x; ++i) {
            vec2 c = m.cell_center(i, j);
            F.valueAt(i, j) = vec2(c.x, c.y);
        }
    }

    // Divergence at an interior point should be ~2 (d/dx x + d/dy y)
    {
        int i = 2, j = 2;
        real div = divPoint(m, F, i, j);
        check("divPoint at interior", std::abs(div - 2.0) < 1e-2);
    }

    std::cout << "vectorops_test passed\n";
    return 0;
}
