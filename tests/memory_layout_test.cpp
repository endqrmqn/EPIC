#include <iostream>
#include <type_traits>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"

using types::real;
using kernels::math::structures::vec2;
using kernels::math::structures::vec3;
using kernels::physics::structures::_2Field;
using kernels::physics::structures::_3Field;
using kernels::physics::structures::_2Mesh;
using kernels::physics::structures::_3Mesh;
using kernels::physics::structures::_2ParticleGroup;
using kernels::physics::structures::_3ParticleGroup;

int main() {
    std::cout << "=== memory layout summary ===\n";

    std::cout << "sizeof(real)      = " << sizeof(real) << " bytes\n";
    std::cout << "sizeof(vec2)      = " << sizeof(vec2) << " bytes\n";
    std::cout << "sizeof(vec3)      = " << sizeof(vec3) << " bytes\n";

    std::cout << "sizeof(_2Mesh)    = " << sizeof(_2Mesh) << " bytes\n";
    std::cout << "sizeof(_3Mesh)    = " << sizeof(_3Mesh) << " bytes\n";

    std::cout << "sizeof(_2Field<real>) header = "
              << sizeof(_2Field<real>) << " bytes\n";
    std::cout << "sizeof(_3Field<real>) header = "
              << sizeof(_3Field<real>) << " bytes\n";

    std::cout << "sizeof(_2Field<vec2>) header = "
              << sizeof(_2Field<vec2>) << " bytes\n";
    std::cout << "sizeof(_3Field<vec3>) header = "
              << sizeof(_3Field<vec3>) << " bytes\n";

    std::cout << "sizeof(_2ParticleGroup) header = "
              << sizeof(_2ParticleGroup) << " bytes\n";
    std::cout << "sizeof(_3ParticleGroup) header = "
              << sizeof(_3ParticleGroup) << " bytes\n";

    // Example per-cell / per-particle memory estimates
    std::cout << "\nApprox per-cell field storage (doubles):\n";
    std::cout << "  scalar field (real): " << sizeof(real) << " bytes/cell\n";
    std::cout << "  vector field 2D (vec2): " << sizeof(vec2) << " bytes/cell\n";
    std::cout << "  vector field 3D (vec3): " << sizeof(vec3) << " bytes/cell\n";

    std::cout << "\nApprox per-particle storage (2D group):\n";
    std::cout << "  mass + charge + pos + vel + acc ~= "
              << sizeof(real) * 2 + sizeof(vec2) * 3 << " bytes/particle\n";

    std::cout << "Approx per-particle storage (3D group):\n";
    std::cout << "  mass + charge + id + pos + vel + acc ~= "
              << sizeof(real) * 2 + sizeof(std::uint32_t)
                 + sizeof(vec3) * 3 << " bytes/particle\n";

    return 0;
}

