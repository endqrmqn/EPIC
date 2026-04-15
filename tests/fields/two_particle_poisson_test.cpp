#include <cmath>
#include <iostream>
#include <chrono>
#include <iomanip>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"

#include "src/fields/fieldManager.cpp"
#include "src/fields/solvers/poissonSolver.cpp"
#include "src/particles/deposition.cpp"
#include "src/particles/integrators.cpp"
#include "src/io/writeBin.cpp"
#include "src/diagnostics/physDiag.cpp"

using types::real;
using kernels::math::structures::vec2;
using kernels::physics::structures::_2Mesh;
using kernels::physics::structures::_2Field;
using kernels::physics::structures::_2ParticleGroup;
using src::fields::_2FieldManager;

// Simple 2-particle electrostatic test:
// - Two identical like-charged particles, initially at rest, placed
//   symmetrically around the domain centre.
// - Each step:
//     1) Deposit rho from particles onto cell-centred grid.
//     2) Solve Poisson for phi and derive E = -grad(phi).
//     3) Push particles with Boris using that E (B = 0).
// - We expect symmetric repulsion: the centre-of-mass stays fixed and
//   x1(t), x2(t) move apart with equal and opposite displacements.
int main(){
    std::cout << "=== 2D two-particle Poisson test ===\n";

    const int nx = 128;
    const int ny = 128;

    _2Mesh mesh(0.0, 1.0, 0.0, 1.0, nx, ny);
    _2FieldManager fm(mesh);

    // Two-particle group.
    _2ParticleGroup p;
    p.mass.resize(2);
    p.charge.resize(2);
    p.pos.resize(2);
    p.vel.resize(2);
    p.acc.resize(2);

    const real m = static_cast<real>(1.0);
    // Small charge: large enough to see motion, small enough to avoid blow-up.
    const real q = static_cast<real>(1e-5);

    const real x_c = static_cast<real>(0.5);
    const real y_c = static_cast<real>(0.5);
    const real dx0 = static_cast<real>(0.05);

    p.mass[0]   = m;
    p.charge[0] = q;
    p.pos[0]    = vec2{x_c - dx0, y_c};
    p.vel[0]    = vec2{0.0, 0.0};
    p.acc[0]    = vec2{0.0, 0.0};

    p.mass[1]   = m;
    p.charge[1] = q;
    p.pos[1]    = vec2{x_c + dx0, y_c};
    p.vel[1]    = vec2{0.0, 0.0};
    p.acc[1]    = vec2{0.0, 0.0};

    // Time step and number of steps.
    const real dt     = static_cast<real>(1e-4);
    const int  nSteps = 200; // total time = 0.2

    // Pre-allocate zero B field for Boris (no magnetic field).
    _2Field<kernels::math::structures::vec3> Bcc;
    Bcc.n_x = nx;
    Bcc.n_y = ny;
    Bcc.data.assign(static_cast<std::size_t>(nx) * ny,
                    kernels::math::structures::vec3{0.0, 0.0, 0.0});

    auto dump = [&](const char* label, const _2ParticleGroup& pg, real t){
        vec2 P = src::diagnostics::total_momentum(pg);
        std::cout << label
                  << " t=" << t
                  << " x1=" << pg.pos[0].x
                  << " x2=" << pg.pos[1].x
                  << " v1=" << pg.vel[0].x
                  << " v2=" << pg.vel[1].x
                  << " Px=" << P.x
                  << " Py=" << P.y
                  << "\n";
    };

    dump("Initial", p, static_cast<real>(0.0));
    src::io::write_particles_2d("two_particles_0.bin", p,
                                static_cast<real>(0.0));

    real t = static_cast<real>(0.0);

    const int printEvery = 10;
    auto t_start = std::chrono::steady_clock::now();

    for (int n = 0; n < nSteps; ++n){
        // 1) Reset sources and deposit rho from particles.
        fm.reset_sources();
        auto& rho = fm.rho();
        auto& Jx  = fm.Jx();
        auto& Jy  = fm.Jy();

        src::particles::deposit_rho_J_2d(mesh, p, rho, Jx, Jy);

        // 2) Solve Poisson for electrostatic E field (cell-centred).
        auto Ecc = src::fields::solvers::poisson_solver(fm);

        // 3) Push particles with Boris using E from Poisson and B=0.
        src::particles::boris_push_2d(p, mesh, Ecc, Bcc, dt);

        t += dt;

        if (((n + 1) % printEvery) == 0 || (n + 1) == nSteps){
            auto t_now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = t_now - t_start;
            double elapsed_s   = elapsed.count();
            double per_step    = elapsed_s / static_cast<double>(n + 1);
            double remaining_s = per_step * static_cast<double>(nSteps - (n + 1));
            double progress    = static_cast<double>(n + 1) /
                                 static_cast<double>(nSteps);

            const int barWidth = 40;
            int filled = static_cast<int>(progress * barWidth + 0.5);

            std::cout << "\r[";
            for (int i = 0; i < barWidth; ++i){
                std::cout << (i < filled ? '=' : ' ');
            }
            std::cout << "] "
                      << std::setw(6) << std::fixed << std::setprecision(1)
                      << (progress * 100.0) << "% "
                      << "elapsed="  << std::setw(6) << std::setprecision(2) << elapsed_s  << "s "
                      << "eta="      << std::setw(6) << std::setprecision(2) << remaining_s << "s"
                      << std::flush;
        }
    }
    std::cout << "\n";

    dump("Final", p, t);
    src::io::write_particles_2d("two_particles_final.bin", p, t);

    std::cout << "2D two-particle Poisson test done\n";
    return 0;
}
