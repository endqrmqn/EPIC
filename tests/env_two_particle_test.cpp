#include <cmath>
#include <iostream>
#include <chrono>
#include <iomanip>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"

#include "src/engine/core/environment.cpp"
#include "src/diagnostics/physDiag.cpp"
#include "src/io/writeBin.cpp"

using types::real;
using kernels::math::structures::vec2;
using kernels::physics::structures::_2Mesh;

using src::engine::env::_2environment;
using src::engine::env::IntegratorKind;
using src::engine::env::FieldMode;
using src::engine::env::FieldUpdateKind;

int main(){
    std::cout << "=== 2D env two-particle Maxwell-Gauss test ===\n";

    const int nx = 128;
    const int ny = 128;

    _2Mesh mesh(0.0, 1.0, 0.0, 1.0, nx, ny);

    const real dt = static_cast<real>(1e-4);
    _2environment env(mesh, dt);

    env.set_integrator(IntegratorKind::Boris);
    env.set_field_mode(FieldMode::SelfConsistent);
    env.set_field_update(FieldUpdateKind::PoissonYee); // explicit Yee update + Gauss projection

    env.enable_gauss_projection(true);

    auto& fm = env.fields();
    auto& pm = env.particles();

    // Ensure initial fields are zero.
    auto& Ex = fm.Ex();
    auto& Ey = fm.Ey();
    auto& Bz = fm.Bz();
    for (int j = 0; j < Ex.n_y; ++j)
        for (int i = 0; i < Ex.n_x; ++i)
            Ex.valueAt(i, j) = 0.0;
    for (int j = 0; j < Ey.n_y; ++j)
        for (int i = 0; i < Ey.n_x; ++i)
            Ey.valueAt(i, j) = 0.0;
    for (int j = 0; j < Bz.n_y; ++j)
        for (int i = 0; i < Bz.n_x; ++i)
            Bz.valueAt(i, j) = 0.0;

    // Single species with two like-charged particles.
    const real m = static_cast<real>(1.0);
    const real q = static_cast<real>(1e-5);

    std::size_t s_idx = pm.add_species_2d(m, q);
    auto& species = pm.species_2d(s_idx);

    const real x_c  = static_cast<real>(0.5);
    const real y_c  = static_cast<real>(0.5);
    const real dx0  = static_cast<real>(0.05);

    species.add_particle(vec2{x_c - dx0, y_c}, vec2{0.0, 0.0});
    species.add_particle(vec2{x_c + dx0, y_c}, vec2{0.0, 0.0});

    // Start from a Gauss-consistent electric field for the initial charge
    // distribution to avoid a spurious Maxwell transient on the first step.
    fm.reset_sources();
    src::particles::deposit_rho_J_2d(mesh, species.particles,
                                     fm.rho(), fm.Jx(), fm.Jy());
    src::fields::solvers::enforce_gauss_law_yee_2d(fm);

    auto dump = [&](const char* label){
        const auto& grp = species.particles;
        vec2 P = src::diagnostics::total_momentum(grp);
        std::cout << label
                  << " t=" << env.time()
                  << " x1=" << grp.pos[0].x
                  << " x2=" << grp.pos[1].x
                  << " v1=" << grp.vel[0].x
                  << " v2=" << grp.vel[1].x
                  << " Px=" << P.x
                  << " Py=" << P.y
                  << "\n";
    };

    dump("Initial");
    src::io::write_particles_2d("env_two_particles_0.bin",
                                species.particles, env.time());

    const int nSteps = 1000;

    const int printEvery = 1;
    auto t_start = std::chrono::steady_clock::now();

    for (int n = 0; n < nSteps; ++n){
        env.step();

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

    dump("Final");
    //src::io::write_particles_2d("env_two_particles_final.bin",
    //                            species.particles, env.time());
    env.print_timing_summary();

    std::cout << "2D env two-particle Maxwell-Gauss test done\n";
    return 0;
}
