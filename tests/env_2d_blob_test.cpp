#include <cmath>
#include <iostream>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"

#include "src/engine/core/environment.cpp"
#include "src/io/writeBin.cpp"

using types::real;
using kernels::math::structures::vec2;
using kernels::physics::structures::_2Mesh;
using kernels::physics::structures::_2ParticleGroup;

using src::engine::env::_2environment;
using src::engine::env::IntegratorKind;
using src::engine::env::FieldMode;

int main() {
    std::cout << "=== 2D env blob-in-E test ===\n";

    // Domain and mesh
    const int nx = 128;
    const int ny = 128;

    _2Mesh mesh(0.0, 1.0, 0.0, 1.0, nx, ny);

    const real dx = mesh.dx();
    const real dy = mesh.dy();

    // Time step: within CFL for c=1, modest dt.
    const real dt = static_cast<real>(0.001);

    _2environment env(mesh, dt);
    env.set_integrator(IntegratorKind::Boris);
    env.set_field_mode(FieldMode::SelfConsistent); // use prescribed Ex, no self-consistent update

    auto& fm = env.fields();

    // Set up a roughly uniform left-to-right Ex field, Ey=Bz=0.
    auto& Ex = fm.Ex();
    auto& Ey = fm.Ey();
    auto& Bz = fm.Bz();

    const real E0 = static_cast<real>(1.0);

    for (int j = 0; j < Ex.n_y; ++j) {
        for (int i = 0; i < Ex.n_x; ++i) {
            Ex.valueAt(i, j) = E0;
        }
    }
    for (int j = 0; j < Ey.n_y; ++j) {
        for (int i = 0; i < Ey.n_x; ++i) {
            Ey.valueAt(i, j) = static_cast<real>(0.0);
        }
    }
    for (int j = 0; j < Bz.n_y; ++j) {
        for (int i = 0; i < Bz.n_x; ++i) {
            Bz.valueAt(i, j) = static_cast<real>(0.0);
        }
    }

    // Create a single 2D species: small charge so that feedback on E
    // is weak and the field remains approximately left-to-right.
    auto& pm = env.particles();

    const real species_mass   = static_cast<real>(1.0);
    const real species_charge = static_cast<real>(0.000001);

    const std::size_t s_idx = pm.add_species_2d(species_mass, species_charge);
    auto& species = pm.species_2d(s_idx);

    // Populate a Gaussian-ish blob around (x,y) ~ (0.25, 0.5) with zero initial velocity.
    const int   n_blob_x = 8;
    const int   n_blob_y = 8;
    const real  blob_dx  = static_cast<real>(0.02);
    const real  blob_dy  = static_cast<real>(0.02);
    const real  x0       = static_cast<real>(0.25);
    const real  y0       = static_cast<real>(0.5);

    for (int jy = 0; jy < n_blob_y; ++jy) {
        for (int ix_p = 0; ix_p < n_blob_x; ++ix_p) {
            const real x = x0 + (static_cast<real>(ix_p) - static_cast<real>(0.5) * (n_blob_x - 1)) * blob_dx;
            const real y = y0 + (static_cast<real>(jy)   - static_cast<real>(0.5) * (n_blob_y - 1)) * blob_dy;
            vec2 pos{x, y};
            vec2 vel{static_cast<real>(0.0), static_cast<real>(0.0)};
            species.add_particle(pos, vel);
        }
    }

    const _2ParticleGroup& grp0 = species.particles;

    auto compute_means = [](const _2ParticleGroup& g,
                            real& mx, real& my,
                            real& mvx, real& mvy) {
        const std::size_t n = kernels::physics::structures::size(g);
        mx = my = mvx = mvy = static_cast<real>(0.0);
        if (n == 0) return;
        for (std::size_t i = 0; i < n; ++i) {
            mx  += g.pos[i].x;
            my  += g.pos[i].y;
            mvx += g.vel[i].x;
            mvy += g.vel[i].y;
        }
        const real inv_n = static_cast<real>(1.0) / static_cast<real>(n);
        mx  *= inv_n;
        my  *= inv_n;
        mvx *= inv_n;
        mvy *= inv_n;
    };

    real x_mean0, y_mean0, vx_mean0, vy_mean0;
    compute_means(grp0, x_mean0, y_mean0, vx_mean0, vy_mean0);

    std::cout << "Initial: <x>=" << x_mean0
              << ", <y>=" << y_mean0
              << ", <vx>=" << vx_mean0
              << ", <vy>=" << vy_mean0 << "\n";

    // Write initial particle snapshot for Python visualization.
    src::io::write_particles_2d("particles_0.bin", grp0, env.time());

    // Run for a modest number of steps.
    const int n_steps = 10000;
    for (int n = 0; n < n_steps; ++n) {
        env.step();
    }

    const _2ParticleGroup& grp = species.particles;

    real x_mean1, y_mean1, vx_mean1, vy_mean1;
    compute_means(grp, x_mean1, y_mean1, vx_mean1, vy_mean1);

    std::cout << "Final:   <x>=" << x_mean1
              << ", <y>=" << y_mean1
              << ", <vx>=" << vx_mean1
              << ", <vy>=" << vy_mean1 << "\n";

    // Write final particle snapshot.
    src::io::write_particles_2d("particles_final.bin", grp, env.time());

    const bool moved_right = (x_mean1 > x_mean0);
    const bool accel_x     = (vx_mean1 > vx_mean0);

    if (moved_right && accel_x) {
        std::cout << "[PASS] blob moves right in +Ex field\n";
    } else {
        std::cout << "[FAIL] blob did not move as expected in +Ex field\n";
    }

    std::cout << "2D env blob-in-E test done\n";
    return 0;
}
