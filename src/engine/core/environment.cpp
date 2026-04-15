#pragma once

// Define EPIC_ENABLE_ENV_TIMING at compile time to enable
// environment step timing diagnostics.
// Example: add -DEPIC_ENABLE_ENV_TIMING to your compile flags.

#include <chrono>
#include <iomanip>
#include <iostream>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"

#include "src/fields/fieldManager.cpp"
#include "src/fields/solvers/maxwell_fdtd_2d.cpp"
#include "src/fields/solvers/maxwell_fdtd_3d.cpp"
#include "src/fields/solvers/poissonSolver.cpp"
#include "src/coupling/yee_coupling.hpp"
#include "src/particles/particleManager.cpp"
#include "src/particles/integrators.cpp"
#include "src/particles/deposition.cpp"

namespace src::engine::env{
    using real = types::real;
    using vec2 = kernels::math::structures::vec2;
    using vec3 = kernels::math::structures::vec3;

    using _2Mesh = kernels::physics::structures::_2Mesh;
    using _3Mesh = kernels::physics::structures::_3Mesh;
    using _2sf   = kernels::physics::structures::_2Field<real>;
    using _2vf2  = kernels::physics::structures::_2Field<vec2>;
    using _2vf3  = kernels::physics::structures::_2Field<vec3>;

    using _2FieldManager = src::fields::_2FieldManager;
    using _3FieldManager = src::fields::_3FieldManager;

    using ParticleManager = src::particles::ParticleManager;

    enum class IntegratorKind {
        Boris,
        Verlet
    };

    // How the fields are evolved during a step.
    enum class FieldMode {
        Frozen,         // fields are held fixed; particles feel E,B but do not update it
        SelfConsistent  // deposit rho,J and advance Maxwell with sources
    };

    // How the fields are updated when FieldMode::SelfConsistent is active.
    enum class FieldUpdateKind {
        Maxwell,    // explicit Yee FDTD update (advance_maxwell_2d/3d)
        PoissonYee  // electrostatic solve: rho -> phi -> Yee E via poisson_solve_yee
    };

    struct StepTimingStats {
        std::size_t steps{0};
        double reset_s{0.0};
        double deposit_s{0.0};
        double projection_s{0.0};
        double build_poisson_s{0.0};
        double build_interp_s{0.0};
        double push_s{0.0};
        double redeposit_s{0.0};
        double field_s{0.0};
        double total_s{0.0};

        void reset(){
            steps = 0;
            reset_s = 0.0;
            deposit_s = 0.0;
            projection_s = 0.0;
            build_poisson_s = 0.0;
            build_interp_s = 0.0;
            push_s = 0.0;
            redeposit_s = 0.0;
            field_s = 0.0;
            total_s = 0.0;
        }

        void record(double reset_time,
                    double deposit,
                    double projection,
                    double build_poisson,
                    double build_interp,
                    double push,
                    double redeposit,
                    double field,
                    double total){
            ++steps;
            reset_s += reset_time;
            deposit_s += deposit;
            projection_s += projection;
            build_poisson_s += build_poisson;
            build_interp_s += build_interp;
            push_s += push;
            redeposit_s += redeposit;
            field_s += field;
            total_s += total;
        }

        void print_average(const char* label, std::ostream& os = std::cout) const{
            if (steps == 0){
                os << "[" << label << " timing] no steps recorded\n";
                return;
            }

            const double inv_steps =
                1.0 / static_cast<double>(steps);
            const auto reset_ms = 1.0e3 * reset_s * inv_steps;
            const auto deposit_ms = 1.0e3 * deposit_s * inv_steps;
            const auto projection_ms = 1.0e3 * projection_s * inv_steps;
            const auto build_poisson_ms = 1.0e3 * build_poisson_s * inv_steps;
            const auto build_interp_ms = 1.0e3 * build_interp_s * inv_steps;
            const auto field_ms = 1.0e3 * field_s * inv_steps;
            const auto push_ms = 1.0e3 * push_s * inv_steps;
            const auto redeposit_ms = 1.0e3 * redeposit_s * inv_steps;
            const auto total_ms = 1.0e3 * total_s * inv_steps;
            const auto other_ms =
                total_ms - (reset_ms + deposit_ms + projection_ms +
                            build_poisson_ms + build_interp_ms +
                            push_ms + redeposit_ms +
                            field_ms);

            const auto old_flags = os.flags();
            const auto old_prec = os.precision();
            os << std::fixed << std::setprecision(3);
            os << "[" << label << " timing avg] "
               << "steps=" << steps << " "
               << "reset=" << reset_ms << "ms "
               << "deposit=" << deposit_ms << "ms "
               << "projection=" << projection_ms << "ms "
               << "build_poisson=" << build_poisson_ms << "ms "
               << "build_interp=" << build_interp_ms << "ms "
               << "redeposit=" << redeposit_ms << "ms "
               << "fields=" << field_ms << "ms "
               << "push=" << push_ms << "ms "
               << "other=" << other_ms << "ms "
               << "total=" << total_ms << "ms"
               << std::endl;
            os.flags(old_flags);
            os.precision(old_prec);
        }
    };

    // ---------------------------------------------------------------------
    // 2D Yee-based environment: mesh + fields + multi-species particles.
    // ---------------------------------------------------------------------
    class _2environment{
    public:
        using Mesh           = _2Mesh;
        using FieldManager   = _2FieldManager;
        using Species2D      = ParticleManager::Species2D;

        _2environment(const Mesh& mesh, real dt)
            : mesh_{mesh},
              fields_{mesh},
              dt_{dt},
              t_{0},
              integrator_{IntegratorKind::Boris},
              field_mode_{FieldMode::SelfConsistent},
              field_update_{FieldUpdateKind::Maxwell},
              gauss_projection_enabled_{false},
              maxwell_velocity_staggered_{false} {
        }

        Mesh&        mesh()        { return mesh_; }
        const Mesh&  mesh()  const { return mesh_; }

        FieldManager&       fields()       { return fields_; }
        const FieldManager& fields() const { return fields_; }

        ParticleManager&       particles()       { return particles_; }
        const ParticleManager& particles() const { return particles_; }

        real time() const { return t_; }
        real dt()   const { return dt_; }

        void set_dt(real dt) {
            dt_ = dt;
            maxwell_velocity_staggered_ = false;
        }

        void set_integrator(IntegratorKind kind){
            integrator_ = kind;
        }

        void set_field_mode(FieldMode mode){
            field_mode_ = mode;
        }

        void set_field_update(FieldUpdateKind kind){
            field_update_ = kind;
            maxwell_velocity_staggered_ = false;
        }

        void enable_gauss_projection(bool enabled = true){
            gauss_projection_enabled_ = enabled;
        }

        void reset_timing_stats(){
            timing_stats_.reset();
        }

        const StepTimingStats& timing_stats() const {
            return timing_stats_;
        }

        void print_timing_summary(std::ostream& os = std::cout) const{
            timing_stats_.print_average("Env2D", os);
        }

        // Advance one full time step:
        //   1) Always reset and deposit rho,J from particles.
        //   2) If field_mode_ == SelfConsistent, advance Maxwell with sources.
        //      If Frozen, leave fields unchanged (external / prescribed E,B).
        // Instrumented with optional per-part timing diagnostics.
        void step(){
            using clock    = std::chrono::high_resolution_clock;
            using duration = std::chrono::duration<double>;

            auto t_start = clock::now();

            // 1) Reset and deposit sources.
            auto t_reset_start = clock::now();
            fields_.reset_sources();
            auto t_reset_end = clock::now();
            auto& rho = fields_.rho();
            auto& Jx  = fields_.Jx();
            auto& Jy  = fields_.Jy();
            _2sf Jx_yee;
            _2sf Jy_yee;

            auto t_dep_start = clock::now();

            const std::size_t ns_dep = particles_.num_species_2d();
            if (field_mode_ == FieldMode::SelfConsistent &&
                field_update_ == FieldUpdateKind::Maxwell){
                Jx_yee.n_x = mesh_.nx_xfaces();
                Jx_yee.n_y = mesh_.ny_xfaces();
                Jy_yee.n_x = mesh_.nx_yfaces();
                Jy_yee.n_y = mesh_.ny_yfaces();
                Jx_yee.data.assign(static_cast<std::size_t>(Jx_yee.n_x) * Jx_yee.n_y, real{0});
                Jy_yee.data.assign(static_cast<std::size_t>(Jy_yee.n_x) * Jy_yee.n_y, real{0});
            }

            for (std::size_t s = 0; s < ns_dep; ++s){
                const auto& grp = particles_.species_2d(s).particles;
                src::particles::deposit_rho_J_2d(mesh_, grp, rho, Jx, Jy);
                if (field_mode_ == FieldMode::SelfConsistent &&
                    field_update_ == FieldUpdateKind::Maxwell){
                    _2sf Jx_species;
                    _2sf Jy_species;
                    src::particles::deposit_J_yee_2d(mesh_, grp, Jx_species, Jy_species);
                    for (std::size_t idx = 0; idx < Jx_yee.data.size(); ++idx){
                        Jx_yee.data[idx] += Jx_species.data[idx];
                    }
                    for (std::size_t idx = 0; idx < Jy_yee.data.size(); ++idx){
                        Jy_yee.data[idx] += Jy_species.data[idx];
                    }
                }
            }

            auto t_dep_end = clock::now();

            if (field_mode_ == FieldMode::SelfConsistent &&
                field_update_ == FieldUpdateKind::Maxwell){
                auto t_proj_start = clock::now();
                if (gauss_projection_enabled_){
                    src::fields::solvers::enforce_gauss_law_yee_2d(fields_);
                }
                auto t_proj_end = clock::now();

                auto t_cc_start = clock::now();
                if (gauss_projection_enabled_){
                    auto Ecc = src::fields::solvers::poisson_solver(fields_);
                    _2vf3 Bcc;
                    Bcc.n_x = mesh_.nx_cells();
                    Bcc.n_y = mesh_.ny_cells();
                    Bcc.data.assign(static_cast<std::size_t>(Bcc.n_x) * Bcc.n_y,
                                    vec3{0.0, 0.0, 0.0});
                    auto t_cc_end   = clock::now();

                    auto t_push_start = clock::now();
                    const std::size_t ns = particles_.num_species_2d();
                    std::vector<std::vector<vec2>> old_positions(ns);
                    for (std::size_t s = 0; s < ns; ++s){
                        auto& grp = particles_.species_2d(s).particles;
                        old_positions[s] = grp.pos;
                        switch (integrator_){
                            case IntegratorKind::Boris:
                                src::particles::boris_push_2d(grp, mesh_, Ecc, Bcc, dt_);
                                break;
                            case IntegratorKind::Verlet:
                                src::particles::verlet_push_2d(grp, mesh_, Ecc, dt_);
                                break;
                        }
                    }
                    auto t_push_end = clock::now();

                    auto t_redep_start = clock::now();
                    fields_.reset_sources();
                    _2sf Jx_yee_new;
                    _2sf Jy_yee_new;
                    Jx_yee_new.n_x = mesh_.nx_xfaces();
                    Jx_yee_new.n_y = mesh_.ny_xfaces();
                    Jy_yee_new.n_x = mesh_.nx_yfaces();
                    Jy_yee_new.n_y = mesh_.ny_yfaces();
                    Jx_yee_new.data.assign(static_cast<std::size_t>(Jx_yee_new.n_x) * Jx_yee_new.n_y, real{0});
                    Jy_yee_new.data.assign(static_cast<std::size_t>(Jy_yee_new.n_x) * Jy_yee_new.n_y, real{0});

                    auto& rho_new = fields_.rho();
                    auto& Jx_new  = fields_.Jx();
                    auto& Jy_new  = fields_.Jy();
                    for (std::size_t s = 0; s < ns; ++s){
                        const auto& grp = particles_.species_2d(s).particles;
                        src::particles::deposit_rho_J_2d(mesh_, grp, rho_new, Jx_new, Jy_new);
                        _2sf Jx_species;
                        _2sf Jy_species;
                        src::particles::deposit_J_yee_2d_charge_conserving(
                            mesh_, old_positions[s], grp, dt_, Jx_species, Jy_species);
                        for (std::size_t idx = 0; idx < Jx_yee_new.data.size(); ++idx){
                            Jx_yee_new.data[idx] += Jx_species.data[idx];
                        }
                        for (std::size_t idx = 0; idx < Jy_yee_new.data.size(); ++idx){
                            Jy_yee_new.data[idx] += Jy_species.data[idx];
                        }
                    }
                    auto t_redep_end = clock::now();

                    auto t_field_start = clock::now();
                    src::fields::solvers::advance_maxwell_2d(fields_, dt_,
                                                            &Jx_yee_new, &Jy_yee_new);
                    if (gauss_projection_enabled_){
                        src::fields::solvers::enforce_gauss_law_yee_2d(fields_);
                    }
                    auto t_field_end = clock::now();

                    t_ += dt_;

                    auto t_end = clock::now();

                    duration reset_time = t_reset_end - t_reset_start;
                    duration dep_time   = t_dep_end   - t_dep_start;
                    duration proj_time  = t_proj_end  - t_proj_start;
                    duration cc_time    = t_cc_end    - t_cc_start;
                    duration push_time  = t_push_end  - t_push_start;
                    duration redep_time = t_redep_end - t_redep_start;
                    duration field_time = t_field_end - t_field_start;
                    duration total_time = t_end       - t_start;

                    timing_stats_.record(reset_time.count(),
                                         dep_time.count(),
                                         proj_time.count(),
                                         cc_time.count(),
                                         0.0,
                                         push_time.count(),
                                         redep_time.count(),
                                         field_time.count(),
                                         total_time.count());
#ifdef EPIC_ENABLE_ENV_TIMING
                    std::cout << "[Env2D step diagnostics] "
                              << "t=" << t_ << " "
                              << "reset=" << reset_time.count() << "s, "
                              << "deposit=" << dep_time.count()   << "s, "
                              << "projection=" << proj_time.count() << "s, "
                              << "redeposit=" << redep_time.count() << "s, "
                              << "fields="  << field_time.count() << "s, "
                              << "cell_center=" << cc_time.count() << "s, "
                              << "push="   << push_time.count()   << "s, "
                              << "total="  << total_time.count()  << "s"
                              << std::endl;
#endif
                    return;
                }

                auto Ecc = src::coupling::make_cell_center_E(fields_);
                auto Bcc = src::coupling::make_cell_center_B(fields_);
                auto t_cc_end   = clock::now();

                auto t_push_start = clock::now();
                const std::size_t ns = particles_.num_species_2d();
                std::vector<std::vector<vec2>> old_positions(ns);
                for (std::size_t s = 0; s < ns; ++s){
                    auto& grp = particles_.species_2d(s).particles;
                    old_positions[s] = grp.pos;

                    switch (integrator_){
                        case IntegratorKind::Boris:
                            src::particles::boris_push_2d_staggered(
                                grp, mesh_, Ecc, Bcc, dt_);
                            break;
                        case IntegratorKind::Verlet:
                            src::particles::verlet_push_2d(grp, mesh_, Ecc, dt_);
                            break;
                    }
                }
                auto t_push_end = clock::now();

                // Re-deposit from the updated particle state so the field
                // update uses sources at the end of the particle step.
                fields_.reset_sources();
                _2sf Jx_yee_new;
                _2sf Jy_yee_new;
                Jx_yee_new.n_x = mesh_.nx_xfaces();
                Jx_yee_new.n_y = mesh_.ny_xfaces();
                Jy_yee_new.n_x = mesh_.nx_yfaces();
                Jy_yee_new.n_y = mesh_.ny_yfaces();
                Jx_yee_new.data.assign(static_cast<std::size_t>(Jx_yee_new.n_x) * Jx_yee_new.n_y, real{0});
                Jy_yee_new.data.assign(static_cast<std::size_t>(Jy_yee_new.n_x) * Jy_yee_new.n_y, real{0});

                auto& rho_new = fields_.rho();
                auto& Jx_new  = fields_.Jx();
                auto& Jy_new  = fields_.Jy();
                for (std::size_t s = 0; s < ns; ++s){
                    const auto& grp = particles_.species_2d(s).particles;
                    src::particles::deposit_rho_J_2d(mesh_, grp, rho_new, Jx_new, Jy_new);
                    _2sf Jx_species;
                    _2sf Jy_species;
                    src::particles::deposit_J_yee_2d_charge_conserving(
                        mesh_, old_positions[s], grp, dt_, Jx_species, Jy_species);
                    for (std::size_t idx = 0; idx < Jx_yee_new.data.size(); ++idx){
                        Jx_yee_new.data[idx] += Jx_species.data[idx];
                    }
                    for (std::size_t idx = 0; idx < Jy_yee_new.data.size(); ++idx){
                        Jy_yee_new.data[idx] += Jy_species.data[idx];
                    }
                }

                auto t_field_start = clock::now();
                src::fields::solvers::advance_maxwell_2d(fields_, dt_,
                                                        &Jx_yee_new, &Jy_yee_new);
                if (gauss_projection_enabled_){
                    src::fields::solvers::enforce_gauss_law_yee_2d(fields_);
                }
                auto t_field_end = clock::now();

                t_ += dt_;

                auto t_end = clock::now();

                duration dep_time   = t_dep_end   - t_dep_start;
                duration cc_time    = t_cc_end    - t_cc_start;
                duration push_time  = t_push_end  - t_push_start;
                duration field_time = t_field_end - t_field_start;
                duration total_time = t_end       - t_start;

#ifdef EPIC_ENABLE_ENV_TIMING
                std::cout << "[Env2D step diagnostics] "
                          << "t=" << t_ << " "
                          << "deposit=" << dep_time.count()   << "s, "
                          << "fields="  << field_time.count() << "s, "
                          << "cell_center=" << cc_time.count() << "s, "
                          << "push="   << push_time.count()   << "s, "
                          << "total="  << total_time.count()  << "s"
                          << std::endl;
#endif
                return;
            }

            // 2) Advance fields (or not) depending on mode.
            auto t_field_start = clock::now();
            if (field_mode_ == FieldMode::SelfConsistent){
                switch (field_update_){
                    case FieldUpdateKind::Maxwell:
                        src::fields::solvers::advance_maxwell_2d(fields_, dt_,
                                                                &Jx_yee, &Jy_yee);
                        if (gauss_projection_enabled_){
                            src::fields::solvers::enforce_gauss_law_yee_2d(fields_);
                        }
                        break;
                    case FieldUpdateKind::PoissonYee:
                        // Electrostatic solve: update Yee E from rho.
                        src::fields::solvers::poisson_solve_yee(fields_);
                        break;
                }
            }
            auto t_field_end = clock::now();

            // Build cell-centred fields for particle push.
            auto t_cc_start = clock::now();
            _2vf2 Ecc;
            _2vf3 Bcc;
            duration build_poisson_time{};
            duration build_interp_time{};

            if (field_mode_ == FieldMode::SelfConsistent &&
                field_update_ == FieldUpdateKind::PoissonYee){
                // Use the stable cell-centred electrostatic solve for particle
                // pushing. The current nodal/Yee reconstruction is not robust
                // for sharply localized particle sources.
                auto t_build_poisson_start = clock::now();
                Ecc = src::fields::solvers::poisson_solver(fields_);
                auto t_build_poisson_end = clock::now();
                build_poisson_time = t_build_poisson_end - t_build_poisson_start;
                Bcc.n_x = mesh_.nx_cells();
                Bcc.n_y = mesh_.ny_cells();
                Bcc.data.assign(static_cast<std::size_t>(Bcc.n_x) * Bcc.n_y,
                                vec3{0.0, 0.0, 0.0});
            } else {
                auto t_build_interp_start = clock::now();
                Ecc = src::coupling::make_cell_center_E(fields_);
                Bcc = src::coupling::make_cell_center_B(fields_);
                auto t_build_interp_end = clock::now();
                build_interp_time = t_build_interp_end - t_build_interp_start;
            }
            auto t_cc_end   = clock::now();

            // Push each 2D species.
            auto t_push_start = clock::now();
            const std::size_t ns = particles_.num_species_2d();
            for (std::size_t s = 0; s < ns; ++s){
                auto& grp = particles_.species_2d(s).particles;

                switch (integrator_){
                    case IntegratorKind::Boris:
                        src::particles::boris_push_2d(grp, mesh_, Ecc, Bcc, dt_);
                        break;
                    case IntegratorKind::Verlet:
                        src::particles::verlet_push_2d(grp, mesh_, Ecc, dt_);
                        break;
                }
            }
            auto t_push_end = clock::now();

            t_ += dt_;

            auto t_end = clock::now();

            duration reset_time = t_reset_end - t_reset_start;
            duration dep_time   = t_dep_end   - t_dep_start;
            duration proj_time{};
            duration field_time = t_field_end - t_field_start;
            duration cc_time    = t_cc_end    - t_cc_start;
            duration push_time  = t_push_end  - t_push_start;
            duration redep_time{};
            duration total_time = t_end       - t_start;

            timing_stats_.record(reset_time.count(),
                                 dep_time.count(),
                                 proj_time.count(),
                                 build_poisson_time.count(),
                                 build_interp_time.count(),
                                 push_time.count(),
                                 redep_time.count(),
                                 field_time.count(),
                                 total_time.count());
#ifdef EPIC_ENABLE_ENV_TIMING
            std::cout << "[Env2D step diagnostics] "
                      << "t=" << t_ << " "
                      << "reset=" << reset_time.count() << "s, "
                      << "deposit=" << dep_time.count()   << "s, "
                      << "projection=" << proj_time.count() << "s, "
                      << "build_poisson=" << build_poisson_time.count() << "s, "
                      << "build_interp=" << build_interp_time.count() << "s, "
                      << "redeposit=" << redep_time.count() << "s, "
                      << "fields="  << field_time.count() << "s, "
                      << "push="   << push_time.count()   << "s, "
                      << "total="  << total_time.count()  << "s"
                      << std::endl;
#endif
        }

    private:
        Mesh            mesh_;
        FieldManager    fields_;
        ParticleManager particles_;

        real t_{0};
        real dt_{0};
        IntegratorKind integrator_;
        FieldMode      field_mode_;
        FieldUpdateKind field_update_;
        bool gauss_projection_enabled_;
        bool maxwell_velocity_staggered_;
        StepTimingStats timing_stats_;
    };

    // ---------------------------------------------------------------------
    // 3D Yee-based environment.
    // ---------------------------------------------------------------------
    class _3environment{
    public:
        using Mesh           = _3Mesh;
        using FieldManager   = _3FieldManager;
        using Species3D      = ParticleManager::Species3D;

        _3environment(const Mesh& mesh, real dt)
            : mesh_{mesh},
              fields_{mesh},
              dt_{dt},
              t_{0},
              integrator_{IntegratorKind::Boris},
              field_mode_{FieldMode::SelfConsistent},
              field_update_{FieldUpdateKind::Maxwell} {
        }

        Mesh&        mesh()        { return mesh_; }
        const Mesh&  mesh()  const { return mesh_; }

        FieldManager&       fields()       { return fields_; }
        const FieldManager& fields() const { return fields_; }

        ParticleManager&       particles()       { return particles_; }
        const ParticleManager& particles() const { return particles_; }

        real time() const { return t_; }
        real dt()   const { return dt_; }

        void set_dt(real dt) { dt_ = dt; }

        void set_integrator(IntegratorKind kind){
            integrator_ = kind;
        }

        void set_field_mode(FieldMode mode){
            field_mode_ = mode;
        }

        void set_field_update(FieldUpdateKind kind){
            field_update_ = kind;
        }

        void reset_timing_stats(){
            timing_stats_.reset();
        }

        const StepTimingStats& timing_stats() const {
            return timing_stats_;
        }

        void print_timing_summary(std::ostream& os = std::cout) const{
            timing_stats_.print_average("Env3D", os);
        }

        // Instrumented with optional per-part timing diagnostics.
        void step(){
            using clock    = std::chrono::high_resolution_clock;
            using duration = std::chrono::duration<double>;

            auto t_start = clock::now();

            // 1) Reset and deposit sources.
            auto t_reset_start = clock::now();
            fields_.reset_sources();
            auto t_reset_end = clock::now();

            auto& rho = fields_.rho();
            auto& Jx  = fields_.Jx();
            auto& Jy  = fields_.Jy();
            auto& Jz  = fields_.Jz();

            auto t_dep_start = clock::now();

            const std::size_t ns_dep = particles_.num_species_3d();
            for (std::size_t s = 0; s < ns_dep; ++s){
                const auto& grp = particles_.species_3d(s).particles;
                src::particles::deposit_rho_J_3d(mesh_, grp,
                                                 rho, Jx, Jy, Jz);
            }

            auto t_dep_end = clock::now();

            // 2) Advance fields (or not) depending on mode.
            auto t_field_start = clock::now();
            if (field_mode_ == FieldMode::SelfConsistent){
                switch (field_update_){
                    case FieldUpdateKind::Maxwell:
                        src::fields::solvers::advance_maxwell_3d(fields_, dt_);
                        break;
                    case FieldUpdateKind::PoissonYee:
                        // Electrostatic 3D solve on the Yee grid.
                        src::fields::solvers::poisson_solve_yee(fields_);
                        break;
                }
            }
            auto t_field_end = clock::now();

            // Build cell-centred fields for particle push.
            auto t_cc_start = clock::now();
            auto Ecc = src::coupling::make_cell_center_E(fields_);
            auto Bcc = src::coupling::make_cell_center_B(fields_);
            auto t_cc_end   = clock::now();

            // Push each 3D species.
            auto t_push_start = clock::now();
            const std::size_t ns = particles_.num_species_3d();
            for (std::size_t s = 0; s < ns; ++s){
                auto& grp = particles_.species_3d(s).particles;

                switch (integrator_){
                    case IntegratorKind::Boris:
                        src::particles::boris_push_3d(grp, mesh_, Ecc, Bcc, dt_);
                        break;
                    case IntegratorKind::Verlet:
                        src::particles::verlet_push_3d(grp, mesh_, Ecc, dt_);
                        break;
                }
            }
            auto t_push_end = clock::now();

            t_ += dt_;

            auto t_end = clock::now();

            duration reset_time = t_reset_end - t_reset_start;
            duration dep_time   = t_dep_end   - t_dep_start;
            duration proj_time{};
            duration field_time = t_field_end - t_field_start;
            duration cc_time    = t_cc_end    - t_cc_start;
            duration push_time  = t_push_end  - t_push_start;
            duration redep_time{};
            duration total_time = t_end       - t_start;

            timing_stats_.record(reset_time.count(),
                                 dep_time.count(),
                                 proj_time.count(),
                                 0.0,
                                 cc_time.count(),
                                 push_time.count(),
                                 redep_time.count(),
                                 field_time.count(),
                                 total_time.count());
#ifdef EPIC_ENABLE_ENV_TIMING
            std::cout << "[Env3D step diagnostics] "
                      << "t=" << t_ << " "
                      << "reset=" << reset_time.count() << "s, "
                      << "deposit=" << dep_time.count()   << "s, "
                      << "projection=" << proj_time.count() << "s, "
                      << "build_poisson=" << 0.0 << "s, "
                      << "build_interp=" << cc_time.count() << "s, "
                      << "redeposit=" << redep_time.count() << "s, "
                      << "fields="  << field_time.count() << "s, "
                      << "push="   << push_time.count()   << "s, "
                      << "total="  << total_time.count()  << "s"
                      << std::endl;
#endif
        }

    private:
        Mesh            mesh_;
        FieldManager    fields_;
        ParticleManager particles_;

        real t_{0};
        real dt_{0};
        IntegratorKind integrator_;
        FieldMode      field_mode_;
        FieldUpdateKind field_update_;
        StepTimingStats timing_stats_;
    };
}
