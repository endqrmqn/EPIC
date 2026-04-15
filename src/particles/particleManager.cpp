#pragma once

#include <vector>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"

namespace src::particles{
    using real = types::real;
    using vec2 = kernels::math::structures::vec2;
    using vec3 = kernels::math::structures::vec3;

    using _2ParticleGroup = kernels::physics::structures::_2ParticleGroup;
    using _3ParticleGroup = kernels::physics::structures::_3ParticleGroup;

    // One species of 2D particles: all particles share the same
    // physical mass/charge, but we still store per-particle arrays
    // for compatibility with existing kernels.
    struct _2Species{
        real species_mass;
        real species_charge;
        _2ParticleGroup particles;

        _2Species() = default;

        _2Species(real m, real q)
            : species_mass{m}, species_charge{q} {}

        // Add a single particle with given phase-space coordinates.
        void add_particle(const vec2& pos,
                          const vec2& vel,
                          const vec2& acc = vec2{}){
            particles.mass.push_back(species_mass);
            particles.charge.push_back(species_charge);
            particles.pos.push_back(pos);
            particles.vel.push_back(vel);
            particles.acc.push_back(acc);
        }

        std::size_t size() const {
            return particles.pos.size();
        }
    };

    // One species of 3D particles.
    struct _3Species{
        real species_mass;
        real species_charge;
        _3ParticleGroup particles;

        _3Species() = default;

        _3Species(real m, real q)
            : species_mass{m}, species_charge{q} {}

        void add_particle(const vec3& pos,
                          const vec3& vel,
                          const vec3& acc = vec3{}){
            particles.mass.push_back(species_mass);
            particles.charge.push_back(species_charge);
            // id field can be used for tagging; for now, use index.
            particles.id.push_back(
                static_cast<std::uint32_t>(particles.pos.size()));
            particles.pos.push_back(pos);
            particles.vel.push_back(vel);
            particles.acc.push_back(acc);
        }

        std::size_t size() const {
            return particles.pos.size();
        }
    };

    // Simple multi-species container. This is a light-weight wrapper
    // that lets the engine manage an arbitrary number of species per
    // dimension while reusing the low-level kernels unchanged.
    class ParticleManager{
    public:
        using Species2D = _2Species;
        using Species3D = _3Species;

        // --- 2D species API -------------------------------------------------

        // Create a new 2D species and return its index.
        std::size_t add_species_2d(real mass, real charge){
            species2d_.emplace_back(mass, charge);
            return species2d_.size() - 1;
        }

        Species2D& species_2d(std::size_t idx){
            return species2d_[idx];
        }

        const Species2D& species_2d(std::size_t idx) const{
            return species2d_[idx];
        }

        std::size_t num_species_2d() const{
            return species2d_.size();
        }

        // --- 3D species API -------------------------------------------------

        std::size_t add_species_3d(real mass, real charge){
            species3d_.emplace_back(mass, charge);
            return species3d_.size() - 1;
        }

        Species3D& species_3d(std::size_t idx){
            return species3d_[idx];
        }

        const Species3D& species_3d(std::size_t idx) const{
            return species3d_[idx];
        }

        std::size_t num_species_3d() const{
            return species3d_.size();
        }

    private:
        std::vector<Species2D> species2d_;
        std::vector<Species3D> species3d_;
    };
}
