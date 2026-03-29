#pragma once

#include <cstdint>
#include <vector>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/mesh.hpp"

namespace kernels::physics::structures{
    template<typename T>
    using vector = std::vector<T>;
    using real = types::real;
    using vec2 = kernels::math::structures::vec2;
    using vec3 = kernels::math::structures::vec3;

    struct _2Particle{
        types::real mass;
        types::real charge;

        kernels::math::structures::vec2 pos;
        kernels::math::structures::vec2 vel;
        kernels::math::structures::vec2 acc;

        _2Particle(const real& mass,
        const real& charge,
        const vec2& pos,
        const vec2& vel,
        const vec2& acc) :
        
        mass{mass}, charge{charge},
        pos{pos}, vel{vel}, acc{acc} {}
    };

    struct _2ParticleGroup{

        vector<real> mass;
        vector<real> charge;

        vector<vec2> pos;
        vector<vec2> vel;
        vector<vec2> acc;

        _2ParticleGroup() = default;
                    
        _2ParticleGroup(const vector<real>& mass,
              const vector<real>& charge,
              const vector<vec2>& pos,
              const vector<vec2>& vel,
              const vector<vec2>& acc): 
            
            mass{mass}, charge{charge},
            pos{pos}, vel{vel}, acc{acc} {}
                
    };

    std::size_t size(const _2ParticleGroup& p){
        return p.pos.size();
    }

    _2Particle getParticle(_2ParticleGroup& p, const int &id){
        return _2Particle(p.mass[id], p.charge[id], p.pos[id], p.vel[id], p.acc[id]);
    }

    vector<vec2> accFromForce(_2ParticleGroup& p, const vec2 &force){
        std::size_t n = size(p);
        vector<vec2> retAcc(n);
        for(std::size_t i = 0; i < n; ++i){
            retAcc[i] = force / p.mass[i];
        }
        return retAcc;
    }
    vector<vec2> accFromForce(_2ParticleGroup& p, const vector<vec2> &force){
        std::size_t n = size(p);
        vector<vec2> retAcc(n);
        for(std::size_t i = 0; i < n; ++i){
            retAcc[i] = force[i] / p.mass[i];
        }
        return retAcc;
    }
    //need to finish
    vector<vec2> accFromField(_2ParticleGroup& p, const _2Mesh &field){
        vector<vec2> retAcc(size(p));

        return retAcc;
    }
    void resetAcceleration(_2ParticleGroup& p){
        for (std::size_t i = 0; i < size(p); ++i){
            p.acc[i] = vec2{};
        }
    }

    void verletStep(_2ParticleGroup& p, const vector<vector<vec2>>& accs, const real &dt){        
        vector<vec2> old_acc = p.acc;
        std::size_t n = size(p);
        
        //update pos
        for (std::size_t i = 0; i < n; ++i){
            p.pos[i] += p.vel[i] * dt + old_acc[i] * (static_cast<real>(0.5) * dt * dt);
        }

        //update acc
        resetAcceleration(p);
        for (const auto& acc_vector : accs){
            for (std::size_t i = 0; i < n; ++i){
                p.acc[i] += acc_vector[i];
            }
        }
        //update vel
        for (std::size_t i = 0; i < n; ++i){
            p.vel[i] += (old_acc[i] + p.acc[i]) * (static_cast<real>(0.5) * dt);
        }
    }

    vector<real> getKineticEnergy(const _2ParticleGroup& p){
        std::size_t n = size(p);
        vector<real> ke(n);
        for (std::size_t i = 0; i < n; ++i){
            ke[i] = static_cast<real> (0.5 * kernels::math::structures::dot(p.vel[i], p.vel[i]) * p.mass[i]);
        }
        return ke;
    }

    struct _3Particle{
        types::real mass;
        types::real charge;
        uint32_t id;

        kernels::math::structures::vec3 pos;
        kernels::math::structures::vec3 vel;
        kernels::math::structures::vec3 acc;

        _3Particle(const real& mass,
        const real& charge,
        const vec3& pos,
        const vec3& vel,
        const vec3& acc) :
        
        mass{mass}, charge{charge},
        pos{pos}, vel{vel}, acc{acc} {}
    };

    struct _3ParticleGroup{

        vector<real> mass;
        vector<real> charge;
        vector<uint32_t> id;

        vector<vec3> pos;
        vector<vec3> vel;
        vector<vec3> acc;

        _3ParticleGroup() = default;
                    
        _3ParticleGroup(const vector<real>& mass,
              const vector<real>& charge,
              const vector<uint32_t>& id,
              const vector<vec3>& pos,
              const vector<vec3>& vel,
              const vector<vec3>& acc): 
            
            mass{mass}, charge{charge}, id{id},
            pos{pos}, vel{vel}, acc{acc} {}
                
    };
    

    std::size_t size(const _3ParticleGroup& p){
        return p.pos.size();
    }

    _3Particle getParticle(_3ParticleGroup& p, const int &id){
        return _3Particle(p.mass[id], p.charge[id], p.pos[id], p.vel[id], p.acc[id]);
    }

    vector<vec3> accFromForce(_3ParticleGroup& p, const vec3 &force){
        std::size_t n = size(p);
        vector<vec3> retAcc(n);
        for(std::size_t i = 0; i < n; ++i){
            retAcc[i] = force / p.mass[i];
        }
        return retAcc;
    }
    vector<vec3> accFromForce(_3ParticleGroup& p, const vector<vec3> &force){
        std::size_t n = size(p);
        vector<vec3> retAcc(n);
        for(std::size_t i = 0; i < n; ++i){
            retAcc[i] = force[i] / p.mass[i];
        }
        return retAcc;
    }
    //need to finish
    vector<vec3> accFromField(_3ParticleGroup& p, const _3Mesh &field){
        vector<vec3> retAcc(size(p));

        return retAcc;
    }
    void resetAcceleration(_3ParticleGroup& p){
        for (std::size_t i = 0; i < size(p); ++i){
            p.acc[i] = vec3{};
        }
    }

    void verletStep(_3ParticleGroup& p, const vector<vector<vec3>>& accs, const real &dt){        
        vector<vec3> old_acc = p.acc;
        std::size_t n = size(p);
        
        //update pos
        for (std::size_t i = 0; i < n; ++i){
            p.pos[i] += p.vel[i] * dt + old_acc[i] * (static_cast<real>(0.5) * dt * dt);
        }

        //update acc
        resetAcceleration(p);
        for (const auto& acc_vector : accs){
            for (std::size_t i = 0; i < n; ++i){
                p.acc[i] += acc_vector[i];
            }
        }
        //update vel
        for (std::size_t i = 0; i < n; ++i){
            p.vel[i] += (old_acc[i] + p.acc[i]) * (static_cast<real>(0.5) * dt);
        }
    }
}