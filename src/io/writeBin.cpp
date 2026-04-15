#pragma once

#include <cstdint>
#include <fstream>
#include <string>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "include/kernels/physics/structures/particlegroup.hpp"

namespace src::io{
    using real = types::real;

    using vec2 = kernels::math::structures::vec2;
    using vec3 = kernels::math::structures::vec3;

    using _2Mesh = kernels::physics::structures::_2Mesh;
    using _3Mesh = kernels::physics::structures::_3Mesh;

    template<typename T>
    using _2Field = kernels::physics::structures::_2Field<T>;
    template<typename T>
    using _3Field = kernels::physics::structures::_3Field<T>;

    using _2ParticleGroup = kernels::physics::structures::_2ParticleGroup;
    using _3ParticleGroup = kernels::physics::structures::_3ParticleGroup;

    // ---------------------------------------------------------------------
    // Binary field writers
    //
    // Layout (2D scalar):
    //   int32 nx, ny
    //   double x0, y0, dx, dy
    //   double time
    //   double data[nx*ny] in row-major: data[j*nx + i]
    //
    // Layout (2D vector):
    //   same header, then double data[nx*ny*2] with components
    //   packed as [j, i, comp], comp = 0:x, 1:y.
    //
    // 3D analogues add nz, z0, dz and pack with nz and 3 components.
    // ---------------------------------------------------------------------

    inline void write_scalar_field_2d(const std::string& filename,
                                      const _2Mesh& mesh,
                                      const _2Field<real>& f,
                                      real time){
        const int32_t nx = f.n_x;
        const int32_t ny = f.n_y;

        std::ofstream os(filename, std::ios::binary);
        if (!os) return;

        const double x0 = static_cast<double>(mesh.x_init);
        const double y0 = static_cast<double>(mesh.y_init);
        const double dx = static_cast<double>(mesh.dx());
        const double dy = static_cast<double>(mesh.dy());

        const double t  = static_cast<double>(time);

        os.write(reinterpret_cast<const char*>(&nx), sizeof(nx));
        os.write(reinterpret_cast<const char*>(&ny), sizeof(ny));
        os.write(reinterpret_cast<const char*>(&x0), sizeof(x0));
        os.write(reinterpret_cast<const char*>(&y0), sizeof(y0));
        os.write(reinterpret_cast<const char*>(&dx), sizeof(dx));
        os.write(reinterpret_cast<const char*>(&dy), sizeof(dy));
        os.write(reinterpret_cast<const char*>(&t),  sizeof(t));

        for (int j = 0; j < ny; ++j){
            for (int i = 0; i < nx; ++i){
                const double val = static_cast<double>(f.valueAt(i, j));
                os.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        }
    }

    inline void write_vector_field_2d(const std::string& filename,
                                      const _2Mesh& mesh,
                                      const _2Field<vec2>& f,
                                      real time){
        const int32_t nx = f.n_x;
        const int32_t ny = f.n_y;

        std::ofstream os(filename, std::ios::binary);
        if (!os) return;

        const double x0 = static_cast<double>(mesh.x_init);
        const double y0 = static_cast<double>(mesh.y_init);
        const double dx = static_cast<double>(mesh.dx());
        const double dy = static_cast<double>(mesh.dy());
        const double t  = static_cast<double>(time);

        os.write(reinterpret_cast<const char*>(&nx), sizeof(nx));
        os.write(reinterpret_cast<const char*>(&ny), sizeof(ny));
        os.write(reinterpret_cast<const char*>(&x0), sizeof(x0));
        os.write(reinterpret_cast<const char*>(&y0), sizeof(y0));
        os.write(reinterpret_cast<const char*>(&dx), sizeof(dx));
        os.write(reinterpret_cast<const char*>(&dy), sizeof(dy));
        os.write(reinterpret_cast<const char*>(&t),  sizeof(t));

        for (int j = 0; j < ny; ++j){
            for (int i = 0; i < nx; ++i){
                const vec2 v = f.valueAt(i, j);
                const double vx = static_cast<double>(v.x);
                const double vy = static_cast<double>(v.y);
                os.write(reinterpret_cast<const char*>(&vx), sizeof(vx));
                os.write(reinterpret_cast<const char*>(&vy), sizeof(vy));
            }
        }
    }

    inline void write_scalar_field_3d(const std::string& filename,
                                      const _3Mesh& mesh,
                                      const _3Field<real>& f,
                                      real time){
        const int32_t nx = f.n_x;
        const int32_t ny = f.n_y;
        const int32_t nz = f.n_z;

        std::ofstream os(filename, std::ios::binary);
        if (!os) return;

        const double x0 = static_cast<double>(mesh.x_init);
        const double y0 = static_cast<double>(mesh.y_init);
        const double z0 = static_cast<double>(mesh.z_init);
        const double dx = static_cast<double>(mesh.dx());
        const double dy = static_cast<double>(mesh.dy());
        const double dz = static_cast<double>(mesh.dz());
        const double t  = static_cast<double>(time);

        os.write(reinterpret_cast<const char*>(&nx), sizeof(nx));
        os.write(reinterpret_cast<const char*>(&ny), sizeof(ny));
        os.write(reinterpret_cast<const char*>(&nz), sizeof(nz));
        os.write(reinterpret_cast<const char*>(&x0), sizeof(x0));
        os.write(reinterpret_cast<const char*>(&y0), sizeof(y0));
        os.write(reinterpret_cast<const char*>(&z0), sizeof(z0));
        os.write(reinterpret_cast<const char*>(&dx), sizeof(dx));
        os.write(reinterpret_cast<const char*>(&dy), sizeof(dy));
        os.write(reinterpret_cast<const char*>(&dz), sizeof(dz));
        os.write(reinterpret_cast<const char*>(&t),  sizeof(t));

        for (int k = 0; k < nz; ++k){
            for (int j = 0; j < ny; ++j){
                for (int i = 0; i < nx; ++i){
                    const double val = static_cast<double>(f.valueAt(i, j, k));
                    os.write(reinterpret_cast<const char*>(&val), sizeof(val));
                }
            }
        }
    }

    inline void write_vector_field_3d(const std::string& filename,
                                      const _3Mesh& mesh,
                                      const _3Field<vec3>& f,
                                      real time){
        const int32_t nx = f.n_x;
        const int32_t ny = f.n_y;
        const int32_t nz = f.n_z;

        std::ofstream os(filename, std::ios::binary);
        if (!os) return;

        const double x0 = static_cast<double>(mesh.x_init);
        const double y0 = static_cast<double>(mesh.y_init);
        const double z0 = static_cast<double>(mesh.z_init);
        const double dx = static_cast<double>(mesh.dx());
        const double dy = static_cast<double>(mesh.dy());
        const double dz = static_cast<double>(mesh.dz());
        const double t  = static_cast<double>(time);

        os.write(reinterpret_cast<const char*>(&nx), sizeof(nx));
        os.write(reinterpret_cast<const char*>(&ny), sizeof(ny));
        os.write(reinterpret_cast<const char*>(&nz), sizeof(nz));
        os.write(reinterpret_cast<const char*>(&x0), sizeof(x0));
        os.write(reinterpret_cast<const char*>(&y0), sizeof(y0));
        os.write(reinterpret_cast<const char*>(&z0), sizeof(z0));
        os.write(reinterpret_cast<const char*>(&dx), sizeof(dx));
        os.write(reinterpret_cast<const char*>(&dy), sizeof(dy));
        os.write(reinterpret_cast<const char*>(&dz), sizeof(dz));
        os.write(reinterpret_cast<const char*>(&t),  sizeof(t));

        for (int k = 0; k < nz; ++k){
            for (int j = 0; j < ny; ++j){
                for (int i = 0; i < nx; ++i){
                    const vec3 v = f.valueAt(i, j, k);
                    const double vx = static_cast<double>(v.x);
                    const double vy = static_cast<double>(v.y);
                    const double vz = static_cast<double>(v.z);
                    os.write(reinterpret_cast<const char*>(&vx), sizeof(vx));
                    os.write(reinterpret_cast<const char*>(&vy), sizeof(vy));
                    os.write(reinterpret_cast<const char*>(&vz), sizeof(vz));
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Particle writers
    //
    // 2D layout:
    //   int32 n
    //   double time
    //   repeated for each particle:
    //       x, y, vx, vy, q, m   (6 doubles)
    //
    // 3D layout:
    //   int32 n
    //   double time
    //   (x, y, z, vx, vy, vz, q, m)  (8 doubles)
    // ---------------------------------------------------------------------

    inline void write_particles_2d(const std::string& filename,
                                   const _2ParticleGroup& p,
                                   real time){
        const std::size_t n_sz = kernels::physics::structures::size(p);
        const int32_t n = static_cast<int32_t>(n_sz);

        std::ofstream os(filename, std::ios::binary);
        if (!os) return;

        const double t = static_cast<double>(time);

        os.write(reinterpret_cast<const char*>(&n), sizeof(n));
        os.write(reinterpret_cast<const char*>(&t), sizeof(t));

        for (std::size_t i = 0; i < n_sz; ++i){
            const vec2 x = p.pos[i];
            const vec2 v = p.vel[i];
            const double px = static_cast<double>(x.x);
            const double py = static_cast<double>(x.y);
            const double vx = static_cast<double>(v.x);
            const double vy = static_cast<double>(v.y);
            const double q  = static_cast<double>(p.charge[i]);
            const double m  = static_cast<double>(p.mass[i]);

            os.write(reinterpret_cast<const char*>(&px), sizeof(px));
            os.write(reinterpret_cast<const char*>(&py), sizeof(py));
            os.write(reinterpret_cast<const char*>(&vx), sizeof(vx));
            os.write(reinterpret_cast<const char*>(&vy), sizeof(vy));
            os.write(reinterpret_cast<const char*>(&q),  sizeof(q));
            os.write(reinterpret_cast<const char*>(&m),  sizeof(m));
        }
    }

    inline void write_particles_3d(const std::string& filename,
                                   const _3ParticleGroup& p,
                                   real time){
        const std::size_t n_sz = kernels::physics::structures::size(p);
        const int32_t n = static_cast<int32_t>(n_sz);

        std::ofstream os(filename, std::ios::binary);
        if (!os) return;

        const double t = static_cast<double>(time);

        os.write(reinterpret_cast<const char*>(&n), sizeof(n));
        os.write(reinterpret_cast<const char*>(&t), sizeof(t));

        for (std::size_t i = 0; i < n_sz; ++i){
            const vec3 x = p.pos[i];
            const vec3 v = p.vel[i];
            const double px = static_cast<double>(x.x);
            const double py = static_cast<double>(x.y);
            const double pz = static_cast<double>(x.z);
            const double vx = static_cast<double>(v.x);
            const double vy = static_cast<double>(v.y);
            const double vz = static_cast<double>(v.z);
            const double q  = static_cast<double>(p.charge[i]);
            const double m  = static_cast<double>(p.mass[i]);

            os.write(reinterpret_cast<const char*>(&px), sizeof(px));
            os.write(reinterpret_cast<const char*>(&py), sizeof(py));
            os.write(reinterpret_cast<const char*>(&pz), sizeof(pz));
            os.write(reinterpret_cast<const char*>(&vx), sizeof(vx));
            os.write(reinterpret_cast<const char*>(&vy), sizeof(vy));
            os.write(reinterpret_cast<const char*>(&vz), sizeof(vz));
            os.write(reinterpret_cast<const char*>(&q),  sizeof(q));
            os.write(reinterpret_cast<const char*>(&m),  sizeof(m));
        }
    }
}
