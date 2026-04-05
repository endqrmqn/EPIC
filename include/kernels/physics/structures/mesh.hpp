#pragma once

#include <cstdint>
#include <vector>

#include "include/types.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/math/structures/vec2.hpp"

namespace kernels::physics::structures{
    template<typename T>
    using vector = std::vector<T>;
    using real = types::real;
    using vec3 = kernels::math::structures::vec3;
    using vec2 = kernels::math::structures::vec2;

    struct _2Mesh{
        real x_init, x_finl, y_init, y_finl;
        int n_x, n_y;
        
        _2Mesh() = default;

        _2Mesh(real x_init, real x_finl,
            real y_init, real y_finl,
            int n_x, int n_y): 
        
            x_init{x_init}, x_finl{x_finl},
            y_init{y_init}, y_finl{y_finl},
            n_x{n_x}, n_y{n_y}
        {}

        inline real dx() const { return (x_finl - x_init) / n_x; }
        inline real dy() const { return (y_finl - y_init) / n_y; }

        inline int nx_cells() const {return n_x;}
        inline int ny_cells() const {return n_y;}

        inline int nx_nodes() const {return n_x+1;}
        inline int ny_nodes() const {return n_y+1;}

        inline int nx_xfaces() const {return n_x+1;}
        inline int nx_yfaces() const {return n_x;}

        inline int ny_xfaces() const {return n_y;}
        inline int ny_yfaces() const {return n_y+1;}

        inline vec2 cell_center(int i, int j) const {
            return vec2(x_init + (i + 0.5) * dx(),
                        y_init + (j + 0.5) * dy());
        }

        inline vec2 node(int i, int j) const {
            return vec2(x_init + i * dx(), y_init + j * dy());
        }

        inline vec2 xf_center(int i, int j) const {
            return vec2(x_init + i * dx(), y_init + (j+0.5) * dy());
        }

        inline vec2 yf_center(int i, int j) const {
            return vec2(x_init + (i+0.5) * dx(), y_init + j * dy());
        }
            
        inline bool pos_to_cell(const vec2& pos, int& i, int& j) const {
            i = static_cast<int>((pos.x - x_init) / dx());
            j = static_cast<int>((pos.y - y_init) / dy());
            return (i >= 0 && i < n_x && j >= 0 && j < n_y);
        }
        
        inline bool pos_to_coords(const vec2& pos, real& i, real& j) const {
            i = (pos.x - x_init) / dx();  // continuous index
            j = (pos.y - y_init) / dy();
            return (i >= 0 && i <= (n_x - 1) && j >= 0 && j <= (n_y - 1));
        }
    };

    struct _3Mesh{
        real x_init, x_finl, y_init, y_finl, z_init, z_finl;
        int n_x, n_y, n_z;
        _3Mesh() = default;
        
        inline real dx() const { return (x_finl - x_init) / n_x; }
        inline real dy() const { return (y_finl - y_init) / n_y; }
        inline real dz() const { return (z_finl - z_init) / n_z; }

        inline int nx_cells() const {return n_x;}
        inline int ny_cells() const {return n_y;}
        inline int nz_cells() const {return n_z;}

        inline int nx_nodes() const {return n_x+1;}
        inline int ny_nodes() const {return n_y+1;}
        inline int nz_nodes() const {return n_z+1;}

        inline int nx_xfaces() const {return n_x+1;}
        inline int ny_xfaces() const {return n_y;}
        inline int nz_xfaces() const {return n_z;}

        inline int nx_yfaces() const {return n_x;}
        inline int ny_yfaces() const {return n_y+1;}
        inline int nz_yfaces() const {return n_z;}

        inline int nx_zfaces() const {return n_x;}
        inline int ny_zfaces() const {return n_y;}
        inline int nz_zfaces() const {return n_z+1;}

        inline vec3 cell_center(int i, int j, int k) const {
            return vec3(x_init + (i + 0.5) * dx(),
                        y_init + (j + 0.5) * dy(),
                        z_init + (k + 0.5) * dz()); 
        }

        inline vec3 node(int i, int j, int k) const {
            return vec3(x_init + i * dx(), y_init + j * dy(), z_init + k * dz());
        }

        inline vec3 xf_center(int i, int j, int k) const {
            return vec3(x_init + i * dx(), y_init + (j+0.5) * dy(), z_init + (k+0.5) * dz());
        }

        inline vec3 yf_center(int i, int j, int k) const {
            return vec3(x_init + (i+0.5) * dx(), y_init + j * dy(), z_init + (k+0.5) * dz());
        }

        inline vec3 zf_center(int i, int j, int k) const {
            return vec3(x_init + (i+0.5) * dx(), y_init + (j+0.5) * dy(), z_init + k * dz());
        }

        inline bool pos_to_cell(const vec3& pos, int& i, int& j, int& k) const {
            i = static_cast<int>((pos.x - x_init) / dx());
            j = static_cast<int>((pos.y - y_init) / dy());
            k = static_cast<int>((pos.z - z_init) / dz());
            return (i >= 0 && i < n_x && j >= 0 && j < n_y && k >= 0 && k < n_z);
        }

        inline bool pos_to_coords(const vec3& pos, real& i, real& j, real& k) const {
            i = (pos.x - x_init) / dx();  // continuous index
            j = (pos.y - y_init) / dy();
            k = (pos.z - z_init) / dz();
            return (i >= 0 && i < n_x && j >= 0 && j < n_y && k >= 0 && k < n_z);
        }
    };
}
