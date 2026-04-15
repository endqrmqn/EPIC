#include <cmath>
#include <chrono>
#include <iostream>

#include "include/types.hpp"
#include "include/constants.hpp"
#include "include/kernels/math/structures/vec2.hpp"
#include "include/kernels/math/structures/vec3.hpp"
#include "include/kernels/physics/structures/mesh.hpp"
#include "include/kernels/physics/structures/field.hpp"
#include "include/kernels/math/algs/vectorops.hpp"
#include "src/fields/fieldManager.cpp"

namespace src::fields::solvers{
    using vec2   = kernels::math::structures::vec2;
    using vec3   = kernels::math::structures::vec3;
    using _2Mesh = kernels::physics::structures::_2Mesh;
    using _3Mesh = kernels::physics::structures::_3Mesh;
    using real   = types::real;

    using _2sf   = kernels::physics::structures::_2Field<real>;
    using _3sf   = kernels::physics::structures::_3Field<real>;

    using _2vf2  = kernels::physics::structures::_2Field<vec2>;
    using _3vf3  = kernels::physics::structures::_3Field<vec3>;

    using _2FieldManager = src::fields::_2FieldManager;
    using _3FieldManager = src::fields::_3FieldManager;
    // -------------------------------------------------------------------------
    // Core Poisson solvers for phi (cell-centred), using the same discrete
    // Laplacian as div(grad(phi)) built from centred differences. This is a
    // "wide" 5-point / 7-point stencil:
    //
    //   (φ_{i+2} - 2φ_i + φ_{i-2}) / (4 h^2)   in each coordinate,
    //
    // so that E = -grad(phi) together with div(E) use a coherent second-order
    // Maxwell calculus (all centred stencils). We solve
    //
    //   ∇² phi = -rho / eps0
    //
    // with Dirichlet phi = 0 on the boundary.
    // -------------------------------------------------------------------------

    // Apply wide 5-point Laplacian (matching div∘grad) in 2D.
    static void apply_laplacian_2d(const _2Mesh& mesh,
                                   const _2sf& phi,
                                   _2sf& out){
        const int nx = phi.n_x;
        const int ny = phi.n_y;

        const real h  = mesh.dx();
        const real h2 = h * h;

        out.n_x = nx;
        out.n_y = ny;
        out.data.assign(static_cast<std::size_t>(nx) * ny, real{0});

        for (int j = 2; j < ny - 2; ++j) {
            for (int i = 2; i < nx - 2; ++i) {
                const real phi_ip2 = phi.valueAt(i + 2, j);
                const real phi_im2 = phi.valueAt(i - 2, j);
                const real phi_jp2 = phi.valueAt(i,     j + 2);
                const real phi_jm2 = phi.valueAt(i,     j - 2);
                const real phi_c   = phi.valueAt(i,     j);

                const real lap =
                    (phi_ip2 + phi_im2 + phi_jp2 + phi_jm2
                     - static_cast<real>(4.0) * phi_c) /
                    (static_cast<real>(4.0) * h2);

                out.valueAt(i, j) = lap;
            }
        }
    }

    void solve_poisson_phi_2d(const _2Mesh& mesh,
                              const _2sf& rho,
                              _2sf& phi_out){
        const int nx = rho.n_x;
        const int ny = rho.n_y;

        const real inv_eps0 =
            static_cast<real>(1.0) / static_cast<real>(constants::epsilon_0);

        _2sf phi, r, p, Ap, b;
        phi.n_x = r.n_x = p.n_x = Ap.n_x = b.n_x = nx;
        phi.n_y = r.n_y = p.n_y = Ap.n_y = b.n_y = ny;
        phi.data.assign(static_cast<std::size_t>(nx) * ny, real{0});
        b.data.assign(static_cast<std::size_t>(nx) * ny, real{0});

        // b = -rho/eps0 on the interior.
        for (int j = 2; j < ny - 2; ++j) {
            for (int i = 2; i < nx - 2; ++i) {
                b.valueAt(i, j) = -rho.valueAt(i, j) * inv_eps0;
            }
        }

        // r = b - A phi; phi starts at 0 => r = b.
        r = b;
        p = r;

        auto dot_2d = [&](const _2sf& a, const _2sf& c) -> real {
            real s = 0;
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 2; i < nx - 2; ++i) {
                    s += a.valueAt(i, j) * c.valueAt(i, j);
                }
            }
            return s;
        };

        real rr = dot_2d(r, r);
        if (rr == real{0}) {
            phi_out.n_x = nx;
            phi_out.n_y = ny;
            phi_out.data = phi.data;
            return;
        }

        // Relative residual tolerance. 1e-5 is well below the truncation
        // error (O(h^2)) for typical grids, and avoids excessive iterations
        // on large meshes.
        const real tol_rel = static_cast<real>(1e-5);
        const real tol2 = tol_rel * tol_rel * rr;

        const int max_iter = 2000;

        for (int it = 0; it < max_iter; ++it) {
            apply_laplacian_2d(mesh, p, Ap);

            const real pAp = dot_2d(p, Ap);
            if (std::abs(pAp) < static_cast<real>(1e-30)) break;

            const real alpha = rr / pAp;

            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 2; i < nx - 2; ++i) {
                    phi.valueAt(i, j) += alpha * p.valueAt(i, j);
                    r.valueAt(i, j)   -= alpha * Ap.valueAt(i, j);
                }
            }

            const real rr_new = dot_2d(r, r);
            if (rr_new < tol2) break;

            const real beta = rr_new / rr;
            rr = rr_new;

            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 2; i < nx - 2; ++i) {
                    p.valueAt(i, j) = r.valueAt(i, j)
                                    + beta * p.valueAt(i, j);
                }
            }
        }

        phi_out.n_x = nx;
        phi_out.n_y = ny;
        phi_out.data = phi.data;
    }

    // Apply wide 7-point Laplacian (matching div∘grad) in 3D.
    static void apply_laplacian_3d(const _3Mesh& mesh,
                                   const _3sf& phi,
                                   _3sf& out){
        const int nx = phi.n_x;
        const int ny = phi.n_y;
        const int nz = phi.n_z;

        const real h  = mesh.dx();
        const real h2 = h * h;

        out.n_x = nx;
        out.n_y = ny;
        out.n_z = nz;

        const std::size_t N = static_cast<std::size_t>(nx) * ny * nz;
        out.data.assign(N, real{0});

        for (int k = 2; k < nz - 2; ++k) {
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 2; i < nx - 2; ++i) {
                    const real phi_ip2 = phi.valueAt(i + 2, j, k);
                    const real phi_im2 = phi.valueAt(i - 2, j, k);
                    const real phi_jp2 = phi.valueAt(i,     j + 2, k);
                    const real phi_jm2 = phi.valueAt(i,     j - 2, k);
                    const real phi_kp2 = phi.valueAt(i,     j,     k + 2);
                    const real phi_km2 = phi.valueAt(i,     j,     k - 2);
                    const real phi_c   = phi.valueAt(i,     j,     k);

                    const real sum_nb =
                        phi_ip2 + phi_im2 +
                        phi_jp2 + phi_jm2 +
                        phi_kp2 + phi_km2;

                    const real lap =
                        (sum_nb - static_cast<real>(6.0) * phi_c) /
                        (static_cast<real>(4.0) * h2);

                    out.valueAt(i, j, k) = lap;
                }
            }
        }
    }

    void solve_poisson_phi_3d(const _3Mesh& mesh,
                              const _3sf& rho,
                              _3sf& phi_out){
        const int nx = rho.n_x;
        const int ny = rho.n_y;
        const int nz = rho.n_z;

        const real inv_eps0 =
            static_cast<real>(1.0) / static_cast<real>(constants::epsilon_0);

        _3sf phi, r, p, Ap, b;
        phi.n_x = r.n_x = p.n_x = Ap.n_x = b.n_x = nx;
        phi.n_y = r.n_y = p.n_y = Ap.n_y = b.n_y = ny;
        phi.n_z = r.n_z = p.n_z = Ap.n_z = b.n_z = nz;

        const std::size_t N = static_cast<std::size_t>(nx) * ny * nz;
        phi.data.assign(N, real{0});
        b.data.assign(N, real{0});

        for (int k = 2; k < nz - 2; ++k) {
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 2; i < nx - 2; ++i) {
                    b.valueAt(i, j, k) =
                        -rho.valueAt(i, j, k) * inv_eps0;
                }
            }
        }

        r = b;
        p = r;

        auto dot_3d = [&](const _3sf& a, const _3sf& c) -> real {
            real s = 0;
            for (int k = 2; k < nz - 2; ++k) {
                for (int j = 2; j < ny - 2; ++j) {
                    for (int i = 2; i < nx - 2; ++i) {
                        s += a.valueAt(i, j, k) * c.valueAt(i, j, k);
                    }
                }
            }
            return s;
        };

        real rr = dot_3d(r, r);
        if (rr == real{0}) {
            phi_out.n_x = nx;
            phi_out.n_y = ny;
            phi_out.n_z = nz;
            phi_out.data = phi.data;
            return;
        }

        const real tol_rel = static_cast<real>(1e-5);
        const real tol2 = tol_rel * tol_rel * rr;

        const int max_iter = 2000;

        for (int it = 0; it < max_iter; ++it) {
            apply_laplacian_3d(mesh, p, Ap);

            const real pAp = dot_3d(p, Ap);
            if (std::abs(pAp) < static_cast<real>(1e-30)) break;

            const real alpha = rr / pAp;

            for (int k = 2; k < nz - 2; ++k) {
                for (int j = 2; j < ny - 2; ++j) {
                    for (int i = 2; i < nx - 2; ++i) {
                        phi.valueAt(i, j, k) += alpha * p.valueAt(i, j, k);
                        r.valueAt(i, j, k)   -= alpha * Ap.valueAt(i, j, k);
                    }
                }
            }

            const real rr_new = dot_3d(r, r);
            if (rr_new < tol2) break;

            const real beta = rr_new / rr;
            rr = rr_new;

            for (int k = 2; k < nz - 2; ++k) {
                for (int j = 2; j < ny - 2; ++j) {
                    for (int i = 2; i < nx - 2; ++i) {
                        p.valueAt(i, j, k) =
                            r.valueAt(i, j, k) +
                            beta * p.valueAt(i, j, k);
                    }
                }
            }
        }

        phi_out.n_x = nx;
        phi_out.n_y = ny;
        phi_out.n_z = nz;
        phi_out.data = phi.data;
    }

    // -------------------------------------------------------------------------
    // Convenience wrappers: return collocated E = -grad(phi) for 2D/3D. These
    // operate on cell-centred fields and are used by existing tests.
    // -------------------------------------------------------------------------

    _2vf2 poisson_solver(const _2FieldManager& fm){
        const _2Mesh& mesh = fm.mesh();
        const _2sf&   rho  = fm.rho();

        _2sf phi;
        solve_poisson_phi_2d(mesh, rho, phi);

        const int nx = phi.n_x;
        const int ny = phi.n_y;

        _2vf2 E;
        E.n_x = nx;
        E.n_y = ny;
        E.data.resize(static_cast<std::size_t>(nx) * ny);

        // E = -grad(phi)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                vec2 g = kernels::math::algs::gradPoint(mesh, phi, i, j);
                E.valueAt(i, j) = vec2{-g.x, -g.y};
            }
        }

        return E;
    }

    // -------------------------------------------------------------------------
    // Yee-compatible nodal Poisson: phi on mesh nodes, E on Yee faces.
    // -------------------------------------------------------------------------

    // Build a nodal charge density by averaging surrounding cells (2D).
    static void build_nodal_rho_2d(const _2Mesh& mesh,
                                   const _2sf& rho_cell,
                                   _2sf& rho_node){
        const int nx_cells = mesh.nx_cells();
        const int ny_cells = mesh.ny_cells();

        const int nx_nodes = mesh.nx_nodes(); // nx_cells + 1
        const int ny_nodes = mesh.ny_nodes(); // ny_cells + 1

        rho_node.n_x = nx_nodes;
        rho_node.n_y = ny_nodes;
        rho_node.data.assign(static_cast<std::size_t>(nx_nodes) * ny_nodes,
                             real{0});

        // Interior nodes: average the four surrounding cell-centred rho's.
        for (int j = 1; j < ny_cells; ++j) {
            for (int i = 1; i < nx_cells; ++i) {
                const real r00 = rho_cell.valueAt(i - 1, j - 1);
                const real r10 = rho_cell.valueAt(i,     j - 1);
                const real r01 = rho_cell.valueAt(i - 1, j);
                const real r11 = rho_cell.valueAt(i,     j);
                rho_node.valueAt(i, j) =
                    static_cast<real>(0.25) * (r00 + r10 + r01 + r11);
            }
        }
    }

    // Apply standard 5-point Laplacian on nodal grid in 2D.
    static void apply_nodal_laplacian_2d(const _2Mesh& mesh,
                                         const _2sf& phi,
                                         _2sf& out){
        const int nx_nodes = mesh.nx_nodes();
        const int ny_nodes = mesh.ny_nodes();

        const real h  = mesh.dx();
        const real h2 = h * h;

        out.n_x = nx_nodes;
        out.n_y = ny_nodes;
        out.data.assign(static_cast<std::size_t>(nx_nodes) * ny_nodes, real{0});

        for (int j = 1; j < ny_nodes - 1; ++j) {
            for (int i = 1; i < nx_nodes - 1; ++i) {
                const real phi_ip = phi.valueAt(i + 1, j);
                const real phi_im = phi.valueAt(i - 1, j);
                const real phi_jp = phi.valueAt(i,     j + 1);
                const real phi_jm = phi.valueAt(i,     j - 1);
                const real phi_c  = phi.valueAt(i,     j);

                const real lap =
                    (phi_ip + phi_im + phi_jp + phi_jm
                     - static_cast<real>(4.0) * phi_c) / h2;

                out.valueAt(i, j) = lap;
            }
        }
    }

    // Solve nodal Poisson in 2D: Lap(phi_node) = -rho_node/eps0 with phi=0 on boundary.
    // Cold-start version: phi starts from zero.
    void solve_poisson_phi_nodes_2d(const _2Mesh& mesh,
                                    const _2sf& rho_cell,
                                    _2sf& phi_node_out){
        const int nx_nodes = mesh.nx_nodes();
        const int ny_nodes = mesh.ny_nodes();

        const real inv_eps0 =
            static_cast<real>(1.0) / static_cast<real>(constants::epsilon_0);

        _2sf rho_node;
        build_nodal_rho_2d(mesh, rho_cell, rho_node);

        _2sf phi, r, p, Ap, b;
        phi.n_x = r.n_x = p.n_x = Ap.n_x = b.n_x = nx_nodes;
        phi.n_y = r.n_y = p.n_y = Ap.n_y = b.n_y = ny_nodes;
        phi.data.assign(static_cast<std::size_t>(nx_nodes) * ny_nodes, real{0});
        b.data.assign(static_cast<std::size_t>(nx_nodes) * ny_nodes, real{0});

        for (int j = 1; j < ny_nodes - 1; ++j) {
            for (int i = 1; i < nx_nodes - 1; ++i) {
                b.valueAt(i, j) = -rho_node.valueAt(i, j) * inv_eps0;
            }
        }

        r = b;
        p = r;

        auto dot = [&](const _2sf& a, const _2sf& c) -> real {
            real s = 0;
            for (int j = 1; j < ny_nodes - 1; ++j) {
                for (int i = 1; i < nx_nodes - 1; ++i) {
                    s += a.valueAt(i, j) * c.valueAt(i, j);
                }
            }
            return s;
        };

        real rr = dot(r, r);
        if (rr == real{0}) {
            phi_node_out = phi;
            return;
        }

        const real tol_rel = static_cast<real>(1e-5);
        const real tol2 = tol_rel * tol_rel * rr;
        const int  max_iter = 2000;

        for (int it = 0; it < max_iter; ++it) {
            apply_nodal_laplacian_2d(mesh, p, Ap);

            const real pAp = dot(p, Ap);
            if (std::abs(pAp) < static_cast<real>(1e-30)) break;

            const real alpha = rr / pAp;

            for (int j = 1; j < ny_nodes - 1; ++j) {
                for (int i = 1; i < nx_nodes - 1; ++i) {
                    phi.valueAt(i, j) += alpha * p.valueAt(i, j);
                    r.valueAt(i, j)   -= alpha * Ap.valueAt(i, j);
                }
            }

            const real rr_new = dot(r, r);
            if (rr_new < tol2) break;

            const real beta = rr_new / rr;
            rr = rr_new;

            for (int j = 1; j < ny_nodes - 1; ++j) {
                for (int i = 1; i < nx_nodes - 1; ++i) {
                    p.valueAt(i, j) =
                        r.valueAt(i, j) + beta * p.valueAt(i, j);
                }
            }
        }

        phi_node_out = phi;
    }

    // Warm-start nodal Poisson in 2D: same equation, but phi_node_io is used
    // as the initial guess (typically from the previous timestep).
    void solve_poisson_phi_nodes_2d_warm(const _2Mesh& mesh,
                                         const _2sf& rho_cell,
                                         _2sf& phi_node_io){
        const int nx_nodes = mesh.nx_nodes();
        const int ny_nodes = mesh.ny_nodes();

        const real inv_eps0 =
            static_cast<real>(1.0) / static_cast<real>(constants::epsilon_0);

        _2sf rho_node;
        build_nodal_rho_2d(mesh, rho_cell, rho_node);

        _2sf r, p, Ap, b;
        r.n_x = p.n_x = Ap.n_x = b.n_x = nx_nodes;
        r.n_y = p.n_y = Ap.n_y = b.n_y = ny_nodes;
        b.data.assign(static_cast<std::size_t>(nx_nodes) * ny_nodes, real{0});

        // Ensure phi_node_io has correct size; if not, reinitialise to zero.
        if (phi_node_io.n_x != nx_nodes || phi_node_io.n_y != ny_nodes ||
            phi_node_io.data.size() != static_cast<std::size_t>(nx_nodes) * ny_nodes){
            phi_node_io.n_x = nx_nodes;
            phi_node_io.n_y = ny_nodes;
            phi_node_io.data.assign(static_cast<std::size_t>(nx_nodes) * ny_nodes, real{0});
        }

        for (int j = 1; j < ny_nodes - 1; ++j) {
            for (int i = 1; i < nx_nodes - 1; ++i) {
                b.valueAt(i, j) = -rho_node.valueAt(i, j) * inv_eps0;
            }
        }

        // r = b - A phi
        apply_nodal_laplacian_2d(mesh, phi_node_io, Ap);

        r.n_x = nx_nodes;
        r.n_y = ny_nodes;
        r.data.assign(static_cast<std::size_t>(nx_nodes) * ny_nodes, real{0});

        for (int j = 1; j < ny_nodes - 1; ++j) {
            for (int i = 1; i < nx_nodes - 1; ++i) {
                r.valueAt(i, j) = b.valueAt(i, j) - Ap.valueAt(i, j);
            }
        }

        p = r;

        auto dot = [&](const _2sf& a, const _2sf& c) -> real {
            real s = 0;
            for (int j = 1; j < ny_nodes - 1; ++j) {
                for (int i = 1; i < nx_nodes - 1; ++i) {
                    s += a.valueAt(i, j) * c.valueAt(i, j);
                }
            }
            return s;
        };

        real rr = dot(r, r);
        if (rr == real{0}) {
            return;
        }

        const real tol_rel = static_cast<real>(1e-5);
        const real tol2 = tol_rel * tol_rel * rr;
        const int  max_iter = 2000;

        for (int it = 0; it < max_iter; ++it) {
            apply_nodal_laplacian_2d(mesh, p, Ap);

            const real pAp = dot(p, Ap);
            if (std::abs(pAp) < static_cast<real>(1e-30)) break;

            const real alpha = rr / pAp;

            for (int j = 1; j < ny_nodes - 1; ++j) {
                for (int i = 1; i < nx_nodes - 1; ++i) {
                    phi_node_io.valueAt(i, j) += alpha * p.valueAt(i, j);
                    r.valueAt(i, j)          -= alpha * Ap.valueAt(i, j);
                }
            }

            const real rr_new = dot(r, r);
            if (rr_new < tol2) break;

            const real beta = rr_new / rr;
            rr = rr_new;

            for (int j = 1; j < ny_nodes - 1; ++j) {
                for (int i = 1; i < nx_nodes - 1; ++i) {
                    p.valueAt(i, j) =
                        r.valueAt(i, j) + beta * p.valueAt(i, j);
                }
            }
        }
    }

    // Given nodal phi, fill Yee Ex/Ey in 2D _2FieldManager.
    void set_yee_E_from_phi_nodes_2d(_2FieldManager& fm,
                                     const _2sf& phi_node){
        const auto& mesh = fm.mesh();
        const int nx_cells = mesh.nx_cells();
        const int ny_cells = mesh.ny_cells();

        auto& Ex = fm.Ex(); // (nx+1, ny)
        auto& Ey = fm.Ey(); // (nx, ny+1)

        const real dx = mesh.dx();
        const real dy = mesh.dy();

        // Ex on x-faces: centred difference in x using nodal planes,
        // averaged vertically. Interior faces use a 2nd-order stencil;
        // boundary faces fall back to one-sided differences.
        for (int j = 0; j < ny_cells; ++j) {
            for (int i = 0; i < nx_cells + 1; ++i) {
                real phi_L, phi_R;

                if (i > 0 && i < nx_cells){
                    // centred: use planes at i-1 and i+1
                    const real phi_L0 = phi_node.valueAt(i - 1, j);
                    const real phi_L1 = phi_node.valueAt(i - 1, j + 1);
                    const real phi_R0 = phi_node.valueAt(i + 1, j);
                    const real phi_R1 = phi_node.valueAt(i + 1, j + 1);

                    phi_L = static_cast<real>(0.5) * (phi_L0 + phi_L1);
                    phi_R = static_cast<real>(0.5) * (phi_R0 + phi_R1);

                    Ex.valueAt(i, j) = -(phi_R - phi_L) / (static_cast<real>(2.0) * dx);
                } else if (i == 0){
                    // forward difference at left boundary
                    const real phi_L0 = phi_node.valueAt(0, j);
                    const real phi_L1 = phi_node.valueAt(0, j + 1);
                    const real phi_R0 = phi_node.valueAt(1, j);
                    const real phi_R1 = phi_node.valueAt(1, j + 1);

                    phi_L = static_cast<real>(0.5) * (phi_L0 + phi_L1);
                    phi_R = static_cast<real>(0.5) * (phi_R0 + phi_R1);

                    Ex.valueAt(i, j) = -(phi_R - phi_L) / dx;
                } else { // i == nx_cells
                    // backward difference at right boundary
                    const real phi_L0 = phi_node.valueAt(nx_cells - 1, j);
                    const real phi_L1 = phi_node.valueAt(nx_cells - 1, j + 1);
                    const real phi_R0 = phi_node.valueAt(nx_cells,     j);
                    const real phi_R1 = phi_node.valueAt(nx_cells,     j + 1);

                    phi_L = static_cast<real>(0.5) * (phi_L0 + phi_L1);
                    phi_R = static_cast<real>(0.5) * (phi_R0 + phi_R1);

                    Ex.valueAt(i, j) = -(phi_R - phi_L) / dx;
                }
            }
        }

        // Ey on y-faces: centred difference in y using nodal planes,
        // averaged horizontally, with one-sided stencils on boundaries.
        for (int j = 0; j < ny_cells + 1; ++j) {
            for (int i = 0; i < nx_cells; ++i) {
                real phi_B, phi_T;

                if (j > 0 && j < ny_cells){
                    // centred in y: planes at j-1 and j+1
                    const real phi_B0 = phi_node.valueAt(i,     j - 1);
                    const real phi_B1 = phi_node.valueAt(i + 1, j - 1);
                    const real phi_T0 = phi_node.valueAt(i,     j + 1);
                    const real phi_T1 = phi_node.valueAt(i + 1, j + 1);

                    phi_B = static_cast<real>(0.5) * (phi_B0 + phi_B1);
                    phi_T = static_cast<real>(0.5) * (phi_T0 + phi_T1);

                    Ey.valueAt(i, j) = -(phi_T - phi_B) / (static_cast<real>(2.0) * dy);
                } else if (j == 0){
                    // forward difference at bottom boundary
                    const real phi_B0 = phi_node.valueAt(i,     0);
                    const real phi_B1 = phi_node.valueAt(i + 1, 0);
                    const real phi_T0 = phi_node.valueAt(i,     1);
                    const real phi_T1 = phi_node.valueAt(i + 1, 1);

                    phi_B = static_cast<real>(0.5) * (phi_B0 + phi_B1);
                    phi_T = static_cast<real>(0.5) * (phi_T0 + phi_T1);

                    Ey.valueAt(i, j) = -(phi_T - phi_B) / dy;
                } else { // j == ny_cells
                    // backward difference at top boundary
                    const real phi_B0 = phi_node.valueAt(i,     ny_cells - 1);
                    const real phi_B1 = phi_node.valueAt(i + 1, ny_cells - 1);
                    const real phi_T0 = phi_node.valueAt(i,     ny_cells);
                    const real phi_T1 = phi_node.valueAt(i + 1, ny_cells);

                    phi_B = static_cast<real>(0.5) * (phi_B0 + phi_B1);
                    phi_T = static_cast<real>(0.5) * (phi_T0 + phi_T1);

                    Ey.valueAt(i, j) = -(phi_T - phi_B) / dy;
                }
            }
        }
    }

    _3vf3 poisson_solver(const _3FieldManager& fm){
        const _3Mesh& mesh = fm.mesh();
        const _3sf&   rho  = fm.rho();

        _3sf phi;
        solve_poisson_phi_3d(mesh, rho, phi);

        const int nx = phi.n_x;
        const int ny = phi.n_y;
        const int nz = phi.n_z;

        _3vf3 E;
        E.n_x = nx;
        E.n_y = ny;
        E.n_z = nz;

        const std::size_t N = static_cast<std::size_t>(nx) * ny * nz;
        E.data.resize(N);

        // E = -grad(phi)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    vec3 g = kernels::math::algs::gradPoint(mesh, phi, i, j, k);
                    E.valueAt(i, j, k) = vec3{-g.x, -g.y, -g.z};
                }
            }
        }

        return E;
    }

    // Forward declaration of multigrid nodal Poisson solver.
    void solve_poisson_nodes_multigrid_2d(const _2Mesh& mesh,
                                          const _2sf& rhs_node,
                                          _2sf& phi_node);

    // High-level 2D Yee Poisson: solve nodal phi and fill Ex/Ey on the Yee grid.
    // Instrumented with simple timing diagnostics.
    void poisson_solve_yee(_2FieldManager& fm){
        using clock    = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<double>;

        auto t_start = clock::now();

        const _2Mesh& mesh = fm.mesh();
        const _2sf&   rho  = fm.rho();

        auto t_solve_start = clock::now();
        static _2sf phi_nodes;

#ifdef EPIC_USE_MULTIGRID_POISSON
        // Multigrid: build nodal RHS: -rho_node/eps0 and solve.
        _2sf rho_node;
        build_nodal_rho_2d(mesh, rho, rho_node);

        const int nx_nodes = mesh.nx_nodes();
        const int ny_nodes = mesh.ny_nodes();

        _2sf rhs_node;
        rhs_node.n_x = nx_nodes;
        rhs_node.n_y = ny_nodes;
        rhs_node.data.assign(static_cast<std::size_t>(nx_nodes) * ny_nodes,
                             real{0});

        const real inv_eps0 =
            static_cast<real>(1.0) / static_cast<real>(constants::epsilon_0);

        for (int j = 1; j < ny_nodes - 1; ++j){
            for (int i = 1; i < nx_nodes - 1; ++i){
                rhs_node.valueAt(i, j) =
                    -rho_node.valueAt(i, j) * inv_eps0;
            }
        }

        solve_poisson_nodes_multigrid_2d(mesh, rhs_node, phi_nodes);
#else
        // CG-based solve (original path).
        if (phi_nodes.data.empty()){
            // First call: cold start to initialise phi.
            solve_poisson_phi_nodes_2d(mesh, rho, phi_nodes);
        } else {
            // Subsequent calls: use previous phi as warm start.
            solve_poisson_phi_nodes_2d_warm(mesh, rho, phi_nodes);
        }
#endif
        auto t_solve_end   = clock::now();

        auto t_yee_start = clock::now();
        set_yee_E_from_phi_nodes_2d(fm, phi_nodes);
        auto t_yee_end   = clock::now();

        auto t_end = clock::now();

        duration solve_time = t_solve_end - t_solve_start;
        duration yee_time   = t_yee_end   - t_yee_start;
        duration total      = t_end       - t_start;

#ifdef EPIC_ENABLE_FIELD_TIMING
        std::cout << "[PoissonYee2D diagnostics] "
                  << "solve_phi_nodes=" << solve_time.count() << "s, "
                  << "set_yee_E="      << yee_time.count()   << "s, "
                  << "total="          << total.count()      << "s"
                  << std::endl;
#endif
    }

    // ---------------------------------------------------------------------
    // 2D multigrid Poisson solver on nodal grid (Dirichlet 0 on edges),
    // using red-black Jacobi smoothing and a simple V-cycle.
    // Enabled when EPIC_USE_MULTIGRID_POISSON is defined.
    // ---------------------------------------------------------------------

#ifdef EPIC_USE_MULTIGRID_POISSON

    namespace {
        // Red-black Jacobi smoother.
        static void smooth_rb_jacobi_2d(_2sf& phi,
                                        const _2sf& rhs,
                                        real h,
                                        int iters){
            const int nx = phi.n_x;
            const int ny = phi.n_y;
            const real h2   = h * h;
            const real inv4 = static_cast<real>(0.25);
            const real omega = static_cast<real>(2.0 / 3.0);

            for (int it = 0; it < iters; ++it){
                for (int color = 0; color < 2; ++color){
                    for (int j = 1; j < ny - 1; ++j){
                        for (int i = 1; i < nx - 1; ++i){
                            if ( ( (i + j) & 1 ) != color ) continue;

                            const real phiN = phi.valueAt(i, j + 1);
                            const real phiS = phi.valueAt(i, j - 1);
                            const real phiE = phi.valueAt(i + 1, j);
                            const real phiW = phi.valueAt(i - 1, j);
                            const real rhs_ij = rhs.valueAt(i, j);

                            const real phi_new =
                                inv4 * (phiN + phiS + phiE + phiW - h2 * rhs_ij);
                            const real old = phi.valueAt(i, j);
                            phi.valueAt(i, j) = old + omega * (phi_new - old);
                        }
                    }
                }
            }
        }

        static void compute_residual_2d(const _2sf& phi,
                                        const _2sf& rhs,
                                        _2sf& res,
                                        real h){
            const int nx = phi.n_x;
            const int ny = phi.n_y;
            const real inv_h2 = static_cast<real>(1.0) / (h * h);

            res.n_x = nx;
            res.n_y = ny;
            res.data.assign(static_cast<std::size_t>(nx) * ny, real{0});

            for (int j = 1; j < ny - 1; ++j){
                for (int i = 1; i < nx - 1; ++i){
                    const real phiC = phi.valueAt(i, j);
                    const real phiN = phi.valueAt(i, j + 1);
                    const real phiS = phi.valueAt(i, j - 1);
                    const real phiE = phi.valueAt(i + 1, j);
                    const real phiW = phi.valueAt(i - 1, j);

                    const real lap_phi =
                        (phiE + phiW + phiN + phiS
                         - static_cast<real>(4.0) * phiC) * inv_h2;

                    res.valueAt(i, j) = rhs.valueAt(i, j) - lap_phi;
                }
            }
        }

        static void restrict_full_weighting_2d(const _2sf& fine,
                                               _2sf& coarse){
            const int nx_f = fine.n_x;
            const int ny_f = fine.n_y;
            const int nx_c = (nx_f + 1) / 2;
            const int ny_c = (ny_f + 1) / 2;

            coarse.n_x = nx_c;
            coarse.n_y = ny_c;
            coarse.data.assign(static_cast<std::size_t>(nx_c) * ny_c, real{0});

            for (int J = 1; J < ny_c - 1; ++J){
                for (int I = 1; I < nx_c - 1; ++I){
                    const int i = 2 * I;
                    const int j = 2 * J;

                    const real c  = fine.valueAt(i, j);
                    const real n  = fine.valueAt(i, j + 1);
                    const real s  = fine.valueAt(i, j - 1);
                    const real e  = fine.valueAt(i + 1, j);
                    const real w  = fine.valueAt(i - 1, j);
                    const real ne = fine.valueAt(i + 1, j + 1);
                    const real nw = fine.valueAt(i - 1, j + 1);
                    const real se = fine.valueAt(i + 1, j - 1);
                    const real sw = fine.valueAt(i - 1, j - 1);

                    const real sum =
                        static_cast<real>(4.0) * c +
                        static_cast<real>(2.0) * (n + s + e + w) +
                        (ne + nw + se + sw);

                    coarse.valueAt(I, J) =
                        sum / static_cast<real>(16.0);
                }
            }
        }

        static void prolong_bilinear_2d(const _2sf& coarse,
                                        _2sf& fine){
            const int nx_c = coarse.n_x;
            const int ny_c = coarse.n_y;
            const int nx_f = fine.n_x;
            const int ny_f = fine.n_y;

            // Zero fine grid first (keeps boundaries at 0).
            fine.data.assign(static_cast<std::size_t>(nx_f) * ny_f, real{0});

            // Inject coarse points.
            for (int J = 0; J < ny_c; ++J){
                for (int I = 0; I < nx_c; ++I){
                    const int i = 2 * I;
                    const int j = 2 * J;
                    if (i < nx_f && j < ny_f){
                        fine.valueAt(i, j) = coarse.valueAt(I, J);
                    }
                }
            }

            // Interpolate in x for even rows.
            for (int j = 0; j < ny_f; j += 2){
                for (int i = 1; i < nx_f - 1; i += 2){
                    fine.valueAt(i, j) = static_cast<real>(0.5) *
                                         (fine.valueAt(i - 1, j) +
                                          fine.valueAt(i + 1, j));
                }
            }

            // Interpolate in y for all columns.
            for (int j = 1; j < ny_f - 1; j += 2){
                for (int i = 0; i < nx_f; ++i){
                    fine.valueAt(i, j) = static_cast<real>(0.5) *
                                         (fine.valueAt(i, j - 1) +
                                          fine.valueAt(i, j + 1));
                }
            }
        }

        static void vcycle_2d(_2sf& phi,
                              const _2sf& rhs,
                              real h){
            const int nx = phi.n_x;
            const int ny = phi.n_y;

            if (nx <= 3 || ny <= 3){
                smooth_rb_jacobi_2d(phi, rhs, h, 30);
                return;
            }

            // Pre-smoothing.
            smooth_rb_jacobi_2d(phi, rhs, h, 3);

            // Residual.
            _2sf res;
            compute_residual_2d(phi, rhs, res, h);

            // Restrict residual to coarse grid.
            _2sf rhs_c;
            restrict_full_weighting_2d(res, rhs_c);

            // Coarse-grid correction.
            _2sf e_c;
            e_c.n_x = rhs_c.n_x;
            e_c.n_y = rhs_c.n_y;
            e_c.data.assign(static_cast<std::size_t>(e_c.n_x) * e_c.n_y,
                            real{0});

            vcycle_2d(e_c, rhs_c, static_cast<real>(2.0) * h);

            // Prolongate and add correction.
            _2sf e_f;
            e_f.n_x = nx;
            e_f.n_y = ny;
            e_f.data.assign(static_cast<std::size_t>(nx) * ny, real{0});
            prolong_bilinear_2d(e_c, e_f);

            for (int j = 1; j < ny - 1; ++j){
                for (int i = 1; i < nx - 1; ++i){
                    phi.valueAt(i, j) += e_f.valueAt(i, j);
                }
            }

            // Post-smoothing.
            smooth_rb_jacobi_2d(phi, rhs, h, 3);
        }

        static void apply_standard_laplacian_2d(const _2sf& phi,
                                                _2sf& out,
                                                real h){
            const int nx = phi.n_x;
            const int ny = phi.n_y;
            const real inv_h2 = static_cast<real>(1.0) / (h * h);

            out.n_x = nx;
            out.n_y = ny;
            out.data.assign(static_cast<std::size_t>(nx) * ny, real{0});

            for (int j = 1; j < ny - 1; ++j){
                for (int i = 1; i < nx - 1; ++i){
                    out.valueAt(i, j) =
                        (phi.valueAt(i + 1, j) + phi.valueAt(i - 1, j) +
                         phi.valueAt(i, j + 1) + phi.valueAt(i, j - 1) -
                         static_cast<real>(4.0) * phi.valueAt(i, j)) * inv_h2;
                }
            }
        }

        static void solve_poisson_standard_2d_warm(const _2Mesh& mesh,
                                                   const _2sf& rhs,
                                                   _2sf& phi_io){
            const int nx = rhs.n_x;
            const int ny = rhs.n_y;
            const real h = mesh.dx();

            if (phi_io.n_x != nx || phi_io.n_y != ny ||
                phi_io.data.size() != static_cast<std::size_t>(nx) * ny){
                phi_io.n_x = nx;
                phi_io.n_y = ny;
                phi_io.data.assign(static_cast<std::size_t>(nx) * ny, real{0});
            }

            _2sf r, p, Ap;
            r.n_x = p.n_x = Ap.n_x = nx;
            r.n_y = p.n_y = Ap.n_y = ny;
            r.data.assign(static_cast<std::size_t>(nx) * ny, real{0});

            apply_standard_laplacian_2d(phi_io, Ap, h);

            for (int j = 1; j < ny - 1; ++j){
                for (int i = 1; i < nx - 1; ++i){
                    r.valueAt(i, j) = rhs.valueAt(i, j) - Ap.valueAt(i, j);
                }
            }

            p = r;

            auto dot = [&](const _2sf& a, const _2sf& b) -> real {
                real sum = real{0};
                for (int j = 1; j < ny - 1; ++j){
                    for (int i = 1; i < nx - 1; ++i){
                        sum += a.valueAt(i, j) * b.valueAt(i, j);
                    }
                }
                return sum;
            };

            real rr = dot(r, r);
            if (rr == real{0}){
                return;
            }

            const real tol_rel = static_cast<real>(1e-6);
            const real tol2 = tol_rel * tol_rel * rr;
            const int max_iter = 2000;

            for (int it = 0; it < max_iter; ++it){
                apply_standard_laplacian_2d(p, Ap, h);

                const real pAp = dot(p, Ap);
                if (std::abs(pAp) < static_cast<real>(1e-30)){
                    break;
                }

                const real alpha = rr / pAp;
                for (int j = 1; j < ny - 1; ++j){
                    for (int i = 1; i < nx - 1; ++i){
                        phi_io.valueAt(i, j) += alpha * p.valueAt(i, j);
                        r.valueAt(i, j) -= alpha * Ap.valueAt(i, j);
                    }
                }

                const real rr_new = dot(r, r);
                if (rr_new < tol2){
                    break;
                }

                const real beta = rr_new / rr;
                rr = rr_new;
                for (int j = 1; j < ny - 1; ++j){
                    for (int i = 1; i < nx - 1; ++i){
                        p.valueAt(i, j) = r.valueAt(i, j) + beta * p.valueAt(i, j);
                    }
                }
            }
        }
    } // namespace

    // Solve nodal Poisson with multigrid V-cycles: Lap(phi) = rhs.
    void solve_poisson_nodes_multigrid_2d(const _2Mesh& mesh,
                                          const _2sf& rhs_node,
                                          _2sf& phi_node){
        const int nx = rhs_node.n_x;
        const int ny = rhs_node.n_y;
        const real h = mesh.dx();

        phi_node.n_x = nx;
        phi_node.n_y = ny;
        if (phi_node.data.size() != static_cast<std::size_t>(nx) * ny){
            phi_node.data.assign(static_cast<std::size_t>(nx) * ny, real{0});
        }

        auto l2_norm = [&](const _2sf& field) -> real {
            real sum = real{0};
            for (int j = 1; j < ny - 1; ++j){
                for (int i = 1; i < nx - 1; ++i){
                    const real value = field.valueAt(i, j);
                    sum += value * value;
                }
            }
            return std::sqrt(sum);
        };

        _2sf res0;
        compute_residual_2d(phi_node, rhs_node, res0, h);
        const real rhs_norm = std::max(l2_norm(rhs_node), static_cast<real>(1e-30));
        real res_norm = l2_norm(res0);

        const real tol_rel = static_cast<real>(1e-6);
        const int max_cycles = 20;

        int k = 0;
        while (k < max_cycles && res_norm > tol_rel * rhs_norm){
            vcycle_2d(phi_node, rhs_node, h);
            ++k;

            _2sf res;
            compute_residual_2d(phi_node, rhs_node, res, h);
            res_norm = l2_norm(res);
        }
    }
#endif

    static void compute_gauss_correction_phi_2d(const _2FieldManager& fm,
                                                _2sf& phi_corr){
        const _2Mesh& mesh = fm.mesh();

        const int nx_cells = mesh.nx_cells();
        const int ny_cells = mesh.ny_cells();

        const auto& Ex = fm.Ex(); // (nx+1, ny)
        const auto& Ey = fm.Ey(); // (nx, ny+1)
        const auto& rho_cell = fm.rho(); // cell-centred rho

        const real dx = mesh.dx();
        const real dy = mesh.dy();
        const real eps0 =
            static_cast<real>(constants::epsilon_0);

        const real inv_eps0 = static_cast<real>(1.0) / eps0;

        // Yee divergence naturally lives on cells, so perform the correction on
        // the cell-centred grid and project the resulting scalar potential back
        // to Yee faces. This keeps the discretisation consistent with the FDTD
        // update and avoids the unstable nodal reconstruction.
        _2sf g_cell;
        g_cell.n_x = nx_cells;
        g_cell.n_y = ny_cells;
        g_cell.data.assign(static_cast<std::size_t>(nx_cells) * ny_cells, real{0});

        for (int j = 0; j < ny_cells; ++j){
            for (int i = 0; i < nx_cells; ++i){
                const real dEx_dx = (Ex.valueAt(i + 1, j) - Ex.valueAt(i, j)) / dx;
                const real dEy_dy = (Ey.valueAt(i, j + 1) - Ey.valueAt(i, j)) / dy;
                g_cell.valueAt(i, j) = dEx_dx + dEy_dy - rho_cell.valueAt(i, j) * inv_eps0;
            }
        }

        solve_poisson_standard_2d_warm(mesh, g_cell, phi_corr);
    }

    static void apply_gauss_correction_yee_2d(_2FieldManager& fm,
                                              const _2sf& phi_corr){
        const _2Mesh& mesh = fm.mesh();
        const int nx_cells = mesh.nx_cells();
        const int ny_cells = mesh.ny_cells();

        auto& Ex = fm.Ex(); // (nx+1, ny)
        auto& Ey = fm.Ey(); // (nx, ny+1)

        const real dx = mesh.dx();
        const real dy = mesh.dy();

        // Add E_phi = -grad(phi_corr) to the Yee fields.
        for (int j = 0; j < ny_cells; ++j){
            for (int i = 0; i < nx_cells + 1; ++i){
                real corr = real{0};
                if (i == 0){
                    corr = -phi_corr.valueAt(0, j) / dx;
                } else if (i == nx_cells){
                    corr = phi_corr.valueAt(nx_cells - 1, j) / dx;
                } else {
                    corr = -(phi_corr.valueAt(i, j) - phi_corr.valueAt(i - 1, j)) / dx;
                }
                Ex.valueAt(i, j) += corr;
            }
        }

        for (int j = 0; j < ny_cells + 1; ++j){
            for (int i = 0; i < nx_cells; ++i){
                real corr = real{0};
                if (j == 0){
                    corr = -phi_corr.valueAt(i, 0) / dy;
                } else if (j == ny_cells){
                    corr = phi_corr.valueAt(i, ny_cells - 1) / dy;
                } else {
                    corr = -(phi_corr.valueAt(i, j) - phi_corr.valueAt(i, j - 1)) / dy;
                }
                Ey.valueAt(i, j) += corr;
            }
        }
    }

    // ---------------------------------------------------------------------
    // 2D Gauss-law projection for Yee fields using Poisson solve.
    //
    // Computes the divergence error g = div(E) - rho/eps0 (mapped to nodes),
    // solves Lap(phi_corr) = g, and corrects Ex,Ey so that div(E) ≈ rho/eps0.
    // ---------------------------------------------------------------------
    void enforce_gauss_law_yee_2d(_2FieldManager& fm){
        static _2sf phi_corr;
        compute_gauss_correction_phi_2d(fm, phi_corr);
        apply_gauss_correction_yee_2d(fm, phi_corr);
    }

    // ---------------------------------------------------------------------
    // 3D nodal-Yee Poisson: phi on mesh nodes, E on Yee faces.
    // ---------------------------------------------------------------------

    // Build a nodal charge density by averaging surrounding cells (3D).
    static void build_nodal_rho_3d(const _3Mesh& mesh,
                                   const _3sf& rho_cell,
                                   _3sf& rho_node){
        const int nx_cells = mesh.nx_cells();
        const int ny_cells = mesh.ny_cells();
        const int nz_cells = mesh.nz_cells();

        const int nx_nodes = mesh.nx_nodes(); // nx_cells + 1
        const int ny_nodes = mesh.ny_nodes(); // ny_cells + 1
        const int nz_nodes = mesh.nz_nodes(); // nz_cells + 1

        rho_node.n_x = nx_nodes;
        rho_node.n_y = ny_nodes;
        rho_node.n_z = nz_nodes;
        rho_node.data.assign(static_cast<std::size_t>(nx_nodes)
                             * ny_nodes * nz_nodes, real{0});

        // Interior nodes: average the eight surrounding cell-centred rho's.
        for (int k = 1; k < nz_cells; ++k) {
            for (int j = 1; j < ny_cells; ++j) {
                for (int i = 1; i < nx_cells; ++i) {
                    real sum = real{0};
                    for (int kk = k - 1; kk <= k; ++kk) {
                        for (int jj = j - 1; jj <= j; ++jj) {
                            for (int ii = i - 1; ii <= i; ++ii) {
                                sum += rho_cell.valueAt(ii, jj, kk);
                            }
                        }
                    }
                    rho_node.valueAt(i, j, k) =
                        sum / static_cast<real>(8.0);
                }
            }
        }
    }

    // Apply standard 7-point Laplacian on the nodal grid in 3D.
    static void apply_nodal_laplacian_3d(const _3Mesh& mesh,
                                         const _3sf& phi,
                                         _3sf& out){
        const int nx_nodes = mesh.nx_nodes();
        const int ny_nodes = mesh.ny_nodes();
        const int nz_nodes = mesh.nz_nodes();

        const real h  = mesh.dx(); // assuming uniform spacing
        const real h2 = h * h;

        out.n_x = nx_nodes;
        out.n_y = ny_nodes;
        out.n_z = nz_nodes;
        out.data.assign(static_cast<std::size_t>(nx_nodes)
                        * ny_nodes * nz_nodes, real{0});

        for (int k = 1; k < nz_nodes - 1; ++k) {
            for (int j = 1; j < ny_nodes - 1; ++j) {
                for (int i = 1; i < nx_nodes - 1; ++i) {
                    const real phi_ip = phi.valueAt(i + 1, j,     k);
                    const real phi_im = phi.valueAt(i - 1, j,     k);
                    const real phi_jp = phi.valueAt(i,     j + 1, k);
                    const real phi_jm = phi.valueAt(i,     j - 1, k);
                    const real phi_kp = phi.valueAt(i,     j,     k + 1);
                    const real phi_km = phi.valueAt(i,     j,     k - 1);
                    const real phi_c  = phi.valueAt(i,     j,     k);

                    const real lap =
                        (phi_ip + phi_im + phi_jp + phi_jm + phi_kp + phi_km
                         - static_cast<real>(6.0) * phi_c) / h2;

                    out.valueAt(i, j, k) = lap;
                }
            }
        }
    }

    // Solve nodal Poisson in 3D: Lap(phi_node) = -rho_node/eps0 with phi=0 on boundary.
    void solve_poisson_phi_nodes_3d(const _3Mesh& mesh,
                                    const _3sf& rho_cell,
                                    _3sf& phi_node_out){
        const int nx_nodes = mesh.nx_nodes();
        const int ny_nodes = mesh.ny_nodes();
        const int nz_nodes = mesh.nz_nodes();

        const real inv_eps0 =
            static_cast<real>(1.0) / static_cast<real>(constants::epsilon_0);

        _3sf rho_node;
        build_nodal_rho_3d(mesh, rho_cell, rho_node);

        _3sf phi, r, p, Ap, b;
        phi.n_x = r.n_x = p.n_x = Ap.n_x = b.n_x = nx_nodes;
        phi.n_y = r.n_y = p.n_y = Ap.n_y = b.n_y = ny_nodes;
        phi.n_z = r.n_z = p.n_z = Ap.n_z = b.n_z = nz_nodes;

        const std::size_t N_nodes =
            static_cast<std::size_t>(nx_nodes) * ny_nodes * nz_nodes;
        phi.data.assign(N_nodes, real{0});
        b.data.assign(N_nodes,  real{0});

        for (int k = 1; k < nz_nodes - 1; ++k) {
            for (int j = 1; j < ny_nodes - 1; ++j) {
                for (int i = 1; i < nx_nodes - 1; ++i) {
                    b.valueAt(i, j, k) =
                        -rho_node.valueAt(i, j, k) * inv_eps0;
                }
            }
        }

        r = b;
        p = r;

        auto dot = [&](const _3sf& a, const _3sf& c) -> real {
            real s = 0;
            for (int k = 1; k < nz_nodes - 1; ++k) {
                for (int j = 1; j < ny_nodes - 1; ++j) {
                    for (int i = 1; i < nx_nodes - 1; ++i) {
                        s += a.valueAt(i, j, k) * c.valueAt(i, j, k);
                    }
                }
            }
            return s;
        };

        real rr = dot(r, r);
        if (rr == real{0}) {
            phi_node_out = phi;
            return;
        }

        const real tol_rel = static_cast<real>(1e-5);
        const real tol2    = tol_rel * tol_rel * rr;
        const int  max_iter = 2000;

        for (int it = 0; it < max_iter; ++it) {
            apply_nodal_laplacian_3d(mesh, p, Ap);

            const real pAp = dot(p, Ap);
            if (std::abs(pAp) < static_cast<real>(1e-30)) break;

            const real alpha = rr / pAp;

            for (int k = 1; k < nz_nodes - 1; ++k) {
                for (int j = 1; j < ny_nodes - 1; ++j) {
                    for (int i = 1; i < nx_nodes - 1; ++i) {
                        phi.valueAt(i, j, k) += alpha * p.valueAt(i, j, k);
                        r.valueAt(i, j, k)   -= alpha * Ap.valueAt(i, j, k);
                    }
                }
            }

            const real rr_new = dot(r, r);
            if (rr_new < tol2) break;

            const real beta = rr_new / rr;
            rr = rr_new;

            for (int k = 1; k < nz_nodes - 1; ++k) {
                for (int j = 1; j < ny_nodes - 1; ++j) {
                    for (int i = 1; i < nx_nodes - 1; ++i) {
                        p.valueAt(i, j, k) =
                            r.valueAt(i, j, k) + beta * p.valueAt(i, j, k);
                    }
                }
            }
        }

        phi_node_out = phi;
    }

    // Given nodal phi, fill Yee Ex/Ey/Ez in 3D _3FieldManager.
    void set_yee_E_from_phi_nodes_3d(_3FieldManager& fm,
                                     const _3sf& phi_node){
        const auto& mesh = fm.mesh();
        const int nx_cells = mesh.nx_cells();
        const int ny_cells = mesh.ny_cells();
        const int nz_cells = mesh.nz_cells();

        auto& Ex = fm.Ex(); // (nx+1, ny,   nz  )
        auto& Ey = fm.Ey(); // (nx,   ny+1, nz  )
        auto& Ez = fm.Ez(); // (nx,   ny,   nz+1)

        const real dx = mesh.dx();
        const real dy = mesh.dy();
        const real dz = mesh.dz();

        // Ex on x-faces: centred in x using nodal planes, averaged over
        // the four nodes of each yz face. Boundary faces use one-sided
        // differences.
        for (int k = 0; k < nz_cells; ++k) {
            for (int j = 0; j < ny_cells; ++j) {
                for (int i = 0; i < nx_cells + 1; ++i) {
                    real phi_L, phi_R;

                    if (i > 0 && i < nx_cells){
                        // planes at i-1 and i+1
                        real sum_L = real{0};
                        real sum_R = real{0};
                        for (int jj = j; jj <= j + 1; ++jj) {
                            for (int kk = k; kk <= k + 1; ++kk) {
                                sum_L += phi_node.valueAt(i - 1, jj, kk);
                                sum_R += phi_node.valueAt(i + 1, jj, kk);
                            }
                        }
                        phi_L = sum_L / static_cast<real>(4.0);
                        phi_R = sum_R / static_cast<real>(4.0);

                        Ex.valueAt(i, j, k) =
                            -(phi_R - phi_L) / (static_cast<real>(2.0) * dx);
                    } else if (i == 0){
                        // forward difference at left boundary: planes at 0 and 1
                        real sum_L = real{0};
                        real sum_R = real{0};
                        for (int jj = j; jj <= j + 1; ++jj) {
                            for (int kk = k; kk <= k + 1; ++kk) {
                                sum_L += phi_node.valueAt(0, jj, kk);
                                sum_R += phi_node.valueAt(1, jj, kk);
                            }
                        }
                        phi_L = sum_L / static_cast<real>(4.0);
                        phi_R = sum_R / static_cast<real>(4.0);

                        Ex.valueAt(i, j, k) = -(phi_R - phi_L) / dx;
                    } else { // i == nx_cells
                        // backward difference at right boundary: planes at nx_cells-1 and nx_cells
                        real sum_L = real{0};
                        real sum_R = real{0};
                        for (int jj = j; jj <= j + 1; ++jj) {
                            for (int kk = k; kk <= k + 1; ++kk) {
                                sum_L += phi_node.valueAt(nx_cells - 1, jj, kk);
                                sum_R += phi_node.valueAt(nx_cells,     jj, kk);
                            }
                        }
                        phi_L = sum_L / static_cast<real>(4.0);
                        phi_R = sum_R / static_cast<real>(4.0);

                        Ex.valueAt(i, j, k) = -(phi_R - phi_L) / dx;
                    }
                }
            }
        }

        // Ey on y-faces: centred in y using nodal planes, averaged over
        // the four nodes of each xz face.
        for (int k = 0; k < nz_cells; ++k) {
            for (int j = 0; j < ny_cells + 1; ++j) {
                for (int i = 0; i < nx_cells; ++i) {
                    real phi_B, phi_T;

                    if (j > 0 && j < ny_cells){
                        // planes at j-1 and j+1
                        real sum_B = real{0};
                        real sum_T = real{0};
                        for (int ii = i; ii <= i + 1; ++ii) {
                            for (int kk = k; kk <= k + 1; ++kk) {
                                sum_B += phi_node.valueAt(ii, j - 1, kk);
                                sum_T += phi_node.valueAt(ii, j + 1, kk);
                            }
                        }
                        phi_B = sum_B / static_cast<real>(4.0);
                        phi_T = sum_T / static_cast<real>(4.0);

                        Ey.valueAt(i, j, k) =
                            -(phi_T - phi_B) / (static_cast<real>(2.0) * dy);
                    } else if (j == 0){
                        // forward difference at bottom boundary
                        real sum_B = real{0};
                        real sum_T = real{0};
                        for (int ii = i; ii <= i + 1; ++ii) {
                            for (int kk = k; kk <= k + 1; ++kk) {
                                sum_B += phi_node.valueAt(ii, 0, kk);
                                sum_T += phi_node.valueAt(ii, 1, kk);
                            }
                        }
                        phi_B = sum_B / static_cast<real>(4.0);
                        phi_T = sum_T / static_cast<real>(4.0);

                        Ey.valueAt(i, j, k) = -(phi_T - phi_B) / dy;
                    } else { // j == ny_cells
                        // backward difference at top boundary
                        real sum_B = real{0};
                        real sum_T = real{0};
                        for (int ii = i; ii <= i + 1; ++ii) {
                            for (int kk = k; kk <= k + 1; ++kk) {
                                sum_B += phi_node.valueAt(ii, ny_cells - 1, kk);
                                sum_T += phi_node.valueAt(ii, ny_cells,     kk);
                            }
                        }
                        phi_B = sum_B / static_cast<real>(4.0);
                        phi_T = sum_T / static_cast<real>(4.0);

                        Ey.valueAt(i, j, k) = -(phi_T - phi_B) / dy;
                    }
                }
            }
        }

        // Ez on z-faces: centred in z using nodal planes, averaged over
        // the four nodes of each xy face.
        for (int k = 0; k < nz_cells + 1; ++k) {
            for (int j = 0; j < ny_cells; ++j) {
                for (int i = 0; i < nx_cells; ++i) {
                    real phi_B, phi_T;

                    if (k > 0 && k < nz_cells){
                        // planes at k-1 and k+1
                        real sum_B = real{0};
                        real sum_T = real{0};
                        for (int ii = i; ii <= i + 1; ++ii) {
                            for (int jj = j; jj <= j + 1; ++jj) {
                                sum_B += phi_node.valueAt(ii, jj, k - 1);
                                sum_T += phi_node.valueAt(ii, jj, k + 1);
                            }
                        }
                        phi_B = sum_B / static_cast<real>(4.0);
                        phi_T = sum_T / static_cast<real>(4.0);

                        Ez.valueAt(i, j, k) =
                            -(phi_T - phi_B) / (static_cast<real>(2.0) * dz);
                    } else if (k == 0){
                        // forward difference at back boundary
                        real sum_B = real{0};
                        real sum_T = real{0};
                        for (int ii = i; ii <= i + 1; ++ii) {
                            for (int jj = j; jj <= j + 1; ++jj) {
                                sum_B += phi_node.valueAt(ii, jj, 0);
                                sum_T += phi_node.valueAt(ii, jj, 1);
                            }
                        }
                        phi_B = sum_B / static_cast<real>(4.0);
                        phi_T = sum_T / static_cast<real>(4.0);

                        Ez.valueAt(i, j, k) = -(phi_T - phi_B) / dz;
                    } else { // k == nz_cells
                        // backward difference at front boundary
                        real sum_B = real{0};
                        real sum_T = real{0};
                        for (int ii = i; ii <= i + 1; ++ii) {
                            for (int jj = j; jj <= j + 1; ++jj) {
                                sum_B += phi_node.valueAt(ii, jj, nz_cells - 1);
                                sum_T += phi_node.valueAt(ii, jj, nz_cells);
                            }
                        }
                        phi_B = sum_B / static_cast<real>(4.0);
                        phi_T = sum_T / static_cast<real>(4.0);

                        Ez.valueAt(i, j, k) = -(phi_T - phi_B) / dz;
                    }
                }
            }
        }
    }

    // High-level 3D Yee Poisson: solve nodal phi and fill Ex/Ey/Ez on the Yee grid.
    void poisson_solve_yee(_3FieldManager& fm){
        const _3Mesh& mesh = fm.mesh();
        const _3sf&   rho  = fm.rho();

        _3sf phi_nodes;
        solve_poisson_phi_nodes_3d(mesh, rho, phi_nodes);
        set_yee_E_from_phi_nodes_3d(fm, phi_nodes);
    }
}
