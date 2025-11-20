Stage 0 — Project Skeleton
branch: skeleton
- folder structure
- minimal CMake
- README stub
------------------------------------------------------

Stage 1 — Math + Mesh Foundation
branch: math-grid
- math utilities (vectors, indexing, kernels)
- domain indexing
- Yee grid layout
- field storage containers
------------------------------------------------------

Stage 2 — Particle Container
branch: particles
- SoA particle storage
- species metadata
- simple initialization
- periodic boundary wrapping
------------------------------------------------------

Stage 3 — CIC Interpolation + Deposition
branch: cic-core
- CIC gather (field → particle)
- CIC scatter (particle → grid)
- charge/mass consistency checks
------------------------------------------------------

Stage 4 — Boris Pusher
branch: boris
- baseline Boris pusher
- free-stream validation
- E-only acceleration test
- uniform-B circular orbit test
------------------------------------------------------

Stage 5 — FDTD Maxwell Solver (Yee)
branch: fdtd
- curl(E), curl(B)
- periodic boundaries
- CFL condition checking
- 1D plane-wave propagation validation
------------------------------------------------------

Stage 6 — Minimal PIC Loop
branch: pic-minimal
Loop:
  1. deposit
  2. solve fields
  3. gather
  4. push
  5. move
  6. periodic bc
- run: Langmuir wave, two-stream instability
------------------------------------------------------

Stage 7 — Diagnostics Expansion
branch: diagnostics
- energy tracker
- field dumps
- particle/phase-space dumps
- plotting scripts
------------------------------------------------------

Stage 8 — Validation Suite
branch: validation-suite
- Langmuir frequency check
- two-stream growth rate
- Weibel instability
- long-time energy conservation
------------------------------------------------------

Stage 9 — Advanced Physics (pick as needed)
branches:
  absorbing-bc
  esirkepov
  relativistic-boris
  ionization
  collisions
  cylindrical-geometry
  traveling-wave-driver
------------------------------------------------------

Stage 10 — Parallelism (Optional)
branch: parallelism
- OpenMP acceleration
- CUDA backend (later)
------------------------------------------------------

Stage 11 — Propulsion Modules
branch: propulsion-driver
- traveling-wave fields
- ion ring acceleration
- injection/orbit tracking
------------------------------------------------------

Stage 12 - Stable I/O

- export to hdf5, vtk, and bin
- import files
- python compatability
------------------------------------------------------

Stage 13 — Publication Release
branch: paper-release
- stable APIs
- reproducible configs
- documentation pass
- tagged release
------------------------------------------------------
