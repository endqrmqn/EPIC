                                                      _..._     
                                                   .-'_..._''.  
       __.....__   _________   _...._      .--.  .' .'      '.\ 
   .-''         '. \        |.'      '-.   |__| / .'            
  /     .-''"'-.  `.\        .'```'.    '. .--.. '              
 /     /________\   \\      |       \     \|  || |              
 |                  | |     |        |    ||  || |              
 \    .-------------' |      \      /    . |  |. '              
  \    '-.____...---. |     |\`'-.-'   .'  |  | \ '.          . 
   `.             .'  |     | '-....-'`    |__|  '. `._____.-'/ 
     `''-...... -'   .'     '.                     `-.______ /  
                   '-----------'                            `   

            Electromagnetic Particle-in-Cell (PIC) Solver

# Electromagnetic Particle-in-Cell (PIC) Solver

A lightweight, modular, multispecies electromagnetic PIC library written in C/C++.
Designed for research use in plasma physics, electric propulsion, nonlinear wave–particle
interactions, and general kinetic simulations.

This project is built to be **readable first**, **correct second**, and **fast third**.
Once the physics is solid, the architecture supports optimization (OpenMP, CUDA, AMReX-style
tiling, etc.) without rewriting the entire codebase.

---

## Features (Current & Planned)

### Core
- Multispecies PIC framework (electrons, ions, optional neutrals)
- Charge + current deposition (Esirkepov scheme planned)
- Field interpolation (CIC → TSC)
- Particle pushers (Boris, Vay planned)
- Boundary conditions (periodic → absorbing → conducting walls)
- Diagnostics system (phase-space dumps, energy tracking, spectra)

### Electromagnetic Solvers
- 3D Yee-grid FDTD Maxwell solver  
- Charge-conserving update  
- Field staggering consistent with standard PIC formulations  

### Physics Modules (Planned)
- Collisions / Monte-Carlo collision models
- Ionization, recombination
- Hybrid PIC–fluid modes
- External E/B field drivers (traveling waves, coils, RF antennas)
- Propulsion-specific modules (thruster geometries, sheath models)

---

## Why This Exists

Most PIC codes fall into one of two categories:
1. **industrial monoliths** (huge, hard to modify)  
2. **student codes** (toy-level, not extensible)  

This project aims for the middle ground:
- easy to read  
- easy to extend  
- physically correct  
- structured like a real research codebase  

The goal is to support high-level projects in plasma physics and electric propulsion
(SToPIT, high-Isp staging architectures, harmonic-resolvent PIC coupling, etc.).

---

