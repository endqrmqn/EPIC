#pragma once
#include<cmath>

/*
    math/constants.hpp
    -------------------

    This header defines core mathematical and physical constants used
    throughout EPIC. All constants are provided as `constexpr` doubles,
    ensuring compile-time evaluation and zero runtime overhead.

    The constants here serve as a central reference for numerical
    algorithms, unit conversions, and physical models. They are kept
    minimal and self-contained: no functions, no heavy dependencies,
    and no state.

    Categories:
      - Mathematical constants   (π, e)
      - Electromagnetic constants (ε₀, μ₀, c)
      - Fundamental constants    (elementary charge, electron mass)
      - Unit reference constants (atomic mass unit)

    These values follow CODATA recommendations and are suitable for
    high-precision scientific computing.

    TODO:
        Nothing! Looks good!
*/

namespace math
{
    
    // Mathematical constants
    constexpr double PI = 3.14159265358979323846;
    constexpr double E  = 2.71828182845904523536;

    // Physical constants
    constexpr double K_B = 1.380649e-23; // in Joules per Kelvin
    constexpr double EPSILON_0 = 8.854187817e-12; // in Farads per meter
    constexpr double MU_0 = 4e-7 * PI; // in Newtons per Ampere squared

    constexpr double C = 299792458.0; // Speed of light in vacuum in m/s

    constexpr double Q_FUND = 1.602176634e-19; // Elementary charge in Coulombs
    constexpr double ME = 9.10938356e-31; // Electron mass in kg

    constexpr double AMU = 1.66053906660e-27; // Atomic mass unit in kg
} // namespace math
