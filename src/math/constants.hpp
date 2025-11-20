#pragma once
#include<cmath>

/**
 * @file constants.hpp
 *
 * math/constants.hpp
 * -------------------
 *
 * Core mathematical and physical constants used throughout EPIC.
 *
 * All constants in this header are provided as `constexpr` doubles for
 * guaranteed compile-time evaluation and zero runtime overhead.
 *
 * Purpose:
 *   - Provide a single authoritative source for numerical constants.
 *   - Avoid magic numbers scattered across the codebase.
 *   - Ensure consistency for physical models and numerical methods.
 *
 * Categories:
 *   - Mathematical constants        (π, e)
 *   - Electromagnetic constants     (ε₀, μ₀, c)
 *   - Fundamental physical values   (e, m_e, m_p)
 *   - Unit reference constants      (atomic mass unit, etc.)
 *
 * All values follow CODATA recommendations and are appropriate for
 * high-precision scientific computing.
 *
 * TODO:
 *   Nothing! This header is intentionally minimal and complete.
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
