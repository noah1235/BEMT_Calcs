# Blade Element Momentum Theory (BEMT) Fan Performance Calculator

This Python program computes and optimizes axial-fan performance using Blade Element Momentum Theory (BEMT). BEMT assumes the flow varies only in the radial direction,
calculates thrust and tangential forces at each radial location, and integrates the results to estimate fan performance.

The solver:
- Takes in airfoil polars to evaluate lift and drag as functions of Reynolds number and angle of attack at each radial station.
- Uses conservation of momentum to estimate axial and tangential induction factors.
- Analyzes fans parameterized by twist-angle and chord-length profiles.
- Includes an optimization loop that tunes twist, chord distribution, and RPM for a specified operating point.
