import numpy as np
from constants import kin_vis, rho
from scipy.optimize import brentq
from SRC.Geometry.Blade_Geometry import Blade_Geometry
import pandas as pd
from SRC.unit_conversion import CFM_2_m3_s
from SRC.Calculations.Airfoil_Data import Airfoil_Data


def calc_Re(u: float, c: float) -> float:
    """
    Reynolds number: Re = u * c / ν
    Uses kin_vis (kinematic viscosity) from external module.
    """
    return (u * c) / kin_vis

def calc_a(phi: float, sigma_prime: float, cn: float, F: float = 1) -> float:
    """
    Induction factor 'a' (kept exactly as given).

    a = 1 / ( (4 F sin^2(phi)) / (sigma' * c_n) - 1 )
    """
    t1 = (4 * F * np.sin(phi)**2) / (sigma_prime * cn)
    return 1 / (t1 - 1)

def calc_a_prime(phi: float, sigma_prime: float, ct: float, F: float = 1) -> float:
    """
    Tangential induction factor 'a_prime' (kept exactly as given).

    a' = 1 / ( (4 F sin(phi) cos(phi)) / (sigma' * c_t) + 1 )
    """
    t1 = (4 * F * np.sin(phi) * np.cos(phi)) / (sigma_prime * ct)
    return 1 / (t1 + 1)

def calc_delta_p(T: float, hub_diameter: float, od: float) -> float:
    """
    Average pressure rise from total thrust T over an annular area proxy (kept as-is).
    """
    r1 = hub_diameter / 2
    r2 = od / 2
    A = np.pi * (r2**2 - r1**2)  # kept as written
    return T / A

def radial_integration(blade_geo: Blade_Geometry, airfoil_data: Airfoil_Data):
    """
    Integrate thrust and torque per unit span over r in blade_geo.r_vals.

    Returns
    -------
    avg_delta_p : float
        Average pressure rise (per calc_delta_p).
    power_loss : float
        Q * omega (mechanical power).
    airfoil_perf_data_list : list
        Per-station performance records from get_force_per_unit_length.
    T_prime_vals, Q_prime_vals : np.ndarray
        Distributions vs r.
    r_vals : np.ndarray
        The corresponding radii used for integration.
    """
    T_prime_vals = []
    p_prime_vals = []
    Q_prime_vals = []
    vd_list = []
    airfoil_perf_data_list = []

    # Loop over radial stations (kept in the same order)
    for r in blade_geo.r_vals:
        T_prime, Q_prime, v_d, a, airfoil_perf_data = get_force_per_unit_length(r, blade_geo, airfoil_data)
        airfoil_perf_data_list.append(airfoil_perf_data)
        T_prime_vals.append(float(T_prime))
        Q_prime_vals.append(float(Q_prime))
        p_prime_vals.append(float(v_d*T_prime))
        vd_list.append(float(a))

    T_prime_vals = np.array(T_prime_vals)
    Q_prime_vals = np.array(Q_prime_vals)
    vd_list = np.array(vd_list)

    T = np.trapezoid(T_prime_vals, blade_geo.r_vals)
    Q = np.trapezoid(Q_prime_vals, blade_geo.r_vals)
    flow_power = np.trapezoid(p_prime_vals, blade_geo.r_vals)

    avg_delta_p = calc_delta_p(T, blade_geo.hub_diameter, blade_geo.od)
    fan_power = Q * blade_geo.omega
    efficiency = (flow_power)/fan_power

    return avg_delta_p, fan_power, efficiency, airfoil_perf_data_list, T_prime_vals, vd_list, blade_geo.r_vals

def get_force_per_unit_length(r: float, blade_geo: Blade_Geometry, airfoil_data: Airfoil_Data):
    """
    Compute per-span thrust and torque contributions at radius r, then package
    a small diagnostics vector for later inspection.

    Kept as-is:
    - Uses w_approx for Re estimation, then recomputes w with (1+a) and (1-a') later.
    - Solves for phi via root-finding of R(phi) in a fixed bracket (1e-6, π/2 - 1e-6).
    """
    # Approximate relative speed for Re (before induction factors)
    w_approx = np.sqrt((blade_geo.omega * r)**2 + blade_geo.v_freestream**2)

    # Local chord and twist
    c = blade_geo.get_arc_chord(r)
    theta = blade_geo.theta_prof(r)

    # Reynolds number at the approximate relative speed
    Re = calc_Re(w_approx, c)

    def calc_parameters(phi: float):
        """
        Compute alpha, cl, cd, normal/tangential coeffs, solidity, and induction factors
        for a given inflow angle phi.
        """
        alpha = theta - phi  # radians

        cl, cd = airfoil_data(Re, np.rad2deg(alpha))  # airfoil table expects degrees

        # Force coefficients resolved along normal (cn) and tangential (ct) directions
        cn = cl * np.cos(phi) - cd * np.sin(phi)
        ct = cl * np.sin(phi) + cd * np.cos(phi)


        # Local solidity
        sigma_prime = (blade_geo.B * c) / (2 * np.pi * r)

        # Induction factors (kept exactly)
        a = calc_a(phi, sigma_prime, cn, F=1)
        a_prime = calc_a_prime(phi, sigma_prime, ct, F=1)

        return alpha, cd, cl, cn, ct, sigma_prime, a, a_prime

    def R(phi: float) -> float:
        """
        Residual for brentq root solve (kept exactly).
        """
        alpha, cd, cl, cn, ct, sigma_prime, a, a_prime = calc_parameters(phi)
        return (np.sin(phi) / (1 + a)) - (blade_geo.v_freestream / (blade_geo.omega * r)) * (np.cos(phi) / (1 - a_prime))

    # Root bracket (open interval near 0 to near π/2)
    phi_lower = 1e-6
    phi_upper = np.pi / 2 - 1e-6

    # Ensure a sign change in the bracket (required by brentq)
    R_low = R(phi_lower)
    R_high = R(phi_upper)
    if R_low * R_high > 0:
        raise ValueError(
            f"R(phi) does not change sign between {phi_lower} and {phi_upper}. "
            f"R_low={R_low}, R_high={R_high}. Cannot use brentq."
        )

    # Solve for inflow angle
    phi_star = brentq(R, phi_lower, phi_upper, xtol=1e-8, rtol=1e-8, maxiter=1000)

    # Recompute parameters at solution
    alpha, cd, cl, cn, ct, sigma_prime, a, a_prime = calc_parameters(phi_star)
    vd = blade_geo.v_freestream * (1 + a)
    #print(a_prime)
    # Relative speed including induction factors (kept exactly as provided)
    w = np.sqrt(vd**2 + (blade_geo.omega * r * (1 - a_prime))**2)

    # Per-span thrust and torque (multiplied by number of blades B)
    T_prime = blade_geo.B * cn * 0.5 * rho * w**2 * c
    Q_prime = blade_geo.B * (ct * 0.5 * rho * w**2 * c) * r

    # Diagnostics vector (kept as the same ordering/content)
    return T_prime, Q_prime, vd, a, [
        np.rad2deg(alpha),  # alpha in degrees
        cd,
        cl,
        cn,
        ct,
        phi_star,
        float(calc_Re(w, c)),
        c,
    ]
