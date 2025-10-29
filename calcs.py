import numpy as np
from cosntants import kin_vis, rho  # NOTE: module name appears misspelled; see notes below
from scipy.optimize import brentq
from dataclasses import dataclass  # imported but not used; retained to avoid behavior change
from geometry import Blade_Geometry
import pandas as pd
from unit_conversion import CFM_2_m3_s


class Airfoil_Data:
    """
    Airfoil lookup and (bilinear) interpolation wrapper.

    Expects a DataFrame with columns:
        ['Re', 'Ncrit', 'alpha', 'CL', 'CD']

    Interpolation logic (kept exactly):
    1) Choose the two Re grid points bracketing the query Re (clamp to ends).
    2) For each of those two Re slices, 1D interpolate CL, CD vs alpha using np.interp.
       (Assumes 'alpha' within each Re-slice is strictly increasing.)
    3) Linearly interpolate the two results across Re.
    """
    def __init__(self, data: pd.DataFrame, Ncrit: float):
        # Filter to the requested Ncrit (exact float equality as in the original)
        df = data[data["Ncrit"] == Ncrit].copy()
        if df.empty:
            raise ValueError(f"No data for Ncrit={Ncrit}")

        # Unique, sorted Reynolds numbers for bracketing
        self.Res = np.sort(df["Re"].unique())
        self.data = df  # store filtered table for later lookups

    def __call__(self, Re: float, alpha: float) -> tuple[float, float]:
        """
        Parameters
        ----------
        Re : float
            Reynolds number at which to query.
        alpha : float
            Angle of attack IN DEGREES (kept as-is; caller passes degrees).

        Returns
        -------
        (CL, CD) : tuple of floats
        """
        # --- 1) Bracket Re (clamped to [min, max]) ---
        if Re <= self.Res[0]:
            Re_lo = Re_hi = self.Res[0]
        elif Re >= self.Res[-1]:
            Re_lo = Re_hi = self.Res[-1]
        else:
            idx = np.searchsorted(self.Res, Re)
            Re_lo, Re_hi = self.Res[idx - 1], self.Res[idx]

        Re_lo = float(Re_lo)
        Re_hi = float(Re_hi)

        # --- 2) Interpolate within each bracket slice vs alpha ---
        def interp_at_Re(R: float) -> tuple[float, float]:
            subset = self.data[self.data["Re"] == R]
            a_arr = subset["alpha"].to_numpy()  # assumed strictly increasing
            cl_arr = subset["CL"].to_numpy()
            cd_arr = subset["CD"].to_numpy()

            # 1D linear interpolation in alpha
            cl_val = np.interp(alpha, a_arr, cl_arr)
            cd_val = np.interp(alpha, a_arr, cd_arr)
            return cl_val, cd_val

        cl_lo, cd_lo = interp_at_Re(Re_lo)
        cl_hi, cd_hi = interp_at_Re(Re_hi)

        # --- 3) Linear interpolation across Re ---
        if Re_lo == Re_hi:
            return float(cl_lo), float(cd_lo)

        t = (Re - Re_lo) / (Re_hi - Re_lo)
        cl = cl_lo + t * (cl_hi - cl_lo)
        cd = cd_lo + t * (cd_hi - cd_lo)
        return float(cl), float(cd)

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

    #print(flow_power, blade_geo.flow_rate * avg_delta_p)

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
    c = blade_geo.get_arc_choord(r)
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


def estimate_dp(CFM, n_fins=81, fin_thickness=2e-4, fan_d=0.2,
                dy_vis=18.6e-6, rho=1.164, L=0.1):
    """
    Plate-fin core assumed square: width=height=fan_d.
    Uses Dh = 2*t (t = clear spacing between plates), laminar f_D = 96/Re.
    Returns delta_p in Pascals.
    """
    Q = CFM_2_m3_s(CFM)
    W = fan_d
    H = fan_d
    open_width = W - n_fins * fin_thickness

    N_c = n_fins + 1
    t = open_width / N_c              # clear gap between plates
    Dh = 2.0 * t                      # per L >> t
    A_open = open_width * H
    U = Q / A_open                    # mean channel velocity
    Re = rho * U * Dh / dy_vis
    fD = 96.0 / Re                    # laminar Darcy friction factor
    delta_p = fD * (L / Dh) * (0.5 * rho * U**2)
    return float(delta_p)
    