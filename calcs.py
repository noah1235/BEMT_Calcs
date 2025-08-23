import numpy as np
from cosntants import kin_vis, rho
from scipy.optimize import brentq
from dataclasses import dataclass
from geometry import Blade_Geometry
import pandas as pd

class Airfoil_Data:
    def __init__(self, data: pd.DataFrame, Ncrit: float):
        """
        data: DataFrame with columns ['Re', 'Ncrit', 'alpha', 'CL', 'CD']
        Ncrit: which Ncrit slice to use
        """
        # Filter to only the requested Ncrit
        df = data[data["Ncrit"] == Ncrit].copy()
        if df.empty:
            raise ValueError(f"No data for Ncrit={Ncrit}")
        
        # Unique, sorted Re values
        self.Res = np.sort(df["Re"].unique())
        # Keep only the filtered subset for later lookups
        self.data = df

    def __call__(self, Re: float, alpha: float):
        # --- 1) Find bracketing Reynolds values ---
        if Re <= self.Res[0]:
            Re_lo = Re_hi = self.Res[0]
        elif Re >= self.Res[-1]:
            Re_lo = Re_hi = self.Res[-1]
        else:
            idx = np.searchsorted(self.Res, Re)
            Re_lo, Re_hi = self.Res[idx-1], self.Res[idx]
        
        Re_lo = float(Re_lo)
        Re_hi = float(Re_hi)


        # --- 2) For each bracketed Re, interpolate in alpha ---
        def interp_at_Re(R):
            subset = self.data[self.data["Re"] == R]
            a_arr = subset["alpha"].to_numpy()
            cl_arr = subset["CL"].to_numpy()
            cd_arr = subset["CD"].to_numpy()

            cl_val = np.interp(alpha, a_arr, cl_arr)
            cd_val = np.interp(alpha, a_arr, cd_arr)
        
            return cl_val, cd_val

        cl_lo, cd_lo = interp_at_Re(Re_lo)
        cl_hi, cd_hi = interp_at_Re(Re_hi)

        # --- 3) Linearly interpolate in Re ---
        if Re_lo == Re_hi:
            return float(cl_lo), float(cd_lo)
        
        t = (Re - Re_lo) / (Re_hi - Re_lo)
        cl = cl_lo + t * (cl_hi - cl_lo)
        cd = cd_lo + t * (cd_hi - cd_lo)
        return float(cl), float(cd)
        
    
def calc_Re(u, c):
    return (u*c)/kin_vis

def calc_a(phi, sigma_prime, cn, F=1):
    t1 = (4*F*np.sin(phi)**2)/(sigma_prime * cn)

    return 1/(t1 - 1)

def calc_a_prime(phi, sigma_prime, ct, F=1):
    t1 = (4*F*np.sin(phi)*np.cos(phi))/(sigma_prime * ct)

    return 1/(t1 + 1)

def calc_delta_p(T, hub_diameter, od):
    r1 = hub_diameter/2
    r2 = od/2
    A = np.pi * (r2 -r1)**2

    return T/A

def radial_integration(blade_geo: Blade_Geometry, airfoil_data):
    """
    Integrate T and Q between r1 and r2.
    Returns (total_T, total_Q).
    """

    T_prime_vals = []
    Q_prime_vals = []

    airfoil_perf_data_list = []

    for r in blade_geo.r_vals:
        T_prime, Q_prime, airfoil_perf_data = get_force_per_unit_length(
            r, blade_geo, airfoil_data
        )
        airfoil_perf_data_list.append(airfoil_perf_data)

        T_prime_vals.append(float(T_prime))
        Q_prime_vals.append(float(Q_prime))

    T_prime_vals = np.array(T_prime_vals)
    Q_prime_vals = np.array(Q_prime_vals)

    # Integrate using trapezoidal rule
    T = np.trapezoid(T_prime_vals, blade_geo.r_vals)
    Q = np.trapezoid(Q_prime_vals, blade_geo.r_vals)

    avg_delta_p = calc_delta_p(T, blade_geo.hub_diameter, blade_geo.od)
    power_loss = Q * blade_geo.omega

    return avg_delta_p, power_loss, airfoil_perf_data_list, T_prime_vals, Q_prime_vals, blade_geo.r_vals

def get_force_per_unit_length(r, blade_geo: Blade_Geometry, airfoil_data: Airfoil_Data):
    w_approx = np.sqrt((blade_geo.omega * r)**2 + blade_geo.v_freestream**2)
    c = blade_geo.get_arc_choord(r)
    theta = blade_geo.theta_prof(r)
    Re = calc_Re(w_approx, c)


    def calc_parameters(phi):
        alpha = theta - phi
        cl, cd = airfoil_data(Re, np.rad2deg(alpha))
        cn = cl * np.cos(phi) - cd * np.sin(phi)
        ct = cl*np.sin(phi) + cd * np.cos(phi)
        sigma_prime = (blade_geo.B*c)/(2*np.pi*r)
        a = calc_a(phi, sigma_prime, cn, F=1)
        a_prime = calc_a_prime(phi, sigma_prime, ct, F=1)

        return alpha, cd, cl, cn, ct, sigma_prime, a, a_prime

    def R(phi):
        alpha, cd, cl, cn, ct, sigma_prime, a, a_prime = calc_parameters(phi)

        return np.sin(phi)/(1+a) - (blade_geo.v_freestream/(blade_geo.omega * r)) * (np.cos(phi)/(1-a_prime))
    

    phi_lower = 1e-6
    phi_upper = np.pi / 2 - 1e-6

    # Check that the bracket actually brackets a root
    R_low = R(phi_lower)
    R_high = R(phi_upper)

    if R_low * R_high > 0:
        raise ValueError(
            f"R(phi) does not change sign between {phi_lower} and {phi_upper}. "
            f"R_low={R_low}, R_high={R_high}. Cannot use brentq."
        )

    # Solve
    phi_star = brentq(R, phi_lower, phi_upper, xtol=1e-8, rtol=1e-8, maxiter=1000)

    alpha, cd, cl, cn, ct, sigma_prime, a, a_prime = calc_parameters(phi_star)
    w = np.sqrt((blade_geo.v_freestream * (1+a))**2 + (blade_geo.omega*r*(1-a_prime))**2)

    T_prime = blade_geo.B * cn * .5 * rho * w**2 * c
    Q_prime = blade_geo.B * (ct * .5 * rho * w**2 * c) * r

    return T_prime, Q_prime, [np.rad2deg(alpha), cd, cl, cn, ct, phi_star, float(calc_Re(w, c)), c]
