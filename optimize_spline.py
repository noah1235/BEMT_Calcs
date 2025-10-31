import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from SRC.read_xfoil_data import load_all_polars
from SRC.Calculations.BEMT_Calcs import radial_integration
from SRC.Calculations.Airfoil_Data import Airfoil_Data
from SRC.Calculations.estimte_delta_p import estimate_dp
from SRC.Geometry.profiles import SplineProfile, Linear_Prof
from SRC.Geometry.Blade_Geometry import Blade_Geometry
# ----------------------------------------------------------------------------
# Thickness lower bound = thickness / thickness_lb_factor
# ----------------------------------------------------------------------------
thickness_lb_factor = 1.5

# ----------------------------------------------------------------------------
# Hard-coded final tip twist (radians)
# ----------------------------------------------------------------------------
FINAL_TIP_ANGLE = np.deg2rad(22)
# ----------------------------------------------------------------------------
# Scaling utilities
# ----------------------------------------------------------------------------
def to_unit(x, lb, ub):
    return (x - lb) / (ub - lb)

def from_unit(u, lb, ub):
    return lb + u * (ub - lb)
# ----------------------------------------------------------------------------
# Performance evaluation
# ----------------------------------------------------------------------------
def evaluate_performance(vars_raw, CFM, airfoil_data,
                         thickness, hub_d, od,
                         r_ctrl_twist, r_ctrl_t,
                         final_tip_angle, nblades):
    """
    vars_raw: concatenated vector
      [theta_ctrl[0..n_twist-2], thickness_ctrl[0..n_t-1], RPM]
    The final twist value at the tip is fixed to final_tip_angle.
    """
    n_twist = len(r_ctrl_twist)
    n_t = len(r_ctrl_t)
    assert vars_raw.shape[0] == (n_twist - 1) + n_t + 1, "vars_raw size mismatch."

    # unpack optimization vars
    i0 = 0
    i1 = i0 + (n_twist - 1)
    theta_raw = vars_raw[i0:i1]               # length n_twist - 1
    i2 = i1 + n_t
    t_ctrl = vars_raw[i1:i2]                  # length n_t

    RPM = float(vars_raw[-1])

    # append hard-coded tip twist
    theta_ctrl = np.concatenate([theta_raw, [final_tip_angle]])

    # build profiles (reuse spline class for thickness as well)
    twist_prof = SplineProfile(r_ctrl_twist, theta_ctrl,
                                    kind='pchip', extrapolate=True)
    t_prof = SplineProfile(r_ctrl_t, t_ctrl,
                                kind='pchip', extrapolate=True)

    blade = Blade_Geometry(
        airfoil_name="",
        Ncrit=9,
        B=nblades,
        max_t=thickness,           # max allowable thickness (cap)
        thickness_prof=t_prof,     # optimized thickness profile
        hub_diameter=hub_d,
        od=od,
        omega_rpm=RPM,
        theta_prof=twist_prof,
        CFM=CFM
    )

    dp, power, efficiency, *_ = radial_integration(blade, airfoil_data)
    print(f"Δp={dp:.2f} Pa | Power={power:.2f} W | Eff={efficiency:.3f} | RPM={RPM:.0f}")
    return dp, power, efficiency

# ----------------------------------------------------------------------------
# Optimization with monotonicity (twist only) and hard tip angle
# ----------------------------------------------------------------------------
def optimize_blade_spline(airfoil_data,
                          hub_d, od,
                          thickness,
                          CFM,
                          r_ctrl_twist,
                          theta_lb,
                          theta_ub,
                          rpm_lb,
                          rpm_ub,
                          dp_reg,
                          th0,           # initial twist controls (length n_twist-1)
                          t0,            # initial thickness controls (length n_t)
                          r_ctrl_t,
                          rpm_const,     # (target_rpm, weight) soft penalty
                          nblades):
    n_twist = len(r_ctrl_twist)
    n_t = len(r_ctrl_t)

    # initial guesses (unit-free / original units)
    rpm0 = 0.5 * (rpm_lb + rpm_ub)
    x0_raw = np.concatenate([th0, t0, [rpm0]])

    # bounds in ORIGINAL units
    lb = np.concatenate([
        np.full(n_twist - 1, theta_lb),      # twist controls (except tip)
        np.full(n_t, thickness/thickness_lb_factor),                   # thickness controls ∈ [0, thickness]
        [rpm_lb]
    ])
    ub = np.concatenate([
        np.full(n_twist - 1, theta_ub),
        np.full(n_t, thickness),
        [rpm_ub]
    ])

    # map to unit cube and use [0,1] box constraints for the optimizer
    x0 = to_unit(x0_raw, lb, ub)
    bounds = Bounds(0.0, 1.0)

    def obj_unit(u):
        x = from_unit(u, lb, ub)

        dp, P, efficiency = evaluate_performance(
            x, CFM, airfoil_data,
            thickness, hub_d, od,
            r_ctrl_twist, r_ctrl_t,
            FINAL_TIP_ANGLE, nblades
        )

        # pressure-drop soft target penalty: weight*(dp - target)^2
        p_dp = dp_reg[1] * (dp - dp_reg[0])**2

        # optional RPM soft penalty
        rpm_pen = rpm_const[1] * (rpm_const[0] - x[-1])**2

        # monotonic-decreasing penalty for TWIST controls only
        pen = 0.0
        theta_raw = x[:n_twist - 1]
        for i in range(len(theta_raw) - 1):
            diff = theta_raw[i] - theta_raw[i + 1]
            if diff < 0:  # penalize increases
                pen += 1e5 * (-diff)**2

        # maximize efficiency -> minimize negative
        return -efficiency + p_dp + rpm_pen + pen

    res = minimize(
        obj_unit, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 50, 'disp': True}
    )

    sol = from_unit(res.x, lb, ub)

    # unpack final solution in ORIGINAL units
    theta_opt = sol[:n_twist - 1]
    t_opt = sol[n_twist - 1:n_twist - 1 + n_t]
    RPM_opt = sol[-1]

    # stash helpful fields on the result object
    res.theta_ctrl_opt = np.concatenate([theta_opt, [FINAL_TIP_ANGLE]])
    res.t_ctrl_opt = t_opt
    res.RPM_opt = float(RPM_opt)
    res.lb = lb
    res.ub = ub
    return res

# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def run_opt():
    df = load_all_polars("airfoil_data/Eppler E63")
    af_data = Airfoil_Data(df, Ncrit=9)

    hub_d, od = 0.085, 0.20
    nblades = 6
    thickness = 0.0225
    CFM = 250

    target_dp = estimate_dp(CFM) * 1.25
    print(f"target delta p = {target_dp:.3f} Pa")
    dp_reg = (target_dp, 1e-1)      # (target, weight)
    RPM_reg = (1500.0, 0.0)        # (target, weight) 

    n_spline_ctrl_pts_twist = 3
    t0 = np.array([thickness/1.1, thickness/1.1, thickness/1.1])
    n_spline_ctrl_pts_t = t0.shape[0]

    r_ctrl_twist = np.linspace(hub_d/2, od/2, n_spline_ctrl_pts_twist)
    r_ctrl_t = np.linspace(hub_d/2, od/2, n_spline_ctrl_pts_t)

    # initial twist (exclude final tip)
    theta_prof_init = Linear_Prof(np.deg2rad(33), np.deg2rad(20),
                                  r_ctrl_twist[0], r_ctrl_twist[-2])
    th0 = np.zeros(n_spline_ctrl_pts_twist - 1)
    for i, r in enumerate(r_ctrl_twist[:-1]):
        th0[i] = theta_prof_init(r)


    theta_lb, theta_ub = FINAL_TIP_ANGLE, np.deg2rad(50)
    rpm_lb, rpm_ub = 1100.0, 1400.0

    res = optimize_blade_spline(
        af_data, hub_d, od,
        thickness, CFM,
        r_ctrl_twist,
        theta_lb, theta_ub,
        rpm_lb, rpm_ub,
        dp_reg,
        th0,
        t0,
        r_ctrl_t,
        rpm_const=RPM_reg,
        nblades=nblades
    )

    print("Optimized twist (deg):", np.rad2deg(res.theta_ctrl_opt))
    print("Optimized twist (rad):", res.theta_ctrl_opt)
    print("Optimized thickness ctrls (m):", res.t_ctrl_opt)
    print("Optimized RPM:", int(res.RPM_opt))

if __name__ == '__main__':
    run_opt()
