
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from read_xfoil_data import load_all_polars
from calcs import Airfoil_Data, radial_integration
from geometry import Blade_Geometry, Max_Chord_Prof, SplineTwistProfile, Linear_Prof
from plotting import plot_airfoil_perf_vs_r

# ----------------------------------------------------------------------------
# Hard-coded final tip twist (radians)
# ----------------------------------------------------------------------------
FINAL_TIP_ANGLE = np.deg2rad(14)

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
                         r_ctrl, final_tip_angle, nblades):
    """
    vars_raw: [theta_ctrl[0..n_ctrl-2], RPM]
    r_ctrl: array of length n_ctrl (including tip location)
    final_tip_angle: scalar
    """
    n_ctrl = len(r_ctrl)
    # separate optimization vars (all but final)
    theta_raw = vars_raw[:n_ctrl-1]
    # append hard-coded tip twist
    theta_ctrl = np.concatenate([theta_raw, [final_tip_angle]])
    RPM = vars_raw[-1]
    # build profiles
    twist_prof = SplineTwistProfile(r_ctrl, theta_ctrl,
                                    kind='pchip', extrapolate=True)
    blade = Blade_Geometry(
        airfoil_name="",
        Ncrit=9,
        B=nblades,
        thickness=thickness,
        hub_diameter=hub_d,
        od=od,
        omega_rpm=RPM,
        theta_prof=twist_prof,
        CFM=CFM
    )

    dp, losses, perf_list, *_ = radial_integration(blade, airfoil_data)
    power = dp * blade.flow_rate + losses
    efficency = (dp * blade.flow_rate)/power
    print(f"Δp={dp:.1f} Pa | Power={power:.1f} W | Eff: {efficency} | RPM={RPM:.0f}")
    return dp, losses, power, efficency

# ----------------------------------------------------------------------------
# Optimization with monotonicity and hard tip angle
# ----------------------------------------------------------------------------
def optimize_blade_spline(airfoil_data,
                          hub_d, od,
                          thickness,
                          CFM,
                          r_ctrl,
                          theta_lb,
                          theta_ub,
                          rpm_lb,
                          rpm_ub,
                          dp_reg,
                          th0,
                          rpm_const,
                          nblades
                          ):
    n_ctrl = len(r_ctrl)
    # initial theta excluding final
    rpm0 = 0.5*(rpm_lb + rpm_ub)
    x0_raw = np.concatenate([th0, [rpm0]])

    # bounds for optimization vars
    lb = np.concatenate([np.full(n_ctrl-1, theta_lb), [rpm_lb]])
    ub = np.concatenate([np.full(n_ctrl-1, theta_ub), [rpm_ub]])
    x0 = to_unit(x0_raw, lb, ub)
    bounds = Bounds(np.zeros_like(x0), np.ones_like(x0))

    def obj_unit(u):
        x = from_unit(u, lb, ub)
        # reuse evaluate
        dp, _, P, effiency = evaluate_performance(x, CFM, airfoil_data,
                                        thickness, hub_d, od,
                                        r_ctrl, FINAL_TIP_ANGLE, nblades)
        # Δp penalty
        p_dp = dp_reg[1]*(dp - dp_reg[0])**2 * 0

        rpm_pen = rpm_const[1] * (rpm_const[0] - x[-1])**2


        # monotonicity penalty for theta_raw
        pen = 0.0
        theta_raw = x[:n_ctrl-1]
        for i in range(len(theta_raw)-1):
            diff = theta_raw[i] - theta_raw[i+1]
            if diff < 0:
                pen += 1e5 * (-diff)**2
        #return P + p_dp + pen + rpm_pen
        return -effiency + p_dp + pen

    res = minimize(obj_unit, x0,
                   method='L-BFGS-B', bounds=bounds,
                   options={'maxiter':20, 'disp':True})

    sol = from_unit(res.x, lb, ub)
    res.theta_ctrl_opt = np.concatenate([sol[:n_ctrl-1], [FINAL_TIP_ANGLE]])
    res.RPM_opt = sol[-1]
    return res

# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def run_opt():
    df = load_all_polars("airfoil_data/Eppler E71")
    af_data = Airfoil_Data(df, Ncrit=9)

    hub_d, od = 0.085, 0.20
    nblades = 4
    thickness = 0.02
    CFM = 200
    dp_reg = (20, 1e-1)
    RPM_reg = (1500, 0)
    n_spline_ctrl_pts = 3
    #th0 = np.array([np.deg2rad(30), np.deg2rad(28), np.deg2rad(20)])
    r_ctrl = np.linspace(hub_d/2, od/2, n_spline_ctrl_pts)

    theta_prof = Linear_Prof(np.deg2rad(29), np.deg2rad(20), r_ctrl[0], r_ctrl[-2])
    th0 = np.zeros(n_spline_ctrl_pts-1)
    for i, r in enumerate(r_ctrl[:-1]):
        th0[i] = theta_prof(r)

    theta_lb, theta_ub = np.deg2rad(0), np.deg2rad(50)
    rpm_lb, rpm_ub = 1300, 3000

    res = optimize_blade_spline(
        af_data, hub_d, od,
        thickness, CFM,
        r_ctrl,
        theta_lb, theta_ub,
        rpm_lb, rpm_ub,
        dp_reg,
        th0,
        rpm_const=RPM_reg,
        nblades=nblades
    )
    print("Optimized thetas (deg):", np.rad2deg(res.theta_ctrl_opt))
    print("Optimized thetas (rad):", (res.theta_ctrl_opt))
    print("Optimized RPM:", int(res.RPM_opt))

if __name__ == '__main__':
    run_opt()
