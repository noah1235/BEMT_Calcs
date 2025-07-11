import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from read_xfoil_data import load_all_polars
from calcs import Airfoil_Data, radial_integration
from geometry import Blade_Geometry, Linear_Prof
from plotting import plot_airfoil_perf_vs_r


def to_unit(x, lb, ub):
    """Scale raw variables x into [0,1] given lb and ub."""
    return (x - lb) / (ub - lb)


def from_unit(u, lb, ub):
    """Map unit variables u in [0,1] back to raw scale."""
    return lb + u * (ub - lb)


def evaluate_performance(vars, CFM, airfoil_data, thickness, od=.2):
    """
    Unpack design variables, run your radial_integration, and return
    Δp, prof_losses, min_Re, and fan_power.
    """
    set_angle, end_angle, hub_d, RPM, choord_end_factor = vars
    coord_set = thickness / np.cos(set_angle)

    blade_geo = Blade_Geometry(
        airfoil_name="NACA 2415",
        Ncrit=9,
        B=3,
        thickness=thickness,
        hub_diameter=hub_d,
        od=od,
        omega_rpm=RPM,
        coord_prof=Linear_Prof(coord_set, coord_set/choord_end_factor, hub_d/2, od/2),
        theta_prof=Linear_Prof(set_angle, end_angle, hub_d/2, od/2),
        CFM=CFM
    )

    delta_p, prof_losses, airfoil_perf_data_list, _, _, _ = radial_integration(blade_geo, airfoil_data)
    fan_power = delta_p * blade_geo.flow_rate + prof_losses

    airfoil_perf_data = np.array(airfoil_perf_data_list)
    min_Re = airfoil_perf_data[:, 6].min()

    print(delta_p, fan_power, min_Re, RPM)
    return delta_p, prof_losses, min_Re, fan_power


def optimize_blade(airfoil_data,
                   x0_raw,
                   raw_lb,
                   raw_ub,
                   thickness,
                   CFM=150,
                   Δp_target=100.0,
                   reynolds_min=5e4):

    x0_unit = to_unit(x0_raw, raw_lb, raw_ub)

    # unit-space bounds [0,1]
    bounds_unit = Bounds(np.zeros(x0_unit.shape[0]), np.ones(x0_unit.shape[0]))

    # original inequality on raw variables
    def con_ineq_raw(vars):
        _, _, min_Re, _ = evaluate_performance(vars, CFM, airfoil_data, thickness)
        return min_Re - reynolds_min

    # unit-space inequality constraint
    def con_ineq_unit(u):
        raw_vars = from_unit(u, raw_lb, raw_ub)
        return con_ineq_raw(raw_vars)

    ineq_cons_unit = {'type': 'ineq', 'fun': con_ineq_unit}

    # original penalized objective on raw variables
    def penalized_obj_raw(vars, μ=5):
        Δp, prof_losses, min_Re, fan_power = evaluate_performance(vars, CFM, airfoil_data, thickness)
        penalty = μ * (Δp - Δp_target)**2
        return fan_power + penalty

    # unit-space objective
    def penalized_obj_unit(u):
        raw_vars = from_unit(u, raw_lb, raw_ub)
        return penalized_obj_raw(raw_vars)

    # run optimization in unit space
    res = minimize(
        penalized_obj_unit,
        x0_unit,
        method='trust-constr',
        bounds=bounds_unit,
        constraints=[ineq_cons_unit],
        #tol=1e-4,
        #options={'verbose': 2},
        options={
            'maxiter': 20,   # ← stop after at most 300 iterations
            #'ftol': 1e-6,
            #'disp': True,
            'verbose': 2
        }
    )

    # map solution back to raw
    res.x_raw = from_unit(res.x, raw_lb, raw_ub)
    return res


def run_opt():
    airfoil_path = f"airfoil_data\\Eppler E168"
    full_df= load_all_polars(airfoil_path)
    airfoil_data = Airfoil_Data(full_df, Ncrit=9)

    # raw bounds for [set_angle(rad), end_angle(rad), hub_d(m), RPM]
    raw_lb = np.array([np.deg2rad(20), np.deg2rad(5), 0.08, 1000, 1])
    raw_ub = np.array([np.deg2rad(40), np.deg2rad(30), 0.1, 3000, 4])

    # transform initial guess into unit space
    x0_raw = np.array([np.deg2rad(30), np.deg2rad(20), 0.1, 1500, 2.1])

    result = optimize_blade(
        airfoil_data,
        x0_raw,
        raw_lb,
        raw_ub,
        thickness=.08,
        CFM=200,
        Δp_target=20.0,
        reynolds_min=5e4
    )

    print("optimal params:")
    print(f"set angle: {np.rad2deg(result.x_raw[0])} | end angle: {np.rad2deg(result.x_raw[1])} | hub_d(m): {result.x_raw[2]} | RPM: {result.x_raw[3]} | choord end factor: {result.x_raw[4]}")
    #print("Unit-space solution:", result.x)
    #print("Raw-space solution:", result.x_raw)
    #print(result)

run_opt()
