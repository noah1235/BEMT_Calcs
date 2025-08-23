import numpy as np
from read_xfoil_data import load_all_polars
from calcs import Airfoil_Data, radial_integration, calc_Re
from geometry import Blade_Geometry, Linear_Prof, Max_Chord_Prof, Power_Law_Prof, SplineTwistProfile
from plotting import plot_airfoil_perf_vs_r, startup
import os
import matplotlib.pyplot as plt
import pandas as pd

#blade geo
hub_diameter = .085 #m
thickness = .02 #m
od = .2 #m

#set_angle = np.deg2rad(35.8) # rads
#end_angle = np.deg2rad(10)
#theta_prof = Linear_Prof(set_angle, end_angle, hub_diameter/2, od/2)
#theta_prof = Power_Law_Prof(set_angle, end_angle, hub_diameter/2, od/2, 1.4)

theta_ctrl = np.array(
    #[np.deg2rad(29), np.deg2rad(26.5), np.deg2rad(19), np.deg2rad(8)]
    [0.56025013, 0.39640558, 0.2443461 ]
    )
r_ctrl = np.linspace(hub_diameter/2, od/2, theta_ctrl.shape[0])
theta_prof = SplineTwistProfile(r_ctrl, theta_ctrl,
                                    kind='pchip', extrapolate=False)

chord_prof = Max_Chord_Prof(theta_prof, thickness)

blade_geo = Blade_Geometry(
    airfoil_name="Eppler E71",
    Ncrit=9,
    B=4,
    thickness=thickness,
    hub_diameter=hub_diameter,
    od=od,
    omega_rpm=1300,
    theta_prof=theta_prof,
    CFM = -1
)



airfoil_path = f"airfoil_data/{blade_geo.airfoil_name}"
full_df = load_all_polars(airfoil_path)
airfoil_data = Airfoil_Data(full_df, blade_geo.Ncrit)


def tabulate_geo(save_path):
    r_vals = np.linspace(hub_diameter/2, od/2, 20)
    arc_chord = np.zeros_like(r_vals)
    lin_chord = np.zeros_like(r_vals)
    twist = np.zeros_like(r_vals)
    x_LE_list = np.zeros_like(r_vals)
    y_LE_list = np.zeros_like(r_vals)
    x_TE_list = np.zeros_like(r_vals)
    y_TE_list = np.zeros_like(r_vals)
    for i, r in enumerate(r_vals):
        arc_chord[i] = blade_geo.get_arc_choord(r)
        x_LE, y_LE = blade_geo.LE_prof(r)
        x_TE, y_TE, z_TE = blade_geo.TE_prof(r)
        lin_chord[i] = np.sqrt((x_LE-x_TE)**2 + (y_LE-y_TE)**2 + z_TE**2)
        twist[i] = np.rad2deg(blade_geo.theta_prof(r))

        x_LE_list[i] = x_LE
        y_LE_list[i] = y_LE
        x_TE_list[i] = x_TE
        y_TE_list[i] = y_TE

    df = pd.DataFrame({
        'radius [m]': r_vals,
        'twist [deg]': twist,
        'arc choord [m]': arc_chord,
        'linear choord [m]': lin_chord,
        'LE x [m]': x_LE_list,
        'LE y [m]': y_LE_list,
        'TE x [m]': x_TE_list,
        'TE y [m]': y_TE_list,
    })

    # Save to Excel
    df.to_excel(os.path.join(save_path, "airfoil_data.xlsx"), index=False)

def save_fan_parameters(blade_geo, CFM, delta_p, eff, save_path):
    param_text = f"""Fan Parameters
--------------------
Airfoil        : {blade_geo.airfoil_name}
Ncrit          : {blade_geo.Ncrit}
Blades (B)     : {blade_geo.B}
Thickness [m]  : {blade_geo.thickness}
Hub Diameter [m]: {blade_geo.hub_diameter}
Outer Diameter [m]: {blade_geo.od}
RPM            : {blade_geo.omega_rpm}
Set Angle [deg]: {np.rad2deg(blade_geo.theta_prof.start):.2f}
Tip Angle [deg]: {np.rad2deg(blade_geo.theta_prof.end):.2f}
Chord Root [m] : {blade_geo.choord_start}
Chord Tip [m]  : {blade_geo.choord_end}
Target CFM     : {CFM}
delta p [Pa]   : {delta_p:.2f}
efficiency      : {eff:.2f}

"""
    with open(os.path.join(save_path, "fan_parameters.txt"), "w") as f:
        f.write(param_text)

def generate_fan_curve(root, CFM_list=np.linspace(100, 300, 10)
                       ):
    delta_p_list = []
    efficency_list = []
    for CFM in CFM_list:
        blade_geo.set_CFM(CFM)
        delta_p, prof_losses, airfoil_perf_data_list, T_prime_vals, Q_prime_vals, r_list = radial_integration(blade_geo, airfoil_data)
        delta_p_list.append(delta_p)
        p_fluid = delta_p * blade_geo.flow_rate
        p_tot = p_fluid + prof_losses
        eff = p_fluid/p_tot
        efficency_list.append(eff)
    
    fig, ax1 = plt.subplots()

    # Primary Y axis: delta_p
    color1 = 'tab:blue'
    ax1.set_xlabel("CFM")
    ax1.set_ylabel("Delta P [Pa]", color=color1)
    ax1.plot(CFM_list, delta_p_list, color=color1, label="Delta P")
    ax1.tick_params(axis='y', labelcolor=color1)

    # Secondary Y axis: efficiency
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel("Efficiency", color=color2)
    ax2.plot(CFM_list, efficency_list, color=color2, linestyle='--', label="Efficiency")
    ax2.tick_params(axis='y', labelcolor=color2)

    # Optionally, add a title and grid
    plt.title("Delta P and Efficiency vs CFM")
    fig.tight_layout()
    plt.grid(True, which='both', axis='both', linestyle=':', alpha=0.5)

    fig.savefig(os.path.join(root, "fan_curve.png"), dpi=600)
    plt.close()

def fan_calc_ctrl(opt_CFM = 200):
    root = os.path.join(airfoil_path, f"blade_geo", f"nblades={blade_geo.B}", f"thickness={blade_geo.thickness}", f"hub_diameter={blade_geo.hub_diameter}", f"target_CFM={opt_CFM}")
    run_single_CFM(opt_CFM, root)
    startup(blade_geo, airfoil_data, root)
    generate_fan_curve(root)
    tabulate_geo(root)

    blade_geo.plot_views(os.path.join(root, "fan_geo.png"))


def run_single_CFM(CFM, root):
    blade_geo.set_CFM(CFM)
    delta_p, prof_losses, airfoil_perf_data_list, T_prime_vals, Q_prime_vals, r_list = radial_integration(blade_geo, airfoil_data)
    print(r_list[0])
    p_fluid = delta_p * blade_geo.flow_rate
    p_tot = p_fluid + prof_losses
    eff = p_fluid/(p_fluid + prof_losses)
    print(f"delta p: {delta_p} | power: {p_tot} | eff: {eff}")
    
    os.makedirs(root, exist_ok=True)
    plot_airfoil_perf_vs_r(airfoil_perf_data_list, T_prime_vals, Q_prime_vals, r_list, root)
    save_fan_parameters(blade_geo, CFM, delta_p, eff, root)


    

if __name__ == "__main__":
    fan_calc_ctrl()
    #generate_fan_curve()