import numpy as np
from read_xfoil_data import load_all_polars
from calcs import Airfoil_Data, radial_integration
from geometry import Blade_Geometry, Linear_Prof
from plotting import plot_airfoil_perf_vs_r
import os

def main():
    set_angle = np.deg2rad(26) # rads
    thickness = .08 #m
    choord_set = thickness / np.cos(set_angle) #m
    hub_diameter = .1 #m
    od = .2 #m

    blade_geo = Blade_Geometry(
        airfoil_name="Eppler E168",
        Ncrit=9,
        B=3,
        thickness=thickness,
        hub_diameter=hub_diameter,
        od=od,
        omega_rpm=1733,
        coord_prof=Linear_Prof(choord_set, choord_set/1.4, hub_diameter/2, od/2),
        theta_prof=Linear_Prof(set_angle, np.deg2rad(5.13), hub_diameter/2, od/2),
        CFM = 200
    )

    airfoil_path = f"airfoil_data\\{blade_geo.airfoil_name}"
    full_df = load_all_polars(airfoil_path)
    airfoil_data = Airfoil_Data(full_df, blade_geo.Ncrit)


    #find_phi(set_angle, omega, .03, choord_set, v_freestream, 3, airfoil_data)
    delta_p, prof_losses, airfoil_perf_data_list, T_prime_vals, Q_prime_vals, r_list = radial_integration(blade_geo, airfoil_data)

    p_fluid = delta_p * blade_geo.flow_rate
    p_tot = p_fluid + prof_losses
    eff = p_fluid/(p_fluid + prof_losses)
    print(f"delta p: {delta_p} | power: {p_tot} | eff: {eff}")
    

    save_path = os.path.join(airfoil_path, f"blade_geo", f"thickness={blade_geo.thickness}", f"hub_diameter={blade_geo.hub_diameter}")
    os.makedirs(save_path, exist_ok=True)
    plot_airfoil_perf_vs_r(airfoil_perf_data_list, T_prime_vals, Q_prime_vals, r_list, save_path)

    

    

if __name__ == "__main__":
    main()