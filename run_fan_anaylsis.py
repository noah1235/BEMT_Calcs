import numpy as np
from SRC.read_xfoil_data import load_all_polars
from SRC.Calculations.BEMT_Calcs import radial_integration
from SRC.Calculations.Airfoil_Data import Airfoil_Data
from SRC.Geometry.Blade_Geometry import Blade_Geometry
from SRC.Geometry.profiles import SplineProfile
from SRC.plotting import plot_airfoil_perf_vs_r, startup
import os
import matplotlib.pyplot as plt
import pandas as pd


# -------------------------------
# Blade geometry (inputs/controls)
# -------------------------------
hub_diameter = 0.085  # [m]
thickness    = 0.025   # [m]
od           = 0.20   # [m]
Blades = 6
RPM = 1365

# Twist (theta) control points (radians) for spline-based profile
theta_ctrl = np.array([0.56812577, 0.4429515,  0.43633231])

# Thickness control points (meters)
t_ctrl = np.array([0.015,      0.01806745, 0.0225])

# Spanwise coordinates corresponding to control points (from hub to tip)
r_ctrl_theta = np.linspace(hub_diameter / 2, od / 2, theta_ctrl.shape[0])
theta_prof = SplineProfile(
    r_ctrl_theta, theta_ctrl, kind="pchip", extrapolate=False
)

r_ctrl_t = np.linspace(hub_diameter / 2, od / 2, t_ctrl.shape[0])
t_prof = SplineProfile(
    r_ctrl_t, t_ctrl, kind="pchip", extrapolate=False
)

# Blade geometry object (B = number of blades, Ncrit = XFOIL/Ncrit, etc.)
blade_geo = Blade_Geometry(
    airfoil_name="Eppler E63",
    Ncrit=9,
    B=Blades,
    thickness_prof=t_prof,
    max_t=thickness,
    hub_diameter=hub_diameter,
    od=od,
    omega_rpm=RPM,
    theta_prof=theta_prof,
    CFM=-1,  # will be set later before analysis
)

# -------------------------------
# Airfoil polars and data wrapper
# -------------------------------
airfoil_path = f"airfoil_data/{blade_geo.airfoil_name}"
full_df = load_all_polars(airfoil_path)     
airfoil_data = Airfoil_Data(full_df, blade_geo.Ncrit) 


def tabulate_geo(save_path):
    """
    Sample the blade geometry along the span and export key geometric data
    to an Excel file for documentation/inspection.

    Parameters
    ----------
    save_path : str
        Directory path where the Excel file will be written.
    """
    r_vals = np.linspace(hub_diameter / 2, od / 2, 20)

    # Preallocate arrays for sampled quantities
    arc_chord = np.zeros_like(r_vals)
    twist     = np.zeros_like(r_vals)
    x_LE_list = np.zeros_like(r_vals)
    y_LE_list = np.zeros_like(r_vals)
    x_TE_list = np.zeros_like(r_vals)
    y_TE_list = np.zeros_like(r_vals)

    # Walk along radius and derive geometry from blade_geo interfaces
    for i, r in enumerate(r_vals):
        arc_chord[i] = blade_geo.get_arc_chord(r)

        # Leading/trailing edge coordinates
        x_LE, y_LE = blade_geo.LE_prof(r)
        x_TE, y_TE, z_TE = blade_geo.TE_prof(r)

        # Local twist (convert to degrees for table)
        twist[i] = np.rad2deg(blade_geo.theta_prof(r))

        # Store coordinates
        x_LE_list[i] = x_LE; y_LE_list[i] = y_LE
        x_TE_list[i] = x_TE; y_TE_list[i] = y_TE

    # Assemble dataframe
    df = pd.DataFrame({
        "radius [m]": r_vals,
        "twist [deg]": twist,
        "arc chord [m]": arc_chord, 
        "LE x [m]": x_LE_list,
        "LE y [m]": y_LE_list,
        "TE x [m]": x_TE_list,
        "TE y [m]": y_TE_list,
    })

    # Write geometry table
    df.to_excel(os.path.join(save_path, "airfoil_data.xlsx"), index=False)


def save_fan_parameters(blade_geo, CFM, delta_p, eff, save_path):
    """
    Write a human-readable summary of key fan parameters and results.

    Parameters
    ----------
    blade_geo : Blade_Geometry
        Geometry object containing setup/inputs.
    CFM : float
        Target volumetric flow (the operating point analyzed).
    delta_p : float
        Predicted pressure rise [Pa].
    eff : float
        Predicted efficiency (0–1 or % per your calcs).
    save_path : str
        Output directory for the text file.
    """
    param_text = f"""Fan Parameters
--------------------
Airfoil         : {blade_geo.airfoil_name}
Ncrit           : {blade_geo.Ncrit}
Blades (B)      : {blade_geo.B}
Thickness [m]   : {blade_geo.max_t}
Hub Diameter [m]: {blade_geo.hub_diameter}
Outer Diameter [m]: {blade_geo.od}
RPM             : {blade_geo.omega_rpm}
Set Angle [deg] : {np.rad2deg(blade_geo.theta_prof.start):.2f}
Tip Angle [deg] : {np.rad2deg(blade_geo.theta_prof.end):.2f}
Chord Root [m]  : {blade_geo.chord_start}
Chord Tip [m]   : {blade_geo.chord_end}
Target CFM      : {CFM}
delta p [Pa]    : {delta_p:.2f}
efficiency      : {eff:.2f}

"""
    with open(os.path.join(save_path, "fan_parameters.txt"), "w") as f:
        f.write(param_text)


def generate_fan_curve(root, CFM_list=np.linspace(100, 300, 20)):
    """
    Sweep CFM setpoints, run the blade model, and plot Δp and efficiency vs CFM.

    Parameters
    ----------
    root : str
        Output directory for figures.
    CFM_list : array-like, optional
        Sequence of target CFMs to analyze. Defaults to 100–300 in 20 steps.
    """
    delta_p_list = []
    efficency_list = []  # keep original spelling to avoid logic/output changes

    # Evaluate each CFM point by setting it into the geometry and integrating
    for CFM in CFM_list:
        blade_geo.set_CFM(CFM)
        delta_p, power, efficency, airfoil_perf_data_list, T_prime_vals, Q_prime_vals, r_list = radial_integration(
            blade_geo, airfoil_data
        )
        delta_p_list.append(delta_p)
        efficency_list.append(efficency)

    # --- Plot Δp and efficiency on twin y-axes ---
    fig, ax1 = plt.subplots()

    # Primary Y axis: Δp(CFM)
    color1 = "tab:blue"
    ax1.set_xlabel("CFM")
    ax1.set_ylabel("Delta P [Pa]", color=color1)
    ax1.plot(CFM_list, delta_p_list, color=color1, label="Delta P")
    ax1.tick_params(axis="y", labelcolor=color1)

    # Secondary Y axis: η(CFM)
    ax2 = ax1.twinx()
    color2 = "tab:green"
    ax2.set_ylabel("Efficiency", color=color2)
    ax2.plot(CFM_list, efficency_list, color=color2, linestyle="--", label="Efficiency")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Layout/formatting
    plt.title("Delta P and Efficiency vs CFM")
    fig.tight_layout()
    plt.grid(True, which="both", axis="both", linestyle=":", alpha=0.5)

    # Save figure
    fig.savefig(os.path.join(root, "fan_curve.png"), dpi=600)
    plt.close()


def fan_calc_ctrl(opt_CFM=250):
    """
    Top-level convenience controller:
      1) Run a single operating point at opt_CFM.
      2) Generate performance plots and geometry tables.
      3) Produce fan curve sweep around the design point.

    Parameters
    ----------
    opt_CFM : float
        Target CFM for the primary operating point.
    """
    # Organize outputs under airfoil/geometry-driven folder structure
    root = os.path.join(
        airfoil_path,
        "blade_geo",
        f"nblades={blade_geo.B}",
        f"thickness={blade_geo.max_t}",
        f"hub_diameter={blade_geo.hub_diameter}",
        f"target_CFM={opt_CFM}",
    )

    # Run one operating point, then startup plots and fan curve
    run_single_CFM(opt_CFM, root)
    startup(blade_geo, airfoil_data, root)
    generate_fan_curve(root)
    tabulate_geo(root)

    # Save multi-view geometry image
    blade_geo.plot_views(os.path.join(root, "fan_geo.png"))


def run_single_CFM(CFM, root):
    """
    Run the solver/integration at a specified CFM and save artifacts.

    Parameters
    ----------
    CFM : float
        Target volumetric flow.
    root : str
        Output directory for plots/parameters.
    """
    # Set operating point on geometry
    blade_geo.set_CFM(CFM)

    # Run radial integration and collect results
    delta_p, power, efficency, airfoil_perf_data_list, T_prime_vals, torque, r_list = radial_integration(
        blade_geo, airfoil_data
    )

    # Console summary
    print(f"delta p: {delta_p} | power: {power} | eff: {efficency}")

    # Ensure output directory exists
    os.makedirs(root, exist_ok=True)

    # Spanwise performance plots and parameter save-out
    plot_airfoil_perf_vs_r(airfoil_perf_data_list, T_prime_vals, torque, r_list, root)
    save_fan_parameters(blade_geo, CFM, delta_p, efficency, root)


if __name__ == "__main__":
    fan_calc_ctrl()
