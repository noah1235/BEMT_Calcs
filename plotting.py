import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter

def plot_airfoil_perf_vs_r(airfoil_perf_data, T_prime_vals, Q_prime_vals, r_vals, save_path):
    airfoil_perf_data = np.array(airfoil_perf_data)
    
    alpha_list = airfoil_perf_data[:, 0]
    cd_list = airfoil_perf_data[:, 1]
    cl_list = airfoil_perf_data[:, 2]
    cn_list = airfoil_perf_data[:, 3]
    ct_list = airfoil_perf_data[:, 4]
    phi_list = airfoil_perf_data[:, 5]
    Re_list = airfoil_perf_data[:, 6]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    # Alpha
    axes[0].plot(r_vals, alpha_list)
    axes[0].set_xlabel("Radius [m]")
    axes[0].set_ylabel("Alpha [deg]")
    axes[0].set_title("Angle of Attack vs Radius")
    axes[0].grid(True)

    # Cl and Cd
    ax1 = axes[1]

    p1, = ax1.plot(r_vals, cl_list/cd_list)
    #p2, = ax1.plot(r_vals, cd_list, color='tab:red', label="Cd")

    ax1.set_xlabel("Radius [m]")
    ax1.set_ylabel("Cl/Cd", color='tab:blue')
    ax1.set_title("Cl/Cd vs Radius")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Cn and Ct together (shared y)
    axes[2].plot(r_vals, cn_list, label="Cn")
    axes[2].plot(r_vals, cl_list*np.cos(phi_list), label="cl * cos(phi)")
    axes[2].plot(r_vals, cl_list*np.sin(phi_list), label="cl * sin(phi)")
    axes[2].plot(r_vals, ct_list, label="Ct")
    axes[2].set_xlabel("Radius [m]")
    axes[2].set_ylabel("Force Coefficient")
    axes[2].set_title("Cn and Ct vs Radius")
    axes[2].legend()
    axes[2].grid(True)

    # T' and Q' with two y-axes
    ax3 = axes[3]
    ax4 = ax3.twinx()

    p3, = ax3.plot(r_vals, T_prime_vals, color='tab:green', label="T'")
    p4, = ax4.plot(r_vals, Q_prime_vals, color='tab:purple', label="Q'")

    ax3.set_xlabel("Radius [m]")
    ax3.set_ylabel("T'", color='tab:green')
    ax4.set_ylabel("Q'", color='tab:purple')
    ax3.set_title("T' and Q' vs Radius")
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax4.tick_params(axis='y', labelcolor='tab:purple')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "airfoil_perf.png"))
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(r_vals, Re_list)
    ax.set_xlabel("Radius [m]")
    ax.set_ylabel("Reynolds #")

    # Force scientific notation on y-axis
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Re_vs_radial_pos.png"))
    plt.close()