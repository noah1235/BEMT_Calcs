import os
import subprocess
import numpy as np

# === Inputs ===
airfoil_path = r"airfoil_data\RAF 30\RAF 30.dat"
alpha_i = -10
alpha_f = 10
alpha_step = 0.25
Re_list = [5e4, 1e5, 5e5]
n_iter = 1000
Ncrit = 9

# === Setup ===
airfoil_dir = os.path.dirname(airfoil_path)
airfoil_base = os.path.splitext(os.path.basename(airfoil_path))[0]

for Re in Re_list:
    print(f"Running XFOIL for Re = {Re:.0e}")

    polar_txt_name = f"{airfoil_base}_polar_Re{int(Re)}_Ncrit{Ncrit}.txt"
    polar_txt_path = os.path.join(airfoil_dir, polar_txt_name)

    # Clean up old files
    if os.path.exists(polar_txt_path):
        os.remove(polar_txt_path)
    if os.path.exists("input_file.in"):
        os.remove("input_file.in")

    # Write input file
    with open("input_file.in", 'w') as input_file:
        input_file.write(f"LOAD {airfoil_path}\n")
        input_file.write(f"{airfoil_base}\n")  # re-enter name if prompted
        input_file.write("PANE\n")
        input_file.write("OPER\n")
        input_file.write(f"Visc {Re:.0f}\n")
        input_file.write(f"Ncrit {Ncrit}\n")
        input_file.write("PACC\n")
        input_file.write(f"{polar_txt_path}\n\n")
        input_file.write(f"ITER {n_iter}\n")
        input_file.write(f"ASeq {alpha_i} {alpha_f} {alpha_step}\n")
        input_file.write("\n\n")
        input_file.write("quit\n")

    # Run XFOIL
    subprocess.call("xfoil.exe < input_file.in", shell=True)

    # Optionally read polar data
    if os.path.exists(polar_txt_path):
        try:
            data = np.loadtxt(polar_txt_path, skiprows=12)
            print(f"Loaded polar data from {polar_txt_name}")
        except Exception as e:
            print(f"Failed to read {polar_txt_name}: {e}")
    else:
        print(f"XFOIL failed: {polar_txt_name} not found")
