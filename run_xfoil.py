import os
import subprocess
import numpy as np

# Inputs
airfoil_path = r"airfoil_data\Eppler E168\Eppler E168.dat"
alpha_i = -10
alpha_f = 10
alpha_step = 0.25
Re_list = [5e4, 1e5, 5e5]
n_iter = 1000
Ncrit = 9  # Added Ncrit parameter

# Extract airfoil directory and base name
airfoil_dir = os.path.dirname(airfoil_path)
airfoil_base = os.path.splitext(os.path.basename(airfoil_path))[0]

for Re in Re_list:
    # Build polar output filename
    polar_txt_name = f"{airfoil_base}_polar_Re{Re}_Ncrit{Ncrit}.txt"
    polar_txt_path = os.path.join(airfoil_dir, polar_txt_name)

    # Remove old polar file if it exists
    if os.path.exists(polar_txt_path):
        os.remove(polar_txt_path)

    # Create XFOIL input script
    with open("input_file.in", 'w') as input_file:
        input_file.write(f"LOAD {airfoil_path}\n")
        input_file.write("\n")  # Accept default airfoil name
        input_file.write("PANE\n")
        input_file.write("OPER\n")
        input_file.write(f"Visc {Re}\n")
        input_file.write(f"Ncrit {Ncrit}\n")   # Set Ncrit
        input_file.write("PACC\n")
        input_file.write(f"{polar_txt_path}\n\n")
        input_file.write(f"ITER {n_iter}\n")
        input_file.write(f"ASeq {alpha_i} {alpha_f} {alpha_step}\n")
        input_file.write("\n\n")
        input_file.write("quit\n")

    # Run XFOIL
    with open("input_file.in", "r") as f:
        subprocess.run(["xfoil.exe"], stdin=f)

    # Optionally: load the data to confirm it was created (or just leave this out)
    polar_data = np.loadtxt(polar_txt_path, skiprows=12)

