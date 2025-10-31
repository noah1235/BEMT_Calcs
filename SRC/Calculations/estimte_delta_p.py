from SRC.unit_conversion import CFM_2_m3_s

def estimate_dp(CFM, n_fins=81, fin_thickness=2e-4, fan_d=0.2,
                dy_vis=18.6e-6, rho=1.164, L=0.1):
    """
    Plate-fin core assumed square: width=height=fan_d.
    Uses Dh = 2*t (t = clear spacing between plates), laminar f_D = 96/Re.
    Returns delta_p in Pascals.
    """
    Q = CFM_2_m3_s(CFM)
    W = fan_d
    H = fan_d
    open_width = W - n_fins * fin_thickness

    N_c = n_fins + 1
    t = open_width / N_c              # clear gap between plates
    Dh = 2.0 * t                      # per L >> t
    A_open = open_width * H
    U = Q / A_open                    # mean channel velocity
    Re = rho * U * Dh / dy_vis
    fD = 96.0 / Re                    # laminar Darcy friction factor
    delta_p = fD * (L / Dh) * (0.5 * rho * U**2)
    return float(delta_p)
    