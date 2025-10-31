import numpy as np

def Straight_LE_Sweep(r: float) -> tuple[float, float]:
    """
    Simple leading-edge param: straight line in y, zero x-offset.

    Parameters
    ----------
    r : float
        Radius/station.

    Returns
    -------
    (x_LE, y_LE) : tuple[float, float]
    """
    return 0.0, r

class Max_Width_TE_prof:
    """
    Trailing-edge profile that attempts to maximize projected width subject to a cap.

    Notes
    -----
    - 'R' is retained (not used) to avoid changing the original API.
    - Uses thickness 't' and local blade angle 'theta_prof(r)' to limit width p.
    - Returns (x, y, z) with z being the axial thickness contribution p*sin(theta).
    """
    def __init__(self, R, t_prof, theta_prof, LE_prof, max_p = .1):
        self.R = R          # retained for API compatibility
        self.t_prof = t_prof
        self.theta_prof = theta_prof
        self.LE_prof = LE_prof
        self.max_p = max_p

    def __call__(self, r: float) -> tuple[float, float, float]:
        theta = self.theta_prof(r)
        # Max projected width based on thickness and user-specified cap
        t = self.t_prof(r)
        p = min(t / np.sin(theta), self.max_p)

        # LE position (x_LE, y_LE) retained for compatibility (not used directly here)
        x_LE, y_LE = self.LE_prof(r)

        # Geometry for a circle projection at radius r
        x = (p / (2 * r)) * np.sqrt(4 * r**2 - p**2)
        y = r - (p**2 / (2 * r))
        z = p * np.sin(theta)

        return x, y, z

