
from unit_conversion import *
import numpy as np
import matplotlib.pyplot as plt
import cmath
import os
from scipy.interpolate import CubicSpline, PchipInterpolator

def central_angle(P, Q, center=(0.0, 0.0), signed=False, degrees=False):
    """
    Angle at the circle center between points P and Q on the circle.

    P, Q: (2,) arrays [x,y]
    center: (2,) center of the circle (default origin)
    signed: if True, CCW-positive angle in (-pi, pi]; else smallest angle in [0, pi]
    degrees: return degrees instead of radians
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    C = np.asarray(center, dtype=float)

    u = P - C
    v = Q - C

    # atan2(cross, dot) is robust and independent of radius
    cross = u[0]*v[1] - u[1]*v[0]
    dot   = u[0]*v[0] + u[1]*v[1]
    theta = np.arctan2(cross, dot)          # signed angle (-pi, pi]

    ang = theta if signed else np.abs(theta) # [0, pi]
    return np.degrees(ang) if degrees else ang

def Straight_LE_Sweep(r):
    return 0, r

class Max_Width_TE_prof():
    def __init__(self, R, t, theta_prof, LE_prof, max_p):
        self.R = R
        self.t = t
        self.theta_prof = theta_prof
        self.LE_prof = LE_prof
        self.max_angle = np.deg2rad(50)
        self.max_p = max_p
    
    def __call__(self, r):
        theta = self.theta_prof(r)
        p = min(self.t/np.sin(theta), self.max_p)
        x_LE, y_LE = self.LE_prof(r)
        x = (p/(2*r)) *np.sqrt(4*r**2 - p**2)
        y = r - (p**2/(2*r))

        z = p * np.sin(theta)
        return x, y, z

       


def plot_tip_arc_center0(
    ax,
    TE_tip, LE_tip,                      # [x, y] for TE_r[-1], LE_r[-1]
    blade_color='k', lw=1.5, n=200,
    center=(0.0, 0.0),
    r=None, use_outer_radius=True,       # if r is None: use max or mean of endpoint radii
    prefer_major=False,                  # False: minor arc; True: major arc
    force_semicircle=False               # if True: draw exactly 180° cap around mid-angle
):
    """
    Draw a circular arc centered at `center` connecting the tip between LE and TE.

    - If force_semicircle=False (default): connects TE -> LE along the circle (minor/major selectable).
    - If force_semicircle=True: draws a 180° cap centered on the mid-angle (endpoints may not
      exactly pass through TE/LE if they aren't diametrically opposite).
    """
    TE_tip = np.asarray(TE_tip, dtype=float)
    LE_tip = np.asarray(LE_tip, dtype=float)
    cx, cy = center

    # Shift to circle center
    TE0 = TE_tip - np.array([cx, cy])
    LE0 = LE_tip - np.array([cx, cy])

    # Radii and chosen plotting radius
    r_TE = np.hypot(TE0[0], TE0[1])
    r_LE = np.hypot(LE0[0], LE0[1])
    if r is None:
        r = max(r_TE, r_LE) if use_outer_radius else 0.5 * (r_TE + r_LE)

    # Angles of endpoints
    th_TE = np.arctan2(TE0[1], TE0[0])
    th_LE = np.arctan2(LE0[1], LE0[0])

    def wrap_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    if force_semicircle:
        # Mid-angle via unit-vector average (robust near wrap-around)
        u_TE = TE0 / (r_TE + 1e-12)
        u_LE = LE0 / (r_LE + 1e-12)
        u_mid = u_TE + u_LE
        if np.allclose(u_mid, 0.0):
            th_mid = th_TE  # fallback if diametrically opposite
        else:
            th_mid = np.arctan2(u_mid[1], u_mid[0])
        th_start = th_mid - np.pi/2
        th_end   = th_mid + np.pi/2
        th = np.linspace(th_start, th_end, n)
    else:
        # Connect TE -> LE along chosen arc
        dtheta = wrap_pi(th_LE - th_TE)
        if prefer_major:
            # take the longer arc
            if dtheta >= 0:
                dtheta = dtheta - 2*np.pi
            else:
                dtheta = dtheta + 2*np.pi
        th = th_TE + np.linspace(0.0, dtheta, n)

    x = cx + r * np.cos(th)
    y = cy + r * np.sin(th)
    ax.plot(x, y, color=blade_color, lw=lw)


class Linear_Prof:
    def __init__(self, y1, y2, r1, r2):
        self.start = y1
        self.end = y2

        self.m = (y2 - y1)/(r2 - r1)
        self.b = y1 - self.m*r1


    def __call__(self, r):
        return self.m * r + self.b
    
class Power_Law_Prof:
    def __init__(self, y1, y2, r1, r2, p):
        self.start = y1
        self.end = y2

        self.r1 = r1
        self.r2 = r2
        self.p = p

    def __call__(self, r):
        return self.start + (self.end - self.start) * ((r - self.r1)/(self.r2 - self.r1))**self.p
    
class SplineTwistProfile:
    """
    A flexible twist profile defined by control points (r, theta).
    Internally uses a 1D spline (cubic by default).
    """
    def __init__(self, r_ctrl, theta_ctrl, *,
                 kind='cubic',
                 extrapolate=True):
        """
        Parameters
        ----------
        r_ctrl : array_like
            Radii of control points, must be strictly increasing.
        theta_ctrl : array_like
            Twist angles (radians) at each r_ctrl.
        kind : {'cubic', 'pchip'}
            'cubic' for CubicSpline, 'pchip' for PchipInterpolator.
        extrapolate : bool
            If True, allow linear extrapolation outside [r_ctrl[0], r_ctrl[-1]].
        """
        self.start = theta_ctrl[0]
        self.end = theta_ctrl[-1]

        r_ctrl = np.asarray(r_ctrl)
        theta_ctrl = np.asarray(theta_ctrl)
        if r_ctrl.ndim != 1 or theta_ctrl.ndim != 1:
            raise ValueError("r_ctrl and theta_ctrl must be 1D arrays.")
        if len(r_ctrl) != len(theta_ctrl):
            raise ValueError("r_ctrl and theta_ctrl must have same length.")
        if np.any(np.diff(r_ctrl) <= 0):
            raise ValueError("r_ctrl must be strictly increasing.")
        
        if kind == 'cubic':
            # natural boundary conditions by default
            self._spline = CubicSpline(r_ctrl, theta_ctrl,
                                       extrapolate=extrapolate)
        elif kind == 'pchip':
            self._spline = PchipInterpolator(r_ctrl, theta_ctrl,
                                             extrapolate=extrapolate)
        else:
            raise ValueError("kind must be 'cubic' or 'pchip'")
        
        self.r_min, self.r_max = r_ctrl[0], r_ctrl[-1]

    def __call__(self, r):
        """
        Evaluate twist angle at r (scalar or array).
        If extrapolate=False, values outside [r_min, r_max] will be nan.
        """
        return self._spline(r)

class Max_Chord_Prof:
    def __init__(self, theta_prof, thickness):
        self.theta_prof = theta_prof
        self.thickness = thickness

        self.start = self.thickness/np.sin(theta_prof.start)
        self.end = self.thickness/np.sin(theta_prof.end)

    def __call__(self, r):
        theta = self.theta_prof(r)
        return self.thickness/np.sin(theta)
    
class Circ_Tip:
    def __init__(self, chord_prof, theta_prof, r_tip):
        self.chord_prof = chord_prof
        self.theta_prof = theta_prof
        self.r_tip = r_tip

        self.start = self.chord_prof.start
        self.end = self.chord_prof.end

    def __call__(self, r):
        chord = self.chord_prof(r)
        theta = self.theta_prof(r)
        
        chord_proj = np.cos(theta) * chord
        x = chord_proj/2
        y = r
        if x**2 + y**2 > self.r_tip**2:
            x = np.sqrt(self.r_tip**2 - y**2)
        
        chord_proj = 2 * x
        chord = chord_proj/np.cos(theta)

        return chord 

class Blade_Geometry:
    def __init__(self, airfoil_name, Ncrit, B, thickness, hub_diameter, od,
                 omega_rpm, theta_prof, CFM, r_calc_tol=.000):
        
        self.airfoil_name = airfoil_name
        self.Ncrit = Ncrit
        self.B = B
        self.thickness = thickness
        self.hub_diameter = hub_diameter
        self.od = od
        self.omega_rpm = omega_rpm
        self.omega = RPM_2_rad_s(omega_rpm)
        self.theta_prof = theta_prof
        self.LE_prof = Straight_LE_Sweep
        #max_p = thickness/np.sin(theta_prof(hub_diameter/2))*2
        max_p = .1
        self.TE_prof = Max_Width_TE_prof(od/2, thickness, self.theta_prof, self.LE_prof, max_p=max_p)
        self.set_CFM(CFM)

        self.r_vals = np.linspace(hub_diameter/2, od/2, 200)
        self.r_vals_no_tol = np.linspace(hub_diameter/2, od/2, 200)

        self.choord_start = self.get_arc_choord(hub_diameter/2)
        self.choord_end = self.get_arc_choord(od/2)
    
    def set_CFM(self, CFM):
        self.flow_rate = CFM_2_m3_s(CFM)
        self.v_freestream = self.flow_rate/self.od**2

    def get_arc_choord(self, r):
        x_LE, y_LE = self.LE_prof(r)
        x_TE, y_TE, z_TE = self.TE_prof(r)
        z_LE = 0
        u = np.array([x_LE, y_LE, 0.0])
        v = np.array([x_TE, y_TE, z_TE])

        # Central angle between u and v (robust, handles wrap)
        cross_norm = np.linalg.norm(np.cross(u, v))
        dot = np.dot(u, v)
        theta = np.arctan2(cross_norm, dot)      # in [0, pi]

        arc_len = r * theta
        return arc_len


    def plot_side_view_on_ax(self, ax, body_color='k',
                              num_radial=100,
                              blade_color='C0'):
        """
        Side view: blade thickness vs. span, improved styling.
        """
        # Radii
        r_hub = self.hub_diameter / 2.0
        r_tip = self.od / 2.0
        r_stations = np.linspace(r_hub, r_tip, num_radial)

        # Leading/Trailing edge curves
        LE = np.zeros((num_radial, 2))
        TE = np.zeros((num_radial, 2))
        for i, r in enumerate(r_stations):
            chord = float(self.coord_prof(r))
            theta = self.theta_prof(r)  # radians
            half_proj = chord * np.sin(theta) / 2
            LE[i] = [ half_proj, r]
            TE[i] = [-half_proj, r]

        # Plot blade edges
        ax.plot(LE[:,0], LE[:,1], color=blade_color, lw=1.5)
        ax.plot(TE[:,0], TE[:,1], color=blade_color, lw=1.5)

        # Draw hub and tip lines (body)
        x0, x1 = -self.thickness/2, self.thickness/2
        ys = [r_hub, -r_hub, r_tip, -r_tip]
        for y in ys:
            ax.hlines(y=y, xmin=x0, xmax=x1,
                      color=body_color, linestyle='-', linewidth=2)

        # Axes styling
        ax.set_aspect('equal', 'box')
        ax.set_xlim(x0*1.2, x1*1.2)
        ax.set_ylim(-r_tip*1.1, r_tip*1.1)
        ax.set_xlabel('Axial (z) [m]', fontsize=10)
        ax.set_ylabel('Span (y) [m]', fontsize=10)
        ax.set_title('Side view', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    @staticmethod
    def get_circ_lin_int(edge_curve, r):
        x1 = edge_curve[1, 0]
        x2 = edge_curve[2, 0]
        y1 = edge_curve[1, 1]
        y2 = edge_curve[2, 1]

        m = (y2 - y1)/(x2 - x1)
        y_int = y1 - m * x1

        a = 1+m**2
        b = 2*m*y_int
        c = y_int**2 - r**2

        D = b*b - 4*a*c
        root1 = (-b + np.sqrt(D)) / (2*a)
        root2 = (-b - np.sqrt(D)) / (2*a)

        return root1, root2

    def plot_views(self, save_path=None):
        """
        Master routine: front + side in a 1×2 subplot.
        """
        fig, (axF, axS) = plt.subplots(1, 2, figsize=(12, 6))

        # front
        axF.set_title(f"Front view: {self.B} blades, tip Ø={self.od:.2f} m")
        self._plot_front_view_on_ax(axF,
)

        # side
        self.plot_side_view_on_ax(axS)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=600)
        else:
            plt.show()

    def plot_side_view_on_ax(self, ax, body_color='k',
                              blade_color='C0'):
        """
        Side view: blade thickness vs. span, improved styling.
        """
        # Radii
        r_hub = self.hub_diameter / 2.0
        r_tip = self.od / 2.0
        r_stations = self.r_vals_no_tol

        # Leading/Trailing edge curves
        LE = np.zeros((r_stations.shape[0], 2))
        TE = np.zeros((r_stations.shape[0], 2))
        for i, r in enumerate(r_stations):
            x_LE, y_LE = self.LE_prof(r)
            x_TE, y_TE, z_TE = self.TE_prof(r)

            theta = self.theta_prof(r)  # radians

            LE[i] = [ z_TE/2, r]
            TE[i] = [-z_TE/2, r]

        # Plot blade edges
        ax.plot(LE[:,0], LE[:,1], color=blade_color, lw=1.5)
        ax.plot(TE[:,0], TE[:,1], color=blade_color, lw=1.5)
        ax.plot([TE[-1, 0], LE[-1, 0]], [TE[-1, 1], LE[-1, 1]], color=blade_color, lw=1.5)

        # Draw hub and tip lines (body)
        x0, x1 = -self.thickness/2, self.thickness/2
        ys = [r_hub, -r_hub, r_tip, -r_tip]
        for y in ys:
            ax.hlines(y=y, xmin=x0, xmax=x1,
                      color=body_color, linestyle='-', linewidth=2)


        ax.axhline(y=self.r_vals[0], color='orange', linestyle='--', linewidth=1)
        ax.axhline(y=self.r_vals[-1], color='orange', linestyle='--', linewidth=1)

        # Axes styling
        ax.set_aspect('equal', 'box')
        ax.set_xlim(x0*1.2, x1*1.2)
        ax.set_ylim(-r_tip*1.1, r_tip*1.1)
        ax.set_xlabel('Axial (z) [m]', fontsize=10)
        ax.set_ylabel('Span (y) [m]', fontsize=10)
        ax.set_title('Side view', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=9)

    def _plot_front_view_on_ax(self, ax,
                                blade_color='C0', circle_color='k'):
        """
        Front view: radial projection of blades, improved styling.
        """
        # Radii
        r_stations = self.r_vals_no_tol
        r_hub = r_stations[0]
        r_tip = r_stations[-1]

        # Draw hub and tip circles
        hub = plt.Circle((0,0), r_hub, fill=False,
                         color=circle_color, lw=1.5, label='Hub/Tip')
        tip = plt.Circle((0,0), r_tip, fill=False,
                         color=circle_color, lw=1.5)
        ax.add_patch(hub)
        ax.add_patch(tip)

        # Build LE/TE curves
        LE = np.zeros((self.r_vals_no_tol.shape[0], 2))
        TE = np.zeros((self.r_vals_no_tol.shape[0], 2))
        for i, r in enumerate(r_stations):
            x_LE, y_LE = self.LE_prof(r)
            x_TE, y_TE, _ = self.TE_prof(r)

        
    
            LE[i] = [x_LE, y_LE]
            TE[i] = [x_TE, y_TE]


        # Plot each blade
        for b in range(self.B):
            phi = 2*np.pi*b/self.B
            c,s = np.cos(phi), np.sin(phi)
            R = np.array([[c,-s],[s,c]])
            LE_r = (R @ LE.T).T
            TE_r = (R @ TE.T).T
            ax.plot(LE_r[:,0], LE_r[:,1], color=blade_color, lw=1.5)
            ax.plot(TE_r[:,0], TE_r[:,1], color=blade_color, lw=1.5)
            # close tip with circle centered at origin
            TE_tip = TE_r[-1]  # [x, y]
            LE_tip = LE_r[-1]  # [x, y]
            plot_tip_arc_center0(ax, TE_tip, LE_tip, blade_color=blade_color, lw=1.5,
                                center=(0.0, 0.0), use_outer_radius=True, prefer_major=False,
                                force_semicircle=False)

        ax.axhline(y=self.r_vals[0], color='orange', linestyle='--', linewidth=1)
        ax.axhline(y=self.r_vals[-1], color='orange', linestyle='--', linewidth=1)

        # Axes styling
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-r_tip*1.3, r_tip*1.3)
        ax.set_ylim(-r_tip*1.3, r_tip*1.3)
        ax.set_xlabel('x [m]', fontsize=10)
        ax.set_ylabel('y [m]', fontsize=10)
        ax.set_title(f'Front view: {self.B} blades', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=9)