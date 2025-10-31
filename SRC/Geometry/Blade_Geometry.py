from SRC.unit_conversion import RPM_2_rad_s, CFM_2_m3_s
from SRC.Geometry.TE_LE_profiles import Straight_LE_Sweep, Max_Width_TE_prof
import numpy as np
import matplotlib.pyplot as plt

class Blade_Geometry:
    """
    Container for blade geometry definitions (LE/TE, twist, tip radius) and plotting utilities.
    """
    def __init__(
        self,
        airfoil_name: str,
        Ncrit: float,
        B: int,
        thickness_prof,
        max_t: float,
        hub_diameter: float,
        od: float,
        omega_rpm: float,
        theta_prof,
        CFM: float,
    ):
        self.airfoil_name = airfoil_name
        self.Ncrit = Ncrit
        self.B = B
        self.max_t = max_t
        self.hub_diameter = hub_diameter
        self.od = od
        self.omega_rpm = omega_rpm
        self.omega = RPM_2_rad_s(omega_rpm)
        self.theta_prof = theta_prof
        self.LE_prof = Straight_LE_Sweep


        self.TE_prof = Max_Width_TE_prof(od / 2, thickness_prof, self.theta_prof, self.LE_prof)

        # Flow specification
        self.set_CFM(CFM)

        # Radial grids
        self.r_vals = np.linspace(hub_diameter / 2, od / 2, 200)

        self.chord_start = self.get_arc_chord(hub_diameter / 2)
        self.chord_end = self.get_arc_chord(od / 2)

    def set_CFM(self, CFM: float) -> None:
        """
        Set flow-related values. Uses od**2 as an area proxy (retained).
        """
        self.flow_rate = CFM_2_m3_s(CFM)   # external helper
        self.v_freestream = self.flow_rate / self.od**2

    def get_arc_chord(self, r: float) -> float:
        """
        Compute arc length between LE and TE at radius r by central angle.

        Returns
        -------
        float
            Arc length r * theta, where theta is the central angle between
            3D vectors u=[x_LE,y_LE,0] and v=[x_TE,y_TE,z_TE].
        """
        x_LE, y_LE = self.LE_prof(r)
        x_TE, y_TE, z_TE = self.TE_prof(r)

        # 3D vectors for cross/dot computation
        u = np.array([x_LE, y_LE, 0.0])
        v = np.array([x_TE, y_TE, z_TE])

        cross_norm = np.linalg.norm(np.cross(u, v))
        dot = np.dot(u, v)
        theta = np.arctan2(cross_norm, dot)  # angle in [0, π]

        arc_len = r * theta
        return arc_len

    # -------------------- FIRST plot_side_view_on_ax (kept, but overridden later) --------------------
    def plot_side_view_on_ax(self, ax, body_color='k', num_radial=100, blade_color='C0'):
        """
        Side view (deprecated/overridden). Retained to avoid changing behavior.
        """
        # Radii
        r_hub = self.hub_diameter / 2.0
        r_tip = self.od / 2.0
        r_stations = np.linspace(r_hub, r_tip, num_radial)

        # Leading/Trailing edge curves (would fail if executed due to self.coord_prof)
        LE = np.zeros((num_radial, 2))
        TE = np.zeros((num_radial, 2))
        for i, r in enumerate(r_stations):
            chord = float(self.coord_prof(r))  # NOTE: coord_prof not defined in class
            theta = self.theta_prof(r)
            half_proj = chord * np.sin(theta) / 2
            LE[i] = [half_proj, r]
            TE[i] = [-half_proj, r]

        ax.plot(LE[:, 0], LE[:, 1], color=blade_color, lw=1.5)
        ax.plot(TE[:, 0], TE[:, 1], color=blade_color, lw=1.5)

        # Hub/tip lines
        x0, x1 = -self.max_t / 2, self.max_t / 2
        ys = [r_hub, -r_hub, r_tip, -r_tip]
        for y in ys:
            ax.hlines(y=y, xmin=x0, xmax=x1, color=body_color, linestyle='-', linewidth=2)

        ax.set_aspect('equal', 'box')
        ax.set_xlim(x0 * 1.2, x1 * 1.2)
        ax.set_ylim(-r_tip * 1.1, r_tip * 1.1)
        ax.set_xlabel('Axial (z) [m]', fontsize=10)
        ax.set_ylabel('Span (y) [m]', fontsize=10)
        ax.set_title('Side view', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=9)
    # -------------------------------------------------------------------------------------------------

    @staticmethod
    def get_circ_lin_int(edge_curve: np.ndarray, r: float) -> tuple[float, float]:
        """
        Intersection x-coordinates of a line (through edge_curve[1:3]) with a circle x^2 + y^2 = r^2.

        Parameters
        ----------
        edge_curve : ndarray, shape (>=3, 2)
            Consecutive points defining an edge. Uses points [1] and [2] to define the line.
        r : float
            Circle radius.

        Returns
        -------
        (x1, x2) : tuple[float, float]
            Intersection x roots with the circle.
        """
        x1 = edge_curve[1, 0]
        x2 = edge_curve[2, 0]
        y1 = edge_curve[1, 1]
        y2 = edge_curve[2, 1]

        m = (y2 - y1) / (x2 - x1)
        y_int = y1 - m * x1

        # Solve (x^2 + (m x + y_int)^2) = r^2
        a = 1 + m**2
        b = 2 * m * y_int
        c = y_int**2 - r**2

        D = b * b - 4 * a * c
        root1 = (-b + np.sqrt(D)) / (2 * a)
        root2 = (-b - np.sqrt(D)) / (2 * a)
        return root1, root2

    def plot_views(self, save_path: str | None = None):
        """
        Master routine: front + side in a 1×2 layout.
        """
        fig, (axF, axS) = plt.subplots(1, 2, figsize=(12, 6))

        # Front view
        axF.set_title(f"Front view: {self.B} blades, tip Ø={self.od:.2f} m")
        self._plot_front_view_on_ax(axF)

        # Side view (calls the *second* definition below)
        self.plot_side_view_on_ax(axS)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=600)
        else:
            plt.show()

    # -------------------- SECOND plot_side_view_on_ax (effective one) --------------------
    def plot_side_view_on_ax(self, ax, body_color='k', blade_color='C0'):
        """
        Side view: blade thickness vs. span using TE_prof axial thickness (z_TE/2 up/down).
        """
        r_hub = self.hub_diameter / 2.0
        r_tip = self.od / 2.0
        r_stations = self.r_vals

        # Build LE/TE curves from TE_prof axial thickness
        LE = np.zeros((r_stations.shape[0], 2))
        TE = np.zeros((r_stations.shape[0], 2))
        for i, r in enumerate(r_stations):
            x_LE, y_LE = self.LE_prof(r)
            x_TE, y_TE, z_TE = self.TE_prof(r)
            theta = self.theta_prof(r)  # retained (unused)

            LE[i] = [z_TE / 2, r]
            TE[i] = [-z_TE / 2, r]

        ax.plot(LE[:, 0], LE[:, 1], color=blade_color, lw=1.5)
        ax.plot(TE[:, 0], TE[:, 1], color=blade_color, lw=1.5)
        ax.plot([TE[-1, 0], LE[-1, 0]], [TE[-1, 1], LE[-1, 1]], color=blade_color, lw=1.5)

        # Hub/tip body lines
        x0, x1 = -self.max_t / 2, self.max_t / 2
        ys = [r_hub, -r_hub, r_tip, -r_tip]
        for y in ys:
            ax.hlines(y=y, xmin=x0, xmax=x1, color=body_color, linestyle='-', linewidth=2)

        # Visual guides for first/last r in r_vals
        ax.axhline(y=self.r_vals[0], color='orange', linestyle='--', linewidth=1)
        ax.axhline(y=self.r_vals[-1], color='orange', linestyle='--', linewidth=1)

        ax.set_aspect('equal', 'box')
        ax.set_xlim(x0 * 1.2, x1 * 1.2)
        ax.set_ylim(-r_tip * 1.1, r_tip * 1.1)
        ax.set_xlabel('Axial (z) [m]', fontsize=10)
        ax.set_ylabel('Span (y) [m]', fontsize=10)
        ax.set_title('Side view', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=9)
    # -------------------------------------------------------------------------------------

    def _plot_front_view_on_ax(self, ax, blade_color='C0', circle_color='k'):
        """
        Front view: radial projection of blades with hub/tip circles and LE/TE curves.
        """
        r_stations = self.r_vals
        r_hub = r_stations[0]
        r_tip = r_stations[-1]

        # Hub and tip circles
        hub = plt.Circle((0, 0), r_hub, fill=False, color=circle_color, lw=1.5, label='Hub/Tip')
        tip = plt.Circle((0, 0), r_tip, fill=False, color=circle_color, lw=1.5)
        ax.add_patch(hub)
        ax.add_patch(tip)

        # Build LE/TE arrays across stations
        LE = np.zeros((self.r_vals.shape[0], 2))
        TE = np.zeros((self.r_vals.shape[0], 2))
        for i, r in enumerate(r_stations):
            x_LE, y_LE = self.LE_prof(r)
            x_TE, y_TE, _ = self.TE_prof(r)
            LE[i] = [x_LE, y_LE]
            TE[i] = [x_TE, y_TE]

        # Draw each blade by rotating LE/TE curves
        for b in range(self.B):
            phi = 2 * np.pi * b / self.B
            c, s = np.cos(phi), np.sin(phi)
            R = np.array([[c, -s], [s, c]])

            LE_r = (R @ LE.T).T
            TE_r = (R @ TE.T).T

            ax.plot(LE_r[:, 0], LE_r[:, 1], color=blade_color, lw=1.5)
            ax.plot(TE_r[:, 0], TE_r[:, 1], color=blade_color, lw=1.5)

            # Close the tip with an arc centered at origin
            TE_tip = TE_r[-1]
            LE_tip = LE_r[-1]
            self.plot_tip_arc_center0(
                ax,
                TE_tip,
                LE_tip,
                blade_color=blade_color,
                lw=1.5,
                center=(0.0, 0.0),
                use_outer_radius=True,
                prefer_major=False,
                force_semicircle=False
            )

        # Visual guides for first/last r in r_vals
        ax.axhline(y=self.r_vals[0], color='orange', linestyle='--', linewidth=1)
        ax.axhline(y=self.r_vals[-1], color='orange', linestyle='--', linewidth=1)

        ax.set_aspect('equal', 'box')
        ax.set_xlim(-r_tip * 1.3, r_tip * 1.3)
        ax.set_ylim(-r_tip * 1.3, r_tip * 1.3)
        ax.set_xlabel('x [m]', fontsize=10)
        ax.set_ylabel('y [m]', fontsize=10)
        ax.set_title(f'Front view: {self.B} blades', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=9)

    @staticmethod
    def plot_tip_arc_center0(
        ax: plt.Axes,
        TE_tip, LE_tip,
        blade_color: str = 'k',
        lw: float = 1.5,
        n: int = 200,
        center: tuple[float, float] = (0.0, 0.0),
        r: float | None = None,
        use_outer_radius: bool = True,
        prefer_major: bool = False,
        force_semicircle: bool = False
    ):
        """
        Draw a circular arc centered at `center` connecting the blade tip between LE and TE.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        TE_tip, LE_tip : array_like shape (2,)
            Tip endpoints for trailing and leading edges.
        blade_color : str
            Line color.
        lw : float
            Line width.
        n : int
            Number of sample points along the arc.
        center : tuple[float, float]
            Center of the arc/circle.
        r : float or None
            Fixed radius to use. If None, uses either max or mean of endpoint radii.
        use_outer_radius : bool
            If r is None, choose max (True) or mean (False) of endpoint radii.
        prefer_major : bool
            Use the major arc instead of the minor arc if True.
        force_semicircle : bool
            If True, draws exactly a 180° cap centered on the mid-angle (endpoints
            may not pass exactly through TE/LE unless diametrically opposite).
        """
        TE_tip = np.asarray(TE_tip, dtype=float)
        LE_tip = np.asarray(LE_tip, dtype=float)
        cx, cy = center

        # Shift coordinates so circle center is origin
        TE0 = TE_tip - np.array([cx, cy])
        LE0 = LE_tip - np.array([cx, cy])

        # Endpoint radii
        r_TE = np.hypot(TE0[0], TE0[1])
        r_LE = np.hypot(LE0[0], LE0[1])
        if r is None:
            r = max(r_TE, r_LE) if use_outer_radius else 0.5 * (r_TE + r_LE)

        # Endpoint angles
        th_TE = np.arctan2(TE0[1], TE0[0])
        th_LE = np.arctan2(LE0[1], LE0[0])

        def wrap_pi(a: float) -> float:
            # Wrap to (-π, π]
            return (a + np.pi) % (2 * np.pi) - np.pi

        if force_semicircle:
            # Compute mid-angle using unit vector average for stability near wrap-around
            u_TE = TE0 / (r_TE + 1e-12)
            u_LE = LE0 / (r_LE + 1e-12)
            u_mid = u_TE + u_LE
            th_mid = th_TE if np.allclose(u_mid, 0.0) else np.arctan2(u_mid[1], u_mid[0])
            th_start = th_mid - np.pi / 2
            th_end = th_mid + np.pi / 2
            th = np.linspace(th_start, th_end, n)
        else:
            # Connect TE -> LE along chosen arc (minor by default, major if requested)
            dtheta = wrap_pi(th_LE - th_TE)
            if prefer_major:
                # Flip to the longer path
                dtheta = dtheta - 2 * np.pi if dtheta >= 0 else dtheta + 2 * np.pi
            th = th_TE + np.linspace(0.0, dtheta, n)

        x = cx + r * np.cos(th)
        y = cy + r * np.sin(th)
        ax.plot(x, y, color=blade_color, lw=lw)


