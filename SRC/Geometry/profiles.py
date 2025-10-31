from scipy.interpolate import CubicSpline, PchipInterpolator
from abc import ABC, abstractmethod
import numpy as np

class Profile(ABC):
    @abstractmethod
    def __call__(self, r: float | np.ndarray) -> float | np.ndarray:
        pass

class Linear_Prof(Profile):
    """
    Linear profile in r: y(r) = m r + b with y(r1)=y1, y(r2)=y2.
    """
    def __init__(self, y1: float, y2: float, r1: float, r2: float):
        self.start = y1
        self.end = y2

        self.m = (y2 - y1) / (r2 - r1)
        self.b = y1 - self.m * r1

    def __call__(self, r: float | np.ndarray) -> float | np.ndarray:
        return self.m * r + self.b

class Power_Law_Prof(Profile):
    """
    Power-law profile in normalized radius:
    y(r) = start + (end - start) * ((r - r1)/(r2 - r1))**p
    """
    def __init__(self, y1: float, y2: float, r1: float, r2: float, p: float):
        self.start = y1
        self.end = y2
        self.r1 = r1
        self.r2 = r2
        self.p = p

    def __call__(self, r: float | np.ndarray) -> float | np.ndarray:
        return self.start + (self.end - self.start) * ((r - self.r1) / (self.r2 - self.r1)) ** self.p

class SplineProfile(Profile):
    """
    A flexible twist profile defined by control points (r, theta),
    implemented via CubicSpline or PCHIP (monotone shape-preserving).
    """
    def __init__(
        self,
        r_ctrl,
        theta_ctrl,
        *,
        kind: str = 'cubic',
        extrapolate: bool = True
    ):
        """
        Parameters
        ----------
        r_ctrl : array_like
            Radii of control points, strictly increasing.
        theta_ctrl : array_like
            Twist angles (radians) at each control radius.
        kind : {'cubic', 'pchip'}
            'cubic' uses CubicSpline; 'pchip' uses PchipInterpolator.
        extrapolate : bool
            If True, allow extrapolation outside the control range.
        """
        self.start = theta_ctrl[0]
        self.end = theta_ctrl[-1]

        r_ctrl = np.asarray(r_ctrl)
        theta_ctrl = np.asarray(theta_ctrl)
        if r_ctrl.ndim != 1 or theta_ctrl.ndim != 1:
            raise ValueError("r_ctrl and theta_ctrl must be 1D arrays.")
        if len(r_ctrl) != len(theta_ctrl):
            raise ValueError("r_ctrl and theta_ctrl must have same length.")
        #if np.any(np.diff(r_ctrl) <= 0):
        #    raise ValueError("r_ctrl must be strictly increasing.")

        if kind == 'cubic':
            # Natural boundary conditions by default
            self._spline = CubicSpline(r_ctrl, theta_ctrl, extrapolate=extrapolate)
        elif kind == 'pchip':
            self._spline = PchipInterpolator(r_ctrl, theta_ctrl, extrapolate=extrapolate)
        else:
            raise ValueError("kind must be 'cubic' or 'pchip'")

        self.r_min, self.r_max = r_ctrl[0], r_ctrl[-1]

    def __call__(self, r: float | np.ndarray) -> float | np.ndarray:
        """Evaluate twist angle at r (scalar or array)."""
        return self._spline(r)

