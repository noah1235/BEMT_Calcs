import numpy as np
import pandas as pd
class Airfoil_Data:
    """
    Airfoil lookup and (bilinear) interpolation wrapper.

    Expects a DataFrame with columns:
        ['Re', 'Ncrit', 'alpha', 'CL', 'CD']

    Interpolation logic (kept exactly):
    1) Choose the two Re grid points bracketing the query Re (clamp to ends).
    2) For each of those two Re slices, 1D interpolate CL, CD vs alpha using np.interp.
       (Assumes 'alpha' within each Re-slice is strictly increasing.)
    3) Linearly interpolate the two results across Re.
    """
    def __init__(self, data: pd.DataFrame, Ncrit: float):
        # Filter to the requested Ncrit (exact float equality as in the original)
        df = data[data["Ncrit"] == Ncrit].copy()
        if df.empty:
            raise ValueError(f"No data for Ncrit={Ncrit}")

        # Unique, sorted Reynolds numbers for bracketing
        self.Res = np.sort(df["Re"].unique())
        self.data = df  # store filtered table for later lookups

    def __call__(self, Re: float, alpha: float) -> tuple[float, float]:
        """
        Parameters
        ----------
        Re : float
            Reynolds number at which to query.
        alpha : float
            Angle of attack IN DEGREES (kept as-is; caller passes degrees).

        Returns
        -------
        (CL, CD) : tuple of floats
        """
        # --- 1) Bracket Re (clamped to [min, max]) ---
        if Re <= self.Res[0]:
            Re_lo = Re_hi = self.Res[0]
        elif Re >= self.Res[-1]:
            Re_lo = Re_hi = self.Res[-1]
        else:
            idx = np.searchsorted(self.Res, Re)
            Re_lo, Re_hi = self.Res[idx - 1], self.Res[idx]

        Re_lo = float(Re_lo)
        Re_hi = float(Re_hi)

        # --- 2) Interpolate within each bracket slice vs alpha ---
        def interp_at_Re(R: float) -> tuple[float, float]:
            subset = self.data[self.data["Re"] == R]
            a_arr = subset["alpha"].to_numpy()  # assumed strictly increasing
            cl_arr = subset["CL"].to_numpy()
            cd_arr = subset["CD"].to_numpy()

            # 1D linear interpolation in alpha
            cl_val = np.interp(alpha, a_arr, cl_arr)
            cd_val = np.interp(alpha, a_arr, cd_arr)
            return cl_val, cd_val

        cl_lo, cd_lo = interp_at_Re(Re_lo)
        cl_hi, cd_hi = interp_at_Re(Re_hi)

        # --- 3) Linear interpolation across Re ---
        if Re_lo == Re_hi:
            return float(cl_lo), float(cd_lo)

        t = (Re - Re_lo) / (Re_hi - Re_lo)
        cl = cl_lo + t * (cl_hi - cl_lo)
        cd = cd_lo + t * (cd_hi - cd_lo)
        return float(cl), float(cd)

