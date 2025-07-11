
from unit_conversion import *
import numpy as np

class Linear_Prof:
    def __init__(self, y1, y2, r1, r2):

        self.m = (y2 - y1)/(r2 - r1)
        self.b = y1 - self.m*r1


    def __call__(self, r):
        return self.m * r + self.b


class Blade_Geometry:
    def __init__(self, airfoil_name, Ncrit, B, thickness, hub_diameter, od,
                 omega_rpm, coord_prof, theta_prof, CFM):
        
        self.airfoil_name = airfoil_name
        self.Ncrit = Ncrit
        self.B = B
        self.thickness = thickness
        self.hub_diameter = hub_diameter
        self.od = od
        self.omega = RPM_2_rad_s(omega_rpm)
        self.coord_prof = coord_prof
        self.theta_prof = theta_prof
        self.flow_rate = CFM_2_m3_s(CFM)
        self.v_freestream = self.flow_rate/self.od**2
