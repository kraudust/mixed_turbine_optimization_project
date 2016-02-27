import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi, tan, cos, acos, sin, sqrt
from Jensen import *
from Tower_Wake import *
from VAWT_power_calculation import *

def obj(xin,params):
    """
	inputs:
		xin - nparray with x, and y locations for each turbine. Form is [xVAWT, yVAWT, xHAWT, yHAWT]
		params - parameters that are kept constant
			nVAWT - number of vertical axis turbines
			rh - radius of horizontal axis turbines
			rv - radius of vertical axis turbines
			rt - radius of tower
			U_dir - wind direction in radians
			U_vel - free stream velocity magnitude
    :return:
    	Ptotal: total power output
    """
    nTurbs = len(xin)/2 #total number of turbines
    nVAWT, rh, rv, rt, U_dir, U_vel = params
    nHAWT = nTurbs - nVAWT #number of horizontal axis turbines
    
    #split xin into x and y locations for VAWT and HAWT
    xVAWT = xin[0 : nVAWT]
    yVAWT = xin[nVAWT: 2*nVAWT]
    xHAWT = xin[nTurbs: nTurbs + nHAWT]
    yHAWT = xin[nTurbs + nVAWT : len(xin)]
    
    xrHAWT, yrHAWT = rotate(xHAWT, yHAWT, U_dir)
    xrVAWT, yrVAWT = rotate(xVAWT, yVAWT, U_dir)
    paramsHAWT = rh, U_vel
    paramsVAWT = rv, rt, U_vel
	
	#calculate power for VAWT and HAWT
    HAWT_Pow = Jensen_Wake_Model(xrHAWT, yrHAWT, paramsHAWT)
    VAWT_Pow = VAWT_Power(xrVAWT, yrVAWT, xrHAWT, yrHAWT, paramsVAWT)
	
    Ptotal = np.sum(HAWT_Pow) + np.sum(VAWT_Pow)
    return Ptotal

if __name__=="__main__":
    xHAWT = np.array([0, 0, 0, 200, 200, 200, 400, 400, 400])
    yHAWT = np.array([0, 200, 400, 0, 200, 400, 0, 200, 400])
    xVAWT = np.array([100, 100, 100, 300, 300, 300])
    yVAWT = np.array([100, 200, 300, 100, 200, 300])

    xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
    nVAWT = len(xVAWT)
    rh = 40.
    rv = 3.
    rt = 5.
    direction = 30.
    dir_rad = (direction+90) * np.pi / 180.
    U_vel = 8.

    params = [nVAWT, rh, rv, rt, dir_rad, U_vel]

    print obj(xin, params)
