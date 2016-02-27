import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi, tan, cos, acos, sin, sqrt
from Jensen import *
from Tower_Wake import *
from VAWT_loss_calculation import *

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
	paramsVAWT = rVAWT, rTOWER, U_vel
	
	#calculate power for VAWT and HAWT
    HAWT_Power = Jensen_Wake_Model(xrHAWT, yrHAWT, paramsHAWT)
    VAWT_Power = VAWT_Power(xrVAWT, yrVAWT, xrHAWT, yrHAWT, paramsVAWT)
	
	Ptotal = np.sum(HAWT_Power) + np.sum(VAWT_Power)
    return Ptotal

if __name__=="__main__"
    
    
