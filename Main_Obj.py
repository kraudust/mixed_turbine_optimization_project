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
	nVAWT = params[0]
	rh = params[1]
	rv = params[2]
	rt = params[3]
	U_dir = params[4]
	U_vel = params[5]
	nHAWT = nTurbs - nVAWT #number of horizontal axis turbines

	#split xin into x and y locations for VAWT and HAWT
	xVAWT = xin[0 : nVAWT]
	yVAWT = xin[nVAWT: 2*nVAWT]
	xHAWT = xin[2*nVAWT: 2*nVAWT + nHAWT]
	yHAWT = xin[2*nVAWT + nHAWT : len(xin)]

	xrHAWT, yrHAWT = rotate(xHAWT, yHAWT, U_dir)
	xrVAWT, yrVAWT = rotate(xVAWT, yVAWT, U_dir)
	paramsHAWT = rh, U_vel
	paramsVAWT = rv, rt, U_vel

	#calculate power for VAWT and HAWT
	HAWT_Pow = Jensen_Wake_Model(xrHAWT, yrHAWT, paramsHAWT)
	VAWT_Pow = VAWT_Power(xrVAWT, yrVAWT, xrHAWT, yrHAWT, paramsVAWT)
	
	Ptotal = np.sum(HAWT_Pow) + np.sum(VAWT_Pow)
	return -Ptotal/1000000


def con(xin, params):
	nTurbs = len(xin)/2 #total number of turbines
	nVAWT = params[0]
	rh = params[1]
	rv = params[2]
	rt = params[3]
	U_dir = params[4]
	U_vel = params[5]
	nHAWT = nTurbs - nVAWT #number of horizontal axis turbines

	#split xin into x and y locations for VAWT and HAWT
	xVAWT = xin[0 : nVAWT]
	yVAWT = xin[nVAWT: 2*nVAWT]
	xHAWT = xin[2*nVAWT: 2*nVAWT + nHAWT]
	yHAWT = xin[2*nVAWT + nHAWT : len(xin)]

	constraints = np.array([])

	for i in range(len(xHAWT)):
		for j in range(len(xHAWT)):
			if i==j:
				constraints = np.append(constraints,0)
			else:
				dx = xHAWT[i]-xHAWT[j]
				dy = yHAWT[i]-yHAWT[j]
				constraints = np.append(constraints, dx**2+dy**2-64*rh**2)

	for i in range(len(xVAWT)):
		for j in range(len(xVAWT)):
			if i==j:
				constraints = np.append(constraints,0)
			else:
				dx = xVAWT[i]-xVAWT[j]
				dy = yVAWT[i]-yVAWT[j]
				constraints = np.append(constraints, dx**2+dy**2-64*rv**2)

	for i in range(len(xVAWT)):
		for j in range(len(xHAWT)):
			dx = xVAWT[i]-xHAWT[j]
			dy = yVAWT[i]-yHAWT[j]
			constraints = np.append(constraints, dx**2+dy**2-16*rh**2)

	return constraints/100000


def bounds(xin, params, xupper, yupper):
	#bounds = numpy array with upper and lower bound on each turbine
	nTurbs = len(xin)/2 #total number of turbines
	nVAWT, rh, rv, rt, U_dir, U_vel = params
	nVAWT = int(nVAWT)
	nHAWT = nTurbs - nVAWT #number of horizontal axis turbines
	
	bounds = np.zeros((nTurbs*2,2))
	boundx = np.array([0,xupper])
	boundy = np.array([0,yupper])
	for i in range(0, nVAWT):
		bounds[i] = boundx
	for i in range(nVAWT, 2*nVAWT):
		bounds[i] = boundy
	for i in range(2*nVAWT, 2*nVAWT + nHAWT):
		bounds[i] = boundx
	for i in range(2*nVAWT + nHAWT, len(xin)):
		bounds[i] = boundy
	return bounds	
	
if __name__=="__main__":
	xHAWT = np.array([0, 200, 400])
	yHAWT = np.array([0, 200, 400])
	xVAWT = np.array([100, 100, 100])
	yVAWT = np.array([100, 200, 300])

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
	print con(xin, params)
	print bounds(xin, params,2000, 1500)
