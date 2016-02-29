import numpy as np
from Main_Obj import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def calc_constraints(xin, params, xupper, yupper):
	xVAWT = xin[0 : nVAWT]
	yVAWT = xin[nVAWT: 2*nVAWT]
	xHAWT = xin[2*nVAWT: 2*nVAWT + nHAWT]
	yHAWT = xin[2*nVAWT + nHAWT : len(xin)]
	const = con(xin, params)
	for i in range(0, len(xVAWT)):
		const = np.append(const, xVAWT)
		const = np.append(const, xupper - xVAWT)
		const = np.append(const, yVAWT)
		const = np.append(const, yupper - yVAWT)
	for i in range(0, len(xHAWT)):
		const = np.append(const, xHAWT)
		const = np.append(const, xupper - xHAWT)
		const = np.append(const, yVAWT)
		const = np.append(const, yupper - yHAWT)
	return const
	
def uncon_obj(xin, params, xupper, yupper):
	const = calc_constraints(xin, params, xupper, yupper)
	sum_log = 0
	for i in range(0, const):
		if const[i] == 0:
			ind_log = 0
		if const[i] <= 0:
			ind_log = -1000.
		ind_log = np.log10(const[i])
		sum_log = sum_log + ind_log
	new_obj = obj(xin, params) - mu*sum_log
	return new_obj
	
if __name__=="__main__":
	xHAWT = np.array([0, 0, 0, 300, 300, 300])
	yHAWT = np.array([0, 150, 300, 0, 150, 300])
	xVAWT = np.array([150, 75])
	yVAWT = np.array([150, 225])
	#set input variable xin
	xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])

	#set input parameters
	nVAWT = len(xVAWT)
	nHAWT = len(xHAWT)
	rh = 40.
	rv = 3.
	rt = 5.
	direction = 30.
	dir_rad = (direction+90) * np.pi / 180.
	U_vel = 8.
	params = [nVAWT, rh, rv, rt, dir_rad, U_vel]
	print "Original Power: ", obj(xin, params)
	
	
	
	
