import numpy as np
from math import sqrt, log10
from math import pi
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from Main_Obj import *

def penalty(xin, params, mu, xupper, yupper):
	f = obj(xin, params)
	constraints = constraints_and_bounds(xin, params, xupper, yupper)
	in_sum = np.zeros(len(constraints))
	for i in range(len(constraints)):
		if constraints[i]>=-1e-7 and constraints[i]<=0:
			in_sum[i] = 0
		elif constraints[i] < 0:
			in_sum[i] = -10.
		else:
			in_sum[i] = log10(constraints[i])
	pi = f-mu*np.sum(in_sum)
	return pi

def constraints_and_bounds(xin, params, xupper, yupper):
	constraints = con(xin, params)
	
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
	
	for i in range(len(xVAWT)):
		constraints = np.append(constraints, xVAWT[i])
		constraints = np.append(constraints, xupper-xVAWT[i])
		constraints = np.append(constraints, yVAWT[i])
		constraints = np.append(constraints, yupper-yVAWT[i])
	for i in range(len(xHAWT)):
		constraints = np.append(constraints, xHAWT[i])
		constraints = np.append(constraints, xupper-xHAWT[i])
		constraints = np.append(constraints, yHAWT[i])
		constraints = np.append(constraints, yupper-yHAWT[i])

	return constraints


if __name__=="__main__":
	xHAWT = np.array([0, 0, 500, 500])
	yHAWT = np.array([0, 500, 0, 500])
	xVAWT = np.array([])
	yVAWT = np.array([])

	xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
	nVAWT = len(xVAWT)
	xupper = 1000
	yupper = 1000
	rh = 40.
	rv = 3.
	rt = 5.
	direction = 48.
	dir_rad = (direction+90) * np.pi / 180.
	U_vel = 8.

	params = tuple([nVAWT, rh, rv, rt, dir_rad, U_vel])
	mu = 1000.
	args = (params, mu, xupper, yupper)
	print "START: ", obj(xin, params)
	
	res = minimize(penalty, xin, args=args, method='BFGS', options = {'disp':True})
	print "FIRST: ", res.x
	print obj(res.x, params)
	difference = res.x
	i=1
	while np.max(abs(difference))>1e-6:
		previous = res.x[:]
		res = minimize(penalty, res.x, args=args, method='BFGS', options={'disp':True})
		mu = mu/2.
		args = (params, mu, xupper, yupper)
		difference = res.x-previous
		print "Iteration: ", i
		i = i+1
		print "Current Function Value: ", obj(res.x, params)

	print res.x
	print "Starting Power: ", obj(xin, params)
	print "Optimized Power: ", obj(res.x, params)


	nVAWT = len(xVAWT)
	nHAWT = len(xHAWT)

	opt_xVAWT = res.x[0 : nVAWT]
	opt_yVAWT = res.x[nVAWT: 2*nVAWT]
	opt_xHAWT = res.x[2*nVAWT: 2*nVAWT + nHAWT]
	opt_yHAWT = res.x[2*nVAWT + nHAWT : len(res.x)]

	plt.scatter(opt_xHAWT,opt_yHAWT,c='b', s=100,label='Optimized HAWT')
	plt.scatter(xHAWT,yHAWT, c='b', s=60, label='Start HAWT')
	plt.scatter(xVAWT,yVAWT, c='r', s=30, label='Start VAWT')
	# plt.legend(loc=2)
	plt.show()
