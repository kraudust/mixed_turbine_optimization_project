import numpy as np
from Main_Obj import *
from scipy.optimize import minimize

if __name__=="__main__":
	#xHAWT = np.array([0, 0, 0, 200, 200, 200, 400, 400, 400])
	#yHAWT = np.array([0, 200, 400, 0, 200, 400, 0, 200, 400])
	#xVAWT = np.array([100, 100, 100, 300, 300, 300])
	#yVAWT = np.array([100, 200, 300, 100, 200, 300])
	xHAWT = np.array([0, 0, 0, 500, 500, 500])
	yHAWT = np.array([0, 500, 1000, 0, 500, 1000])
	xVAWT = np.array([250, 250])
	yVAWT = np.array([250, 750])
	#set input variable xin
	xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])

	#set input parameters
	nVAWT = len(xVAWT)
	rh = 40.
	rv = 3.
	rt = 5.
	direction = 30.
	dir_rad = (direction+90) * np.pi / 180.
	U_vel = 8.
	params = [nVAWT, rh, rv, rt, dir_rad, U_vel]

	print "Original Power: ", obj(xin, params)
	#-------------------------Optimization----------------------------------
	#set bound constraints
	xupper = 2000 #upper bound on x direction field coordinate
	yupper = 1950 #upper bound on y direction field coordinate
	bound_constraints = bounds(xin, params, xupper, yupper)
	constraints = {'type': 'ineq', 'fun': con, 'args': params}
	options = {'dist': True, 'maxiter': 2000}
	res = minimize(obj, xin, args = params, method = 'SLSQP', jac = False, bounds = bound_constraints, constraints = constraints, tol = 1e-6, options = options)

    
