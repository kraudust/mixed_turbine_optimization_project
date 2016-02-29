import numpy as np
from Main_Obj import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
	
	#-------------------------Optimization----------------------------------
	#set bound constraints
	xupper = 2000 #upper bound on x direction field coordinate
	yupper = 1950 #upper bound on y direction field coordinate
	bound_constraints = bounds(xin, params, xupper, yupper)
	ineq_constraints = {'type': 'ineq', 'fun': con, 'args': (params,)}
	options = {'disp': True, 'maxiter': 2000}
	res = minimize(obj, xin, args = (params,), method = 'SLSQP', jac = False, bounds = bound_constraints, constraints = ineq_constraints, tol = 1e-6, options = options)
	x_opt = res.x
	con_opt = con(x_opt, params)
	print x_opt
	print con_opt
	
	#--------------------------Make Plots-----------------------------------
	xVAWT_opt = x_opt[0 : nVAWT]
	yVAWT_opt = x_opt[nVAWT: 2*nVAWT]
	xHAWT_opt = x_opt[2*nVAWT: 2*nVAWT + nHAWT]
	yHAWT_opt = x_opt[2*nVAWT + nHAWT : len(xin)]
	
	plt.figure()
	plt.scatter(xVAWT, yVAWT, c = 'k')
	plt.scatter(xVAWT_opt, yVAWT_opt, c = 'r')
	plt.scatter(xHAWT, yHAWT, c = 'c')
	plt.scatter(xHAWT_opt, yHAWT_opt, c = 'g')
	plt.show()
	
    
