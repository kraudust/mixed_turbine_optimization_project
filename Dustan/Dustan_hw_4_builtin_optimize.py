import numpy as np
from Main_Obj import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt

if __name__=="__main__":
	xHAWT = np.array([250, 260, 270, 350, 360, 370])
	yHAWT = np.array([0, 100, 200, 0, 100, 200])
	xVAWT = np.array([0, 350])
	yVAWT = np.array([175, 525])
	#xHAWT = np.array([0, 0, 0, 95, 95, 95])
	#yHAWT = np.array([0, 50, 100, 0, 50, 100])
	#xVAWT = np.array([50, 50])
	#yVAWT = np.array([0, 50])
	#set input variable xin
	xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])

	#set input parameters
	nVAWT = len(xVAWT)
	nHAWT = len(xHAWT)
	rh = 40.
	rv = 3.
	rt = 5.
	direction = 10. #wind coming from 
	dir_rad = (direction+90) * np.pi / 180.
	U_vel = 8.
	params = [nVAWT, rh, rv, rt, dir_rad, U_vel]
	print "Original Power: ", -obj(xin, params)*1.e6
	
	#-------------------------Optimization----------------------------------
	#set bound constraints
	xupper = 1000 #upper bound on x direction field coordinate
	yupper = 1000 #upper bound on y direction field coordinate
	bound_constraints = bounds(xin, params, xupper, yupper)
	ineq_constraints = {'type': 'ineq', 'fun': con, 'args': (params,)}
	options = {'disp': True, 'maxiter': 2000}
	res = minimize(obj, xin, args = (params,), method = 'SLSQP', jac = False, bounds = bound_constraints, constraints = ineq_constraints, tol = 1e-9, options = options)
	x_opt = res.x
	con_opt = con(x_opt, params)
	
	#--------------------------Make Plots-----------------------------------
	xVAWT_opt = x_opt[0 : nVAWT]
	yVAWT_opt = x_opt[nVAWT: 2*nVAWT]
	xHAWT_opt = x_opt[2*nVAWT: 2*nVAWT + nHAWT]
	yHAWT_opt = x_opt[2*nVAWT + nHAWT : len(xin)]
	
	plt.figure()
	plt.scatter(xVAWT, yVAWT, c = 'r', s = 60, label = "VAWT init")
	plt.scatter(xVAWT_opt, yVAWT_opt, c = 'r', s = 100, label = "VAWT opt")
	plt.scatter(xHAWT, yHAWT, c = 'b', s = 60, label = "HAWT init")
	plt.scatter(xHAWT_opt, yHAWT_opt, c = 'b', s = 100, label = "HAWT opt")
	plt.legend()
	plt.xlabel("x dir (m)")
	plt.ylabel("y dir (m)")
	plt.show()
	print "New Power: ", -obj(x_opt,params)*1.e6
