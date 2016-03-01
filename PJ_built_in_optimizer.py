import numpy as np
from math import sqrt
from math import pi
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from Main_Obj import *

if __name__=="__main__":
	"Define Variables"
	xHAWT = np.array([0, 0])
	yHAWT = np.array([0, 500])
	xVAWT = np.array([])
	yVAWT = np.array([])

	xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
	nVAWT = len(xVAWT)
	rh = 40.
	rv = 3.
	rt = 5.
	direction = 2
	dir_rad = (direction+90) * np.pi / 180.
	U_vel = 8.

	params = tuple([nVAWT, rh, rv, rt, dir_rad, U_vel])
	

	boundaries = bounds(xin, params, 1000, 1000)
	
	options = {'disp': True, 'maxiter': 1000}
	constraints = {'type': 'ineq', 'fun': con, 'args': (params,)}

	print "Start Power: ", obj(xin, params)
	print "Start Constraints: ", con(xin, params)

	res = minimize(obj, xin, args=(params,), method='SLSQP', jac=False, bounds=boundaries, tol=1e-9, constraints=constraints, options=options)
	
	
	print "Optimized Power: ", obj(res.x, params)
	print "Optimized Constraints: ", con(res.x, params)
	print res.x
	
	nVAWT = len(xVAWT)
	nHAWT = len(xHAWT)

	opt_xVAWT = res.x[0 : nVAWT]
	opt_yVAWT = res.x[nVAWT: 2*nVAWT]
	opt_xHAWT = res.x[2*nVAWT: 2*nVAWT + nHAWT]
	opt_yHAWT = res.x[2*nVAWT + nHAWT : len(res.x)]

	plt.scatter(opt_xVAWT,opt_yVAWT,c='r', s=60,label='Optimized VAWT')
	plt.scatter(opt_xHAWT,opt_yHAWT,c='b', s=100,label='Optimized HAWT')
	plt.scatter(xHAWT,yHAWT, c='b', s=60, label='Start HAWT')
	plt.scatter(xVAWT,yVAWT, c='r', s=30, label='Start VAWT')
	# plt.legend(loc=2)
	plt.show()

