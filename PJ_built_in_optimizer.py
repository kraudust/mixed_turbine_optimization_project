import numpy as np
from math import sqrt
from math import pi
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from Main_Obj import *

if __name__=="__main__":
	"Define Variables"
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

	params = nVAWT, rh, rv, rt, dir_rad, U_vel
	

	print obj(xin, params)
	print con(xin, params)

	boundaries = bounds(xin, params, 1500, 1500)
	print boundaries
	options = {'disp': True}
	constraints = {'type': 'ineq', 'fun': con, 'args': (params)}
	res = minimize(obj, xin, args=[params], method='SLSQP', jac=False, bounds=boundaries, tol=1e-6, constraints=constraints, options=options)

	print res.x
	print con(res.x)

	xcoordinates = np.zeros(nTurbines)
	ycoordinates = np.zeros(nTurbines)

	xcoordinates = res.x[0:nTurbines]
	ycoordinates = res.x[nTurbines:nTurbines*2.]

	plt.scatter(x,y,label='Start')
	plt.scatter(xcoordinates,ycoordinates, c='r', label='Optimized')
	plt.legend()
	plt.show()

	print "Start Power: ", Jensen_Wake_Model(xin, params)
	print "Optimized Power: ", Jensen_Wake_Model(res.x, params)
