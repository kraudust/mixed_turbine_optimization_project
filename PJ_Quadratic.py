import numpy as np
from scipy.optimize import minimize
from Main_Obj import *


def optimizeQuadPenalty(x0, params, func, cons):
	rho = 1e-1
	args = (params, func, cons, rho)
	xin = x0
	difference = 1
	iterations = 1
	while abs(difference) > 1e-9:
		previous = quadPenalty(xin, args)
		res = minimize(quadPenalty, xin, args=(args,), method='BFGS', options={})
		rho = rho*2.
		difference = quadPenalty(res.x, args) - previous
		args = (params, func, cons, rho)
		xin = res.x
		print "Iterations Number: ", iterations
		iterations += 1
		print "Current Penalty Function Value: ", quadPenalty(xin, args)
		print "RHO: ", rho
	return xin


def test_func2(xin, params):
	return exp(xin[0]**21-12*xin[1]**2)+42*x[2]


def quadPenalty(xin, args):
	params, func, cons, rho = args
	f = func(xin, params)
	c = cons(xin, params)
	#for i in range(len(c)):
	#	if c[i] < 0:
	#		c[i] = -3e2
	sum = 0
	for i in range(len(c)):
		sum += (np.min([0, c[i]]))**2

	return f+rho*sum



def test_func(xin, params):
	sum = 0
	for i in range(len(xin)):
		sum += xin[i]**2
	return sum


def test_con(xin, params):
	constraints = np.zeros(np.size(xin))
	for i in range(len(xin)):
		constraints[i] = xin[i] - i
	return constraints


def withBounds(xin, params):
	constraints = con(xin, params)
	

	xupper = 2000
	yupper = 2000
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
	
	# for i in range(len(xVAWT)):
	#	constraints = np.append(constraints, xVAWT[i]/1.)
	#	constraints = np.append(constraints, xupper-xVAWT[i]/1.)
	#	constraints = np.append(constraints, yVAWT[i]/1.)
	#	constraints = np.append(constraints, yupper-yVAWT[i]/1.)
	#for i in range(len(xHAWT)):
	#	constraints = np.append(constraints, xHAWT[i]/1.)
	#	constraints = np.append(constraints, xupper-xHAWT[i]/1.)
	#	constraints = np.append(constraints, yHAWT[i]/1.)
	#	constraints = np.append(constraints, yupper-yHAWT[i]/1.)
	for i in range(len(xVAWT)):
		constraints = np.append(constraints, xVAWT[i])
		constraints = np.append(constraints, yVAWT[i])
		constraints = np.append(constraints, xupper-xVAWT[i])
		constraints = np.append(constraints, yupper-yVAWT[i])
	for i in range(len(xHAWT)):
		constraints = np.append(constraints, xHAWT[i])
		constraints = np.append(constraints, yHAWT[i])
		constraints = np.append(constraints, xupper-xHAWT[i])
		constraints = np.append(constraints, yupper-yHAWT[i])

	return constraints


if __name__=="__main__":
	"Define Variables"
	xHAWT = np.array([0, 0, 0, 500, 500, 500, 1000, 1000, 1000])
	yHAWT = np.array([0, 50, 100, 0, 500, 1000, 0, 500, 1000])
	xVAWT = np.array([])
	yVAWT = np.array([])

	xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
	nVAWT = len(xVAWT)
	rh = 40.
	rv = 3.
	rt = 5.
	direction = 10.
	dir_rad = (direction+90) * np.pi / 180.
	U_vel = 8.

	params = tuple([nVAWT, rh, rv, rt, dir_rad, U_vel])
	
	x0 = np.array([15,14,12,14])

	sol = optimizeQuadPenalty(xin, params, obj, withBounds)
	print "LOCATIONS: ", sol

	print "CONSTRAINT VALUES: ", withBounds(sol, params)
	print "START POWER: ", obj(x0, params)
	print "OPTIMIZED POWER: ", obj(sol, params)


	nVAWT = len(xVAWT)
	nHAWT = len(xHAWT)

	opt_xVAWT = sol[0 : nVAWT]
	opt_yVAWT = sol[nVAWT: 2*nVAWT]
	opt_xHAWT = sol[2*nVAWT: 2*nVAWT + nHAWT]
	opt_yHAWT = sol[2*nVAWT + nHAWT : len(sol)]

	plt.scatter(opt_xVAWT,opt_yVAWT,c='r', s=60,label='Optimized VAWT')
	plt.scatter(opt_xHAWT,opt_yHAWT,c='b', s=100,label='Optimized HAWT')
	plt.scatter(xHAWT,yHAWT, c='b', s=60, label='Start HAWT')
	plt.scatter(xVAWT,yVAWT, c='r', s=30, label='Start VAWT')
	# plt.legend(loc=2)
	plt.show()
	
