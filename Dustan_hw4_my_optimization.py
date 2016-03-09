import numpy as np
from Main_Obj import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def calc_constraints(xin, params):
	xupper = 1000
	yupper = 1000
	
	#split into x and y loactions for VAWT and HAWT
	xVAWT = xin[0 : nVAWT]
	yVAWT = xin[nVAWT: 2*nVAWT]
	xHAWT = xin[2*nVAWT: 2*nVAWT + nHAWT]
	yHAWT = xin[2*nVAWT + nHAWT : len(xin)]
	const = con(xin, params)
	for i in range(0, len(xVAWT)):
		const = np.append(const, xVAWT[i])
		const = np.append(const, xupper - xVAWT[i])
		const = np.append(const, yVAWT[i])
		const = np.append(const, yupper - yVAWT[i])
	for i in range(0, len(xHAWT)):
		const = np.append(const, xHAWT[i])
		const = np.append(const, xupper - xHAWT[i])
		const = np.append(const, yHAWT[i])
		const = np.append(const, yupper - yHAWT[i])
	return const
	
def penalty_obj(xin, args):
	"""
	Inputs:
		xin: nparray of form -  (xVAWT, yVAWT, xHAWT, yHAWT)
		args: nparray containing:
			nVAWT - number of vertical axis turbines
			rh - radius of horizontal axis turbines
			rv - radius of vertical axis turbines
			rt - radius of tower
			U_dir - wind direction in radians
			U_vel - free stream velocity magnitude
			mu - scaling parameter to penalize constraint violations
			obj_han - objective function handle
			con_han - constraint function handle
	Returns:
		new_obj: previous objective function plus a log penalty on constraints
	"""
	params = args[0]
	mu = args[1]
	obj_han = args[2]
	con_han = args[3]
	const = con_han(xin, params)
	sum_quad = 0
	for i in range(0, len(const)):
	#	if const[i] <= 0:
	#		const[i] = -1000 #really penalize bad constraints
		sum_quad += np.min([0, const[i]])**2
	new_obj = obj_han(xin, params) + (mu/2)*sum_quad
	return new_obj

def run_optimizer(xin, params, obj_han, con_han):
	"""
	Inputs:
		xin: nparray of form -  (xVAWT, yVAWT, xHAWT, yHAWT)
		params: nparray containing:
			nVAWT - number of vertical axis turbines
			rh - radius of horizontal axis turbines
			rv - radius of vertical axis turbines
			rt - radius of tower
			U_dir - wind direction in radians
			U_vel - free stream velocity magnitude
		obj_han - objective function handle
		con_han - constraint function handle
	Returns:
		x_opt: optimum x value (nparray)
	"""
	mu = 1. 
	options = {'disp': True}
	args = (params, mu, obj_han, con_han)
	dif = 1.
	while abs(dif) >= 1.e-6:
		prev = penalty_obj(xin, args)
		options = {'disp': True}
		res = minimize(penalty_obj, xin, args = (args,), method = 'BFGS', options = options)
		mu = mu*2.
		args = (params, mu, obj_han, con_han)
		xin = res.x
		dif = penalty_obj(xin, args) - prev
	return xin

def plot_func(xin, x_opt):
	xVAWT_opt = x_opt[0 : nVAWT]
	yVAWT_opt = x_opt[nVAWT: 2*nVAWT]
	xHAWT_opt = x_opt[2*nVAWT: 2*nVAWT + nHAWT]
	yHAWT_opt = x_opt[2*nVAWT + nHAWT : len(xin)]
	
	xVAWT = xin[0 : nVAWT]
	yVAWT = xin[nVAWT: 2*nVAWT]
	xHAWT = xin[2*nVAWT: 2*nVAWT + nHAWT]
	yHAWT = xin[2*nVAWT + nHAWT : len(xin)]
	
	plt.figure()
	plt.scatter(xVAWT, yVAWT, c = 'r', s = 60, label = "VAWT init")
	plt.scatter(xVAWT_opt, yVAWT_opt, c = 'r', s = 100, label = "VAWT opt")
	plt.scatter(xHAWT, yHAWT, c = 'b', s = 60, label = "HAWT init")
	plt.scatter(xHAWT_opt, yHAWT_opt, c = 'b', s = 100, label = "HAWT opt")
	plt.legend()
	plt.xlabel("x dir (m)")
	plt.ylabel("y dir (m)")
	plt.show()
	
def test_func(xin, params):
	sum = 0
	for i in range(len(xin)):
		sum += xin[i]**2
	return sum

def test_con(xin, params):
	constraints = np.zeros(len(xin))
	for i in range(len(xin)):
		constraints[i] = xin[i] - i
	return constraints

if __name__=="__main__":
	#xHAWT = np.array([0, 0, 0, 95, 95, 95])
	#yHAWT = np.array([0, 50, 100, 0, 50, 100])
	#xVAWT = np.array([50, 50])
	#yVAWT = np.array([0, 50])
	
	xHAWT = np.array([250, 260, 270, 350, 360, 370])
	yHAWT = np.array([0, 100, 200, 0, 100, 200])
	xVAWT = np.array([0, 350])
	yVAWT = np.array([175, 525])
	
	#set input variable xin
	xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
	
	#set input parameters
	nVAWT = len(xVAWT)
	nHAWT = len(xHAWT)
	rh = 40.
	rv = 3.
	rt = 5.
	direction = 10.
	dir_rad = (direction+90) * np.pi / 180.
	U_vel = 8.
	params = [nVAWT, rh, rv, rt, dir_rad, U_vel]
	print "Original Power: ", -obj(xin, params)*1.e6
	print "Old x", xin
	x_opt = run_optimizer(xin, params, obj, calc_constraints)
	print "New Power: ", -obj(x_opt, params)*1.e6
	print "New x", x_opt
	plot_func(xin, x_opt)
	
	
	
	
	
