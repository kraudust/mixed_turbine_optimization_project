import numpy as np
from Main_Obj import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def calc_constraints(xin, params):
	xupper = 1000
	yupper = 1000
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
	params = args[0:6]
	mu = args[6]
	obj_han = args[7]
	con_han = args[8]
	const = con_han(xin, params)
	sum_quad = 0
	for i in range(0, len(const)):
	#	if const[i] <= 0:
	#		const[i] = -1000 #really penalize bad constraints
		ind_quad = np.min([0, const[i]])**2
		sum_quad = sum_quad + ind_quad
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
	options = {'disp': True, 'maxiter': 2000}
	args = np.hstack([params, mu, obj_han, con_han])
	res = minimize(penalty_obj, xin, args = (args,), method = 'BFGS', jac = False, tol = 1e-9, options = options)
	x_opt = res.x
	x_dif = 1
	while x_dif >= 1.e-6:
		x_prev = x_opt
		mu = mu*2
		options = {'disp': True, 'maxiter': 2000}
		args = np.hstack([params, mu, obj_han, con_han])
		res = minimize(penalty_obj, x_prev, args = (args,), method = 'BFGS', jac = False, tol = 1e-9, options = options)
		x_opt = res.x
		x_dif = obj_han(x_opt, params) - obj_han(x_prev, params)
	return x_opt

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
	plt.scatter(xVAWT, yVAWT, c = 'r', s = 60)
	plt.scatter(xVAWT_opt, yVAWT_opt, c = 'r', s = 100)
	plt.scatter(xHAWT, yHAWT, c = 'b', s = 60)
	plt.scatter(xHAWT_opt, yHAWT_opt, c = 'b', s = 100)
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
	#xHAWT = np.array([0, 0, 0, 700, 700, 700])
	#yHAWT = np.array([0, 350, 700, 0, 350, 700])
	#xVAWT = np.array([350, 350])
	#yVAWT = np.array([175, 525])
	xHAWT = np.array([0, 700])
	yHAWT = np.array([0, 350])
	xVAWT = np.array([350])
	yVAWT = np.array([350])
	#set input variable xin
	xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
	xupper = 1000
	yupper = 1000
	x0 = [7,11, 17, 23,76]
	#set input parameters
	nVAWT = len(xVAWT)
	nHAWT = len(xHAWT)
	rh = 40.
	rv = 3.
	rt = 5.
	direction = -95.
	dir_rad = (direction+90) * np.pi / 180.
	U_vel = 8.
	params = [nVAWT, rh, rv, rt, dir_rad, U_vel]
	#print "Original Power: ", -obj(xin, params)*1.e6
	x_opt = run_optimizer(x0, params, obj, con)
	print "Old x", x0
	print "New x", x_opt
	#plot_func(xin, x_opt)
	
	
	
	
	
