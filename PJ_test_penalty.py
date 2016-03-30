import numpy as np
from PJ_penalty_method_QUADRATIC import *

def quadratic(xyz, params):
	return xyz[0]**2+xyz[1]**2+xyz[2]**2
	

def constraints(xyz, params):
	return ([xyz[0]-3, xyz[1]-4, xyz[2]-1])


if __name__=="__main__":
	xin = ([15, 9, 4.99])
	xupper = 30
	yupper = 30
	params = tuple([5, 4, 5, 5, 6, 5])
	mu = 10000
	args = (params, mu, xupper, yupper, quadratic, constraints, constraints)
	print "START: ", quadratic(xin, params)
	
	res = minimize(penalty, xin, args=args, method='BFGS', options = {'disp':True})
	print "FIRST: ", res.x
	print obj(res.x, params)
	difference = res.x
	i=1
	while np.max(abs(difference))>1e-6:
		previous = res.x[:]
		res = minimize(penalty, res.x, args=args, method='BFGS', options={'disp':True})
		mu = mu*2.
		args = (params, mu, xupper, yupper, obj, constraints, constraints)
		difference = res.x-previous
		print "Iteration: ", i
		i = i+1
		print "Current Function Value: ", obj(res.x, params)

	# print res.x
	print "Starting Power: ", quadratic(xin, params)
	print "Optimized Power: ", quadratic(res.x, params)
