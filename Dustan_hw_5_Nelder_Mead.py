import numpy as np
from math import sqrt
from scipy.optimize import minimize

global func_calls
def mul_dim_rosen(x):
	global func_calls
	func_calls += 1
	n = len(x) #number of dimensions
	if n < 2:
		return "Error, there need to be at least 2 dimensions"
	f = 0
	for i in range(0, n-1):
		f += 100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2
	return f

def init_tetrahedron(x0, c):
	#x is a list (length n+1) of arrays (length n) which are the vertices of the hypertetrahedron
	x = [x0]
	n = len(x0)
	b = (c/(n*sqrt(2)))*(sqrt(n+1)-1)
	a = b + c/sqrt(2)
	var = b*np.ones(n)
	for i in range(0, n):
		var[i] = a
		x.append(x0+var)
		var[i] = b
	return x
	
def simplex(fun, x0, c):
	#fun takes as inputs x (a point with n design variables) and outputs function value
	
	#initialize hypertetrahedron and calculate function values at vertices
	n = len(x0)
	xin = init_tetrahedron(x0, c)
	
	vertices = np.array([]) #this is the function value at each vertice of the tetrahedron
	for i in range(n+1):
		vertices = np.append(vertices, fun(xin[i]))
	
	#calculate worst, best, and lousy points
	worst = np.argmax(vertices) #index of the worst point
	best = np.argmin(vertices) #index of the best point
	fw = vertices[worst] #function value of the worst point
	fb = vertices[best] #function value of the best point
	while (fw - fb) > 1e-6:
		vertices = np.array([]) #this is the function value at each vertice of the tetrahedron
		for i in range(n+1):
			vertices = np.append(vertices, fun(xin[i]))
		
		#calculate worst, best, and lousy points
		worst = np.argmax(vertices) #index of the worst point
		best = np.argmin(vertices) #index of the best point
		fw = vertices[worst] #function value of the worst point
		fb = vertices[best] #function value of the best point
		xw = xin[worst] #worst point
		xb = xin[best] #best point
		xin.pop(worst) #remove the worst point to calculate the lousy point
		vertices = np.delete(vertices, worst) #remove the worst function value to calculate the lousy point
		lousy = np.argmax(vertices) #index of the lousy point
		fl = vertices[lousy] #function value of the lousy point
		xl = xin[lousy] #lousy point
	
		#Evaluate x average and perform reflection
		xa = (1./n)*sum(xin) #average of points excluding the worst
		alpha_r = 1. 
		xr = xa + alpha_r*(xa-xw)
		fr = fun(xr)
	
		#If reflection is better than best, look at expanding
		if fr < fb:
			alpha_e = 1. 
			xe = xr + alpha_e*(xr-xa)
			fe = fun(xe)
			if fe < fb: #accept expansion
				xin.append(xe)
				vertices = np.append(vertices, fe)
			else: #stick to the original reflection
				xin.append(xr)
				vertices = np.append(vertices, fr)
			
		#If reflection is better than lousy, accept reflection		
		elif fr <= fl:
			xin.append(xr)
			vertices = np.append(vertices, fr)
		
		#if reflection is worse than worst, perform contraction	
		else:	
			if fr > fw: #do an inside contraction
				beta = 0.5
				xc = xa - beta*(xa-xw)
				fc = fun(xc)
				if fc < fw: #accept contraction
					xin.append(xc)
					vertices = np.append(vertices, fc)
				else: #shrink the simplex
					rho = 0.5
					xin.append(xw) #add the worst point back on so I can shrink it
					for i in range(n+1): 
						if np.array_equal(xin[i], xb): #exclude the best point from the shrink
							xin[i] = xb
						else:
							xin[i] = xb + rho*(xin[i] - xb)
			else: #do an outside contraction
				beta = 0.5
				xo = xa + beta*(xa-xw)
				fo = fun(xo)
				if fo <= fr: #accept contraction
					xin.append(xo)
					vertices = np.append(vertices, fo)
				else: #shrink the simplex
					rho = 0.5
					xin.append(xw) #add the worst point back on so I can shrink it
					for i in range(n+1): 
						if np.array_equal(xin[i], xb): #exclude the best point from the shrink
							xin[i] = xb
						else:
							xin[i] = xb + rho*(xin[i] - xb)			
	return xb
	
if __name__=='__main__':
	global func_calls
	func_calls = 0
	n = 8
	x0 = 0.*np.ones(n)
	c = 1.
	xopt = simplex(mul_dim_rosen, x0, c)
	
	print "------------------------My Nelder-Mead--------------------------"
	print "xopt: ", xopt
	print "optimial function value: ", mul_dim_rosen(xopt)
	print "function calls: ", func_calls, '\n\n'
	
	print "--------------Scipy.optimize.minimize Nelder-Mead---------------"
	options = {'maxfev': 15000}
	res = minimize(mul_dim_rosen, x0, method = 'Nelder-Mead', options = options)
	print "xopt: ", res.x
	print "optimal function value: ", res.fun
	print "function calls: ", res.nfev
	
