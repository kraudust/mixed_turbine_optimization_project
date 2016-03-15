import numpy as np
from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, ALPSO, SLSQP

def mul_dim_rosen(x, part):
	n = len(x) #number of dimensions
	if n < 2:
		return "Error, there need to be at least 2 dimensions"
	f = 0
	for i in range(0, n-1):
		f += 100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2
	if part == "a" or part == "b":
		return f, []
	elif part == "c":
		g = np.zeros(n)
		g[0] = -400*(x[0]*x[1]) + 400*x[0]**3 - 2 + 2*x[0]
		for i in range(1, n-1):
			g[i] = 200*x[i] - 200*x[i-1]**2 - 400*x[i+1]*x[i] + 400*x[i]**3 - 2 + 2*x[i]
		g[n-1] = 200*x[n-1] - 200*x[n-2]**2
		return f, [], g, []
	
	
if __name__ == '__main__':
	x = [0, 0]
	f = []
	part = ["a", "b", "c"]
	for i in range(0, 3):
		print "----------------- Part: ", part[i], " -------------------"
		for j in range(0, 6):
			ub = 5.0*np.ones(len(x))
			lb = -5.0*np.ones(len(x))
			if part[i] == "a":
				optimizer = NSGA2()
				#optimizer = ALPSO()
				optimizer.setOption('maxGen', 200)
				func, _ = mul_dim_rosen(x, part[i])
			if part[i] == "b":
				func, _ = mul_dim_rosen(x, part[i])
				optimizer = SNOPT()
			elif part[i] == "c":
				optimizer = SNOPT()
				func, _, _, _ = mul_dim_rosen(x, part[i])
				
			xopt, fopt, info = optimize(mul_dim_rosen, x, lb, ub, optimizer, args = [part[i]])
			print "n: ", len(x)
			print "x_in: ", x
			print "f_in: ", func
			print "xopt: ", xopt
			print "fopt: ", fopt
			print "info: ", info, "\n\n"
			x = np.append(x, x)
		x = [0, 0]
		
	
	
