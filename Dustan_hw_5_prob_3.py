import numpy as np
from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, ALPSO, SLSQP
import matplotlib.pyplot as plt

global function_calls

def mul_dim_rosen(x, part):
	global function_calls
	function_calls += 1
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
	global function_calls
	function_calls = 0
	x = [0., 0.]
	f = []
	part = ["a", "b", "c"]
	plt.figure()
	n = np.array([2, 4, 8, 16, 32, 64])
	function_calls_array = np.array([])
	
	for i in range(2, 3):
		print "----------------- Part: ", part[i], " -------------------"
		for j in range(0, 6):
			ub = 5.0*np.ones(len(x))
			lb = -5.0*np.ones(len(x))
			#Somthing is wrong with NSGA2 that returns segmentation fault (core dumped) when run multiple times
			#...so I did it outside the loop..
			#if part[i] == "a":
			#	optimizer = NSGA2()
			#	optimizer.setOption('maxGen', 500)
			#	optimizer.setOption('PopSize', len(x)*10)				
			#	func, _ = mul_dim_rosen(x, part[i])
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
			print "function calls: ", function_calls
			print "info: ", info, "\n\n"
			function_calls_array = np.append(function_calls_array, function_calls)
			x = np.append(x, x)
			function_calls = 0
		if part[i] == "a":
			plt.plot(n, function_calls_array, 'ro-', label = "Genetic Algorithm Gradient Free")
		elif part[i] == "b":
			plt.plot(n, function_calls_array, 'go-', label = "Finite Difference Gradient")
		elif part[i] == "c":
			plt.plot(n, function_calls_array, 'ko-', label = "Analytic Gradient")
		x = [0., 0.]
		function_calls_array = np.array([])
	
	
	x = np.zeros(2)
	enn = len(x)
	ub = 5.*np.ones(enn)
	lb = -5.*np.ones(enn)	
	fin = mul_dim_rosen(x, "a")
	optimizer = NSGA2()
	optimizer.setOption('maxGen', 1500)
	optimizer.setOption('PopSize', enn*20)	
	xopt, fopt, info = optimize(mul_dim_rosen, x, lb, ub, optimizer, args = [part[0]])
	print "----------------- Part: ", part[0], " -------------------"
	print "xin: ", x
	print "fin: ", fin
	print "xopt: ", xopt
	print "fopt: ", fopt
	print "function calls: ", function_calls 
	print info, "\n\n"
	
	
	# I hard coded these in cuz they were taking forever each time running it
	function_calls_b = np.array([94, 48774, 86692, 159464, 294067, 512269])
	function_calls_c = np.array([7151, 19242, 39978, 80003, 160003, 320003]) #pop size = 10*n
	function_calls_c2 = np.array([15247, 39768, 79980, 160003, 320003, 640003]) #pop size = 20*n
	function_calls_c3 = np.array([40512, 119236, 239935, 480009, 960009, 1920009]) #maxGen of 1500 instead of 500
	
	plt.plot(n, function_calls_c3, 'ro-', label = "NSGA2")
	plt.plot(n, function_calls_b, 'go-', label = "Finite Difference Gradient")
	plt.yscale('log')
	plt.ylabel('Function Calls')
	plt.xlabel('Number of design variables')
	plt.title('Multi-Dimensional Rosenbrock Function')
	plt.xlim([0, 70])
	plt.legend(loc = 0)
	plt.show()
	

		
	
	
