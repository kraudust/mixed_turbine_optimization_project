import numpy as np
from datetime import datetime
from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT


def rosenbrock(x):
	n = len(x)
	if n < 2:
		return "ERROR: there needs to be at least 2 dimensions"

	f = 0
	for i in range(n-1):
		f += 100*(x[i+1]-x[i]**2)**2+(1-x[i])**2

	g = np.zeros([n])
	
	g[0] = -400*x[0]*x[1]+400*x[0]**3-2+2*x[0]

	for i in range(1, n-1):
		g[i] = 200*x[i]-200*x[i-1]**2-400*x[i]*x[i+1]+400*x[i]**3-2+2*x[i]

	g[n-1] = 200*x[n-1]-200*x[n-2]**2

	return f, [], g, []


if __name__=="__main__":
	x = np.array([5,5,5])

	print "Starting Function Value: ", rosenbrock(x)[0]
	print "Start Gradients: ", rosenbrock(x)[2]
	lower = np.ones(len(x))*-3
	upper = np.ones(len(x))*3
	startTime = datetime.now()
	optimizer = NSGA2()
	#optimizer.setOption('maxGen',1000)
	xopt, fopt, info = optimize(rosenbrock, x, lower, upper, optimizer)
	print "Time to run: ", datetime.now()-startTime

	print 'NSGA2:'
	print 'xopt: ', xopt
	print 'fopt: ', fopt
	print 'info: ', info
