import numpy as np
from math import sqrt
def mul_dim_rosen(x):
	n = len(x) #number of dimensions
	if n < 2:
		return "Error, there need to be at least 2 dimensions"
	f = 0
	for i in range(0, n-1):
		f += 100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2
	return f, []

def gen_init_point(x0, c):
	x = np.array([])
	x = np.append(x, x0)
	n = len(x0)
	b = (c/(n*sqrt(2)))*(sqrt(n+1)-1)
	a = b + c/sqrt(2)
	var = b*np.ones(n)
	for i in range(0, n):
		var[i] = a
		x = [x, x0+var]
		var[i] = b
	return x
	
def simplex(fun, xin, n):
	#xin is an array of starting point arrays of length n
	vertices = np.array([]) #this is the function value at each vertice of the tetrahedron
	for i in range(n+1):
		vertices = np.append(vertices, fun(xin[i]))
	worst = np.argmax(vertices) #returns the index of the worst point
	xw = vertices[worst] #this is the value of the worst point
	vertices = np.delete(vertices, worst) #delete the worst point
	return vertices
	
if __name__=='__main__':
	n = 2
	x0 = np.array([0.0, 0.0])
	c = 1.
	xin = gen_init_point(x0, c)
	print xin
	print simplex(mul_dim_rosen, xin, n)
