import numpy as np
from math import sqrt
from scipy.optimize import minimize

global func_calls

def rosenbrock(x):
    global func_calls
    func_calls += 1
    f = 0
    for i in range(0,len(x)-1):
        f += (1 - x[i])**2 + 100*(x[i+1] - x[i]**2)**2
    return f


def vertices(x0):
    c = 1.
    xnew = [x0]
    n = len(x0)
    b = (c/(n*sqrt(2.))) * (sqrt(n+1.) - 1.)
    a = b + c/sqrt(2.)

    vertices = b*np.ones(n)
    for i in range(n):
        vertices[i] = a
        xnew.append(x0+vertices)
        vertices[i] = b
    return xnew


def simplex(f, x):

    c = 1.
    n = len(x)
    func_vertices = np.array([])
    x_vert = vertices(x)

    for i in range(n+1):
        func_vertices = np.append(func_vertices,f(x_vert[i]))

    worst = np.argmax(func_vertices)
    f_w = func_vertices[worst]
    x_w = x_vert[worst]
    best = np.argmin(func_vertices)
    f_b = func_vertices[best]
    x_b = x_vert[best]

    while (f_w - f_b) > 1e-8:

        func_vertices = np.array([]) #this is the function value at each vertice of the tetrahedron
        for i in range(n+1):
            func_vertices = np.append(func_vertices, f(x_vert[i]))
             #Rank Vertices
        # WORST
        worst = np.argmax(func_vertices)
        f_w = func_vertices[worst]
        x_w = x_vert[worst]
        func_vertices = np.delete(func_vertices, worst) # Delete worst indice
        x_vert.pop(worst)
        #  LOUSY
        lousy = np.argmax(func_vertices)
        f_l = func_vertices[lousy]
        x_l = x_vert[lousy]
        # BEST
        best = np.argmin(func_vertices)
        f_b = func_vertices[best]
        x_b = x_vert[best]
        # Find Average with Worst
        x_a = (1./n) * sum(x_vert)

        # Reflection
        alpha = 1.
        x_r = x_a + alpha *(x_a - x_w)
        f_r = f(x_r)

        if f_r < f_b:  #Expand
            x_e = x_r + alpha*(x_r - x_a)
            f_e = f(x_e)
            if f_e < f_b:
                x_vert.append(x_e)
                func_vertices = np.append(func_vertices, f_e )
            else: # accept reflection
                x_vert.append(x_r)
                func_vertices = np.append(func_vertices, f_r )
        elif f_r <= f_l:
            x_vert.append(x_r)
            func_vertices = np.append(func_vertices, f_r )
        else:
            if f_r > f_w:
                x_c = x_a - 0.5*(x_a - x_w)
                f_c = f(x_c)
                if f_c < f_w:
                    x_vert.append(x_c)
                    func_vertices = np.append(func_vertices, f_c )
                else:
                    x_vert.append(x_w)
                    for i in range(n+1):
                        if np.array_equal(x_vert[i], x_b):
                            x_vert[i] = x_b
                        else:
                            x_vert[i] = x_b + 0.5*( x_vert[i] - x_b)

            else:
                x_o = x_a + 0.5*(x_a - x_w)
                f_o = f(x_o)
                if f_o<= f_r:
                    x_vert.append(x_o)
                    func_vertices = np.append(func_vertices, f_o )
                else:
                    x_vert.append(x_w)
                    for i in range(n+1):
                        if np.array_equal(x_vert[i], x_b):
                            x_vert[i] = x_b
                        else:
                            x_vert[i] = x_b + 0.5*( x_vert[i] - x_b)

    return x_b

if __name__ == '__main__':
    global func_calls
    func_calls = 0
    n = 8
    x = -1.*np.ones(n)

    #print vertices(x)
    xb = simplex(rosenbrock,x)
    print 'Converged X value:', xb
    print 'Number of Function Calls:', func_calls

    print "--------------BUILT IN  Nelder-Mead SIMPLEX---------------"
    options = {'maxfev': 15000}
    res = minimize(rosenbrock, x, method = 'Nelder-Mead', options = options)
    print "xopt: ", res.x
    print "function calls: ", res.nfev
