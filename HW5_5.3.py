from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, NLPQLP, ALPSO, SLSQP
import numpy as np
from datetime import datetime

def rosenbrock(x):
    f = 0
    g = np.zeros(len(x))
    for i in range(len(x)-1):
        f += (1 - x[i])**2 + 100*(x[i+1] - x[i]**2)**2


    g[0] = -400.0*x[0]*x[1] + 400.0*x[0]**3 - 2 + 2*x[0]
    for i in range(1,len(x)-1):
        g[i] = 200.0*x[i] - 200.0*x[i-1]**2 - 400.0*x[i]*x[i+1] + 400.*x[i]**3 - 2 +2*x[i]

    g[len(x)-1] = 200.*x[len(x)-1] - 200.*x[len(x)-2]**2

    c = []
    gc = []
    return f,c,g,gc

if __name__ == '__main__':
# ----------------GRADIENT FREE-----------------------------
    print "GRADIENT FREE METHOD - NSGA2"
    x = [0,0,0]
    #print rosenbrock(x)
    lb = np.ones(len(x))*-5.
    ub = np.ones(len(x))*5.
    starttime = datetime.now()
    optimizer = NSGA2()
    xopt, fopt, info = optimize(rosenbrock, x, lb, ub, optimizer)
    print xopt
    print fopt
    print info
    print "Time to run", datetime.now()-starttime

#------------------GRADIENT BASED SNOPT ---------
    # print "GRADIENT BASED METHOD - SNOPT"
    # starttime = datetime.now()
    # optimizer = SNOPT()
    # xopt, fopt, info = optimize(rosenbrock, x, lb, ub, optimizer)
    # print xopt
    # print fopt
    # print info
    # print "Time to run", datetime.now()-starttime

