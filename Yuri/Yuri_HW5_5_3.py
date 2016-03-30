from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, NLPQLP, ALPSO, SLSQP
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

global func_call, func_call2

def rosenbrock(x):
    global func_call
    func_call += 1
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

def rosenbrock2(x):
    global func_call2
    func_call2 += 1
    f = 0
    g = []
    for i in range(len(x)-1):
        f += (1 - x[i])**2 + 100*(x[i+1] - x[i]**2)**2


    c = []
    gc = []
    return f,c



if __name__ == '__main__':
    x = np.zeros(2)

    func_call2 = 0
    func_call = 0
    funcall_a = np.array([])
    funcall_b = np.array([])
    funcall_c = np.array([])

#----------------GRADIENT FREE-----------------------------
    # print "GRADIENT FREE METHOD - NSGA2"
    # lb = np.ones(len(x))*-5.
    # ub = np.ones(len(x))*5.
    # starttime = datetime.now()
    # optimizer = NSGA2()
    # optimizer.setOption('maxGen', 500)
    # optimizer.setOption('PopSize', len(x)*20)
    # xopt, fopt, info = optimize(rosenbrock, x, lb, ub, optimizer)
    # print 'n:', len(x)
    # print 'Function Calls:', func_call
    # print xopt
    # print fopt
    # print info
    # print "Time to run", datetime.now()-starttime,'\n'
#Manual function calls *20pop: [15246, 39767, 79980, 160002, 320002, 640002]
#Manual function calls *10pop: [7150, 19241, 39977, 80002, 160002, 320002]




    while len(x)<1:
        lb = np.ones(len(x))*-5
        ub = np.ones(len(x))*5
        dimen = len(x)

#------------------GRADIENT BASED SNOPT W/ EXACT GRADIENT---------
        print "GRADIENT BASED METHOD W/ Exact Gradient - SNOPT"
        starttime = datetime.now()
        optimizer = SNOPT()
        xopt, fopt, info = optimize(rosenbrock, x, lb, ub, optimizer)
        print 'n:', dimen
        print 'Function Calls:', func_call
        print xopt
        print fopt
        print info
        print "Time to run", datetime.now()-starttime,'\n'
        funcall_b = np.append(funcall_b,func_call)
        func_call = 0
        #manual func calls = [ 33, 50,68 , 109, 234,437]
#------------------GRADIENT BASED SNOPT W/O EXACT GRADIENT---------
        print "GRADIENT BASED METHOD W/O Exact Gradient - SNOPT"
        starttime = datetime.now()
        optimizer = SNOPT()
        xopt, fopt, info = optimize(rosenbrock2, x, lb, ub, optimizer)
        print 'n:', dimen
        print 'Function Calls:', func_call2
        print xopt
        print fopt
        print info
        print "Time to run", datetime.now()-starttime, '\n'
        funcall_c = np.append(funcall_c,func_call2)
        func_call2 = 0
        x = np.append(x,x)
        #manual func calls = [ 92,48773 ,86691 , 15943, 294066, 512268]

funcall_a_10pop = [7150, 19241,39977, 80002, 160002, 320002]
funcall_a_20pop = [15246, 39767, 79980, 160002, 320002, 640002]
funcall_b = [ 33, 50,68 , 109, 234,437]
funcall_c = [ 92,48773 ,86691 , 159463, 294066, 512268]
n = [2,4,8,16,32,64]
plt.figure()
plt.plot(n,funcall_a_20pop,'ro-', label = 'NSGA2 with 20*population')
plt.plot(n,funcall_b,'go-', label = 'Exact Gradient (SNOPT)')
plt.plot(n,funcall_c,'bo-', label = 'Finite Difference (SNOPT)')
plt.yscale('log')
plt.ylabel('Function Calls')
plt.xlabel('Dimensions (n)')
plt.title('Increasing Dimension Size vs Function Calls')
plt.legend(loc = 0)
plt.show()

