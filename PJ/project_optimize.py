import numpy as np
from math import sqrt
from math import pi
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from Jensen import *


def obj(xin):
    theta = 0.1
    alpha = np.tan(theta)
    rho = 1.1716
    a = 1. / 3.
    U_velocity = 8.
    U_direction = 45.
    r_0 = 40.
    U_direction_radians = (U_direction+90.) * np.pi / 180.
    params = (r_0, alpha, U_direction_radians, a, U_velocity, rho, Cp)

    # step = 1e-2
    # grad = np.zeros(len(xin))
    #
    # for i in range(len(xin)):
    #     xnew = xin
    #     xnew[i] = xnew[i]+step
    #     grad[i] = (Jensen_Wake_Model(xnew, params)-Jensen_Wake_Model(xin,params))/step
    # print grad

    return -Jensen_Wake_Model(xin, params)/1000000


def con(xin):
    r_0 = 40.
    x = np.zeros(nTurbines)
    y = np.zeros(nTurbines)
    x[:] = xin[0:nTurbines]
    y[:] = xin[nTurbines:nTurbines*2]
    constraints = np.array([])
    for i in range(len(x)):
        for j in range(len(x)):
            constraints = np.append(constraints, sqrt(x[i]**2+y[j]**2)-10*r_0)

    return constraints

if __name__=="__main__":
    "Define Variables"
    theta = 0.1
    alpha = np.tan(theta)
    x = np.array([0,0,0,400,400,400,800,800,800]) #x coordinates of the turbines
    y = np.array([0,400,800,0,400,800,0,400,800]) #y coordinates of the turbines
    rho = 1.1716
    a = 1. / 3.
    U_velocity = 8.
    "0 degrees is coming from due North. +90 degrees means the wind is coming from due East, -90 from due West"
    U_direction = 45.
    r_0 = 40.

    U_direction_radians = (U_direction+90) * np.pi / 180.
    Cp = 4.*a*(1-a)**2.
    nTurbines = len(x)

    xin = np.hstack([x, y])
    params = (r_0, alpha, U_direction_radians, a, U_velocity, rho, Cp)

    bounds = np.empty((nTurbines*2.,2))
    bounds[0:nTurbines, :] = np.array([0, 1000])
    bounds[nTurbines:nTurbines*2, :] = np.array([0, 1000])
    options = {'disp': True}
    constraints = {'type': 'ineq', 'fun': con}
    res = minimize(obj, xin, method='SLSQP', jac=False, bounds=bounds, tol=1e-6, constraints=constraints, options=options)

    print res.x
    print con(res.x)

    xcoordinates = np.zeros(nTurbines)
    ycoordinates = np.zeros(nTurbines)

    xcoordinates = res.x[0:nTurbines]
    ycoordinates = res.x[nTurbines:nTurbines*2.]

    plt.scatter(x,y,label='Start')
    plt.scatter(xcoordinates,ycoordinates, c='r', label='Optimized')
    plt.legend()
    plt.show()

    print "Start Power: ", Jensen_Wake_Model(xin, params)
    print "Optimized Power: ", Jensen_Wake_Model(res.x, params)