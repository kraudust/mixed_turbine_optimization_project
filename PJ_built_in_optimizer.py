import numpy as np
from math import sqrt
from math import pi
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from Main_Obj import *

def con(xin):
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
    xHAWT = np.array([0, 0, 0, 200, 200, 200, 400, 400, 400])
    yHAWT = np.array([0, 200, 400, 0, 200, 400, 0, 200, 400])
    xVAWT = np.array([100, 100, 100, 300, 300, 300])
    yVAWT = np.array([100, 200, 300, 100, 200, 300])

    xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
    nVAWT = len(xVAWT)
    rh = 40.
    rv = 3.
    rt = 5.
    direction = 30.
    dir_rad = (direction+90) * np.pi / 180.
    U_vel = 8.

    params = [nVAWT, rh, rv, rt, dir_rad, U_vel]

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
