from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, NLPQLP, ALPSO, SLSQP
import numpy as np
from Yuri_main_test import *
from datetime import datetime






if __name__ == '__main__':


    xHAWT = np.array([0,50,100,250])
    yHAWT = np.array([100,150,100,300])
    xVAWT = np.array([150])
    yVAWT = np.array([100])


    #Define Xin so that the optimizer understands it
    xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
    #set input parameters
    nVAWT = len(xVAWT)
    nHAWT = len(xHAWT)
    rh = 40.
    rv = 3.
    rt = 5.
    direction = -95.   # Degrees
    dir_rad = (direction+90) * np.pi / 180.
    U_vel = 8.
    params = [nVAWT, rh, rv, rt, dir_rad, U_vel]

    lb = np.zeros(len(xin))
    ub = np.ones(len(xin))*1000.
    optimizer = NSGA2()

    starttime = datetime.now()
    xopt, fopt, info = optimize(obj, xin, lb, ub, optimizer,args = [params,])
    print xopt
    print fopt
    print info
    print "Time to run", datetime.now()-starttime
