from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, NLPQLP, ALPSO, SLSQP
import numpy as np
from Yuri_main_test import *
from datetime import datetime
import matplotlib.pyplot as plt






if __name__ == '__main__':


    xHAWT = np.array([0,0,0,100,100,100,200,200,200])
    yHAWT = np.array([0,100,200,0,100,200,0,100,200])
    xVAWT = np.array([])
    yVAWT = np.array([])


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
    numDir = 5
    numSpeed = 4  # AEP does not work for 3 and under
    params = [nVAWT, rh, rv, rt, dir_rad, U_vel, numDir, numSpeed]

    aep_orig, cons = AEP(xin,params) # Cal original power


    #
    # lb = np.zeros(len(xin))
    # ub = np.ones(len(xin))*1000.
    #
    # optimizer = SNOPT()
    #
    # starttime = datetime.now()
    # #optimizer.setOption('maxGen',50)
    # xopt, fopt, info = optimize(AEP, xin, lb, ub, optimizer,args = [params,])
    # print 'New Tower Possitions:', xopt
    # print 'New Power', fopt * -1e12
    # print info
    # print "Time to run", datetime.now()-starttime
    # print 'Original Tower Positions:', xin
    # print 'Original Power:', aep_orig*-1e12
    # xnew =   [4.72165043e+02,   4.73309307e+01,   7.56915814e+02,   2.26688059e+01,
    #              9.99710288e+02,   5.24272522e+02,   5.13852648e+02,   9.97395555e+02, 2.403921e1]
    # ynew = [4.05614528e-01,   4.66323803e+02,   6.37089449e+02,
    #          1.13793976e+02,   9.14463656e+02,   3.58957375e+02,   9.99178313e+02,   2.35830506e+02,   8.82911914e+02]
    xnew = [   29.9246249,      4.1357555 ,     5.77865275,   553.82061138 ,  347.24998121,
   551.1732353 ,   953.25111564  , 953.22590622  , 951.05982987]
    ynew = [0.,   438.8618946  ,  758.85779237  , 106.92291692 ,  351.31723546 , 1000.,
   225.55159351,   556.23402223 ,  902.74458321]
    plt.figure()
    plt.plot(xnew,ynew,'bo', label = 'Optmized Position')
    plt.plot(xHAWT,yHAWT,'ro',label = 'Original Position')
    axes = plt.gca()
    axes.set_xlim([-100,1100])
    axes.set_ylim([-100,1100])
    plt.ylabel('y-tower position (m)')
    plt.xlabel('x-tower position (m)')
    plt.legend()
    plt.show()