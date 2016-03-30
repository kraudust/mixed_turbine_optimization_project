import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi, tan, cos, acos, sin, sqrt
from Jensen import *
from Tower_Wake import *
from VAWT_power_calculation import *
from Main_Obj import *
from calcAEP import *
from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, ALPSO

global func_call

def AEP(xin, params):
    global func_call
    func_call += 1
    nVAWT = params[0]
    rh = params[1]
    rv = params[2]
    rt = params[3]
    numDir = params[4]
    numSpeed = params[5]
    freqDir = frequ(numDir)
    freqSpeed = speed_frequ(numSpeed)
    AEP = 0
    for i in range(numDir):
        binSizeDir = 2.*pi/numDir
        direction = i*binSizeDir+binSizeDir/2.
        print "Direction: ", i
        for j in range(numSpeed):
            print "Speed: ", j
            binSizeSpeed = 27./numSpeed
            speed = 3+j*binSizeSpeed+binSizeSpeed/2.
            params_2 = tuple([nVAWT, rh, rv, rt, direction, speed])
            AEP += freqDir[i]*freqSpeed[j]*-1.e6*obj(xin, params_2)*24.*365.
    constraints = con(xin, params_2)
    print func_call, -AEP/1e11
    return -AEP/1e11, -constraints
	
if __name__=="__main__":
    global func_call
    func_call = 0
    xHAWT = np.array([580,580])
    yHAWT = np.array([500,580])
    xVAWT = np.array([400, 400])
    yVAWT = np.array([400, 450])

    xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
    n = len(xin)
    nVAWT = len(xVAWT)
    nHAWT = len(xHAWT)
    rh = 40.
    rv = 3.
    rt = 5.
    numDir = 18
    numSpeed = 18 #why can I only do 4 or more speeds?
    params = [nVAWT, rh, rv, rt, numDir, numSpeed]
    print AEP(xin, params)
    '''
    lb = np.zeros(len(xin))
    ub = np.ones(len(xin))*1000.

    forig, cons = AEP(xin, params)

    #optimizer = NSGA2()
    optimizer = SNOPT()'''
    #optimizer.setOption('SwarmSize', 20)
    #optimizer.setOption('maxGen', 1)
    #optimizer.setOption('PopSize', n*10)	
    xopt, fopt, info = optimize(AEP, xin, lb, ub, optimizer, args = [params,])

    print "Original positions: ", xin
    print "Original AEP: ", forig*-1e11
    print "Optimal positions: ", xopt
    print "New AEP: ", fopt*-1e11
    print "Function Calls: ", func_call
    print info

    #-------------------------------Plots---------------------------------------
    plt.figure()
    xVAWT_opt = xopt[0 : nVAWT]
    yVAWT_opt = xopt[nVAWT: 2*nVAWT]
    xHAWT_opt = xopt[2*nVAWT: 2*nVAWT + nHAWT]
    yHAWT_opt = xopt[2*nVAWT + nHAWT : len(xin)]
    plt.scatter(xVAWT, yVAWT, c = 'r', s = 30, label = "VAWT init")
    plt.scatter(xVAWT_opt, yVAWT_opt, c = 'r', s = 120, label = "VAWT opt")
    plt.scatter(xHAWT, yHAWT, c = 'b', s = 30, label = "HAWT init")
    plt.scatter(xHAWT_opt, yHAWT_opt, c = 'b', s = 120, label = "HAWT opt")
    #plt.legend()
    plt.xlabel("x dir (m)")
    plt.ylabel("y dir (m)")
    plt.title("Mixed Wind Farm Optimization")
    #move legend outside plot
    plt.legend(loc='upper center', bbox_to_anchor = (0.5, -0.05), fancybox = True, shadow=True, ncol = 2)
    plt.show()

	
	
	
	
	
