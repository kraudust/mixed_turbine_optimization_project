import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi, tan, cos, acos, sin, sqrt
from Jensen import *
from Tower_Wake import *
from VAWT_power_calculation import *
from Main_Obj import bounds
from calcAEP import *
from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT

def AEP(xin, params):
	nVAWT = params[0]
	rh = params[1]
	rv = params[2]
	rt = params[3]
	U_dir = params[4]
	U_vel = params[5]
	numDir = params[6]
	numSpeed = params[7]
	freqDir = frequ(numDir)
	freqSpeed = speed_frequ(numSpeed)
	AEP = 0
	print "cheers"
	for i in range(numDir):
		binSizeDir = 2.*pi/numDir
		direction = i*binSizeDir+binSizeDir/2.
		#print "Direction: ", i
		for j in range(numSpeed):
			#print "Speed: ", j
			binSizeSpeed = 27./numSpeed
			speed = 3+j*binSizeSpeed+binSizeSpeed/2.
			params_2 = tuple([nVAWT, rh, rv, rt, direction, speed])
			AEP += freqDir[i]*freqSpeed[j]*-1.e6*obj(xin, params_2)*24.*365.
	constraints = con(xin, params_2)
	return -AEP/1e12, -constraints
	
if __name__=="__main__":
	xHAWT = np.array([0, 0, 0, 100, 100, 100, 200, 200, 200])
	yHAWT = np.array([0, 100, 200, 0, 100, 200, 0, 100, 200])
	xVAWT = np.array([])
	yVAWT = np.array([])

	xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
	nVAWT = len(xVAWT)
	rh = 40.
	rv = 3.
	rt = 5.
	direction = 30.
	dir_rad = (direction+90) * np.pi / 180.
	U_vel = 8.
	numDir = 5
	numSpeed = 4 #why can I only do 4 or more speeds?
	params = [nVAWT, rh, rv, rt, dir_rad, U_vel, numDir, numSpeed]
	params_2 = [nVAWT, rh, rv, rt, dir_rad, U_vel]
	lb = np.zeros(len(xin))
	ub = np.ones(len(xin))*1000.
	
	forig, cons = AEP(xin, params)
	print "Original AEP: ", forig*-1e12
	optimizer = NSGA2()
	optimizer.setOption('maxGen', 100)
	xopt, fopt, info = optimize(AEP, xin, lb, ub, optimizer, args = [params,])
	
	print "Original AEP: ", forig*-1e12
	print "New AEP: ", fopt*-1e12
	
	
	
	
	
	
