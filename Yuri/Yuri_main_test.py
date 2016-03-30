import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi, tan, cos, acos, sin, sqrt
from Jensen import *
from Tower_Wake import *
from VAWT_power_calculation import *
from Main_Obj import *
from calcAEP import *


def AEP(xin, params):
	nTurbs = len(xin)/2
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
	for i in range(numDir):
		binSizeDir = 2.*pi/numDir
		direction = i*binSizeDir+binSizeDir/2.
		print "Direction: ", i
		for j in range(numSpeed):
			print "Speed: ", j
			binSizeSpeed = 27./numSpeed
			speed = 3+j*binSizeSpeed+binSizeSpeed/2.
			param = tuple([nVAWT, rh, rv,rt, direction, speed])
			AEP += freqDir[i]*freqSpeed[j]*-1.e6*obj(xin, param)*24.*365.
########          ADD CONSTRAINTS                ######
	nHAWT = nTurbs - nVAWT #number of horizontal axis turbines

	#split xin into x and y locations for VAWT and HAWT
	xVAWT = xin[0 : nVAWT]
	yVAWT = xin[nVAWT: 2*nVAWT]
	xHAWT = xin[2*nVAWT: 2*nVAWT + nHAWT]
	yHAWT = xin[2*nVAWT + nHAWT : len(xin)]
	constraints = np.array([])

	for i in range(len(xHAWT)):
		for j in range(len(xHAWT)):
			if i==j:
				constraints = np.append(constraints,0)
			else:
				dx = xHAWT[i]-xHAWT[j]
				dy = yHAWT[i]-yHAWT[j]
				constraints = np.append(constraints, dx**2+dy**2-64*rh**2)

	for i in range(len(xVAWT)):
		for j in range(len(xVAWT)):
			if i==j:
				constraints = np.append(constraints,0)
			else:
				dx = xVAWT[i]-xVAWT[j]
				dy = yVAWT[i]-yVAWT[j]
				constraints = np.append(constraints, dx**2+dy**2-64*rv**2)

	for i in range(len(xVAWT)):
		for j in range(len(xHAWT)):
			dx = xVAWT[i]-xHAWT[j]
			dy = yVAWT[i]-yHAWT[j]
			constraints = np.append(constraints, dx**2+dy**2-16*rh**2)

	return -AEP/1e12, -constraints/1e5





#if __name__=="__main__":
	# xHAWT = np.array([0, 200, 400])
	# yHAWT = np.array([0, 200, 400])
	# xVAWT = np.array([100, 100, 100])
	# yVAWT = np.array([100, 200, 300])
    #
	# xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
	# nVAWT = len(xVAWT)
	# rh = 40.
	# rv = 3.
	# rt = 5.
	# direction = 30.
	# dir_rad = (direction+90) * np.pi / 180.
	# U_vel = 8.
	# numDir = 10
	# numSpeed = 5
	# params = [nVAWT, rh, rv, rt, dir_rad, U_vel, numDir, numSpeed]
    #
	# print obj(xin, params)
	# print con(xin, params)
	# print bounds(xin, params,2000, 1500)
