import numpy as np
from math import sqrt, pi, exp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from Main_Obj import *
from datetime import datetime
from scipy.interpolate import interp1d
from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, ALPSO


def weibull_prob(x):
	a = 1.8
	avg = 8.
	lamda = avg/(((a-1)/a)**(1/a))
	return a/lamda*(x/lamda)**(a-1)*exp(-(x/lamda)**a)


def speed_frequ(speeds):
	x = speeds
	size = 30./(speeds)
	dx = 0.01
	x1 = 0.
	x2 = x1+dx
	location = size
	frequency = np.zeros(speeds)
	for i in range(3, speeds):
		while x1 <= location:
			dfrequency = dx*(weibull_prob(x1)+weibull_prob(x2))/2
			frequency[i] += dfrequency
			x1 = x2
			x2 += dx
		location += size
	return frequency


def wind_frequency_funcion():
	input_file = open("amalia_windrose_8.txt")
	wind_data = np.loadtxt(input_file)

	length_data = np.linspace(0,72.01,len(wind_data))
	f = interp1d(length_data, wind_data)
	return f


def frequ(bins):
	f = wind_frequency_funcion()
	bin_size = 72./bins
	dx = 0.01
	x1 = 0.
	x2 = x1+dx
	bin_location = bin_size
	frequency = np.zeros(bins)
	for i in range(0, bins):
		while x1 <= bin_location:
			dfrequency = dx*(f(x1)+f(x2))/2
			frequency[i] += dfrequency
			x1 = x2
			x2 += dx
		bin_location += bin_size
	return frequency


def calc_AEP(xin):
	
	# nVAWT, rh, rv, rt, U_dir, U_vel, numDir, numSpeed = params
	nVAWT = 0
    
	rh = 40.
	rv = 3.
	rt = 5.
	direction = 5.
	dir_rad = (direction+90) * np.pi / 180.
	U_vel = 8.
	numDir = 18
	numSpeed = 18
	freqDir = frequ(numDir)
	freqSpeed = speed_frequ(numSpeed)
	nTurbs = len(xin)/2
	params1 = tuple([nVAWT, rh, rv, rt, dir_rad, U_vel])
	#print "Direction Frequency Vector: ", freqDir
	#print "Speed Fequency Vector: ", freqSpeed
	AEP = 0
	for i in range(numDir):
		binSizeDir = 2.*pi/numDir
		direction = i*binSizeDir+binSizeDir/2.
		print "Direction: ", i
		for j in range(numSpeed):
			print "Speed: ", j
			binSizeSpeed = 27./numSpeed
			speed = 3+j*binSizeSpeed+binSizeSpeed/2.
			params = tuple([nVAWT, rh, rv,rt, direction, speed])
			AEP += freqDir[i]*freqSpeed[j]*-1.e6*obj(xin, params1)*24.*365.
   
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
	constraints = -constraints




	return -AEP/1e9, constraints/1e6


if __name__=="__main__":
	xHAWT = np.array([0,0,0,500,500])
	yHAWT = np.array([0,500,750,0,500])
	rh = 40.
	nRows = 10   # number of rows and columns in grid
	spacing = 5     # turbine grid spacing in diameters

	"""points = np.linspace(start=0, stop=(nRows-1)*spacing*rh, num=nRows)
	xpoints, ypoints = np.meshgrid(points, points)
	xHAWT = np.ndarray.flatten(xpoints)
	yHAWT = np.ndarray.flatten(ypoints)"""
	
	xVAWT = np.array([])
	yVAWT = np.array([])
	# xVAWT = np.array([250,250,250,750,750,750,1250,1250,1250])
	# yVAWT = np.array([250,750,1250,250,750,1250,250,750,1250])

	xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
	nVAWT = len(xVAWT)
    
	rv = 3.
	rt = 5.
	direction = 5.
	dir_rad = (direction+90) * np.pi / 180.
	U_vel = 8.
	numDir = 18
	numSpeed = 18
	lower = np.zeros(len(xin))
	upper = np.ones(len(xin))*np.max(xin)

	params = tuple([nVAWT, rh, rv, rt, dir_rad, U_vel, numDir, numSpeed])
	print "Running..."


	startTime = datetime.now()
	optimizer = SNOPT()
	#optimizer.setOption('maxGen',100)
	xopt, fopt, info = optimize(calc_AEP, xin, lower, upper, optimizer)
	print "Time to run: ", datetime.now()-startTime

	print 'SNOPT:'
	print 'xopt: ', xopt
	print 'fopt: ', fopt
	print 'info: ', info

	#print 'Start:'
	#print calc_AEP(xin)[0]

	x = xopt[0:len(xopt)/2]
	y = xopt[len(xopt)/2:len(xopt)]
	plt.scatter(x,y)
	plt.title('Optimized Turbine Layout Using Particle Swarm')
	plt.xlabel('x coordinates (m)')
	plt.ylabel('y coordinates (m)')
	plt.show()
	



