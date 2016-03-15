import numpy as np
from PJ_HW5_builtin import *
from PJ_HW5_3 import *
import matplotlib.pyplot as plt
import time

def particleSwarm(func, xin, lb, ub):
	tol = 1e-6
	nVariables = len(xin)
	nParticles = 10*nVariables
	vMax = (ub-lb)/10.
	particle = np.empty([nParticles, nVariables])
	particle_sum = np.zeros(nParticles)
	velocity = np.empty([nParticles, nVariables])
	w = 3.0
	c1 = 0.5
	c2 = 1.5
	#initialize particle locations and velocities
	for i in range(10):
		for j in range(nVariables):
			particle[nVariables*i+j] = np.random.rand(nVariables)*(ub-lb)+lb
			particle_sum[nVariables*i+j] = np.sum(abs(particle[nVariables*i+j]))
			velocity[nVariables*i+j] = 2.*(np.random.rand(nVariables)-0.5)*vMax
			plt.scatter(particle[nVariables*i+j][0], particle[nVariables*i+j][1])
	#plt.show()
	#print particle_sum
	#print velocity
	#print particle
	#initial function values of each particle, as well as the best function value
	memory = np.empty([nParticles, nVariables])
	social = np.zeros(nVariables)
	memoryVal = np.zeros(nParticles)
	socialVal = func(particle[0])[0]
	for i in range(nParticles):
		funcValue = func(particle[i])
		memoryVal[i] = funcValue[0]+1e6*np.sum(funcValue[1])
		memory[i] = particle[i]
		#print "Memory Value: ", memoryVal[i]
		if memoryVal[i] < socialVal:
			social = memory[i]
			socialVal = memoryVal[i]
		counter = 0

	while counter < 5:
		oldPos = socialVal
		oldVel = velocity
		plt.cla()
		for i in range(nParticles):
			#new velocity
			velocity[i] = w*oldVel[i]+c1*(memory[i]-particle[i])+c2*(social-particle[i])
			for j in range(nVariables):
				if velocity[i][j] < -vMax[j]:
					velocity[i][j] = -vMax[j]
				if velocity[i][j] > vMax[j]:
					velocity[i][j] = vMax[j]
			#new position
			particle[i] = particle[i] + velocity[i]
			for j in range(nVariables):
				if particle[i][j] < lb[j]:
					particle[i][j] = lb[j]
				if particle[i][j] > ub[j]:
					particle[i][j] = ub[j]
			particle_sum[i] = np.sum(abs(particle[i]))
			#new function value
			funcVal = func(particle[i])
			newVal = funcVal[0]+1e6*np.sum(funcVal[1])
			if newVal < memoryVal[i]:
				memoryVal[i] = newVal
				memory[i] = particle[i]
			#update best value
			if memoryVal[i] < socialVal:
				social = memory[i]
				socialVal = memoryVal[i]
		#print "OLD: ", oldPos, "NEW: ", socialVal
		if abs(oldPos-socialVal) < tol:
			counter += 1
		else:
			counter = 0
		print counter
		print socialVal
		#w = w*0.9
				
		
	return socialVal, social



if __name__=="__main__":
	xHAWT = np.array([0,0,500,500])
	yHAWT = np.array([0,500,0,500])
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
	#xin = np.array([1,1,1])
	lb = np.ones(len(xin))*0.
	ub = np.ones(len(xin))*500.
	swarm = particleSwarm(calc_AEP, xin, lb, ub)
	print "Function Value: ", swarm[0]
	print "Location: ", swarm[1]
	print calc_AEP(swarm[1])[0]
