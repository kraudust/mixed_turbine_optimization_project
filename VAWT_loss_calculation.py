from math import sqrt
import numpy as np

def VAWT_loss(x, y, norm_vel_func, velf, dia, tsr, solidity):
	"""
	inputs:
		x: numpy array with VAWT turbine x locations as follows [x1, x2, ..., xn]
		y: numpy array with VAWT turbine y locations as follows [y1, y2, ..., yn]
		norm_vel_func: function handle that calculates the normalized velocity at each location 
		velf: free stream velocity
		dia: turbine diameter
		tsr: tip speed ratio
		solidity: 
	returns:
		loss: numpy array with loss of each turbine as follows [L1, L2, ..., Ln]
	"""
	n = np.size(x) #number of turbines
	loss = np.zeros(n) #total loss on each turbine
	#loss on a turbine from all the other turbines [Lj2, Lj3, Lj4, ..., Ljn] where Ljn = loss on turbine j from turbine n
	ind_loss = np.zeros(n-1) 
	for i in range(0, n):
		for j in range(0, n-1):
			x_dist = x[j+1] - x[j]
			y_dist = y[j+1] - y[j]
			ind_loss[j] = 1 - norm_vel_func(x[i], y[i], x[i] + x_dist, y[i] + y_dist, velf, dia, tsr, solidity)
		loss[i] = np.linalg.norm(ind_loss,2) #calculate the sum of the squares
	print loss
	return loss
	
