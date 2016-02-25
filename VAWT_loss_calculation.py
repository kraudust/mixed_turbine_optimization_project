from math import sqrt
import numpy as np
from VAWT_Wake_Model import velocity_field
import matplotlib.pyplot as plt

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
			x_dist = x[i] - x[j+1]
			y_dist = y[i] - y[j+1]
			ind_loss[j] = 1 - norm_vel_func(x[j+1], y[j+1], x[j+1] + x_dist, y[j+1] + y_dist, velf, dia, tsr, solidity)
		loss[i] = np.linalg.norm(ind_loss,2) #calculate the sum of the squares (the 2 norm)
		ind_loss = np.zeros(n-1)
	return loss
	
if __name__=="__main__":
	velf = 15.0 # free stream wind speed (m/s)
	dia = 6.0  # turbine diameter (m)
	tsr = 4.0  # tip speed ratio
	B = 3. # number of blades
	chord = 0.25 # chord lenth (m)
	solidity = (chord*B)/(dia/2.)
	#x = np.array([0, 0, 0, 10, 10, 10, 20, 20, 20])
	#y = np.array([0, 10, 20, 0, 10, 20, 0, 10, 20])
	#x = np.array([0, 0, 0, 50, 50, 50, 100, 100, 100])
	#y = np.array([0, 50, 100, 0, 50, 100, 0, 50, 100])
	x = np.array([0., 50.])
	y = np.array([0.,0.])
	loss = VAWT_loss(x, y, velocity_field, velf, dia, tsr, solidity)
	print loss
	plt.figure()
	plt.scatter(x,y)
	plt.show()
	
