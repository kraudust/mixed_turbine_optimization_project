from math import sqrt
import numpy as np
from VAWT_Wake_Model import velocity_field
import matplotlib.pyplot as plt
from Tower_Wake import *
from math import pi

def VAWT_loss(x_v, y_v, x_h, norm_vel_func, velf, dia, tsr, solidity, r_tower, theta):
	"""
	inputs:
		x_v: numpy array with VAWT x locations as follows [x1, x2, ..., xn]
		y_v: numpy array with VAWT y locations as follows [y1, y2, ..., yn]
		x_h: numpy array with HAWT x locations as follows [x1, x2, ..., xm]
		norm_vel_func: function handle that calculates the normalized velocity at each location 
		velf: free stream velocity
		dia: turbine diameter
		tsr: tip speed ratio
		solidity: 
	returns:
		loss: numpy array with loss of each turbine as follows [L1, L2, ..., Ln]
	"""
	n = np.size(x_v) #number of turbines
	loss_VT = np.zeros(n) #total loss on each turbine
	tot_loss = np.zeros(n)
	#loss on a turbine from all the other turbines [Lj1, Lj2, Lj3, Lj4, ..., Lji] where Ljn = loss on turbine j from turbine n
	ind_loss = np.zeros(n)
	for i in range(0, n):
		for j in range(0, n):
			if i == j:
				ind_loss[j] = 0 #i.e. L11 = L22 = L33 = 0... turbine loss from itself is zero
			else:
				ind_loss[j] = 1. - norm_vel_func(x_v[j], y_v[j], x_v[i], y_v[i], velf, dia, tsr, solidity)
		loss_VT[i] = np.linalg.norm(ind_loss,2) #calculate the sum of the squares (the 2 norm)
		ind_loss = np.zeros(n)
	
	loss_HT = loss_cylinder(x_h, x_v, overlap_cyl,r_tower,theta) #loss from HAWT towers
	for z in range(0, n):
		tot_loss[z] = (loss_HT[z]**2. + loss_VT[z]**2.)**0.5
	return tot_loss
	
#def VAWT_Power():
	
	
if __name__=="__main__":
	velf = 15.0 # free stream wind speed (m/s)
	dia = 6.0  # turbine diameter (m)
	tsr = 4.0  # tip speed ratio
	B = 3. # number of blades
	chord = 0.25 # chord lenth (m)
	solidity = (chord*B)/(dia/2.)
	#x = np.array([0, 0, 0, 10, 10, 10, 20, 20, 20])
	#y = np.array([0, 10, 20, 0, 10, 20, 0, 10, 20])
	x_v = np.array([0, 0, 0, 50, 50, 50, 100, 100, 100])
	y_v = np.array([0, 50, 100, 0, 50, 100, 0, 50, 100])
	x_h = np.array([7, 30])
	#x = np.array([0., 50.])
	#y = np.array([0.,0.])
	theta = 10.*pi/180.
	loss = VAWT_loss(x_v, y_v, x_h, velocity_field, velf, dia, tsr, solidity, r_tower, theta)
	print loss
	plt.figure()
	plt.scatter(x,y)
	plt.show()
	
