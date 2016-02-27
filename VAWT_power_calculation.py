from math import sqrt
import numpy as np
from VAWT_Wake_Model import velocity_field
import matplotlib.pyplot as plt
from Tower_Wake import *
from math import pi

def VAWT_Power(x_v, y_v, x_h, y_h, params):
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
	velf, r_tower, r_VAWT = params
	dia = 2*r_VAWT  # turbine diameter (m)
	tsr = 4.0  # tip speed ratio
	B = 3. # number of blades
	chord = 0.25 # chord lenth (m)
	solidity = (chord*B)/(dia/2.)
	n = np.size(x_v) #number of turbines
	loss_VT = np.zeros(n) #total loss on each turbine
	tot_loss = np.zeros(n)
	power = np.zeros(n)
	cp = 0.3
	rho = 1.1716
	h = 2.*dia
	#loss on a turbine from all the other turbines [Lj1, Lj2, Lj3, Lj4, ..., Lji] where Ljn = loss on turbine j from turbine n
	ind_loss = np.zeros(n)
	for i in range(0, n):
		for j in range(0, n):
			if i == j:
				ind_loss[j] = 0 #i.e. L11 = L22 = L33 = 0... turbine loss from itself is zero
			else:
				ind_loss[j] = 1. - velocity_field(x_v[j], y_v[j], x_v[i], y_v[i], velf, dia, tsr, solidity)
		loss_VT[i] = np.linalg.norm(ind_loss,2) #calculate the sum of the squares (the 2 norm)
		ind_loss = np.zeros(n)
	
	loss_HT = loss_cylinder(x_h, y_h, x_v, y_v, r_tower, r_VAWT) #loss from HAWT towers
	for z in range(0, n):
		tot_loss[z] = (loss_HT[z]**2. + loss_VT[z]**2.)**0.5
		if tot_loss[z] > 1.:
			tot_loss[z] = 1.
		vel = (1-tot_loss[z])*velf
		power[z] = 0.5*rho*cp*dia*h*vel**3
	return power
	
	
if __name__=="__main__":
	velf = 15.0 # free stream wind speed (m/s)
	#x = np.array([0, 0, 0, 10, 10, 10, 20, 20, 20])
	#y = np.array([0, 10, 20, 0, 10, 20, 0, 10, 20])
	x_v = np.array([0, 0, 0, 50, 50, 50, 100, 100, 100])
	y_v = np.array([0, 50, 100, 0, 50, 100, 0, 50, 100])
	x_h = np.array([7, 30])
	y_h = np.array([7, 30])
	#x = np.array([0., 50.])
	#y = np.array([0.,0.])
	r_tower = 5.
	r_VAWT = 6.
	params = [velf, r_tower, r_VAWT]
	power = VAWT_Power(x_v, y_v, x_h, y_h, params)
	print power
	plt.figure()
	plt.scatter(x_v,y_v)
	plt.scatter(x_h, y_h, c='r')
	plt.show()
	
