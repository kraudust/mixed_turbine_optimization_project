from Main_Obj import *
from Jensen import *
import numpy as np

if __name__=="__main__":
	y = np.linspace(0, 500, num=500)
	power = np.zeros(len(y))
	for i in range(len(y)):
		xHAWT = ([0, 500])
		yHAWT = ([250, y[i]])
		nVAWT = 0
		rh = 40.
		rv = 3.
		rt = 5.
		direction = 30.
		dir_rad = (direction+90) * np.pi / 180.
		U_vel = 8.

		params = [rh, U_vel]
		power[i] = Jensen_Wake_Model(xHAWT, yHAWT, params)

	plt.plot(power, y)
	plt.show()
