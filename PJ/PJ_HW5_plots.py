import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
	#MY PARTICLE SWARMS
	#4 Directions and Speeds
	#1
	"""x1 = np.array([459.6,723.0,248.2,587.1,342.6])
	y1 = np.array([57.8,269.2,537.8,137.4,84.7])

	#2
	x2 = np.array([0.7,501.9,326.7,223.1,699.7])
	y2 = np.array([277.7,206.0,720.4,661.8,228.6])

	#3
	#18 directions and speeds
	x3 = np.array([454.3,103,571.0,119.6,709.8])
	y3 = np.array([128.7,319.1,236.6,286.6,176.1])

	#4
	x4 = np.array([420.8,556.0,189.1,712.2,311.8])
	y4 = np.array([534.5,147.8,728.8,307.7,516.3])

	plt.figure(1)
	plt.scatter(x1, y1)
	plt.title('Optimized Turbine Layout Using My Particle Swarm')
	plt.xlabel('x coordinate (m)')
	plt.ylabel('y coordinate (m)')

	plt.figure(2)
	plt.scatter(x2, y2)
	plt.title('Optimized Turbine Layout Using My Particle Swarm')
	plt.xlabel('x coordinate (m)')
	plt.ylabel('y coordinate (m)')

	plt.figure(3)
	plt.scatter(x3, y3)
	plt.title('Optimized Turbine Layout Using My Particle Swarm')
	plt.xlabel('x coordinate (m)')
	plt.ylabel('y coordinate (m)')

	plt.figure(4)
	plt.scatter(x4, y4)
	plt.title('Optimized Turbine Layout Using My Particle Swarm')
	plt.xlabel('x coordinate (m)')
	plt.ylabel('y coordinate (m)')

	plt.figure(5)
	plt.scatter(np.array([213.9,102.7,0,686.4,413.8]), np.array([0,350.5,750,0,619.7]))
	plt.title('Optimized Turbine Layout Using SNOPT')
	plt.xlabel('x coordinate (m)')
	plt.ylabel('y coordinate (m)')"""
	n = np.array([2,4,8,16,32,64])
	SNOPT_AN = np.array([52,80.4,136.2,238,393.6,818.8])
	SNOPT_FD = np.array([17470.2,37781,62094,104886.4,191680,145174])
	ALPSO = np.array([1750,13098,48850,118050,113602,326274])

	plt.figure(1)
	plt.plot(n, SNOPT_AN, c='b', label="Analytic Gradients")
	plt.plot(n, SNOPT_FD, c ='r', label="Finite Difference Gradients")
	plt.plot(n, ALPSO, c='y', label="Gradient Free Method")
	plt.semilogy()
	plt.title('Function Calls As a Function of Dimensionality')
	plt.xlabel('Number of Dimensions')
	plt.ylabel('Function Calls')
	plt.legend(loc=4)

	plt.show()

