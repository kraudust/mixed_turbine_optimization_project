import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from turbines import turbines

if __name__=="__main__":
    """
    #Starting Points 1 and 2
    xHAWT = np.array([0,0,0,200,200,200,400,400,400,600,600,600,800,800,800])
    yHAWT = np.array([0,200,400,600,800,0,200,400,600,800,0,200,400,600,800])
    xVAWT = np.array([50,50,50,50,150,150,150,150,250,250,250,250,350,350,350,350,450,450,450,450,550,550,550,550,650,650,650,650,750,750,750,750])
    yVAWT = np.array([50,150,250,350,450,550,650,750,50,150,250,350,450,550,650,750,50,150,250,350,450,550,650,750,50,150,250,350,450,550,650,750])
    """
    """
    #Starting Points 3
    xHAWT = np.array([0,0,0,250,250,250,500,500,500])
    yHAWT = np.array([0,250,500,0,250,500,0,250,500])
    xVAWT = np.array([62.5,62.5,62.5,62.5,62.5,62.5,125,125,125,125,125,125,187.5,187.5,187.5,187.,187.5,187.5,312.5,312.5,312.5,312.5,312.5,312.5,375,375,375,375,375,375,437.5,437.5,437.5,437.5,437.5,437.5])
    yVAWT = np.array([62.5,125,187.5,312.5,375,437.5,62.5,125,187.5,312.5,375,437.5,62.5,125,187.5,312.5,375,437.5,62.5,125,187.5,312.5,375,437.5,62.5,125,187.5,312.5,375,437.5,62.5,125,187.5,312.5,375,437.5])
    """

    #Starting Points 4
    xHAWT = np.array([0,0,0,250,250,250,500,500,500])
    yHAWT = np.array([0,250,500,0,250,500,0,250,500])
    xVAWT = np.array([0,0,0,0,83.33,83.33,83.33,83.33,83.33,83.33,83.33,166.66,166.66,166.66,166.66,166.66,166.66,166.66,250,250,250,250,333.33,333.33,333.33,333.33,333.33,333.33,333.33,416.66,416.66,416.66,416.66,416.66,416.66,416.66,500,500,500,500])
    yVAWT = np.array([83.33,166.66,333.33,416.66,0,83.33,166.66,250,333.33,416.66,500,0,83.33,166.66,250,333.33,416.66,500,83.33,166.66,333.33,416.66,0,83.33,166.66,250,333.33,416.66,500,0,83.33,166.66,250,333.33,416.66,500,83.33,166.66,333.33,416.66])
    """
    #Starting Points 5
    xHAWT = np.array([])
    yHAWT = np.array([])
    xVAWT = np.array([0,0,11,20,20,30])
    yVAWT = np.array([0,20,10,0,20,40])
    """
    """
    #Optimal Points
    filename = "Optimization_4.txt"
    """
    """
    #Dustan's starting point for 25 turbines
    xHAWT = np.array([0,0,250,500,500])
    yHAWT = np.array([0,500,250,0,500])
    xVAWT = np.array([0, 0, 0, 125, 125, 125, 125, 125, 250, 250, 250, 250, 375, 375, 375, 375, 375, 500, 500, 500])
    yVAWT = np.array([125, 250, 375, 0, 125, 250, 375, 500, 0, 125, 375, 500, 0, 125, 250, 375, 500, 125, 250, 375])
    """
    #Dustan 16 Turbines
    xHAWT = np.array([0,0, 300,300])
    yHAWT = np.array([0,300,0,300])
    xVAWT = np.array([0, 0, 100, 100, 100, 100, 200, 200, 200, 200, 300, 300])
    yVAWT = np.array([100, 200, 0, 100, 200, 300, 0, 100, 200, 300, 100, 200])
    
    #Optimal Points
    filename = "dustan_16_turbines.txt"
    file = open(filename)
    xin = np.loadtxt(file)
    nVAWT = len(xVAWT)
    nHAWT = len(xHAWT)
    xVAWT_opt, yVAWT_opt, xHAWT_opt, yHAWT_opt = turbines(xin, nVAWT)

    plt.figure(1)
    plt.scatter(xHAWT, yHAWT, c='r', s=50, label='HAWT')
    plt.scatter(xVAWT, yVAWT, c='b', s=30, label='VAWT')
    plt.title('Starting Locations')
    plt.legend(bbox_to_anchor=(1.1, 1.1))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    plt.figure(2)
    plt.scatter(xHAWT_opt, yHAWT_opt, c='r', s=50, label='HAWT')
    plt.scatter(xVAWT_opt, yVAWT_opt, c='b', s=30, label='VAWT')
    plt.title('Optimized Locations')
    plt.legend(bbox_to_anchor=(1.1, 1.1))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 15}
    matplotlib.rc('font', **font)

    plt.show()
    
