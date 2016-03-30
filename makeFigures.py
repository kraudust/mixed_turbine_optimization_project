import matplotlib
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    #Starting Points
    xHAWT = np.array([0,0,0,200,200,200,400,400,400,600,600,600,800,800,800])
    yHAWT = np.array([0,200,400,600,800,0,200,400,600,800,0,200,400,600,800])
    xVAWT = np.array([50,50,50,50,150,150,150,150,250,250,250,250,350,350,350,350,450,450,450,450,550,550,550,550,650,650,650,650,750,750,750,750])
    yVAWT = np.array([50,150,250,350,450,550,650,750,50,150,250,350,450,550,650,750,50,150,250,350,450,550,650,750,50,150,250,350,450,550,650,750])
    
    #Optimal Points
    filename = "Optimization_1.txt"
    file = open(filename)
    xin = np.loadtxt(file)
    nVAWT = len(xVAWT)
    nHAWT = len(xHAWT)
    xVAWT_opt = xin[0:nVAWT]
    yVAWT_opt = xin[nVAWT:2*nVAWT]
    xHAWT_opt = xin[2*nVAWT:2*nVAWT+nHAWT]
    yHAWT_opt = xin[2*nVAWT+nHAWT:2*(nVAWT+nHAWT)]

    plt.figure(1)
    plt.scatter(xHAWT, yHAWT, c='r', s=50)
    plt.scatter(xVAWT, yVAWT, c='b', s=30)
    plt.title('Starting Locations')
    
    plt.figure(2)
    plt.scatter(xHAWT_opt, yHAWT_opt, c='r', s=50)
    plt.scatter(xVAWT_opt, yVAWT_opt, c='b', s=30)
    plt.title('Optimized Locations')
    
    
    font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 15}
    matplotlib.rc('font', **font)

    plt.show()


   

