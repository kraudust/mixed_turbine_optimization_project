import matplotlib
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    filename = "numDirConvergence.txt"
    file = open(filename)
    data = np.loadtxt(file)
    x = np.zeros(len(data))
    y = np.zeros(len(data))
    variance = np.zeros(len(data))
    error = np.zeros(len(data))

    """for i in range(len(data)):
        x[i] = data[i][0]
        y[i] = data[i][1]"""

    # plt.plot(speed_points, AEPspeed, 'g', label='Santiago')
    y = np.linspace(0,71,71)
    plt.plot(y,data, c='b')
    # plt.xlim([0,72])
    # plt.ylim([2000, 3000])
    plt.xlabel('Direction Bins')
    plt.ylabel('AEP(MWhr)')
    plt.title('')
    plt.show()

    converged = y[len(y)-1]
    
    top1percent = converged+0.01*converged
    bottom1percent = converged - 0.01*converged
    
    topy = (top1percent, top1percent)
    bottomy = (bottom1percent, bottom1percent)	
    topx = (0,150)
    
    font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 15}
    matplotlib.rc('font', **font)


   

