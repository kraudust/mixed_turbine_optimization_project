import numpy as np
from Main_Obj import *
from scipy.optimize import minimize






if __name__=="__main__":
    xHAWT = np.array([0, 0, 0, 200, 200, 200, 400, 400, 400])
    yHAWT = np.array([0, 200, 400, 0, 200, 400, 0, 200, 400])
    xVAWT = np.array([100, 100, 100, 300, 300, 300])
    yVAWT = np.array([100, 200, 300, 100, 200, 300])

    xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
    nVAWT = len(xVAWT)
    rh = 40.
    rv = 3.
    rt = 5.
    direction = 30.
    dir_rad = (direction+90) * np.pi / 180.
    U_vel = 8.

    params = [nVAWT, rh, rv, rt, dir_rad, U_vel]

    print obj(xin, params)
