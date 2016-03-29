import numpy as np
from Main_Obj import *

if __name__=="__main__":

    x = np.linspace(0,500,num=500)
    power = np.zeros(len(x))


    for i in range(len(power)):
        xHAWT = np.array([0, 500])
        yHAWT = np.array([250, x[i]])
        xVAWT = np.array([])
        yVAWT = np.array([])

        xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
        nVAWT = len(xVAWT)
        xupper = 1000
        yupper = 1000
        rh = 40.
        rv = 3.
        rt = 5.
        direction = -90.
        dir_rad = (direction+90) * np.pi / 180.
        U_vel = 8.

        params = tuple([nVAWT, rh, rv, rt, dir_rad, U_vel])
        power[i] = -1.*obj(xin,params)

    plt.figure(1)
    plt.plot(power,x)
    plt.title('HAWT Wake Model Validation')
    plt.xlabel('Power (MW)')
    plt.ylabel('Location of the Second Turbine (m)')
    plt.show()

    for i in range(len(power)):
        xHAWT = np.array([0])
        yHAWT = np.array([250])
        xVAWT = np.array([100])
        yVAWT = np.array([x[i]])

        xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
        nVAWT = len(xVAWT)
        xupper = 1000
        yupper = 1000
        rh = 40.
        rv = 3.
        rt = 5.
        direction = -90.
        dir_rad = (direction+90) * np.pi / 180.
        U_vel = 8.

        params = tuple([nVAWT, rh, rv, rt, dir_rad, U_vel])
        power[i] = -1.*obj(xin,params)

    plt.figure(2)
    plt.plot(power,x)
    plt.title('Tower Wake Model Validation')
    plt.xlabel('Power (MW)')
    plt.ylabel('Location of the Second Turbine (m)')
    plt.show()



    x = np.linspace(0, 500,num=500)

    for i in range(len(power)):
        xVAWT = np.array([0, 100])
        yVAWT = np.array([250, x[i]])
        xHAWT = np.array([])
        yHAWT = np.array([])

        xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
        nVAWT = len(xVAWT)
        xupper = 1000
        yupper = 1000
        rh = 40.
        rv = 3.
        rt = 5.
        direction = -158.
        dir_rad = (direction+90) * np.pi / 180.
        U_vel = 8.

        params = tuple([nVAWT, rh, rv, rt, dir_rad, U_vel])
        power[i] = -1.*obj(xin,params)


    plt.figure(3)
    plt.plot(power,x)
    plt.title('VAWT Wake Model Validation')
    plt.xlabel('Power (MW)')
    plt.ylabel('Location of the Second Turbine (m)')
    plt.figure(4)
    plt.scatter(0,250)
    plt.scatter(np.ones(500), x)
    plt.show()


    
