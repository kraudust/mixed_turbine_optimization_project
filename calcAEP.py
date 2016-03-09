import numpy as np
from math import sqrt, pi, exp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from Main_Obj import *
from datetime import datetime
from scipy.interpolate import interp1d


def weibull_prob(x):
    a = 1.8
    avg = 8.
    lamda = avg/(((a-1)/a)**(1/a))
    return a/lamda*(x/lamda)**(a-1)*exp(-(x/lamda)**a)


def speed_frequ(speeds):
    x = speeds
    size = 30./(speeds)
    dx = 0.01
    x1 = 0.
    x2 = x1+dx
    location = size
    frequency = np.zeros(speeds)
    for i in range(0, speeds):
        while x1 <= location:
            dfrequency = dx*(weibull_prob(x1)+weibull_prob(x2))/2
            frequency[i] += dfrequency
            x1 = x2
            x2 += dx
        location += size
    return frequency


def wind_frequency_funcion():
    input_file = open("amalia_windrose_8.txt")
    wind_data = np.loadtxt(input_file)

    length_data = np.linspace(0,72.01,len(wind_data))
    f = interp1d(length_data, wind_data)
    return f


def frequ(bins):
    f = wind_frequency_funcion()
    bin_size = 72./bins
    dx = 0.01
    x1 = 0.
    x2 = x1+dx
    bin_location = bin_size
    frequency = np.zeros(bins)
    for i in range(0, bins):
        while x1 <= bin_location:
            dfrequency = dx*(f(x1)+f(x2))/2
            frequency[i] += dfrequency
            x1 = x2
            x2 += dx
        bin_location += bin_size
    return frequency


def calc_AEP(xin, params, numDir, numSpeed):
    nVAWT = params[0]
    rh = params[1]
    rv = params[2]
    rt = params[3]
    U_dir = params[4]
    U_vel = params[5]
    freqDir = frequ(numDir)
    freqSpeed = speed_frequ(numSpeed)
    AEP = 0
    for i in range(numDir):
        binSizeDir = 2.*pi/numDir
        direction = i*binSizeDir+binSizeDir/2.
        for j in range(numSpeed):
            binSizeSpeed = 2.*pi/numSpeed
            speed = j*binSizeSpeed+binSizeSpeed/2.
            params = tuple([nVAWT, rh, rv,rt, direction, speed])
            AEP += freqDir[i]*freqSpeed[j]*-1.e6*obj(xin, params)*24.*365.

    return AEP/1e6


if __name__=="__main__":

    xHAWT = np.array([0,1000,2000,3000,4000,5000,6000,7000,8000])
    yHAWT = np.array([0,500,1000,0,500,1000,0,500,1000])
    xVAWT = np.array([])
    yVAWT = np.array([])


    xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
    nVAWT = len(xVAWT)
    rh = 40.
    rv = 3.
    rt = 5.
    direction = 5.
    dir_rad = (direction+90) * np.pi / 180.
    U_vel = 8.

    params = tuple([nVAWT, rh, rv, rt, dir_rad, U_vel])

    AEP = np.array([])
    for i in range(1, 72):
        AEP = np.append(AEP, calc_AEP(xin, params, i, 30))
        print calc_AEP(xin, params, i, 30)
        print i
        np.savetxt("numDirConvergence.txt", np.c_[AEP])
    
    
