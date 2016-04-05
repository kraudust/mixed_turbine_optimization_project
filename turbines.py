import numpy as np

def turbines(xin, nVAWT):
    nHAWT = len(xin)/2-nVAWT
    xVAWT = xin[0:nVAWT]
    yVAWT = xin[nVAWT:2*nVAWT]
    xHAWT = xin[2*nVAWT:2*nVAWT+nHAWT]
    yHAWT = xin[2*nVAWT+nHAWT:2*(nVAWT+nHAWT)]
    return xVAWT, yVAWT, xHAWT, yHAWT
