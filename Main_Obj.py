import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi, tan, cos, acos, sin, sqrt
from Jensen import *
from Tower_Wake import *
from VAWT_loss_calculation import *

def obj(xin,params):
    """

    :param xin:
    :param params:
    :return:
    """
    nTurbs = len(xin)/2
    nVAWT, r_h, r_v, r_t, U_direction_radians, U_velocity = params
    nHAWT = nTurbs - nVAWT
    xVAWT = xin[0 : nVAWT]
    xHAWT = xin[nVAWT : nTurbs]
    yVAWT = xin[nTurbs : nTurbs + nVAWT]
    yHAWT = xin[nTurbs + nVAWT : len(xin)]

    HAWT_Power = Jensen_Wake_Model(xHAWT, yHAWT, params)
    VAWT_Power = VAWT_Power_Func(xVAWT, yVAWT, params)

    return HAWT_Power+VAWT_Power