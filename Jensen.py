"""THINGS TO FIX"
1. overlap function gives a NaN if the x values do not increase
FIXED - added the 'else: return 0' at the end of the function definition
2. loss gives a value greater than one for some things
problem 2 is likely due to this problem: overlap_fraction needs to be a matrix, not a vector. overlap from each turbine
to each turbine
FIXED
3. allow the wind to come from any direction, rather than always from the West
OVERLAP AND LOSS ARE FIXED
STILL NEED TO FIX THE GRAPHIC::
FIXED-(It's a little lopsided and doesn't keep constant distance between points, but it works!

THINGS TO MAKE IT BETTER
1. put more things into separate functions that can be used in any wake model
    a. Effective Velocity
    b. Power Calculation
    c. Rotation Matrix
    d. Plotting
    """

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def Jensen_Wake_Model(xHAWT, yHAWT, params):
    theta = 0.1
    alpha = sp.tan(theta)
    rho = 1.1716
    a = 1. / 3.
    Cp = 4.*a*(1-a)**2.
    # nTurbines = len(xin)/2.
    r_0, U_velocity = params
    "Make the graphic for the turbines and wakes"
    # jensen_plot(x, y, r_0, alpha, U_direction_radians)

    "Calculate power from each turbine"
    return jensen_power(overlap, loss, xHAWT, yHAWT, r_0, alpha, a, U_velocity, rho, Cp)

    # plt.show()


#Determine how much of the turbine is in the wake of the other turbines
def overlap(x, xdown, y, ydown, r, alpha):
    overlap_fraction = np.zeros(np.size(x))
    for i in range(0, np.size(x)):
        #define dx as the upstream x coordinate - the downstream x coordinate then rotate according to wind direction
        dx = xdown - x[i]
        #define dy as the upstream y coordinate - the downstream y coordinate then rotate according to wind direction
        dy = ydown - y[i]
        R = r+dx*alpha #The radius of the wake depending how far it is from the turbine
        A = r**2*np.pi #The area of the turbine

        if dx > 0:
            if np.abs(dy) <= R-r:
                overlap_fraction[i] = 1 #if the turbine is completely in the wake, overlap is 1, or 100%
            elif np.abs(dy) >= R+r:
                overlap_fraction[i] = 0 #if none of it touches the wake, the overlap is 0
            else:
                #if part is in and part is out of the wake, the overlap fraction is defied by the overlap area/rotor area
                overlap_area = r**2.*sp.arccos((dy**2.+r**2.-R**2.)/(2.0*dy*r))+R**2.*sp.arccos((dy**2.+R**2.-r**2.)/(2.0*dy*R))-0.5*sp.sqrt((-dy+r+R)*(dy+r-R)*(dy-r+R)*(dy+r+R))
                overlap_fraction[i] = overlap_area/A
        else:
            overlap_fraction[i] = 0 #turbines cannot be affected by any wakes that start downstream from them

    # print overlap_fraction
    return overlap_fraction #retrun the n x n matrix of how each turbine is affected by all of the others
                            #for example [0, 0.5]
                                        #[0, 0] means that the first turbine (row one) has half of its area in the
                                        #wake of the second turbine (row two). The overlap_fraction on the second
                                        #turbine is zero, so we can conclude that it is upstream of the first


#Jensen wake decay to determine the total velocity deficit at each turbine
def loss(r_0, a, alpha, x_focus, x, overlap):
    loss = np.zeros(np.size(x))
    loss_squared = np.zeros(np.size(x))
    dx = np.zeros(np.size(x))
    for i in range(0, np.size(x)):
        dx[i] = x_focus-x[i]
        if dx[i] > 0:
            loss[i] = overlap[i]*2.*a*(r_0/(r_0+alpha*(dx[i])))**2
            loss_squared[i] = loss[i]**2
        else:
            loss[i] = 0
            loss_squared[i] = 0
    total_loss = sp.sqrt(np.sum(loss_squared))
    return total_loss


def jensen_power(overlap, loss, x, y, r_0, alpha, a, U_velocity, rho, Cp):
    "Effective velocity at each turbine"
    A = r_0**2*np.pi
    V = np.zeros([np.size(x)])
    total_loss = np.zeros([np.size(x)])
    # print "\nOverlap Fraction Matrix"
    for i in range(0, np.size(x)):
        overlap_fraction = overlap(x, x[i], y, y[i], r_0, alpha)
        total_loss[i] = loss(r_0, a, alpha, x[i], x, overlap_fraction)
        V[i] = (1-total_loss[i])*U_velocity
    # print V
    # print "Total Loss: ", total_loss
    "Calculate Power from each turbine and the total"
    P = np.zeros([np.size(x)])
    P = 0.5*rho*A*Cp*V**3

    P_total = np.sum(P)

    # print "\nPower Output From Each Turbine"
    # print P
    # print "\nTotal Power Output"
    # print P_total
    return P_total


def jensen_plot(x, y, r_0, alpha, U_direction_radians):
    #plt.plot([x], [y], 'ro', markersize=10)

    wakes = np.linspace(0, 1000, num=101)

    for i in range(0, np.size(y)):
        turbine_y_top = y[i]+r_0
        turbine_y_bottom = y[i]-r_0
        turbine_x = [x[i], x[i], x[i]]
        turbine_y = [turbine_y_bottom, y[i], turbine_y_top]
        plt.plot(turbine_x, turbine_y, linewidth=2, c='r')
        for j in range(1, np.size(wakes)):
            wake_x = x[i]+wakes[j]
            wake_top_y = y[i]+r_0+wakes[j]*alpha
            wake_bottom_y = y[i]-r_0-wakes[j]*alpha
            plt.plot(wake_x, wake_top_y, 'b.', markersize=2)
            plt.plot(wake_x, wake_bottom_y, 'b.', markersize=2)


def rotate(x, y, U_direction_radians):
    x_r = x*sp.cos(U_direction_radians)-y*sp.sin(U_direction_radians)
    y_r = x*sp.sin(U_direction_radians)+y*sp.cos(U_direction_radians)
    return x_r, y_r


if __name__ == '__main__':

    "Define Variables"
    theta = 0.1
    alpha = sp.tan(theta)
    x = np.array([1000, 1000, 1000, 2000, 2000, 2000, 3000, 3000, 3000]) #x coordinates of the turbines
    y = np.array([1500, 2000, 3000, 1000, 2000, 3000, 1000, 2000, 3000]) #y coordinates of the turbines
    rho = 1.1716
    a = 1. / 3.
    U_velocity = 8.
    "0 degrees is coming from due North. +90 degrees means the wind is coming from due East, -90 from due West"
    U_direction = 45.
    r_0 = 40

    U_direction_radians = (U_direction+90) * np.pi / 180.
    #print U_direction_radians
    Cp = 4.*a*(1-a)**2.

    # x_r, y_r = rotate(x, y, U_direction_radians)

    # Jensen_Wake_Model(overlap, loss, jensen_power, jensen_plot, x, y, r_0, alpha, U_direction_radians)
    # xin = np.hstack([x, y])
    params = (r_0, U_velocity)
    print Jensen_Wake_Model(x, y, params)

    # wakes = np.linspace(0, 1000, num=101)
    #
    # plt.figure(2)
    # for i in range(0, np.size(y)):
    #     turbine_x_top = ((wakes[0])*sp.cos(-U_direction_radians)-(r_0+alpha*wakes[0])*sp.sin(-U_direction_radians))+x[i]
    #     turbine_y_top = ((wakes[0])*sp.sin(-U_direction_radians)+(r_0+alpha*wakes[0])*sp.cos(-U_direction_radians))+y[i]
    #     turbine_x_bottom = ((wakes[0])*sp.cos(-U_direction_radians)-(-r_0-alpha*wakes[0])*sp.sin(-U_direction_radians))+x[i]
    #     turbine_y_bottom = ((wakes[0])*sp.sin(-U_direction_radians)+(-r_0-alpha*wakes[0])*sp.cos(-U_direction_radians))+y[i]
    #     turbine_x = [turbine_x_bottom, x[i], turbine_x_top]
    #     turbine_y = [turbine_y_bottom, y[i], turbine_y_top]
    #     plt.plot(turbine_x, turbine_y, linewidth=2, c='r')
    #     for j in range(1, np.size(wakes)):
    #         top_x = ((wakes[j])*sp.cos(-U_direction_radians)-(r_0+alpha*wakes[j])*sp.sin(-U_direction_radians))+x[i]
    #         top_y = ((wakes[j])*sp.sin(-U_direction_radians)+(r_0+alpha*wakes[j])*sp.cos(-U_direction_radians))+y[i]
    #         bottom_x = ((wakes[j])*sp.cos(-U_direction_radians)-(-r_0-alpha*wakes[j])*sp.sin(-U_direction_radians))+x[i]
    #         bottom_y = ((wakes[j])*sp.sin(-U_direction_radians)+(-r_0-alpha*wakes[j])*sp.cos(-U_direction_radians))+y[i]
    #         plt.plot([top_x], [top_y], 'b.', markersize=2)
    #         plt.plot([bottom_x], [bottom_y], 'b.', markersize=2)
    # plt.show()
