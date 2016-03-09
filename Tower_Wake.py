import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi, tan, cos, acos, sin, sqrt, atan


#def loss_on_VAWT_from_HAWT(xin, params):



def overlap_cylinder(x_h,y_h,x_v,y_v,r_tower,r_vawt):
    #radius tower, vertical ,wake
    theta = 10.0 * pi/180.0   # angle to radians
    n_cyl = len(x_v)   # number of cylinders
    overlap_cyl = np.zeros([n_cyl,len(x_h)])
    area_vawt = pi * r_vawt**2.
    for i in range(n_cyl):
        for j in range(len(x_h)):
            dx = x_v[i] - x_h[j]
            R_wake = tan(theta) * dx + r_tower
            dy = abs(y_v[i] - y_h[j])
            if dx > 0:
                if abs(dy) >= R_wake + r_tower:   #Rwake +rtower is equal to total Rwake
                    overlap_cyl[i][j] = 0
                elif abs(dy) <= R_wake - r_tower:
                    overlap_cyl[i][j] = 1.0
                else:
                    beta = dx * tan(theta) - (dy - r_tower)
                    if beta > 0:
                        d = beta * cos(theta)
                        phi = 2. * acos(d/r_vawt)
                        area_overlap = (r_vawt**2./2.) * (phi - sin(phi))
                        overlap_cyl[i][j] = 1 - area_overlap/area_vawt
                    else:
                        d = beta * cos(theta)
                        phi = 2. * acos(d/r_vawt)
                        area_overlap = area_vawt - (r_vawt**2./2.) * (phi - sin(phi))
                        overlap_cyl[i][j] = area_overlap/area_vawt
            else:
                overlap_cyl[i][j] = 0

    return overlap_cyl


def loss_cylinder(x_h, y_h, x_v, y_v, r_tower, r_vawt):

    theta = 10.0*pi/180.
    loss = np.zeros(np.size(x_h))
    loss_squared = np.zeros(np.size(x_h))
    total_loss = np.zeros(np.size(x_v))
    Cd = 0.3
    overlap_cyl = overlap_cylinder(x_h, y_h, x_v, y_v, r_tower, r_vawt)
    for i in range(len(x_v)):
        for j in range(len(x_h)):
            dy = abs(y_v[i] - y_h[j])
            dx = x_v[i] - x_h[j]
            R_wake = tan(theta) * dx + r_tower
            if dx > 0:
                loss[j] = overlap_cyl[i][j] * sqrt((3*r_tower+Cd*r_tower)/(3*r_tower + 2*dx*tan(theta)))*(cos(-dy*pi/(R_wake+4*r_vawt)))
                loss_squared [j] = loss[j]**2
            else:
                loss[j] = 0
                loss_squared[j] = 0
        total_loss[i] = sqrt(np.sum(loss_squared))
    return total_loss



def cylinder_plot(x, y,r_tower, alpha, U_direction_radians):

    wakes = np.linspace(0, 1000, num=101)
    fig, ax = plt.subplots(1,1)

    for i in range(0, np.size(y)):
        ax.scatter(x, y, s=r_tower**2,color = 'r')
        for j in range(1, np.size(wakes)):
            top_x = ((wakes[j])*sp.cos(-U_direction_radians)-(r_tower+alpha*wakes[j])*sp.sin(-U_direction_radians))+x[i]
            top_y = ((wakes[j])*sp.sin(-U_direction_radians)+(r_tower+alpha*wakes[j])*sp.cos(-U_direction_radians))+y[i]
            bottom_x = ((wakes[j])*sp.cos(-U_direction_radians)-(-r_tower-alpha*wakes[j])*sp.sin(-U_direction_radians))+x[i]
            bottom_y = ((wakes[j])*sp.sin(-U_direction_radians)+(-r_tower-alpha*wakes[j])*sp.cos(-U_direction_radians))+y[i]
            plt.plot([top_x], [top_y], 'b.', markersize=2)
            plt.plot([bottom_x], [bottom_y], 'b.', markersize=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':

    "Define Variables"
    theta = 1./18.*pi
    alpha = sp.tan(theta)
    x_h = np.array([0,500,1000,1500]) #x coordinates of the turbines
    y_h = np.array([0,20,200,500]) #y coordinates of the turbines
    x_v = np.array([200, 300, 400,955, 1500])
    y_v = np.array([20, 120, 220,90, 400])
    rho = 1.1716
    a = 1. / 3.
    U_velocity = 8.
    "0 degrees is coming from due North. +90 degrees means the wind is coming from due East, -90 from due West"
    U_direction = -90.
    r_tower = 40.
    r_vawt = 50.
    U_direction_radians = (U_direction+90) * pi / 180.

    # TEST
    r_tower = 40.
    r_vawt = 50.
    print overlap_cylinder(x_h,y_h, x_v, y_v, r_tower, r_vawt)
    #print loss_cylinder(x_h, y_h, x_v, y_v, r_tower, r_vawt)
    #plot = cylinder_plot(x,y,r_tower,alpha,U_direction_radians)
    plt.scatter(x_h,y_h)
    plt.scatter(x_v,y_v,c='r')
    plt.show()
