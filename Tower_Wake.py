import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi, tan, cos, acos, sin


def overlap_cylinder(x,y,r_tower,r_vawt):
    #radius tower, vertical ,wake
    theta = 15.0 * pi/180.0   # angle to radians
    n_cyl = len(x)   # number of cylinders
    overlap_cyl = np.zeros([n_cyl,n_cyl])
    area_vawt = pi * r_vawt**2.
    for i in range(n_cyl):
        for j in range(n_cyl):
            dx = x[i] - x[j]
            R_wake = tan(theta) * dx + r_tower
            dy = y[i] - y[j]
            print R_wake, dy
            if dx > 0:
                if abs(dy) >= R_wake + r_tower:
                    overlap_cyl[i][j] = 0
                elif abs(dy) <= R_wake - r_tower:
                    overlap_cyl[i][j] = 1.0
                else:
                    beta = dx * tan(theta) - dy
                    if beta > 0:
                        d = beta * cos(theta)
                        phi = 2. * acos(d/r_vawt)
                        area_overlap = (r_vawt**2./2.) * (phi - sin(phi))
                        overlap_cyl[i][j] = area_overlap/area_vawt
                    else:
                        d = abs(beta) * cos(theta)
                        phi = 2. * acos(d/r_vawt)
                        area_overlap = area_vawt - (r_vawt**2./2.) * (phi - sin(phi))
                        overlap_cyl[i][j] = area_overlap/area_vawt
            else:
                overlap_cyl[i][j] = 0

    return overlap_cyl

# def loss_cylinder()
#
#
# def cylinder_plot(x, y, r_0 => r_tower, alpha, U_direction_radians):
#
#     wakes = np.linspace(0, 1000, num=101)
#     fig, ax = plt.subplots(1,1)
#
#     for i in range(0, np.size(y)):
#         ax.scatter(x, y, s=r_0,color = 'r')
#         for j in range(1, np.size(wakes)):
#             top_x = ((wakes[j])*sp.cos(-U_direction_radians)-(r_0+alpha*wakes[j])*sp.sin(-U_direction_radians))+x[i]
#             top_y = ((wakes[j])*sp.sin(-U_direction_radians)+(r_0+alpha*wakes[j])*sp.cos(-U_direction_radians))+y[i]
#             bottom_x = ((wakes[j])*sp.cos(-U_direction_radians)-(-r_0-alpha*wakes[j])*sp.sin(-U_direction_radians))+x[i]
#             bottom_y = ((wakes[j])*sp.sin(-U_direction_radians)+(-r_0-alpha*wakes[j])*sp.cos(-U_direction_radians))+y[i]
#             plt.plot([top_x], [top_y], 'b.', markersize=2)
#             plt.plot([bottom_x], [bottom_y], 'b.', markersize=2)
#
#
def rotate(x, y, U_direction_radians):
    x_r = x*sp.cos(U_direction_radians)-y*sp.sin(U_direction_radians)
    y_r = x*sp.sin(U_direction_radians)+y*sp.cos(U_direction_radians)
    return x_r, y_r


if __name__ == '__main__':

    "Define Variables"
    theta = 0.1
    alpha = sp.tan(theta)
    x = np.array([0,80,80,80,80,80,80,80,80]) #x coordinates of the turbines
    y = np.array([0,20,21,22,23,25,30,35,40]) #y coordinates of the turbines
    rho = 1.1716
    a = 1. / 3.
    U_velocity = 8.
    "0 degrees is coming from due North. +90 degrees means the wind is coming from due East, -90 from due West"
    U_direction = -90.
    r_0 = 40.

    U_direction_radians = (U_direction+90) * pi / 180.

    x_r, y_r = rotate(x,y,U_direction_radians)
    # TEST
    r_tower = 40.
    r_vawt = 50.
    print overlap_cylinder(x_r,y_r, r_tower, r_vawt)

    plt.scatter(x,y)
    plt.show()