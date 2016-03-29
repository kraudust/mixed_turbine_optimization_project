"""
Parameterized VAWT Wake Model using CFD vorticity data
Developed by Eric Tingey at Brigham Young University

This code models the wake behind a vertical-axis wind turbine based on
parameters like tip-speed ratio, solidity and wind speed by converting the
vorticity of the wake into velocity information. The model uses CFD data
obtained from STAR-CCM+ of simulated turbines to make the wake model as
accurate as possible.

Only valid for tip-speed ratios between 1.5 and 7.0 and solidities between
0.15 and 1.0. Reynolds numbers should also be around the range of 200,000 to
6,000,000.

In this code, the x and y coordinates are made according to:

--------------->--------------------------------------------------------
--------------->--------------------------------------------------------
--------------->---------=====--------#################-----------Y-----
--------------->------//       \\#############################----|-----
-FREE-STREAM--->-----|| TURBINE ||########## WAKE ###############-|___X-
----WIND------->-----||         ||###############################-------
--------------->------\\       //#############################----------
--------------->---------=====--------#################-----------------
--------------->--------------------------------------------------------
--------------->--------------------------------------------------------

The imported vorticity data also assumes symmetry in the wake and therefore
rotation direction is irrelevant.

"""


import csv
from os import path
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi,exp,fabs,sqrt
from scipy.integrate import dblquad
from scipy.interpolate import RectBivariateSpline

import _vortmodel

# from matplotlib import rcParams
# rcParams['font.family'] = 'Times New Roman'


def vorticity(tsr,solidity):
    """
    Using EMG distribution parameters to define the vorticity strength and shape
    
    Parameters
    ----------
    tsr : float
        tip-speed ratio
    solidity : float
        turbine solidity
    
    Returns
    ----------
    loc : array
        array of the location parameter (3 values)
    spr : array
        array of the spread parameter (2 values)
    skw : array
        array of the skew parameter (2 values)
    scl : array
        array of the scale parameter (3 values)
    """
    
    # Reading in csv file (vorticity database)
    basepath = path.join(path.dirname(path.realpath(__file__)),'data')
    fdata = basepath + path.sep + 'vortdatabase.csv'
    f = open(fdata)
    csv_f = csv.reader(f)
    
    i = 0
    sol_d = np.array([])
    for row in csv_f:
        if i == 0:
            raw = row
            raw = np.delete(raw,0)
            vortdat = raw
            tsr_d = raw # range of tip-speed ratios included
        if row[0] == 'solidity':
            sol_d = np.append(sol_d,float(row[1])) # range of solidities included
        elif row[0] != 'TSR' and row[0] != 'solidity':
            raw = row
            raw = np.delete(raw,0)
            vortdat = np.vstack([vortdat,raw]) # adding entry to vorticity database array
        i += 1
    f.close()
    
    vortdat = np.delete(vortdat,(0),axis=0) # eliminating first row used as a placeholder
    tsr_d = tsr_d.astype(np.float) # converting tip-speed ratio entries into floats
    vortdat = vortdat.astype(np.float) # converting vorticity database entries into floats
    
    # Creating arrays for each EMG parameter
    for i in range(np.size(sol_d)):
        sol = str(i+1)
        
        exec('s'+sol+'_loc1 = vortdat[i*10]\ns'+sol+'_loc2 = vortdat[i*10+1]\ns'+sol+'_loc3 = vortdat[i*10+2]\ns'+sol+'_spr1 = vortdat[i*10+3]\ns'+sol+'_spr2 = vortdat[i*10+4]\ns'+sol+'_skw1 = vortdat[i*10+5]\ns'+sol+'_skw2 = vortdat[i*10+6]\ns'+sol+'_scl1 = vortdat[i*10+7]\ns'+sol+'_scl2 = vortdat[i*10+8]\ns'+sol+'_scl3 = vortdat[i*10+9]\n')
    
    # BIVARIATE SPLINE FITTING
    
    iz = np.size(sol_d)
    jz = np.size(tsr_d)
    
    # Initializing rectangular matrices
    Z_loc1 = np.zeros((iz,jz))
    Z_loc2 = np.zeros((iz,jz))
    Z_loc3 = np.zeros((iz,jz))
    Z_spr1 = np.zeros((iz,jz))
    Z_spr2 = np.zeros((iz,jz))
    Z_skw1 = np.zeros((iz,jz))
    Z_skw2 = np.zeros((iz,jz))
    Z_scl1 = np.zeros((iz,jz))
    Z_scl2 = np.zeros((iz,jz))
    Z_scl3 = np.zeros((iz,jz))
    
    # Transferring raw data into rectangular matrices
    for i in range(iz):
        for j in range(jz):
            sol = str(i+1)
            exec('Z_loc1[i,j] = s'+sol+'_loc1[j]')
            exec('Z_loc2[i,j] = s'+sol+'_loc2[j]')
            exec('Z_loc3[i,j] = s'+sol+'_loc3[j]')
            exec('Z_spr1[i,j] = s'+sol+'_spr1[j]')
            exec('Z_spr2[i,j] = s'+sol+'_spr2[j]')
            exec('Z_skw1[i,j] = s'+sol+'_skw1[j]')
            exec('Z_skw2[i,j] = s'+sol+'_skw2[j]')
            exec('Z_scl1[i,j] = s'+sol+'_scl1[j]')
            exec('Z_scl2[i,j] = s'+sol+'_scl2[j]')
            exec('Z_scl3[i,j] = s'+sol+'_scl3[j]')
    
    # Creating a rectangular bivariate spline of the parameter data
    s_loc1 = RectBivariateSpline(sol_d,tsr_d,Z_loc1)
    s_loc2 = RectBivariateSpline(sol_d,tsr_d,Z_loc2)
    s_loc3 = RectBivariateSpline(sol_d,tsr_d,Z_loc3)
    s_spr1 = RectBivariateSpline(sol_d,tsr_d,Z_spr1)
    s_spr2 = RectBivariateSpline(sol_d,tsr_d,Z_spr2)
    s_skw1 = RectBivariateSpline(sol_d,tsr_d,Z_skw1)
    s_skw2 = RectBivariateSpline(sol_d,tsr_d,Z_skw2)
    s_scl1 = RectBivariateSpline(sol_d,tsr_d,Z_scl1)
    s_scl2 = RectBivariateSpline(sol_d,tsr_d,Z_scl2)
    s_scl3 = RectBivariateSpline(sol_d,tsr_d,Z_scl3)
    
    # Selecting the specific parameters to use for TSR and solidity
    loc1 = s_loc1(solidity,tsr)
    loc2 = s_loc2(solidity,tsr)
    loc3 = s_loc3(solidity,tsr)
    spr1 = s_spr1(solidity,tsr)
    spr2 = s_spr2(solidity,tsr)
    skw1 = s_skw1(solidity,tsr)
    skw2 = s_skw2(solidity,tsr)
    scl1 = s_scl1(solidity,tsr)
    scl2 = s_scl2(solidity,tsr)
    scl3 = s_scl3(solidity,tsr)
    
    # Creating arrays of the parameters
    loc = np.array([loc1[0,0],loc2[0,0],loc3[0,0]])
    spr = np.array([spr1[0,0],spr2[0,0]])
    skw = np.array([skw1[0,0],skw2[0,0]])
    scl = np.array([scl1[0,0],scl2[0,0],scl3[0,0]])
    
    return loc,spr,skw,scl


def velocity(tsr,solidity):
    """
    Using SMG distribution parameters to define the velocity strength and shape
    
    Parameters
    ----------
    tsr : float
        tip-speed ratio
    solidity : float
        turbine solidity
    
    Returns
    ----------
    men : array
        array of the mean parameter (3 values)
    spr : array
        array of the spread parameter (3 values)
    scl : array
        array of the scale parameter (3 values)
    rat : array
        array of the rate parameter (2 values)
    tns : array
        array of the translation parameter (2 values)
    """
    # Reading in csv file (vorticity database)
    basepath = path.join(path.dirname(path.realpath(__file__)),'data')
    fdata = basepath + path.sep + 'velodatabase.csv'
    f = open(fdata)
    csv_f = csv.reader(f)
    
    i = 0
    sol_d = np.array([])
    for row in csv_f:
        if i == 0:
            raw = row
            raw = np.delete(raw,0)
            velodat = raw
            tsr_d = raw # range of tip-speed ratios included
        if row[0] == 'solidity':
            sol_d = np.append(sol_d,float(row[1])) # range of solidities included
        elif row[0] != 'TSR' and row[0] != 'solidity':
            raw = row
            raw = np.delete(raw,0)
            velodat = np.vstack([velodat,raw]) # adding entry to vorticity database array
        i += 1
    f.close()
    
    velodat = np.delete(velodat,(0),axis=0) # eliminating first row used as a placeholder
    tsr_d = tsr_d.astype(np.float) # converting tip-speed ratio entries into floats
    velodat = velodat.astype(np.float) # converting vorticity database entries into floats
    
    # Creating arrays for each EMG parameter
    for i in range(np.size(sol_d)):
        sol = str(i+1)
        
        exec('s'+sol+'_men1 = velodat[i*13]\ns'+sol+'_men2 = velodat[i*13+1]\ns'+sol+'_men3 = velodat[i*13+2]\ns'+sol+'_spr1 = velodat[i*13+3]\ns'+sol+'_spr2 = velodat[i*13+4]\ns'+sol+'_spr3 = velodat[i*13+5]\ns'+sol+'_scl1 = velodat[i*13+6]\ns'+sol+'_scl2 = velodat[i*13+7]\ns'+sol+'_scl3 = velodat[i*13+8]\ns'+sol+'_rat1 = velodat[i*13+9]\ns'+sol+'_rat2 = velodat[i*13+10]\ns'+sol+'_tns1 = velodat[i*13+11]\ns'+sol+'_tns2 = velodat[i*13+12]\n')
    
    # BIVARIATE SPLINE FITTING
    
    iz = np.size(sol_d)
    jz = np.size(tsr_d)
    
    # Initializing rectangular matrices
    Z_men1 = np.zeros((iz,jz))
    Z_men2 = np.zeros((iz,jz))
    Z_men3 = np.zeros((iz,jz))
    Z_spr1 = np.zeros((iz,jz))
    Z_spr2 = np.zeros((iz,jz))
    Z_spr3 = np.zeros((iz,jz))
    Z_scl1 = np.zeros((iz,jz))
    Z_scl2 = np.zeros((iz,jz))
    Z_scl3 = np.zeros((iz,jz))
    Z_rat1 = np.zeros((iz,jz))
    Z_rat2 = np.zeros((iz,jz))
    Z_tns1 = np.zeros((iz,jz))
    Z_tns2 = np.zeros((iz,jz))
    
    # Transferring raw data into rectangular matrices
    for i in range(iz):
        for j in range(jz):
            sol = str(i+1)
            exec('Z_men1[i,j] = s'+sol+'_men1[j]')
            exec('Z_men2[i,j] = s'+sol+'_men2[j]')
            exec('Z_men3[i,j] = s'+sol+'_men3[j]')
            exec('Z_spr1[i,j] = s'+sol+'_spr1[j]')
            exec('Z_spr2[i,j] = s'+sol+'_spr2[j]')
            exec('Z_spr3[i,j] = s'+sol+'_spr3[j]')
            exec('Z_scl1[i,j] = s'+sol+'_scl1[j]')
            exec('Z_scl2[i,j] = s'+sol+'_scl2[j]')
            exec('Z_scl3[i,j] = s'+sol+'_scl3[j]')
            exec('Z_rat1[i,j] = s'+sol+'_rat1[j]')
            exec('Z_rat2[i,j] = s'+sol+'_rat2[j]')
            exec('Z_tns1[i,j] = s'+sol+'_tns1[j]')
            exec('Z_tns2[i,j] = s'+sol+'_tns2[j]')
    
    # Creating a rectangular bivariate spline of the parameter data
    s_men1 = RectBivariateSpline(sol_d,tsr_d,Z_men1)
    s_men2 = RectBivariateSpline(sol_d,tsr_d,Z_men2)
    s_men3 = RectBivariateSpline(sol_d,tsr_d,Z_men3)
    s_spr1 = RectBivariateSpline(sol_d,tsr_d,Z_spr1)
    s_spr2 = RectBivariateSpline(sol_d,tsr_d,Z_spr2)
    s_spr3 = RectBivariateSpline(sol_d,tsr_d,Z_spr3)
    s_scl1 = RectBivariateSpline(sol_d,tsr_d,Z_scl1)
    s_scl2 = RectBivariateSpline(sol_d,tsr_d,Z_scl2)
    s_scl3 = RectBivariateSpline(sol_d,tsr_d,Z_scl3)
    s_rat1 = RectBivariateSpline(sol_d,tsr_d,Z_rat1)
    s_rat2 = RectBivariateSpline(sol_d,tsr_d,Z_rat2)
    s_tns1 = RectBivariateSpline(sol_d,tsr_d,Z_tns1)
    s_tns2 = RectBivariateSpline(sol_d,tsr_d,Z_tns2)
    
    # Selecting the specific parameters to use for TSR and solidity
    men1 = s_men1(solidity,tsr)
    men2 = s_men2(solidity,tsr)
    men3 = s_men3(solidity,tsr)
    spr1 = s_spr1(solidity,tsr)
    spr2 = s_spr2(solidity,tsr)
    spr3 = s_spr3(solidity,tsr)
    scl1 = s_scl1(solidity,tsr)
    scl2 = s_scl2(solidity,tsr)
    scl3 = s_scl3(solidity,tsr)
    rat1 = s_rat1(solidity,tsr)
    rat2 = s_rat2(solidity,tsr)
    tns1 = s_tns1(solidity,tsr)
    tns2 = s_tns2(solidity,tsr)
    
    # Creating arrays of the parameters
    men = np.array([men1[0,0],men2[0,0],men3[0,0]])
    spr = np.array([spr1[0,0],spr2[0,0],spr3[0,0]])
    scl = np.array([scl1[0,0],scl2[0,0],scl3[0,0]])
    rat = np.array([rat1[0,0],rat2[0,0]])
    tns = np.array([tns1[0,0],tns2[0,0]])

    return men,spr,scl,rat,tns


def velocity_field(xt,yt,x0,y0,velf,dia,tsr,solidity,cfd_data,param):
    """
    Calculating normalized velocity from the vorticity data at (x0,y0) in global flow domain
    
    Parameters
    ----------
    xt : float
        downstream position of turbine domain (m)
    yt : float
        lateral position of turbine in flow domain (m)
    x0 : float
        downstream position in flow domain to be calculated (m)
    y0 : float
        lateral position in flow domain to be calculated (m)
    velf : float
        free stream velocity (m/s)
    dia : float
        turbine diameter (m)
    tsr : float
        tip-speed ratio; [rotation rate (rad/s)]*[turbine radius (m)]/[free stream velocity (m/s)]
    solidity : float
        turbine solidity; [number of turbine blades]*[blade chord length (m)]/[turbine radius (m)]
    cfd_data : string
        specifying to use CFD vorticity ('vort') or velocity ('velo') for the basis of the wake model
    
    Returns
    ----------
    vel : float
        final normalized velocity at (x0,y0) with respect to the free stream velocity (m/s)
    """
    rad = dia/2.
    rot = tsr*velf/rad
    
    # Translating the turbine position
    x0t = x0 - xt
    y0t = y0 - yt
    
    if cfd_data == 'vort':
        # Calculating EMG distribution parameters
        # loc,spr,skw,scl = vorticity(tsr,solidity)
        loc = param[0]
        spr = param[1]
        skw = param[2]
        scl = param[3]
        
        # Integration of the vorticity profile using Fortran code (vorticity.f90)
        vel_vs = dblquad(_vortmodel.integrand,0.,35.*dia,lambda x: -4.*dia,lambda x: 4.*dia, args=(x0t,y0t,dia,loc[0],loc[1],loc[2],spr[0],spr[1],skw[0],skw[1],scl[0],scl[1],scl[2]))
        
        # Calculating velocity deficit
        vel = (vel_vs[0]*(rot))/(2.*pi)
        vel = (vel + velf)/velf # normalization of velocity
        
    elif cfd_data == 'velo':
        # Normalizing the downstream and lateral positions by the turbine diameter
        x0d = x0/dia
        y0d = y0/dia
        
        # Calculating SMG distribution parameters
        # men,spr,scl,rat,tns = velocity(tsr,solidity)
        # men = param[0]
        # spr = param[1]
        # scl = param[2]
        # rat = param[3]
        # tns = param[4]
        
        #men = np.array( [-0.0007448786610163438, 0.011700465818493566, -0.005332505770684337] )
        #spr = np.array( [6.462355161093711, 7.079901300173991, 12.102886237210939] )
        #scl = np.array( [8.509272717226171, 7.023483471068396, 27.707846411384697] )
        #rat = np.array( [-2.107186196351149, 44.93845180541949] )
        #tns = np.array( [-1.4660542829002265, 30.936653231840257] )
         
        men = np.array( [-0.00059737414699399, 0.009890587474506057, -0.0016721254639608882] )
        spr = np.array( [-0.005652314031253564, 0.06923002880544946, 0.526304136118912] )
        scl = np.array( [6.639808894608728, 5.477607580787858, 21.13678312202297] )
        rat = np.array( [-2.0794010451530873, 44.798557035611] )
        tns = np.array( [-1.43164706657537, 30.761785195818447] )

        #men = np.array( [-0.0006344223751663201, 0.01055675755786011, -0.004073212523707764] )
        #spr = np.array( [-0.005187125854670714, 0.06397918461247416, 0.543874357807372] )
        #scl = np.array( [6.667328694868336, 5.617498827673229, 21.520026361522778] )
        #rat = np.array( [-2.129054494312758, 45.17191461412915] )
        #tns = np.array( [-1.5569348878268718, 31.913143231782648] )
        #param = np.array([men,spr,scl,rat,tns])

        
        men_v = men[0]*x0d**2 + men[1]*x0d + men[2]
        if men_v > 0.5:
            men_v = 0.5
        elif men_v < -0.5:
            men_v = -0.5
        
        # spr_v = spr[2]/(spr[1]*sqrt(2.*pi))*exp(-(x0d-spr[0])**2/(2.*spr[1]**2))
        spr_v = spr[0]*x0d**2 + spr[1]*x0d + spr[2]
        if spr_v < 0.35:
            spr_v = 0.35
        
        scl_v = scl[2]/(scl[1]*sqrt(2.*pi))*exp(-(x0d-scl[0])**2/(2.*scl[1]**2))
        
        rat_v = rat[0]*x0d + rat[1]
        if rat_v < 0.:
            rat_v = 0.
        
        tns_v = tns[0]*x0d + tns[1]
        if tns_v < 0.:
            tns_v = 0.
        
        vel = (-scl_v/(spr_v*sqrt(2.*pi))*exp(-(y0d+men_v)**2/(2.*spr_v**2)))*(1./(1 + exp(rat_v*fabs(y0d)-tns_v))) + 1. # Normal distribution with sigmoid weighting
        
        if x0 < xt:
            vel = 1. # Velocity is free stream in front and to the sides of the turbine
    
    return vel
    
if __name__ == '__main__':
	velf = 15.0 # free stream wind speed (m/s)
	dia = 6.0  # turbine diameter (m)
	tsr = 4.0  # tip speed ratio
	B = 3. # number of blades
	chord = 0.25 # chord lenth (m)
	solidity = (chord*B)/(dia/2.)
	cfd_data = 'velo'
	param = []
	print 1. - velocity_field(0., 1000000., 0., -1000000., velf, dia, tsr, solidity, cfd_data, param)   


