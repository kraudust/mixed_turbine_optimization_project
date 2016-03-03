from Main_Obj import *
from scipy.optimize import minimize



def added_con(xin,params):
    xupper = 1200.
    yupper = 1200.
    xVAWT = xin[0 : nVAWT]
    yVAWT = xin[nVAWT: 2*nVAWT]
    xHAWT = xin[2*nVAWT: 2*nVAWT + nHAWT]
    yHAWT = xin[2*nVAWT + nHAWT : len(xin)]
    cons = con(xin, params)

    for i in range(0, len(xVAWT)):
        cons = np.append(cons, xVAWT[i])
        cons = np.append(cons, xupper - xVAWT[i])
        cons = np.append(cons, yVAWT[i])
        cons = np.append(cons, yupper - yVAWT[i])
    for i in range(0, len(xHAWT)):
        cons = np.append(cons, xHAWT[i])
        cons = np.append(cons, xupper - xHAWT[i])
        cons = np.append(cons, yHAWT[i])
        cons = np.append(cons, yupper - yHAWT[i])
    return cons


def un_obj(xin,args):   #unconstrained power
    c_handle = args[3]
    f_handle = args[2]
    mu = args[1]
    params = args[0]

    const = c_handle(xin, params)
    sum_quad = 0
    for i in range(0, len(const)):
        sum_quad += np.min([0, const[i]])**2
    unc_obj = f_handle(xin, params) + (mu/2)*sum_quad
    return unc_obj

def unc_optimizer(xin, params, f_handle, c_handle):
    mu = 1.0
    dif = 1.0
    args = (params, mu, f_handle, c_handle)
    while np.abs(dif) > 1e-6:
        previous = un_obj(xin,args)
        options = {'disp': False, 'maxiter': 300}
        unc_opt = minimize(un_obj, xin, args=(args,), method='BFGS', options= options)
        mu = mu*2.
        args = (params, mu, f_handle, c_handle)
        xin = unc_opt.x
        dif = un_obj(xin, args) - previous

    return xin

def test(xin,params):
    sum = 0
    for i in range(len(xin)):
        sum += xin[i]**2
    return sum

def test_con(xin,params):
    cons = np.zeros(len(xin))
    for i in range(len(xin)):
        cons[i] = xin[i] - i
    return cons



if __name__ == '__main__':
    xHAWT = np.array([0,1,2])
    yHAWT = np.array([0,1,2])
    xVAWT = np.array([0])
    yVAWT = np.array([5])
    xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])


    nVAWT = len(xVAWT)
    nHAWT = len(xHAWT)
    rh = 40.
    rv = 3.
    rt = 5.
    direction = -91.
    dir_rad = (direction+90) * np.pi / 180.
    U_vel = 8.
    params = [nVAWT, rh, rv, rt, dir_rad, U_vel]



    print "Original Power: ", -obj(xin, params)*1.e6
    print "Old X:", xin
    optX = unc_optimizer(xin, params, obj, added_con)
    print "New OptX:", optX
    print "New Power", -obj(optX,params)*1.e6


#--------------------------Plots-----------------------------------
    xVAWT_opt = optX[0 : nVAWT]
    yVAWT_opt = optX[nVAWT: 2*nVAWT]
    xHAWT_opt = optX[2*nVAWT: 2*nVAWT + nHAWT]
    yHAWT_opt = optX[2*nVAWT + nHAWT : len(xin)]

    plt.figure()
    plt.scatter(xVAWT, yVAWT,s=(np.pi*rv**2.), c = 'k',label= "Starting Horizontal")
    plt.scatter(xVAWT_opt, yVAWT_opt,s=(np.pi*rv**2.), c = 'r',label = "Opt. Horizontal")
    plt.scatter(xHAWT, yHAWT,s=(15*np.pi*rt**2.), c = 'c',label = "Starting Vertical")
    plt.scatter(xHAWT_opt, yHAWT_opt,s=(15*np.pi*rt**2.), c = 'g', label = "Opt. Vertical")
    plt.legend(loc=2)
    plt.show()