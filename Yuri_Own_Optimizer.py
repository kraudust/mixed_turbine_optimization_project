from Main_Obj import *
from scipy.optimize import minimize



def added_con(xin,params):
    xupper = 2000.
    yupper = 2000.
    cons = con(xin,params)
    xVAWT = xin[0 : nVAWT]
    yVAWT = xin[nVAWT: 2*nVAWT]
    xHAWT = xin[2*nVAWT: 2*nVAWT + nHAWT]
    yHAWT = xin[2*nVAWT + nHAWT : len(xin)]

    for i in range(len(xHAWT)):
        cons = np.append(cons,xHAWT[i])
        cons = np.append(cons,xupper - xHAWT[i])
        cons = np.append(cons,yHAWT[i])
        cons = np.append(cons,yupper - yHAWT[i])
    for i  in range(len(yVAWT)):
        cons = np.append(cons,xVAWT[i])
        cons = np.append(cons,xupper - xVAWT[i])
        cons = np.append(cons,yVAWT[i])
        cons = np.append(cons,yupper - yVAWT[i])

        return cons


def un_obj(xin,mu,params):   #power
    cons = added_con(xin,params)
    nlog = np.zeros(len(cons))
    for i in range(len(cons)):
        if cons[i] == 0:
            nlog[i] = 0
        elif cons[i] < 0:
            nlog[i] = -1000.
        else:
            nlog[i] = np.log10(cons[i])

    log_sum = np.sum(nlog)
    fun = obj(xin,params)
    un_obj = fun - mu*log_sum
    print un_obj
    return un_obj





if __name__ == '__main__':
    xHAWT = np.array([0, 200])
    yHAWT = np.array([0, 200])
    xVAWT = np.array([100])
    yVAWT = np.array([0])

    xin = np.hstack([xVAWT, yVAWT, xHAWT, yHAWT])
    nVAWT = len(xVAWT)
    nHAWT = len(xin)/2. - nVAWT
    rh = 40.
    rv = 3.
    rt = 5.
    direction = 30.
    dir_rad = (direction+90) * np.pi / 180.
    U_vel = 8.
    mu = 1000.
    params = [nVAWT, rh, rv, rt, dir_rad, U_vel]
    args = mu,params
    print "Original Power: ", obj(xin, params)

    #--------------- Unconstrained Optimizer --------------------
    options = {'disp': True, 'maxiter': 300}
    unc_opt = minimize(un_obj, xin, args=args, method='BFGS', options= options)
    optX = unc_opt.x


    while np.max(np.abs(optX)) > 1e-6:
        previous = unc_opt.x
        print np.max(previous)
        mu = mu/2.
        options = {'disp': False, 'maxiter': 300}
        unc_opt = minimize(un_obj, xin, args=args, method='BFGS', options= options)
        optX = unc_opt.x - previous
        print optX
        print np.max(optX)
        print mu

    print unc_opt.x
