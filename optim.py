# Optimization routines  # Author: Aurelien Lucchi

from math import log
from math import sqrt

import random
import time
import os
from datetime import datetime


import numpy as np
from scipy import linalg
from sklearn.utils.extmath import randomized_svd

def set_global_variables(gt):
    global full_ground_truth
    full_ground_truth = gt

###############################################################################
# Helpers functions
# including loss, stochastic gradient, Hessian, ...
###############################################################################

# X = data matrix
# Y = groundtruth
# w = parameter
# opt = structure containing parameters
def loss(X, Y, w, opt):
    n = X.shape[0]
    d = X.shape[1]
    l = 0
    if opt['loss_type'] == 'square_loss': #1/2 ||Y-Xw||^2
        #P = [np.dot(w, X[i]) for i in range(n)]  # prediction <w, x> (row of X 'times' w)
        P = X.dot(w)
        l = 0.5 * np.average([(Y[i] - P[i]) ** 2 for i in range(len(Y))])
        #l = 0.5 * np.dot(Y-P,Y-P)/n
        l = l + 0.5 * opt['lambda'] * (np.linalg.norm(w) ** 2)
    elif opt['loss_type'] == 'hinge_loss':
        P = [np.dot(w, X[i]) for i in range(n)]  # prediction <w, x>
        l = np.sum([max(0, 1 - Y[i] * P[i]) for i in range(len(Y))]) / n
        l = l + 0.5 * opt['lambda'] * (np.linalg.norm(w) ** 2)
    elif opt['loss_type'] == 'binary_regression': #!This loss function is only valid for y in {1,0}!
            z = X.dot(w)  # prediction <w, x>
            #h = phi(z)
            #l= - (np.dot(np.log(h),Y)+np.dot(np.ones(n)-Y,np.log(np.ones(n)-h)))/n
            l= - (np.dot(log_phi(z),Y)+np.dot(np.ones(n)-Y,one_minus_log_phi(z)))/n

            l = l + 0.5*  opt['lambda'] * (np.linalg.norm(w) ** 2)
    elif opt['loss_type'] == 'non_convex':
        alpha = opt['non_cvx_reg_alpha']
        z = X.dot(w)  # prediction <w, x>
        h = phi(z)
        l= - (np.dot(np.log(h),Y)+np.dot(np.ones(n)-Y,np.log(np.ones(n)-h)))/n
        l= l + opt['lambda']*np.dot(alpha*w**2,1/(1+alpha*w**2))

    elif opt['loss_type'] == 'multiclass_regression':
        w=np.matrix(w.reshape(opt['n_classes'],d).T)
        z=np.dot(X,w) #activation of each i for class c
        z-=np.max(z,axis=1)  # model is overparametrized. allows to subtract maximum to prevent overflow for each datapoint!
        h = np.exp(z)
        P= h/np.sum(h,axis = 1) #gives matrix with with [P]i,j = probability of sample i to be in class j (n x nC)
        error=np.multiply(full_ground_truth, np.log(P))
        l = -(np.sum(error) / n)
        l += 0.5*opt['lambda']*(np.sum(np.multiply(w,w))) #weight decay
        
    elif opt['loss_type'] == 'monkey':
        l = w[0] ** 3 - 3 * w[0] * w[1] ** 2
    elif opt['loss_type'] == 'rosenbrock':
        l = (1. - w[0]) ** 2 + 100. * (w[1] - w[0] ** 2) ** 2
    elif opt['loss_type'] == 'quadratic':
        l = w[0] ** 2 + 1*w[1] ** 2
    elif opt['loss_type'] == 'non-quadratic':
        l = (3-w[0]) ** 3 + w[1] ** 3
    elif opt['loss_type'] == 'nesterov_non_convex_coercive':
        l = 0.5 * w[0] ** 2 + 0.25 * w[1] ** 4 - 0.5 * w[1] ** 2

    return l



# X = data matrix
# Y = groundtruth
# w = parameter
# opt = structure containing parameters
def true_pos(X, Y, w, opt):
    n = X.shape[0]
    d = X.shape[1]
    TP = 0
    if opt['loss_type'] == 'square_loss':
        P = [np.dot(w, X[i]) for i in range(n)] # prediction <w, x>
        TP = np.sum([round(Y[i],1) == P[i] for i in range(len(Y))])
    elif opt['loss_type'] == 'hinge_loss':
        P = [np.dot(w, X[i]) for i in range(n)] # prediction <w, x>
        TP = np.sum([round(Y[i],1) == P[i] for i in range(len(Y))])
    elif opt['loss_type'] == 'binary_regression':
        z = [np.dot(w, X[i]) for i in range(n)] # prediction <w, x>
        h = [(1.0/(np.exp(-z[i]) + 1)) for i in range(n)]
        #print('sum',np.sum([(h[i] > 0.5) for i in range(n)]))
        P = [(1.0/(np.exp(-z[i]) + 1)) > 0.5 for i in range(n)]        
        TP = np.sum([round(Y[i],1) == P[i] for i in range(len(Y))])
    elif opt['loss_type'] == 'multiclass_regression':
        # assume w contains a weight vector for each class c, i.e. w=[w_1 ... w_c ... w_C]
        nC = opt['n_classes']
        dC = d
        wc = [w[i*dC:(i+1)*dC] for i in range(nC)]

        P = [np.argmax([np.dot(wc[c],X[i]) for c in range(nC)]) for i in range(n)]
        TP = np.sum([round(Y[i],1) == P[i] for i in range(len(Y))])

    return TP 




# Get one sample from X
# Also reshapes it so that it's still a 2d array
def get_batch(X, idx):
    d = X.shape[1]
    return np.reshape(X[idx, :], (-1, d))


# Compute a stochastic gradient
# X = data matrix
# Y = groundtruth
# w = parameter
# idx = index of the point to sample
# opt = structure containing parameters
def stochastic_gradient(_X, _Y, w, idx, opt):
    n = _X.shape[0]
    d = _X.shape[1]
    X = get_batch(_X, idx)
    nb = X.shape[0]

    if type(_Y)==list:
        _Y=np.array(_Y)

    #Y = [_Y[idx]] if len(idx) == 1 else _Y[idx]  <- this makes either SGD or SAGA crash depending on Y being an array or list. 
    Y = _Y[idx] if len(idx) == 1 else _Y[idx]

    if opt['loss_type'] == 'square_loss':

        grad = np.mean([np.multiply(np.dot(w, X[i]) - Y[i], X[i],dtype=np.longdouble) for i in range(nb)], axis=0)
        grad = grad + opt['lambda'] * w
    elif opt['loss_type'] == 'hinge_loss':
        cond = [j for j in [np.dot(w, X[i]) * Y[i] for i in range(nb)] if j < 1]
        if not cond:
            grad = np.zeros(d)
        else:
            grad = np.mean([np.multiply(-Y[i], X[i]) for i in range(nb) if cond[i] < 1], axis=0)
        grad = grad + opt['lambda'] * w
    elif opt['loss_type'] == 'binary_regression':
        z = X.dot(w)   # prediction <w, x>
        h = phi(z)
        grad= X.T.dot(h-Y)/nb
        grad = grad + opt['lambda'] * w
    elif opt['loss_type'] == 'non_convex':
        alpha = opt['non_cvx_reg_alpha']
        z = X.dot(w)   # prediction <w, x>
        h = phi(z)
        grad= X.T.dot(h-Y)/nb
        grad = grad + opt['lambda']*np.multiply(2*alpha*w,(1+alpha*w**2)**(-2))
    elif opt['loss_type'] == 'monkey':
        grad = np.array([3 * (w[0] ** 2 - w[1] ** 2), -6 * w[0] * w[1]])
        # add noise to simulate a stochastic gradient
        mu, sigma = 0, 1e-6  # mean and standard deviation
        grad = grad + np.random.normal(mu, sigma, d)

    elif opt['loss_type'] == 'rosenbrock':
        grad = np.array([-2 + 2 * w[0] - 400 * w[0] * w[1] + 400 * w[0] ** 3, 200 * w[1] - 200 * w[0] ** 2])
        # add noise to simulate a stochastic gradient
        mu, sigma = 0, 1e-6  # mean and standard deviation
        grad = grad + np.random.normal(mu, sigma, d)

    elif opt['loss_type'] == 'quadratic':
        grad = np.array([2 * w[0], 2 * w[1]])
        # add noise to simulate a stochastic gradient
        mu, sigma = 0, 1e-6  # mean and standard deviation
        grad = grad + np.random.normal(mu, sigma, d)

    elif opt['loss_type'] == 'nesterov_non_convex_coercive':
        grad = np.array([w[0], w[1] ** 3 - w[1]])
        # add noise to simulate a stochastic gradient
        mu, sigma = 0, 2e-2  # mean and standard deviation

        grad = grad + np.random.normal(mu, sigma, 2)

    elif opt['loss_type'] == 'multiclass_regression':
        w=np.matrix(w.reshape(opt['n_classes'],d).T)
        # construct P (this could be passed over by the loss actually)
        z=np.dot(X,w) #activation of each i for class c
        z-=np.max(z,axis=1)  # model is overparametrized. allows to subtract maximum to prevent overflow for each datapoint!
        h = np.exp(z)
        P= h/np.sum(h,axis = 1) #gives matrix with with [P]i,j = probability of sample i to be in class j (n x nC)
        if opt['method'] == 'SAGA' or opt['method']== 'SGD':
            grad = -np.dot(X.T, (opt['ground_truth'] - P))
            grad = grad / nb  + opt['lambda']* w

        else:
            grad = -np.dot(X.T, (full_ground_truth - P))
            grad = grad / n  + opt['lambda']* w
        grad = np.array(grad)
        grad = grad.flatten(('F'))

    return grad


# Return a stochastic Hessian matrix as a 2d numpy array
# X = data matrix
# Y = groundtruth
# w = parameter
# idx = index of the point to sample
# opt = structure containing parameters
def stochastic_hessian(_X, _Y, w, idx, opt):
    alpha = opt['non_cvx_reg_alpha']

    n = _X.shape[0]
    d = _X.shape[1]
    X = get_batch(_X, idx)
    nb = X.shape[0]
    Y = [_Y[idx]]
    if opt['loss_type'] == 'square_loss':
        # lambda_rescaled = opt['lambda']*n/2
        lambda_rescaled = opt['lambda']
        H = np.dot(np.transpose(X), X) / nb + (lambda_rescaled * np.eye(d, d))


    elif opt['loss_type'] == 'binary_regression':
        #B = np.zeros([opt['sub_newton_s'], opt['sub_newton_s']]) # this only works as long as hessian sampling size is not increased
        #for i in range(opt['sub_newton_s']):
        #    a = 1.0 / (1 + np.exp(-np.dot(w, X[i])))
        #    B[i, i] = a * (1 - a)
        z= X.dot(w)
        h= phi(z)*(1-phi(z))
        B = np.diag(h)
        #H = np.dot(np.dot(np.transpose(X), B), X) / nb    # in sub newton we divide by s later. -> twice ??!!
        H = np.dot(np.dot(np.transpose(X), B), X)
        H = H + opt['lambda'] * np.eye(d, d)  # add regularizer

    elif opt['loss_type'] == 'multiclass_regression':
        nC = opt['n_classes']
        w=np.matrix(w.reshape(nC,d).T)
        z=np.dot(X,w) #activation of each i for class c
        z-=np.max(z,axis=1)  # model is overparametrized. allows to subtract maximum to prevent overflow for each datapoint!
        h = np.exp(z)
        P= h/np.sum(h,axis = 1) #gives matrix with with [P]i,j = probability of sample i to be in class j (n x nC)
        H=np.zeros([d*nC,d*nC])
        for c in range(nC):
            for k in range (nC):

                if c==k:
                    D=np.diag(np.multiply(P[:,c],1-P[:,c]).A1)
                    Hcc = np.dot(np.dot(np.transpose(X), D), X) #/n
                    H[c*d:(c+1)*d,c*d:(c+1)*d]=Hcc
                else:
                    D=np.diag(-np.multiply(P[:,c],P[:,k]).A1)
                    Hck = np.dot(np.dot(np.transpose(X), D), X) #/ n <- should be added!!!
                    H[c*d:(c+1)*d,k*d:(k+1)*d]=Hck
                    H[k*d:(k+1)*d,c*d:(c+1)*d,]=Hck

        H = H/n + opt['lambda']*np.eye(d*nC,d*nC) # add regularizer     

    elif opt['loss_type'] == 'non_convex':
        alpha = opt['non_cvx_reg_alpha']
        z= X.dot(w)
        q=phi(z)
        h= q*(1-phi(z))
        D = np.diag(h)
        H = np.dot(np.dot(np.transpose(X), D), X) / n
        H = H + opt['lambda'] * np.eye(d,d)*np.multiply(2*alpha-6*alpha**2*w**2,(alpha*w**2+1)**(-3))
    elif opt['loss_type'] == 'monkey':
        H = np.array([[6 * w[0], -6 * w[1]], [-6 * w[1], -6 * w[0]]])
    elif opt['loss_type'] == 'rosenbrock':
        H = np.array([[2 - 400 * w[1] + 1200 * w[0] ** 2, -400 * w[0]], [-400 * w[0], 200]])
    elif opt['loss_type'] == 'quadratic':
        H = 2 * np.eye(d, d)
    elif opt['loss_type'] == 'nesterov_non_convex_coercive':
        H = np.array([[1, 0], [0, 3 * w[1] ** 2 - 1]])
    return H


# Return the gradient as a numpy array
# X = data matrix
# Y = groundtruth
# w = parameter
# opt = structure containing parameters
def gradient( X, Y,w, opt):
    n = X.shape[0]
    d = X.shape[1]
    # print('gradient ' + str(n) + ' ' + str(d))
    if opt['loss_type'] == 'square_loss':
        #grad = np.mean([np.multiply(np.dot(w, X[i]) - Y[i], X[i]) for i in range(n)], axis=0)
        grad = (-X.T.dot(Y)+np.dot(X.T,X.dot(w)))/n
        grad = grad + opt['lambda'] * w
    elif opt['loss_type'] == 'binary_regression':
            z = X.dot(w)   # prediction <w, x>
            h = phi(z)
            grad= X.T.dot(h-Y)/n
            grad = grad + opt['lambda'] * w


    elif opt['loss_type'] == 'multiclass_regression':

        w=np.matrix(w.reshape(opt['n_classes'],d).T)
        z=np.dot(X,w) #activation of each i for class c
        z-=np.max(z,axis=1)  # model is overparametrized. allows to subtract maximum to prevent overflow for each datapoint!
        h = np.exp(z)
        P= h/np.sum(h,axis = 1) #gives matrix with with [P]i,j = probability of sample i to be in class j (n x nC)

        flag=opt.get('ARC_subproblem_solver', 'none')
        if flag=='sub_lanczos_ada':
            grad = -np.dot(X.T, (opt['ground_truth'] - P))
        else:
            grad = -np.dot(X.T, (full_ground_truth - P))


        grad = grad / n  + opt['lambda']* w
        grad = np.array(grad)
        grad = grad.flatten(('F'))

    elif opt['loss_type'] == 'non_convex':
        alpha = opt['non_cvx_reg_alpha']
        z = X.dot(w)   # prediction <w, x>
        h = phi(z)
        grad= X.T.dot(h-Y)/n
        grad = grad + opt['lambda']*np.multiply(2*alpha*w,(1+alpha*w**2)**(-2))
    elif opt['loss_type'] == 'monkey':
        grad = np.array([3 * (w[0] ** 2 - w[1] ** 2), -6 * w[0] * w[1]])
    elif opt['loss_type'] == 'rosenbrock':
        grad = np.array([-2 + 2. * w[0] - 400 * w[0] * w[1] + 400 * w[0] ** 3, 200. * w[1] - 200 * w[0] ** 2])
    elif opt['loss_type'] == 'quadratic':
        grad = np.array([2 * w[0], 1*2 * w[1]])
    elif opt['loss_type'] == 'non-quadratic':
        grad = np.array([-3*(3-w[0])**2, 3 * w[1]**2])

        print(grad)

    elif opt['loss_type'] == 'nesterov_non_convex_coercive':
        grad = np.array([w[0], w[1] ** 3 - w[1]])

    return grad

# Return the Hessian matrix as a 2d numpy array
condition_numbers=[]
def hessian(X, Y, w, opt):
    global condition_numbers
    n = X.shape[0]
    d = X.shape[1]
    if opt['loss_type'] == 'square_loss':
        H = np.dot(X.T, X) / n + (opt['lambda'] * np.eye(d, d))

    elif opt['loss_type'] == 'binary_regression':
        z= X.dot(w)
        q=phi(z)
        h= np.array(q*(1-phi(z)))
        #D = np.diag(h)
        #H = np.dot(np.dot(np.transpose(X), D), X) / n
        H = np.dot(np.transpose(X),h[:, np.newaxis]* X) / n  #<- avoids creation of nxn matrix D which quickly causes memory errors
        H = H + opt['lambda'] * np.eye(d, d)  # add regularizer

        #In case you are interested in the conditioning of the problem at w:
        if False:
            ew,ev= np.linalg.eigh(H)
            cn=abs(max(ew))/abs(min(ew))
            print ('Condition= ', cn)
            condition_numbers.append(cn)


    elif opt['loss_type'] == 'multiclass_regression':
        nC = opt['n_classes']
        w=np.matrix(w.reshape(nC,d).T)
        z=np.dot(X,w) #activation of each i for class c
        z-=np.max(z,axis=1)  # model is overparametrized. allows to subtract maximum to prevent overflow for each datapoint!
        h = np.exp(z)
        P= h/np.sum(h,axis = 1) #gives matrix with with [P]i,j = probability of sample i to be in class j (n x nC)
        H=np.zeros([d*nC,d*nC])
        for c in range(nC):
            for k in range (nC):

                if c==k:
                    #h=np.multiply(P[:,c],1-P[:,c]).A1
                    D=np.diag(np.multiply(P[:,c],1-P[:,c]).A1)
                    Hcc = np.dot(np.dot(np.transpose(X), D), X) #/n
                    #Hcc = np.dot(np.transpose(X),h[:, np.newaxis]* X) #/n
                    H[c*d:(c+1)*d,c*d:(c+1)*d]=Hcc
                else:
                    #h=-np.multiply(P[:,c],P[:,k]).A1
                    D=np.diag(-np.multiply(P[:,c],P[:,k]).A1)
                    Hck = np.dot(np.dot(np.transpose(X), D), X) #/ n <- should be added!!!
                    #Hck = np.dot(np.transpose(X),h[:, np.newaxis]* X) #/n
                    H[c*d:(c+1)*d,k*d:(k+1)*d]=Hck
                    H[k*d:(k+1)*d,c*d:(c+1)*d,]=Hck

        H = H/n + opt['lambda']*np.eye(d*nC,d*nC) # add regularizer 
        if True:
            ew,ev= np.linalg.eigh(H)
            cn=abs(max(ew))/abs(min(ew))
            print ('Condition= ', cn)
            condition_numbers.append(cn)   

    elif opt['loss_type'] == 'non_convex':
        alpha = opt['non_cvx_reg_alpha']
        z= X.dot(w)
        q=phi(z)
        h= q*(1-phi(z))
        #D = np.diag(h)
        #H = np.dot(np.dot(np.transpose(X), D), X) / n
        H = np.dot(np.transpose(X),h[:, np.newaxis]* X) / n  #<- avoids creation of nxn matrix D which quickly causes memory errors
        H = H + opt['lambda'] * np.eye(d,d)*np.multiply(2*alpha-6*alpha**2*w**2,(alpha*w**2+1)**(-3))
         #In case you are interested in the conditioning of the problem at w:
        if True:
            ew,ev= np.linalg.eigh(H)
            cn=abs(max(ew))/abs(min(ew))
            print ('Condition= ', cn)
            condition_numbers.append(cn)

    elif opt['loss_type'] == 'monkey':
        H = np.array([[6 * w[0], -6 * w[1]], [-6 * w[1], -6 * w[0]]])
    elif opt['loss_type'] == 'rosenbrock':
        H = np.array([[2 - 400 * w[1] + 1200 * w[0] ** 2, -400 * w[0]], [-400 * w[0], 200]])
    elif opt['loss_type'] == 'quadratic':
        #H = 2 * np.eye(d, d)
        H= 2* np.array([[1,0],[0,1]])
    elif opt['loss_type'] == 'non-quadratic':
        H= np.array([[6.*(3.-w[0]),0],[0,6*w[1]]])
        print (H)

    elif opt['loss_type'] == 'nesterov_non_convex_coercive':
        H = np.array([[1, 0], [0, 3 * w[1] ** 2 - 1]])
    return H

def hv(X, Y, w, opt,v): # efficient matrix vector product for binary_regression. All others calculate H explicitly.
    n = X.shape[0]
    d = X.shape[1]
    if opt['loss_type']=='binary_regression':        
        wa = d_binary * X.dot(v)
        Hv = X.T.dot(wa)/n
        out = Hv + opt['lambda'] * v
        return out
    if opt['loss_type']=='non_convex':
        alpha = opt['non_cvx_reg_alpha']

        wa = d_binary * X.dot(v)
        Hv = X.T.dot(wa)/n
        out = Hv + opt['lambda'] *np.multiply(np.multiply(2*alpha-6*alpha**2*w**2,(alpha*w**2+1)**(-3)), v)
        return out

    if opt['loss_type']=='square_loss':
        Xv=np.dot(X,v)
        Hv=np.dot(X.T,Xv)/n + opt['lambda'] * v
        return Hv
    if opt['loss_type']=='multiclass_regression':
        nC = opt['n_classes']
        v = v.reshape(nC, -1)
        r_yhat = np.dot(X, v.T)
        r_yhat += (-P_multi * r_yhat).sum(axis=1)[:, np.newaxis]
        r_yhat *= P_multi
        hessProd = np.zeros((nC, d))
        hessProd[:, :d] = np.dot(r_yhat.T, X)/n
        hessProd[:, :d] += v * opt['lambda']
        return hessProd.ravel()
    else:
        H=hessian(X, Y, w, opt)
        return np.dot(H,v)




######## Robut Sigmoid, log(sigmoid) and 1-log(sigmoid) computations ########
def phi(t): #Author: Fabian Pedregosa
    # logistic function returns 1 / (1 + exp(-t))
    idx = t > 0
    out = np.empty(t.size, dtype=np.float)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out
def log_phi(t):
    # log(Sigmoid): log(1 / (1 + exp(-t)))

    idx = t>0
    out = np.empty(t.size, dtype=np.float)
    out[idx]=-np.log(1+np.exp(-t[idx]))
    out[~idx]= t[~idx]-np.log(1+np.exp(t[~idx]))
    return out
def one_minus_log_phi(t):
    # log(1-Sigmoid): log(1-1 / (1 + exp(-t)))

    idx = t>0
    out = np.empty(t.size, dtype=np.float)
    out[idx]= -t[idx]-np.log(1+np.exp(-t[idx]))
    out[~idx]=-np.log(1+np.exp(t[~idx]))
    return out


###############################################################################
# Gradient descent
###############################################################################

def GD(X, Y, w, opt):
    from datetime import datetime

    print ('--- GD ---')
    n = X.shape[0]
    d = X.shape[1]
    n_passes = opt['n_passes']
    eta = opt['learning_rate']

    loss_file = open(opt['log_dir'] + "gd_loss.txt", "w")
    param_file = open(opt['log_dir'] + "gd_param.txt", "w")
    if opt['3d_plot'] == True:
        plot_file = open("GD_plot_file.txt", "w")  

    n_samples_per_step = n
    n_steps = int((n_passes * n) / n_samples_per_step)
    n_samples = 0  # number of samples processed so far
    start = datetime.now()
    for i in range(n_steps):
        grad = gradient(X, Y, w, opt)
        n_samples += n
        if n_samples >= opt['recording_step']:
            timing=(datetime.now() - start).total_seconds()
            n_samples = 0
            _loss = loss(X, Y, w, opt)
            print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(np.linalg.norm(grad)))
            if opt['x_axis_samples'] == True:
                loss_file.write(str((i + 1) * n) + '\t' + str(_loss) + '\n')
            else:
                loss_file.write(str(timing) + '\t' + str(_loss) + '\n')
        

        #for 3d plots: save current iterate 'w' and loss 'current_f' + subproblem solver
        if opt['3d_plot'] == True:
            plot_file.write('\t'.join(['%.5f' % num for num in w]) + '\t' + str(_loss) + '\n')

        #w_str = ','.join(['%.5f' % num for num in w])
        #param_file.write(w_str + '\n')
        w = w - eta * grad

    loss_file.close()
    param_file.close()

    return w


###############################################################################
# Stochastic Gradient descent
###############################################################################

def SGD(X, Y, w, opt):
    import time

    print ('--- SGD ---')
    n = X.shape[0]
    d = X.shape[1]
    n_passes = opt['n_passes_SGD']

    eta = opt['learning_rate']
    bs = opt['batch_size_SGD']


    #loss_file = open(opt['log_dir'] + "sgd_loss.txt", "w")
    param_file = open(opt['log_dir'] + "sgd_param.txt", "w")

    n_samples_per_step = bs
    n_steps = int((n_passes * n) / n_samples_per_step)
    n_samples = 0  # number of samples processed so far
    t_sgd = time.clock()

    if opt['3d_plot'] == True:
        plot_file = open("SGD_plot_file.txt", "w")  

    loss_collector=[]
    timings_collector=[]
    samples_collector=[]
    for i in range(n_steps):
        idx = np.random.randint(0, high=n, size=bs)
        if opt['loss_type']=='multiclass_regression':
            opt['ground_truth']=full_ground_truth[idx,:]
        

        grad = stochastic_gradient(X, Y, w, idx, opt)
        n_samples += bs
        
        if (i == 0 or n_samples >= opt['recording_step']) or opt['3d_plot'] == True:
            n_samples = 0
            _loss = loss(X, Y, w, opt)
            print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(np.linalg.norm(grad)))

            t = time.clock()
            if opt['x_axis_samples'] == True:
                #loss_file.write(str((i + 1) * n_samples_per_step) + '\t' + str(_loss) + "\n")
                timings_collector.append((i+1)*n_samples_per_step)
            else:
                #loss_file.write(str(t - t_sgd) + "\t" + str(_loss) + "\n")
                timings_collector.append(t-t_sgd)
                samples_collector.append((i+1)*n_samples_per_step)


            loss_collector.append(_loss)
                
            w_str = ','.join(['%.5f' % num for num in w])
            param_file.write(w_str + '\n')

        if opt['3d_plot'] == True:
            plot_file.write('\t'.join(['%.5f' % num for num in w]) + '\t' + str(_loss) + '\n')

        w = w - eta * grad

    #loss_file.close()
    param_file.close()

    return w, timings_collector, loss_collector, samples_collector


###############################################################################
# Importance sampling
###############################################################################

# Sampling code

def log_add(x, y):
    maximum = max(x, y)
    minimum = min(x, y)
    if (abs(maximum - minimum) > 30):
        # the difference is too small, return the just the maximum
        return maximum
    return maximum + log(1 + pow(2, minimum - maximum), 2)


# Takes as input a log probability vector. The base of the logarithm is 2.
# Returns a sampled position according to the corresponding log probabilities
# Uses the formula from
# http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers
def sample_from_log_prob(A):
    max_log_prob = max(A)
    C = [A[0]]
    for a in A[1:]:
        C.append(log_add(C[-1], a))
    C_pos = [-c for c in reversed(C)]
    r = log(random.random(), 2)
    pos = np.searchsorted(C_pos, -r)
    return len(C) - pos


# A = [1/10.0] * 10
# lst = [0] * 10
# for i in range(0, 10000):
#    idx = sample_from_log_prob(np.log2(A))
#    lst[idx] = lst[idx] + 1
# print (lst)

def SGD_importance_sampling(X, Y, w, opt):
    print ('--- SGD importance sampling ---')
    n = X.shape[0]
    d = X.shape[1]
    n_passes = opt['n_passes']
    eta = opt['learning_rate']

    loss_file = open(opt['log_dir'] + "sgd_is_loss.txt", "w")
    param_file = open(opt['log_dir'] + "sgd_is_param.txt", "w")

    n_samples_per_step = 1
    n_steps = int((n_passes * n) / n_samples_per_step)
    n_samples = 0  # number of samples processed so far

    for i in range(n_steps):

        # Compute all the gradients
        # grad_norms = [np.linalg.norm(stochastic_gradient(X, Y, w, j)) for j in range(n)]
        grad_norms = [np.linalg.norm(X[j, :]) for j in range(n)]
        sum_grad_norms = np.sum(grad_norms)
        # probs = [(0.5/n) + (0.5*(grad_norms[j]/sum_grad_norms)) for j in range(n)]
        probs = [grad_norms[j] / sum_grad_norms for j in range(n)]
        # print(np.sum(probs))

        # idx = np.argmax(grad_norms)
        idx = sample_from_log_prob(np.log2(probs))
        # pi = grad_norms[idx]/sum_grad_norms
        pi = probs[idx]
        # print('pi ' + str(idx) + ' ' + str(pi) + ' = ' + str(grad_norms[idx]) + '/' + str(sum_grad_norms))
        # print('pi ' + str(idx) + str(grad_norms[10]) + ' < ' + str(grad_norms[idx]))

        grad = stochastic_gradient(X, Y, w, idx, opt)
        n_samples += 1

        if n_samples >= opt['recording_step']:
            n_samples = 0
            _loss = loss(X, Y, w)
            print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' eta = ' + str(eta) + ' norm_grad = ' + str(
                np.linalg.norm(grad)))
            loss_file.write(str(i + 1) + '\t' + str(_loss) + '\n')
            w_str = ','.join(['%.5f' % num for num in w])
            param_file.write(w_str + '\n')
        # eta = backtracking_line_search(X,Y,w,grad,-grad)
        w = w - (eta / (n * pi)) * grad

    loss_file.close()
    param_file.close()

    return w


###################
# SAGA
###################
def SAGA(X, Y, w, opt):
    if True: #old version
        print ('--- SAGA ---')
        n = X.shape[0]
        d = X.shape[1] 
       
        n_passes = opt['n_passes_SAGA']
        eta = opt['learning_rate_SAGA']

        # Store past gradients in a table
        mem_gradients = {}
        nGradients = 0  # no gradient stored in mem_gradients at initialization
        if opt['loss_type']== 'multiclass_regression':
            avg_mg=np.zeros(d*opt['n_classes'])
        else:
            avg_mg = np.zeros(d)

        start = datetime.now()
        timing=0
    
        # Fill in table
        a = 1.0 / n
        print ('starting to build avg mg')
        bool_idx = np.zeros(n,dtype=bool)

        for i in range(n):
            bool_idx[i]=True
            _X=np.zeros((1,d))
            _X=np.compress(bool_idx,X,axis=0)
            _Y=np.compress(bool_idx,Y,axis=0)
            grad = gradient(_X, _Y, w, opt)

            if opt['loss_type']== 'multiclass_regression':
                opt['ground_truth']=full_ground_truth[i,:]
            #grad = stochastic_gradient(X, Y, w, np.array([i]), opt)
            mem_gradients[i] = grad
            # avg_mg = avg_mg + (grad*a)
            avg_mg = avg_mg + grad

        print ('done building avg mg')

        avg_mg = avg_mg / n
        nGradients = n

        n_samples_per_step = 1
        n_steps = int((n_passes * n) / n_samples_per_step)
        n_samples_seen = 0  # number of samples processed so far

       
        k=0

        loss_collector=[]
        timings_collector=[]
        samples_collector=[]

        print ('numer of steps', n_steps)

        for i in range(n_steps):
            list_idx = np.random.randint(0, high=n, size=1)
            idx = list_idx[0]
            if opt['loss_type']== 'multiclass_regression':
                opt['ground_truth']=full_ground_truth[idx,:]


            bool_idx = np.zeros(n,dtype=bool)
            bool_idx[idx]=True
             
            _X=np.zeros((1,d))
            _X=np.compress(bool_idx,X,axis=0)
            _Y=np.compress(bool_idx,Y,axis=0)
            grad = gradient(_X, _Y, w, opt)
            #if i==1:
            #    print ('starting to calc grad')
            #    t1=time.clock()
            #grad = stochastic_gradient(X, Y, w, list_idx, opt)
            #if i==1:
            #    t2=time.clock()
            #    print ('done calc grad', t2-t1)

            n_samples_seen += 1

            if (n_samples_seen >= n*k)  == True:
                k+=1
                _loss = loss(X, Y, w, opt)

                _timing=timing
                timing=(datetime.now() - start).total_seconds()  

                print ('Epoch ' + str(k) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(np.linalg.norm(grad)), 'time=',round(timing-_timing,3))

            
                timings_collector.append(timing)
                samples_collector.append(i+n)
                loss_collector.append(_loss)

            # Parameter update
            if idx in mem_gradients:
                w = w - eta * (grad - mem_gradients[idx] + avg_mg)  # SAGA step
            else:
                w = w - eta * grad  # SGD step
            
            # Update average gradient
            if idx in mem_gradients:
                delta_grad = grad - mem_gradients[idx]
                a = 1.0 / nGradients
                avg_mg = avg_mg + (delta_grad * a)
            else:
                # Gradient for datapoint idx does not exist yet
                nGradients = nGradients + 1  # increment number of gradients
                a = 1.0 / nGradients
                b = 1.0 - a
                avg_mg = (avg_mg * b) + (grad * a)

            # Update memorized gradients
            mem_gradients[idx] = grad


        return w, timings_collector,loss_collector, samples_collector
    else: #new version
        import time
        import math

        print ('--- SAGA new ---')
        n = X.shape[0]
        d = X.shape[1] 
        n_passes = opt['n_passes_SAGA']
        eta = opt['learning_rate_SAGA']

        param_file = open(opt['log_dir'] + "saga_param.txt", "w")

        # Store past gradients in a table
        mem_gradients= np.zeros((n,d))
        nGradients = 0  # no gradient stored in mem_gradients at initialization
        if opt['loss_type']== 'multiclass_regression':
            avg_mg=np.zeros(d*opt['n_classes'])
        else:
            avg_mg = np.zeros(d)

        # Fill in table
        a = 1.0 / n
        print ('starting to build avg mg')
        for i in range(n):

            bool_idx = np.zeros(n,dtype=bool)
            bool_idx[i]=True
            _X=np.zeros((1,d))
            _X=np.compress(bool_idx,X,axis=0)
            _Y=np.compress(bool_idx,Y,axis=0)
            grad = gradient(_X, _Y, w, opt)

            if opt['loss_type']== 'multiclass_regression':
                opt['ground_truth']=full_ground_truth[i,:]
            grad = gradient(X, Y, w, opt)
            mem_gradients[i,:] = grad
            # avg_mg = avg_mg + (grad*a)
            avg_mg = avg_mg + grad

        print ('done building avg mg')

        avg_mg = avg_mg / n
        nGradients = n

        n_samples_per_step = 1
        n_steps = int((n_passes * n) / n_samples_per_step)
        n_samples = 0  # number of samples processed so far
        t_saga = time.clock()
    


        loss_collector=[]
        timings_collector=[]
        samples_collector=[]

        print ('numer of steps', n_steps)

        for i in range(n_steps):
            list_idx = np.random.randint(0, high=n, size=1)
            idx = list_idx[0]
            if opt['loss_type']== 'multiclass_regression':
                opt['ground_truth']=full_ground_truth[idx,:]


            bool_idx = np.zeros(n,dtype=bool)
            bool_idx[idx]=True
            _X=np.zeros((1,d))
            _X=np.compress(bool_idx,X,axis=0)
            _Y=np.compress(bool_idx,Y,axis=0)
            grad = gradient(_X, _Y, w, opt)

            print ('starting to calc grad')
            t1=time.clock()
            grad = gradient(X, Y, w, opt)
            t2=time.clock()

            print ('done calc grad', t2-t1)


            n_samples += 1

            if i == 0 or n_samples >= opt['recording_step']:
                n_samples = 0
                _loss = loss(X, Y, w, opt)
                print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(
                    np.linalg.norm(grad)) + ' avg_mg = ' + str(np.linalg.norm(avg_mg)))
                t = time.clock()
                if opt['x_axis_samples'] == True:
                    #loss_file.write(str(i) + '\t' + str(_loss) + '\n')
                    timings_collector.append(i+n)
                else:
                    #loss_file.write(str(t - t_saga) + '\t' + str(_loss) + '\n')
                    timings_collector.append(t - t_saga)
                    samples_collector.append(i+n)

                loss_collector.append(_loss)
                w_str = ','.join(['%.5f' % num for num in w])
                param_file.write(w_str + '\n')

            # Parameter update
            if idx in mem_gradients:
                w = w - eta * (grad - mem_gradients[idx] + avg_mg)  # SAGA step
            else:
                w = w - eta * grad  # SGD step
            if math.isnan(w[0]):
                print ('Iteration', i)
                raise ValueError('w,g ~ nan, mem_grad, avg ~ +/- inf')

            # Update average gradient
            if idx in mem_gradients:
                delta_grad = grad - mem_gradients[idx,:]
                a = 1.0 / nGradients
                avg_mg = avg_mg + (delta_grad * a)
            else:
                # Gradient for datapoint idx does not exist yet
                nGradients = nGradients + 1  # increment number of gradients
                a = 1.0 / nGradients
                b = 1.0 - a
                avg_mg = (avg_mg * b) + (grad * a)

            # Update memorized gradients
            mem_gradients[idx,:] = grad

        #loss_file.close()
        param_file.close()

        return w, timings_collector,loss_collector, samples_collector



###############################################################################
# SDCA
###############################################################################

def dual(X, Y, alphas, reg):
    n = X.shape[0]
    # print([alphas[i]*X[i,:] for i in alphas])
    # print(np.sum([alphas[i]*X[i,:] for i in alphas], axis=0))
    # reg_term = (reg/2.0)*np.linalg.norm(np.sum([alphas[i]*X[i,:] for i in alphas], axis=0)/(reg*n))
    reg_term = (1 / (reg * 2.0 * n * n)) * np.linalg.norm(np.sum([alphas[i] * X[i, :] for i in alphas], axis=0)) ** 2
    loss_term = np.sum([alphas[i] * Y[i] for i in alphas]) / n
    _dual = loss_term - reg_term
    return _dual


def SDCA(X, Y, opt):
    print ('--- SDCA ---')
    n = X.shape[0]
    d = X.shape[1]
    n_passes = opt['n_passes']
    eta = opt['learning_rate']
    reg = opt['lambda']

    dual_file = open(opt['log_dir'] + "sdca_dual.txt", "w")
    loss_file = open(opt['log_dir'] + "sdca_loss.txt", "w")
    param_file = open(opt['log_dir'] + "sdca_param.txt", "w")

    dual_var_file = open(opt['log_dir'] + "sdca_dual_details.txt", "w")
    # dual_var_file.write('idx\talphas[idx]\tdual\tprimal\t||w||\n')

    # Dual variables
    alphas = {}
    # Primal vector
    w = np.zeros(d)

    n_samples_per_step = 1
    n_steps = int((n_passes * n) / n_samples_per_step)
    n_samples = 0  # number of samples processed so far

    for i in range(n_steps):
        idx = np.random.randint(0, high=n, size=1)[0]
        y_i = Y[idx]
        x_i = X[idx, :]

        a_i = 0
        if idx in alphas:
            a_i = alphas[idx]

        if opt['loss_type'] == 'hinge_loss':
            p_i = y_i * np.dot(w, x_i)
            xi_sq_norm = np.dot(x_i, x_i)
            tmp = ((reg * n) / xi_sq_norm) * (1 - p_i) + a_i * y_i
            d_alpha = y_i * max(0, min(1, tmp)) - a_i
        elif opt['loss_type'] == 'square_loss':
            xi_sq_norm = np.dot(x_i, x_i)
            tmp = y_i - np.dot(w, x_i) - 0.5 * a_i
            d_alpha = tmp / (0.5 + (xi_sq_norm / (reg * n)))

        alphas[idx] = a_i + d_alpha
        w = w + d_alpha * x_i / (reg * n)

        # print('alphas[idx]', idx, alphas[idx], dual(X, Y, alphas, reg), loss(X, Y, w, opt), np.linalg.norm(w)**2)
        # dual_var_file.write(str(idx) + '\t' + str(alphas[idx]) + '\t' + str(dual(X, Y, alphas, reg)) + '\t' + str(loss(X, Y, w, opt)) + '\t' + str(np.linalg.norm(w)) + '\n')

        if n_samples >= opt['recording_step']:
            n_samples = 0
            _loss = loss(X, Y, w, opt)
            _dual = dual(X, Y, alphas, reg)
            # print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' dual = ' + str(_dual))
            loss_file.write(str(i) + '\t' + str(_loss) + '\n')
            dual_file.write(str(i) + '\t' + str(_dual) + '\n')
            w_str = ','.join(['%.5f' % num for num in w])
            param_file.write(w_str + '\n')

    dual_file.close()
    # dual_var_file.close()
    loss_file.close()
    param_file.close()

    return w


###############################################################################
# Newton
###############################################################################

# g gradient (should be exact gradient)
# dd descent direction
# dd being a descent direction means that dot(dd,g) < 0
# Goal: find a step size such that new function value fwt is smaller than current function value fw
# by a certain amount given by dot(dd,g)
def backtracking_line_search(X, Y, w, g, dd, opt):
    t = 1.0  # step size
    alpha = 0.1  # < 0.5
    beta = 0.9  # < 1
    wt = w + t * dd
    fw = loss(X, Y, w, opt)
    fwt = loss(X, Y, wt, opt)
    inc = alpha * np.dot(dd, g)  # amount by which we want to decrease the function
    while (fwt > (fw + t * inc) and (not np.isnan(fwt)) and (t > 1e-10)):
        t = t * beta
        wt = w + t * dd
        fwt = loss(X, Y, wt, opt)
    return t


def Newton(X, Y, w, opt):
    print ('--- Newtons method ---')
    from datetime import datetime

    n = X.shape[0]
    d = X.shape[1]
    n_passes = opt['n_passes_N']
    eta = opt['learning_rate']
    bs = opt['batch_size_N']
    
    loss_collector=[]
    timings_collector=[]
    samples_collector=[]

    #loss_file = open(opt['log_dir'] + "newton_loss.txt", "w")
    param_file = open(opt['log_dir'] + "newton_param.txt", "w")
    if opt['3d_plot'] == True:
        plot_file = open("plot_file_newton.txt", "w")

    n_samples_per_step = 2 * n
    n_steps = int((n_passes * n) / n_samples_per_step)
    n_samples = 0  # number of samples processed so far


    #start Timer and save initial values
    _loss = loss(X, Y, w, opt)
    loss_collector.append(_loss)
    timings_collector.append(0)
    samples_collector.append(0)


    #loss_file.write(str(0) + '\t' + str(_loss) + '\n')
    w_str = ','.join(['%.5f' % num for num in w])
    param_file.write(w_str + '\n')
    start = datetime.now()
    grad = gradient(X, Y, w, opt)

    for i in range(n_passes):
        H = hessian(X, Y, w, opt)
        if np.linalg.norm(H) < 1e-20:
            print ('Iteration ' + str(i) + ': det(H) = ' + str(np.linalg.det(H)))
            invH = np.eye(d, d)
        else:
            invH = np.linalg.inv(H)  # might be faster with cholesky decomp

        dd = - np.dot(invH, grad)  # descent direction

        n_samples += n_samples_per_step

        if opt['3d_plot'] == True:
            current_f = loss(X, Y, w, opt)
            plot_file.write('\t'.join(['%.5f' % num for num in w]) + '\t' + str(current_f) + '\n')
        # debug
        if np.dot(grad, dd) > 0:
            print('gBg = ' + str(np.dot(grad, dd)))
        # print('norm_grad = ' + str(np.linalg.norm(grad)) + ' norm_dd = ' + str(np.linalg.norm(dd)))
        eta = backtracking_line_search(X, Y, w, grad, dd, opt)
        eta=1
        w = w + eta * dd
        grad = gradient(X, Y, w, opt)  # full gradient
        ### Checkpoint ###
        if n_samples >= opt['recording_step']:
            n_samples = 0
            _loss = loss(X, Y, w, opt)
            timing=(datetime.now() - start).total_seconds()
            print ('Iteration ' + str(i+1) + ': loss = ' + str(_loss) + ' eta = ' + str(eta) + ' norm_grad = ' + str(
                np.linalg.norm(grad)), 'time= ', timing)
            print ('Iteration ' + str(i+1) + ': norm(H^-1) = ' + str(np.linalg.norm(invH)) + ' norm_dd = ' + str(
                np.linalg.norm(dd)) + ' norm_w = ' + str(np.linalg.norm(w)))
            if opt['x_axis_samples'] == True:
                timings_collector.append((i+1)*n_samples_per_step)
                #loss_file.write(str((i+1) * n_samples_per_step) + '\t' + str(_loss) + '\n')
            else:
                #loss_file.write(str(timing) + '\t' + str(_loss) + '\n')
                timings_collector.append(timing)
                samples_collector.append((i+1)*n_samples_per_step)

            loss_collector.append(_loss)

            w_str = ','.join(['%.5f' % num for num in w])
            param_file.write(w_str + '\n')


    if opt['3d_plot'] == True:
        current_f = loss(X, Y, w, opt)
        plot_file.write('\t'.join(['%.5f' % num for num in w]) + '\t' + str(current_f) + '\n')

    #loss_file.close()
    param_file.close()

    return w, timings_collector, loss_collector, samples_collector


# Truncated Newton
def TruncatedNewton(X, Y, w, opt):
    print ('--- Truncated Newtons method ---')
    n = X.shape[0]
    d = X.shape[1]
    n_passes = opt['n_passes_tN']
    eta = opt['learning_rate']

    loss_file = open("truncated_newton_loss.txt", "w")
    param_file = open("truncated_newton_param.txt", "w")

    n_samples_per_step = 2 * n
    n_steps = int((n_passes * n) / n_samples_per_step)
    n_samples = 0  # number of samples processed so far

    #start Timer and save initial values
    _loss = loss(X, Y, w, opt)
    loss_file.write(str(0) + '\t' + str(_loss) + '\n')
    w_str = ','.join(['%.5f' % num for num in w])
    param_file.write(w_str + '\n')
    start = datetime.now()
    
    for i in range(n_steps):
        grad = gradient(X, Y, w, opt)
        H = hessian(X, Y, w, opt)

        # Truncate negative eigenvalues
        U, s, V = np.linalg.svd(H)
        for i in range(d):
            if s[i] < 0:
                s[i] = 0
        S = np.zeros((d, d))
        S[:d, :d] = np.diag(s)
        tH = np.dot(U, np.dot(S, V))

        if np.absolute(np.linalg.det(tH)) < 1e-20:
            print ('Iteration ' + str(i) + ': det(H) = ' + str(np.linalg.det(tH)))
            invH = np.eye(d, d)
        else:
            invH = np.linalg.inv(tH)  # might be faster with cholesky decomp!

        dd = - np.dot(invH, grad)  # descent direction

        n_samples += n_samples_per_step

        
        # debug
        if np.dot(grad, dd) > 0:
            print('gBg = ' + str(np.dot(grad, dd)))
        # print('norm_grad = ' + str(np.linalg.norm(grad)) + ' norm_dd = ' + str(np.linalg.norm(dd)))
        eta = backtracking_line_search(X, Y, w, grad, dd, opt)
        w = w + eta * dd

        ### Checkpoint ###
        if n_samples >= opt['recording_step']:
            n_samples = 0
            _loss = loss(X, Y, w, opt)
            timing=(datetime.now() - start).total_seconds()
            print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' eta = ' + str(eta) + ' norm_grad = ' + str(
                np.linalg.norm(grad)), 'time =', timing)
            print ('Iteration ' + str(i) + ': det(H^-1) = ' + str(np.linalg.det(invH)) + ' norm_dd = ' + str(
                np.linalg.norm(dd)))

            t = time.clock()
            if opt['x_axis_samples'] == True:
                loss_file.write(str((i+1) * n_samples_per_step) + '\t' + str(_loss) + '\n')
            else:
                loss_file.write(str(timing) + '\t' + str(_loss) + '\n')

            w_str = ','.join(['%.5f' % num for num in w])
            param_file.write(w_str + '\n')


    loss_file.close()
    param_file.close()

    return w


# Adagrad
# J. Duchi, E. Hazan, and Y. Singer
# "Adaptive subgradient methods for online leaning and stochastic optimization" in COLT, 2010
def Adagrad(X, Y, w, opt):
    print ('--- Adagrad ---')
    n = X.shape[0]
    d = X.shape[1]
    n_passes = opt['n_passes']
    eta = opt['learning_rate']

    loss_file = open(opt['log_dir'] + "adagrad_loss.txt", "w")
    param_file = open(opt['log_dir'] + "adagrad_param.txt", "w")

    # term premultiplying the gradient
    mu = np.ones(d) * 1e-8

    n_samples_per_step = 1
    n_steps = int((n_passes * n) / n_samples_per_step)
    n_samples = 0  # number of samples processed so far

    for i in range(n_steps):
        idx = np.random.randint(0, high=n, size=1)[0]
        grad = stochastic_gradient(X, Y, w, idx, opt)

        # Update adagrad step-size term
        mu = mu + np.multiply(grad, grad)

        if n_samples >= opt['recording_step']:
            n_samples = 0
            _loss = loss(X, Y, w, opt)
            print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(np.linalg.norm(grad)))
            loss_file.write(str(i) + '\t' + str(_loss) + '\n')
            w_str = ','.join(['%.5f' % num for num in w])
            param_file.write(w_str + '\n')
        w = w - eta * np.divide(grad, np.sqrt(mu))

    loss_file.close()
    param_file.close()

    return w


# Sub-sampled Newton method
# Erdogdu, Murat A., and Andrea Montanari.
# "Convergence rates of sub-sampled Newton methods." NIPS. 2015.
# TODO: Needs more debugging!!
def sub_sampled_Newton(X, Y, w, opt):
    from datetime import datetime
    n = X.shape[0]
    d = X.shape[1]
    n_samples = opt['n_passes_sN'] * n  # number of samples to process

    eta = opt['learning_rate']
    bs = opt['hessian_sample_size_sub_newton']  # =n/r <- might want to be changed to n for deterministic gradients
    bs_0 = opt['hessian_sample_size_sub_newton']  # start value for geometric increase in batch size
    loss_file = open(opt['log_dir'] + "sub_newton_loss.txt", "w")
    param_file = open(opt['log_dir'] + "sub_newton_param.txt", "w")

    r = opt['sub_newton_r']  # rank Q matrix
    s = opt['sub_newton_s']  # number of samples to compute approximate Hessian
    gamma = log(d) / s
    n_samples_seen = 0
    i = 0
    k = 0

    print ('--- Sub-sampled Newton method ---' + ' initial batch:' + str(bs_0) + ' -- gradients: ' + str(
        opt['stochastic_gradients']) + ' -- steps: ' + str(eta))

   #start Timer and save initial values
    _loss = loss(X, Y, w, opt)
    loss_file.write(str(0) + '\t' + str(_loss) + '\n')
    w_str = ','.join(['%.5f' % num for num in w])
    param_file.write(w_str + '\n')
    start = datetime.now()

    ### adapt x axis of error plot (number of samples looked at)
    while (n_samples_seen < n_samples):
        ### Compute (approximate) gradient
        idx_gradient = np.random.randint(0, high=n, size=bs)
        if (opt['stochastic_gradients'] == 'deterministic'):
            grad = gradient(X, Y, w, opt)  # full gradient
            n_samples_per_step = n + s  # adapt x axis of error plot
        else:
            grad = stochastic_gradient(X, Y, w, idx_gradient, opt)
            n_samples_per_step = bs + s  # adapt x axis of error plot

        ### Compute approximate Hessian
        H_s = np.zeros((d, d))
        # indices = np.random.randint(0, high=n, size=s)
        indices = np.random.choice(range(n), s, replace=False)
        
        #for j in range(s):
        #    idx = indices[j]
        #    H_s = H_s + stochastic_hessian(X, Y, w, idx, opt)
        #H_s = H_s / s
        
        _X=get_batch(X,indices)
        _Y = Y[[indices]]
        H_s = hessian(_X, _Y, w, opt)

        # H = hessian(X, Y, w, opt)
        # print('|H-H_s| = ', np.linalg.norm(H-H_s))

        # Rank (r+1) truncated SVD
        U, Sigma, VT = randomized_svd(H_s, n_components=r + 1,
                                      n_iter=10,
                                      random_state=None)

        Ur = U[:, range(r)]
        Sigma_r = Sigma[range(r)]
        invSigma_r = np.linalg.inv(np.diag(Sigma_r))

        # # Constuct approximate Hessian matrix
        # # we fill its 0 eigenvalues with the (r +1)-th eigenvalue
        Q = np.eye(d, d) / Sigma[r] + np.dot(Ur, np.dot(invSigma_r - np.eye(r, r) / Sigma[r], np.transpose(
            Ur)))  # uncomment this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        dd = - np.dot(Q, grad)  # descent direction


        ###learning rate:
        # eta = backtracking_line_search(X,Y,w,grad,dd,opt)
        # dynamic lr as in NewSwamp

        if (opt['dynamic_learning_rate'] == True):
            evalues, evectors = np.linalg.eig(H_s)
            lambda_min_Hs = np.real(min(evalues))  # O(p^2)
            eta = (2 / (1 + lambda_min_Hs / Sigma[r] + (5) * log(d) / s))

        w = w + eta * dd

        ###dynamic sampling of gradients
        if opt['stochastic_gradients'] == 'geometric':
            bs = int(min(n, bs_0 + 1.25 ** (i + 1)))
        elif opt['stochastic_gradients'] == 'adaptive':
            bs = adaptive_sampling(idx_gradient, d, X, Y, w, opt, grad, bs, n)

        n_samples_seen = n_samples_seen + n_samples_per_step

        ### Checkpoint ###
        if n_samples_seen >= opt['recording_step'] * k:
            _loss = loss(X, Y, w, opt)
            timing=(datetime.now() - start).total_seconds()
            print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' eta = ' + str(eta) + ' norm_grad = ' + str(
                np.linalg.norm(grad)) + 'grad_batch = ' + str(bs))
            print ('Iteration ' + str(i) + ': |H_s^-1 - Q| = ' + str(
                np.linalg.norm(np.linalg.inv(H_s) - Q)) + ' norm(Q) = ' + str(np.linalg.norm(Q)) + ' norm_dd = ' + str(
                np.linalg.norm(dd)))
            
            if opt['x_axis_samples'] == True:
                loss_file.write(str(n_samples_seen) + '\t' + str(_loss) + '\n') 
            else:
                loss_file.write(str(timing) + '\t' + str(_loss) + '\n')
            w_str = ','.join(['%.5f' % num for num in w])
            param_file.write(w_str + '\n')
            k = k + 1

        i = i + 1

    loss_file.close()
    param_file.close()

    return w


# adaptive sample size as in: "Sample Size Selection in Optimization Methods for Machine Learning" Byrd, Chin, Nocedal and Wu 2001
def adaptive_sampling(idx_gradient, d, X, Y, w, opt, grad, bs, n):
    sigma = 0.75
    sample_variance = np.zeros(d)
    for j in range(len(idx_gradient)):
        idx = [idx_gradient[j]]
        grad_j = stochastic_gradient(X, Y, w, idx, opt)
        delta = grad_j - grad
        delta_sq = [x * x for x in delta]
        sample_variance += delta_sq
    sample_variance = np.multiply(1.0 / (bs - 1), sample_variance)  # may give zeroDivisionError
    norm_grad = np.linalg.norm(grad)

    # 2. increase S if not ensured that descent directions are produced sufficiently often (<- need to change as we have 2nd order info?)
    if np.linalg.norm(sample_variance, 1) / bs * (
                (n - bs) / (n - 1.0)) > sigma ** 2 * norm_grad ** 2:  # *((n-bs)/(n-1)) for small samples
        bs = int(min(n, np.linalg.norm(sample_variance, 1) / (sigma ** 2 * norm_grad ** 2)))
    return bs

def Conjugate_Gradients(X,Y,w,opt): ## not sure if correct initialization
    print ('--- Conjugate_Gradients ---')
    H=hessian(X, Y, w, opt)
    d=X.shape[1]
    grad=gradient(X, Y, w, opt)
    g=grad
    p=-g
    for i in range(d):
        Hp=np.dot(H, p)
        pHp=np.dot(p,Hp)
        alpha = np.dot(g, g) / pHp
        _loss = loss(X, Y, w, opt)
        print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(np.linalg.norm(g)))
        w= w + alpha*p
        g_next=g+alpha*Hp
        beta= np.dot(g_next, g_next) / np.dot(g, g)
        g=g_next
        p=-g_next+beta*p

    return w

###############################################################################
#  Newton Trust Region Method
###############################################################################
#  Jorge Nocedal and Stephen Wright, Numerical Optimization, second edition, Springer-Verlag, 2006, page 171.
def Trust_Region(X, Y, w, opt):
    #import time
    from datetime import datetime

    print ('--- Trust-region method ' + opt['subproblem_solver'] + ' ---')

    #loss_file = open("trust_loss.txt", "w")
    param_file = open("trust_param.txt", "w")

    # for 3D plots
    if opt['3d_plot'] == True:
        name= "TR_plot_file_"+str(opt['subproblem_solver']+".txt") 
        plot_file = open(name, "w")       

    n = X.shape[0]
    n_iterations = opt['n_iterations']
    eta = opt['success_treshold']
    eta2 = opt['very_success_treshold']
    gamma = opt['penalty_increase_multiplier']
    tr_radius = opt['initial_trust_radius']  # intial tr radius
    max_tr_radius = opt['max_trust_radius']  # max tr radius
    assert (tr_radius > 0 and max_tr_radius > 0), "negative radius"
    

    k = 0
    n_samples_seen = 0

    # samples on x-axis. Needs to be adapted if: stochastic gradients or increasing hessian sample size!

    if opt['subproblem_solver'] in{'cauchy_point','cg','cg_no_H'}:
        n_samples_per_step = n
    elif opt['subproblem_solver'] in {'sub_cg_no_H','sub_exact_exp','sub_GLTR_exp'}:
        n_samples_per_step = n ##TODO
    else:
        n_samples_per_step = 2 * n 
        

    grad = gradient(X, Y, w, opt)
    
    loss_collector=[]
    timings_collector=[]
    samples_collector=[]

    #start Timer and save initial values
    _loss = loss(X, Y, w, opt)

    loss_collector.append(_loss)
    timings_collector.append(0)
    samples_collector.append(0)

    print (_loss,np.linalg.norm(grad))

    #loss_file.write(str(0) + '\t' + str(_loss) + '\t'+ str(0) + '\t' + str(tr_radius) + '\t' + str(int(n*0.05 + 2)) +'\t'+ str( np.linalg.norm(grad))+ '\n')
    #w_str = ','.join(['%.5f' % num for num in w])
    #param_file.write(w_str + '\n')
    start = datetime.now()
    


    # while (n_samples_seen <n_samples):
    if opt['subproblem_solver'] == 'cauchy_point':
        n_iterations = n_iterations*3
    elif opt['subproblem_solver'] in {'sub_exact_exp' , 'sub_GLTR_exp'}:
        n_iterations= int(n_iterations*1)

    timing= 0
    for i in range(n_iterations):
        bad_approx = False           

        opt['TR_sample_size'] = int(min(n, n*0.05 + 2 ** (i + 1)))
        
        #H = hessian(X, Y, w, opt)
        # solve subproblem

         ## draw samples ##

        if opt['subproblem_solver'] in {'sub_GLTR_exp','sub_exact_exp'}:
            indices=np.random.choice(range(n), opt['TR_sample_size'], replace=False)
            _X=get_batch(X, indices)
            _Y = Y[[indices]]
        else:
            _X=X
            _Y=Y

        if opt['loss_type']=='multiclass_regression':
                nC = opt['n_classes']
                d=X.shape[1]
                global P_multi
                w_multi=np.matrix(w.reshape(nC,d).T)
                z_multi=np.dot(_X,w_multi) #activation of each i for class c
                z_multi-=np.max(z_multi,axis=1)  # model is overparametrized. allows to subtract maximum to prevent overflow. 
                h_multi = np.exp(z_multi)
                P_multi= np.array(h_multi/np.sum(h_multi,axis = 1)) #gives matrix with with [P]i,j = probability of sample i to be in class j (n x nC)
        elif opt['loss_type']in{'binary_regression','non_convex'}:
                global d_binary
                _z=_X.dot(w)
                _z = phi(-_z) #TODO warum ist hier ein -? gegenchecken mit np.dot(H,v)
                d_binary = _z * (1 - _z)

        s = solve_TR_subproblem(grad,tr_radius, opt, _X, _Y, w)


        

        # 'Compute ratio actual reduction/predicted reduction for the trial step
        ## function decrease
        current_f = loss(X, Y, w, opt)

        # for 3d plots: save current iterate 'w' and loss 'current_f' + subproblem solver
        if opt['3d_plot'] == True:
            plot_file.write('\t'.join(['%.5f' % num for num in w]) + '\t' + str(current_f) + '\n')


        fd = current_f - loss(X, Y, w + s, opt)
        ##model decrease (f(x) cancels out)

        #if opt['explicit_hessian'] == True:
        #    H = hessian(X, Y, w, opt)
        #    Hs = np.dot(H, s)
        #else:
        Hs=hv(_X, _Y, w, opt,s)


        md = -(np.dot(grad, s) + 0.5 * np.dot(s, Hs))
        if md <= 0:
            print ('bad approx: zero or negative model decrease. Causes failure in rho, breaking out')
            bad_approx = True
        ##ratio
        rho = fd / md
        # update the TR radius according to actual/predicted ratio
        if rho < eta:
            tr_radius *= 1/gamma
            print ('unscuccesful iteration')
        else:
            if rho > eta2 and (np.linalg.norm(s) - tr_radius < 1e-10) and bad_approx == False:
                tr_radius = min(opt['penalty_derease_multiplier'] * tr_radius, max_tr_radius)

        # Update if ratio is high enough
        if rho > eta and bad_approx == False:
            w = w + s
            grad = gradient(X, Y, w, opt) # inside for performance
            
        if np.linalg.norm(grad) < opt['g_tol']:
            break
        n_samples_seen += n_samples_per_step


        ### CHECKPOINT ###
        # if i % opt['recording_step'] == 0:
        if True:
            _loss = loss(X, Y, w, opt) # we could also just save the w's and timing and then reconstruct the loss later. (would not be fair compared to scipy though)
            _timing=timing
            timing=(datetime.now() - start).total_seconds()
            
            _sn=np.linalg.norm(s)

            print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' eta = ' + str(eta) + ' norm_grad = ' + str(
                np.linalg.norm(grad)), 'time= ', timing-_timing, 'tr_radius=',tr_radius, "stepnorm=",_sn)
            if opt['x_axis_samples'] == True:
                timings_collector.append(n_samples_seen)
                #loss_file.write(str(n_samples_seen) + '\t' + str(_loss) + '\t'+ str(_sn) + '\t'+ str(tr_radius) + '\t' + str(opt['TR_sample_size']) + '\t'+ str(np.linalg.norm(grad))+'\n')
            else:
                timings_collector.append(timing)
                samples_collector.append(n_samples_seen)

                #loss_file.write(str(timing) + '\t' + str(_loss) + '\t'+ str(_sn) + '\t'+ str(tr_radius) + '\t'+ str(opt['TR_sample_size']) +'\t'+ str(np.linalg.norm(grad))+ '\n')
            #w_str = ','.join(['%.5f' % num for num in w])

            #param_file.write(w_str +'\n')
            loss_collector.append(_loss)
            k += 1

     # for 3d plots: save current iterate 'w' and loss 'current_f' + subproblem solver
    if opt['3d_plot'] == True:
        plot_file.write('\t'.join(['%.5f' % num for num in w]) + '\t' + str(current_f) + '\n')        
    #loss_file.close()
    param_file.close()

    return w,timings_collector,loss_collector, samples_collector


def solve_TR_subproblem(grad, tr_radius, opt, X, Y, w):
    from scipy import linalg

    subproblem_solver = opt['subproblem_solver']
    tolerance = min(opt['subproblem_tolerance_cg/GLTR'], sqrt(linalg.norm(grad)) * linalg.norm(grad)) # <- this guy may become 0
    if tolerance == 0:
        tolerance = opt['subproblem_tolerance_cg/GLTR']
    eps_exact = opt['subproblem_tolerance_exact']
    

    if (subproblem_solver == 'cauchy_point'):
        Hg=hv(X, Y, w, opt,grad)
        gBg = np.dot(grad, Hg)
        tau = 1
        ## Compute Cauchy point

        # if model is convex quadratic the unconstrained minimizer may be inside the TR
        if gBg > 0:
            tau = min(linalg.norm(grad) ** 3 / (tr_radius * gBg), 1)
        pc = - tau * tr_radius * np.divide(grad, linalg.norm(grad))
        return pc

    elif (subproblem_solver == 'dog_leg'):
        H = hessian(X, Y, w, opt)
        gBg = np.dot(grad, np.dot(H, grad))
        if gBg <= 0:
            raise ValueError(
                'dog_leg requires H to be positive definite!')  # Dogleg requires H to be positive definite.

        ## Compute the Newton Point and return it if inside the TR
        cholesky_B = linalg.cho_factor(H)
        pn = -linalg.cho_solve(cholesky_B, grad)
        if (linalg.norm(pn) < tr_radius):
            return pn
        # Compute the 'unconstrained Cauchy Point'
        pc = -(np.dot(grad, grad) / gBg) * grad
        pc_norm = linalg.norm(pc)
        # if it is outside the TR, return the point where the path intersects the boundary
        if pc_norm >= tr_radius:
            p_boundary = pc * (tr_radius / pc_norm)
            return p_boundary
        # else, give intersection of path from pc to pn with tr radius.
        # Requieres finding a line-sphere intersection, i.e. solving the quadratic equation ||pc + t*(pn - pc)||^2 != trust_radius^2
        t_lower, t_upper = solve_quadratic_equation(pc, pn, tr_radius)
        p_boundary = pc + t_upper * (pn - pc)
        return p_boundary

    elif subproblem_solver in {'cg', 'cg_no_H', 'sub_cg_no_H'}:  # this is algo 7.2 from Nocedal & Wright which is the Steihaug-Toint CG method (w/p precoditioning) see also Conn et al 2000 Alg 7.5.1
        grad_norm = linalg.norm(grad)
        p_start = np.zeros_like(grad)

        # stop if torelance already satisfied (thats not good for 2nd order points!!!)
        if grad_norm < tolerance:
            return p_start
        # initialise
        z = p_start
        r = grad
        d = -r
        k = 0
        if subproblem_solver == 'cg':
            H = hessian(X, Y, w, opt)
        elif (subproblem_solver=='sub_cg_no_H'):
            n = X.shape[0]
            indices = np.random.choice(range(n), opt['TR_sample_size'], replace=False)
            X=get_batch(X, indices)
            Y = Y[[indices]]            
        while True:



            if subproblem_solver in {'cg_no_H','sub_cg_no_H'}:
                Bd=hv(X, Y, w, opt,d)
            else:
                Bd = np.dot(H, d)

            dBd = np.dot(d, Bd)
            # terminate when encountering a direction of negative curvature with lowest boundary point along current search direction
            if dBd <= 0:
                t_lower, t_upper = solve_quadratic_equation(z, d, tr_radius)
                p_low = z + t_lower * d
                p_up = z + t_upper * d
                m_p_low = loss(X, Y, w + p_low, opt) + np.dot(grad, p_low) + 0.5 * np.dot(p_low, np.dot(H, p_low))
                m_p_up = loss(X, Y, w + p_up, opt) + np.dot(grad, p_up) + 0.5 * np.dot(p_up, np.dot(H, p_up))
                if m_p_low < m_p_up:
                    return p_low
                else:
                    return p_up

            alpha = np.dot(r, r) / dBd
            z_next = z + alpha * d
            # terminate if z_next violates TR bound and
            if linalg.norm(z_next) >= tr_radius:
                # return intersect of current search direction w/ boud
                t_lower, t_upper = solve_quadratic_equation(z, d, tr_radius)
                return z + t_upper * d
            r_next = r + alpha * Bd
            if linalg.norm(r_next) < tolerance:
                return z_next

            beta_next = np.dot(r_next, r_next) / np.dot(r, r)
            d_next = -r_next + beta_next * d
            # update iterates
            z = z_next
            r = r_next
            d = d_next
            k = k + 1


    elif subproblem_solver in {'GLTR','GLTR_no_H','sub_GLTR_exp'}:

        g_norm = linalg.norm(grad)
        s = np.zeros_like(grad)

        if g_norm == 0:
            # escape along the direction of the leftmost eigenvector as far as tr_radius allows us to. (exact solver handles that) ## cheapter in full krylov space??
            print ('zero gradient encountered')
            H = hessian(X, Y, w, opt)
            s = exact_TR_suproblem_solver(grad, H, tr_radius, eps_exact)

        else:
            # initialise
            g = grad
            p = -g
            gamma = g_norm
            T = np.zeros((1, 1))
            alpha_k = []
            beta_k = []
            interior_flag = True
            k = 0

            if (subproblem_solver == 'GLTR'):  #gltr is always more inefficient but just as affective as gltr_no_H. So there's no point in making a subsampled version.
                H = hessian(X, Y, w, opt)

            #elif (subproblem_solver=='sub_GLTR_exp'):
             #   n = X.shape[0]
              #  indices = np.random.choice(range(n), opt['TR_sample_size'], replace=False)
               # X=get_batch(X, indices)
                #Y = Y[[indices]]           
            while True:
                if subproblem_solver in {'GLTR_no_H' ,'sub_GLTR_exp'}:
                    Hp=hv(X, Y, w, opt,p)
                else:
                    Hp = np.dot(H, p)

                pHp = np.dot(p, Hp)

                alpha = np.dot(g, g) / pHp
               
                alpha_k.append(alpha)

                ###Lanczos Step 1: Build up subspace (needs g and H) for subproblem 7.5.58
                # a) Create g_lanczos = gamma*e_1
                e_1 = np.zeros(k + 1)
                e_1[0] = 1.0
                g_lanczos = gamma * e_1
                # b) Create T for Lanczos Model (Eq 5.2.21 in Conn et al. 2000). <- More efficient ways than np.ix_ ???
                T_new = np.zeros((k + 1, k + 1))
                if k == 0:
                    T[k, k] = 1. / alpha
                    #T_new[np.ix_(range(0, k), range(0, k))] = T
                    T_new[0:k,0:k]=T #redundant???

                else:
                    #T_new[np.ix_(range(0, k), range(0, k))] = T  # recycle T-1
                    T_new[0:k,0:k]=T
                    T_new[k, k] = 1. / alpha + beta / alpha_k[k - 1]
                    T_new[k - 1, k] = sqrt(beta) / abs(alpha_k[k - 1])
                    T_new[k, k - 1] = sqrt(beta) / abs(alpha_k[k - 1])
                    T = T_new

                if (interior_flag == True and alpha < 0) or linalg.norm(s + alpha * p) >= tr_radius:
                    interior_flag = False

                if interior_flag == True:
                    s = s + alpha * p
                else:
                    ###Lanczos Step 2: solve problem in subspace
                    h = exact_TR_suproblem_solver(g_lanczos, T, tr_radius, eps_exact)

                g_next = g + alpha * Hp

                # test for convergence
                e_k = np.zeros(k + 1)
                e_k[k] = 1.0
                if interior_flag == True and linalg.norm(g_next) < tolerance :
                    break

                if interior_flag == False and linalg.norm(g_next) * abs(np.dot(h, e_k)) < tolerance: 
                                #if linalg.norm(y)*abs(np.dot(u,e_k))< min(0.0001,np.linalg.norm(u)/max(1, sigma))*grad_norm:
                    break

                #if interior_flag == True and linalg.norm(g_next) < opt['krylov_tol'] :
                #    break

                #if opt['krylov_stop_as_in_ARC'] == True:
                #if False:
                #    if interior_flag == False and linalg.norm(g_next) * abs(np.dot(h, e_k)) < min(opt['krylov_tol'],np.linalg.norm(h)/max(1, 1/tr_radius))*grad_norm:
                #        break
                #else:               
                #    if interior_flag == False and linalg.norm(g_next) * abs(np.dot(h, e_k)) < opt['krylov_tol']: 
                #        break

                if k==X.shape[1]:
                    print ('Krylov dimensionality reach full space! Breaking out..')
                    break
                beta = np.dot(g_next, g_next) / np.dot(g, g)
                beta_k.append(beta)
                p = -g_next + beta * p
                g = g_next
                k = k + 1
            if interior_flag == False:
                #### Recover Q by building up the lanczos space (Eq 5.2.18  q_k=g/gn)
                n = np.size(grad)
                Q1 = np.zeros((n, k + 1))

                g = grad
                p = -g
                for j in range(0, k + 1):
                    gn = np.linalg.norm(g)
                    if j == 0:
                        sigma = 1
                    else:
                        sigma = -np.sign(alpha_k[j - 1]) * sigma
                    Q1[:, j] = sigma * g / gn  # see equation 5.2.18 or p.229

                    if not j == k:
                        if (subproblem_solver == 'GLTR'):
                            g = g + alpha_k[j] * np.dot(H, p)
                        else:
                            Hp=hv(X, Y, w, opt,p)
                            g= g + alpha_k[j] * Hp
                        p = -g + beta_k[j] * p

                # compute final step in R^n
                s = np.dot(Q1, np.transpose(h))
        return s


    elif (subproblem_solver == 'exact'):
        H = hessian(X, Y, w, opt)
        s = exact_TR_suproblem_solver(grad, H, tr_radius, eps_exact)
        return s

    elif (subproblem_solver == 'sub_exact_exp'):
        
        H = hessian(X, Y, w, opt)
        s = exact_TR_suproblem_solver(grad, H, tr_radius, eps_exact)
        return s

    else:
        raise ValueError('solver unknown')


def exact_TR_suproblem_solver(grad, H, tr_radius, eps_exact):
    from scipy import linalg
### This is an extended Version of Algorithm 7.3.2 as in Conn, Toint, Gould: Trust Region Methods
## Solves TR subproblems exactly.
    s = np.zeros_like(grad)
    ## Step 0: initialize safeguards
    H_ii_min = min(np.diagonal(H))
    H_max_norm = sqrt(H.shape[0] ** 2) * np.absolute(H).max()
    H_fro_norm = np.linalg.norm(H, 'fro')
    gerschgorin_l = max([H[i, i] + (np.sum(np.abs(H[i, :])) - np.abs(H[i, i])) for i in range(len(H))])
    gerschgorin_u = max([-H[i, i] + (np.sum(np.abs(H[i, :])) - np.abs(H[i, i])) for i in range(len(H))])

    lambda_lower = max(0, -H_ii_min, np.linalg.norm(grad) / tr_radius - min(H_fro_norm, H_max_norm, gerschgorin_l))
    lambda_upper = max(0, np.linalg.norm(grad) / tr_radius + min(H_fro_norm, H_max_norm, gerschgorin_u))
    # print ('lambda lower=', lambda_lower, 'lambda upper=', lambda_upper)

    if lambda_lower == 0:  # allow for fast convergence in case of inner solution
        lambda_j = lambda_lower
    else:
        lambda_j = random.uniform(lambda_lower, lambda_upper)
    i=0
    # Root Finding
    while True:
        i+=1
        lambda_in_N = False
        lambda_plus_in_N = False
        B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
        try:
            # 1 Factorize B
            L = np.linalg.cholesky(B)
            # 2 Solve LL^Ts=-g
            Li = np.linalg.inv(L)
            s = - np.dot(np.dot(Li.T, Li), grad)
            sn = np.linalg.norm(s)
            ## 2.1 Termination: Lambda in F, if q(s(lamda))<eps_opt q(s*) and sn<eps_tr tr_radius -> stop. By Conn: Lemma 7.3.5:
            phi_lambda = 1. / sn - 1. / tr_radius
            #if (abs(sn - tr_radius) <= eps_exact * tr_radius):
            if (abs(phi_lambda)<=eps_exact): #
                break;

            # 3 Solve Lw=s
            w = np.dot(Li, s)
            wn = np.linalg.norm(w)

            
            ##Step 1: Lambda in L
            #if lambda_j > 0 and (phi_lambda) < 0 and np.any(grad != 0): <-- this should be wrong. If cholesky succeeds we are to the right of -lambda_1. If then phi<0: no hard case
            if lambda_j > 0 and (phi_lambda) < 0:
                # print ('lambda: ',lambda_j, ' in L')
                lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)
                lambda_j = lambda_plus


            ##Step 2: Lambda in G    (sn<tr_radius)
            elif (phi_lambda) > 0 and lambda_j > 0 and np.any(grad != 0): #TODO: remove grad
                # print ('lambda: ',lambda_j, ' in G')
                lambda_upper = lambda_j
                lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)

                ##Step 2a: If factorization succeeds: lambda_plus in L
                if lambda_plus > 0:
                    try:
                        # 1 Factorize B
                        B_plus = H + lambda_plus * np.eye(H.shape[0], H.shape[1])
                        L = np.linalg.cholesky(B_plus)
                        lambda_j = lambda_plus
                        # print ('lambda+', lambda_plus, 'in L')


                    except np.linalg.LinAlgError:
                        lambda_plus_in_N = True

                ##Step 2b/c: If not: Lambda_plus in N
                if lambda_plus <= 0 or lambda_plus_in_N == True:
                    # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
                    ##is H positive definite?
                    try:
                        U = np.linalg.cholesky(H)
                        H_pd = True
                    except np.linalg.LinAlgError:
                        H_pd = False

                    if lambda_lower == 0 and H_pd == True and phi_lambda >= 0: #cannot happen in ARC!
                        lambda_j = 0
                        #print ('inner solution found')
                        break
                    # 2. Else, choose a lambda within the safeguard interval
                    else:
                        # print ('lambda_plus', lambda_plus, 'in N')
                        lambda_lower = max(lambda_lower, lambda_plus)  # reset lower safeguard
                        lambda_j = max(sqrt(lambda_lower * lambda_upper),
                                       lambda_lower + 0.01 * (lambda_upper - lambda_lower))

                        lambda_upper = np.float32(
                            lambda_upper)  # for some reason lambda_l is np.float and lambda_u is float and then if == fails
                        
                        # no iterate falls into L. Then, the length of the interval of uncertainty shrinks every iteration and thus converges to zero. This can only happen
                        # in hard case or interior convergence, but we have checked the latter before.
                        if lambda_lower == lambda_upper:
                            lambda_j = lambda_lower
                            ## Hard case: now lambda_j=lambda_l=lambda_u=-lambda_1=lambda^M and s=s_crit
                            # we have -lambda_1 but we need its corresponding unit eigenvector
                            ew, ev = linalg.eigh(H, eigvals=(0, 0))
                            d = ev[:, 0]
                            dn = np.linalg.norm(d)
                            assert (ew == -lambda_j), "Ackward: in hard case but lambda_j != -lambda_1"
                            #tao_lower, tao_upper = solve_quadratic_equation(s, d / dn, tr_radius) # this may be wrong. gives a=norm(u1-s)^2 while a should be norm(u1)^2???
                            tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-tr_radius**2)
                            s=s + tao_lower * d   
                            print ('hard case resolved inside')

                            return s

            # (phi_lambda)=0, i.e. we're done.
            elif (phi_lambda) == 0: 
                break
            else:      #TODO:  move into if lambda+ column #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                lambda_in_N = True
        ##Step 3: Lambda in N
        except np.linalg.LinAlgError:
            lambda_in_N = True
        if lambda_in_N == True:
            # print ('lambda: ',lambda_j, ' in N')
            try:
                U = np.linalg.cholesky(H)
                H_pd = True
            except np.linalg.LinAlgError:
                H_pd = False

            # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
            if lambda_lower == 0 and H_pd == True and phi_lambda >= 0: #what phi_lambda is referenced here??
                lambda_j = 0
                #print ('inner solution found')
                break
            # 2. Else, choose a lambda within the safeguard interval
            else:
                lambda_lower = max(lambda_lower, lambda_j)  # reset lower safeguard
                lambda_j = max(sqrt(lambda_lower * lambda_upper),
                               lambda_lower + 0.01 * (lambda_upper - lambda_lower))  # eq 7.3.14
                lambda_upper = np.float32(lambda_upper)  
                # Check for Hard Case:
                if lambda_lower == lambda_upper:
                    lambda_j = lambda_lower
                    ew, ev = linalg.eigh(H, eigvals=(0, 0))
                    d = ev[:, 0]
                    dn = np.linalg.norm(d)
                    assert (ew == -lambda_j), "Ackward: in hard case but lambda_j != -lambda_1"
                    tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-tr_radius**2)
                    s=s + tao_lower * d 

                    print ('hard case resolved outside')
                    return s




    # compute final step
    B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
    # 1 Factorize B
    L = np.linalg.cholesky(B)
    # 2 Solve LL^Ts=-g
    Li = np.linalg.inv(L)
    s = - np.dot(np.dot(Li.T, Li), grad)
    #print (i,' exact solver iterations')
    return s


def solve_quadratic_equation(pc, pn, tr_radius):
    # solves ax^2+bx+c=0
    a = np.dot(pn - pc, pn - pc)
    b = 2 * np.dot(pc, pn - pc)
    c = np.dot(pc, pc) - tr_radius ** 2
    sqrt_discriminant = sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper

def mitternachtsformel(a,b,c):
    sqrt_discriminant = sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper

# Adaptive Cubic Regularization

def ARC(X,Y,w,opt):
    from datetime import datetime
    from math import isnan
    print ('--- Adaptive Cubic Regularization ' + opt['ARC_subproblem_solver'] + ' ---')
    #loss_file = open("arc_loss.txt", "w")
    param_file = open("arc_param.txt", "w")
    n = X.shape[0] 
    d = X.shape[1]
    opt['full_sample']=n
    n_iterations = opt['n_iterations']

    if opt['ARC_subproblem_solver'] in {'cauchy_point','GD'}:
        n_iterations= n_iterations*3
    #elif opt['ARC_subproblem_solver']=='sub_exact_lin':
       # n_iterations= int(n_iterations*0.7)
    elif opt['ARC_subproblem_solver']=='sub_exact_ada':
        n_iterations= int(n_iterations*7)

    elif opt['ARC_subproblem_solver'] in {'sub_lanczos_exp'}:
        n_iterations= int(opt['n_iterations']*1.5)     
        #n_iterations= 45
    elif opt['ARC_subproblem_solver']=='sub_lanczos_ada':
        n_iterations= int(opt['n_iterations']*1.5)
        

    eta_1 = opt['success_treshold']
    eta_2 = opt['very_success_treshold']
    gamma = opt['penalty_increase_multiplier']
    gamma2= opt['penalty_derease_multiplier']
    sigma = opt['initial_penalty_parameter']
    new_sampling_flag=opt.get('new_sampling_scheme', False)
    k=0
    n_samples_seen=0
    n_samples_per_step=2*n

    lambda_k=0
    successful_flag=True


    if opt['loss_type'] == 'multiclass_regression':
        opt['ground_truth']=full_ground_truth

    grad = gradient(X, Y, w, opt)
    grad_norm=np.linalg.norm(grad)
    #start Timer and save initial values

    loss_collector=[]
    timings_collector=[]
    steps_collector=[]
    samples_collector=[]
    
    _loss = loss(X, Y, w, opt)
    loss_collector.append(_loss)
    timings_collector.append(0)
    samples_collector.append(0)

    #loss_file.write(str(0) + '\t' + str(_loss) +'\t'+ str(0) + '\t' + str(sigma) + '\t' + str(int(n*0.05 + 2)) +'\t'+ str(np.linalg.norm(grad))+  '\n')
    #w_str = ','.join(['%.5f' % num for num in w])
    #param_file.write(w_str + '\n')

    # for 3D plots
    if opt['3d_plot'] == True:
        name= "ARC_plot_file_"+str(opt['ARC_subproblem_solver']+".txt") 
        plot_file = open(name, "w")    


    start = datetime.now()
    timing=0
    exp_growth_constant=((1-opt['initial_sample_size'])*n)**(1/n_iterations)

    for i in range(n_iterations):
        bad_approx = False

        #### SUBSAMPLING #####

        ## a) determine batch size ##
        if opt['ARC_subproblem_solver'] in {'sub_exact_exp','sub_lanczos_exp'}:
            #opt['ARC_sample_size'] = int(min(n, n*opt['initial_sample_size']+ 2 ** (i + 1)))

            opt['ARC_sample_size'] = int(min(n, n*opt['initial_sample_size']+ exp_growth_constant**(i+1)))+1
            opt['ARC_sample_size_gradient']=n


        elif opt['ARC_subproblem_solver'] in {'sub_exact_lin','sub_lanczos_lin'}:
            opt['ARC_sample_size'] = int(min(n, max(n*opt['initial_sample_size'], n/n_iterations*(i+1)))) 
            opt['ARC_sample_size_gradient']=n
            opt['ARC_sample_size'] = int(n*opt['initial_sample_size']) 
            opt['ARC_sample_size_gradient']=int(n*opt['initial_sample_size']) 

           

        elif opt['ARC_subproblem_solver'] in{'sub_exact_ada', 'sub_lanczos_ada'}:
            if i==0 or i==1:
                opt['ARC_sample_size']=int(opt['initial_sample_size']*n)
                opt['ARC_sample_size_gradient']=opt['gradient_sampling']*int(opt['initial_sample_size_gradient']*n)+(1-opt['gradient_sampling'])*n

            else:
                #adjust sampling C such that the first step, gives a sample size of opt['initial sample size']
                if i==2:
                    if new_sampling_flag == False:
                        c=(opt['initial_sample_size']*n*sn**2)/log(d)
                    else:
                        c=(opt['initial_sample_size']*n*sn**2)/log(d)/36*sn/Hs_norm

                    c_grad=(opt['initial_sample_size_gradient']*n*sn**4)/log(d)
                if successful_flag==False:
                    opt['ARC_sample_size']=min(n,int(opt['ARC_sample_size']*opt['unsuccessful_sample_scaling']))
                    opt['ARC_sample_size_gradient']=opt['gradient_sampling']*min(n,int(opt['ARC_sample_size_gradient']*opt['unsuccessful_sample_scaling'])) +(1-opt['gradient_sampling'])*n

                else:
                    opt['ARC_sample_size_gradient']=opt['gradient_sampling']*min(n,int(max((c*log(d)/(sn**4)*opt['sample_scaling_gradient']),opt['initial_sample_size_gradient']*n))) +(1-opt['gradient_sampling'])*n
                    if new_sampling_flag == False:
                        opt['ARC_sample_size']=min(n,int(max((c*log(d)/(sn**2)*opt['sample_scaling']),opt['initial_sample_size']*n)))
                    else:
                        print('hs norm/sn',Hs_norm/sn)
                        opt['ARC_sample_size']=min(n,int(max((c*(36*(Hs_norm/sn)*log(d))/(sn**2)*opt['sample_scaling']),opt['initial_sample_size']*n)))


        else: # no subsampling!
            opt['ARC_sample_size']=n
            opt['ARC_sample_size_gradient']=n

        n_samples_per_step=opt['ARC_sample_size']+opt['ARC_sample_size_gradient']
        n = X.shape[0]
        ## b) 1. draw hessian samples
        if opt['ARC_sample_size'] <n:
            
            #indices = np.random.choice(range(n), opt['ARC_sample_size'], replace=False)
            #_X=np.zeros((len(indices),d))
            #_X=get_batch(X, indices)
            #_Y=np.zeros(len(indices))
            #_Y = Y[[indices]]

            #1. get boolean index array
            int_idx=np.random.permutation(n)[0:opt['ARC_sample_size']]
            bool_idx = np.zeros(n,dtype=bool)
            bool_idx[int_idx]=True
            #2. sample rows 
            _X=np.zeros((opt['ARC_sample_size'],d))
            _X=np.compress(bool_idx,X,axis=0)
            _Y=np.compress(bool_idx,Y,axis=0)



        else: 
            _X=X
            _Y=Y
 
        ## b) 2. draw gradient samples
        if opt['ARC_sample_size_gradient'] < n:
            #indices2 =np.random.choice(range(n), opt['ARC_sample_size_gradient'], replace=False)
            #_X2=np.zeros((len(indices2),d))
            #_X2=get_batch(X, indices2)
            #_Y2=np.zeros(len(indices2))
            #_Y2 = Y[[indices2]]

            #1. get boolean index array
            int_idx2=np.random.permutation(n)[0:opt['ARC_sample_size_gradient']]
            bool_idx2 = np.zeros(n,dtype=bool)
            bool_idx2[int_idx2]=True
            #2. sample rows 
            _X2=np.zeros((opt['ARC_sample_size_gradient'],d))
            _X2=np.compress(bool_idx2,X,axis=0)
            _Y2=np.compress(bool_idx2,Y,axis=0)

            if opt['loss_type']=='multiclass_regression':
                opt['ground_truth']=np.compress(bool_idx2,full_ground_truth,axis=0)

        else:
            _X2=X
            _Y2=Y
            if opt['gradient_sampling']==True and opt['loss_type']=='multiclass_regression':
                opt['ground_truth']=full_ground_truth

           

        
        #Precompute results for later Hv calculation that happens in each krylov subspace            
        if opt['loss_type']=='multiclass_regression':
            nC = opt['n_classes']
            global P_multi
            w_multi=np.matrix(w.reshape(nC,d).T)
            z_multi=np.dot(_X,w_multi) #activation of each i for class c
            z_multi-=np.max(z_multi,axis=1)  # model is overparametrized. allows to subtract maximum to prevent overflow. 
            h_multi = np.exp(z_multi)
            P_multi= np.array(h_multi/np.sum(h_multi,axis = 1)) #gives matrix with with [P]i,j = probability of sample i to be in class j (n x nC)
        elif opt['loss_type'] in {'binary_regression','non_convex'} :
            global d_binary
            _z=_X.dot(w)
            _z = phi(-_z)
            d_binary = _z * (1 - _z)


        #Recompute gradient either because of accepted step or because of re-sampling (or both)
        if opt['gradient_sampling']==True or successful_flag==True:
            grad = gradient(_X2, _Y2, w, opt) # need to recompute for unsuccessful iterations too since gradient is resampled!
            grad_norm =np.linalg.norm(grad)   

        (s,lambda_k) = solve_ARC_subproblem(grad,sigma, opt, _X, _Y, w, successful_flag, lambda_k)

        sn=np.linalg.norm(s)

        #Assess step:
        current_f = loss(X, Y, w, opt)

        # for 3d plots: save current iterate 'w' and loss 'current_f' + subproblem solver
        if opt['3d_plot'] == True:
            plot_file.write('\t'.join(['%.5f' % num for num in w]) + '\t' + str(current_f) + '\n')

        fd = current_f - loss(X, Y, w + s, opt)
        Hs=hv(_X, _Y, w, opt,s)
        Hs_norm=np.linalg.norm(Hs)
        md=-(np.dot(grad, s) + 0.5 * np.dot(s, Hs)+1/3*sigma*sn**3)
       
        rho = fd / md
        if md < 0 or isnan(rho):
            print ('rho=',rho)
            print ('bad approx: zero or negative model decrease. Causes failure in rho, skipping step')
            bad_approx = True
        # Update w
        if rho > eta_1 and bad_approx == False:
            w = w + s                        

        n_samples_seen += n_samples_per_step

        #Update penalty parameter
        if rho >= eta_2 and bad_approx == False:
            #sigma=max(sigma/4.,np.nextafter(0,1))
            if opt['sigma_decrease_as_in_ARC'] == True:
                sigma= max(min(grad_norm,sigma),np.nextafter(0,1)) # see ARC part I chap Exp
            else:
                sigma=max(sigma/gamma2,1e-16)
        elif rho < eta_1 or bad_approx == True:
            sigma = gamma*sigma
            successful_flag=False   
            print ('unscuccesful iteration')


        if rho >= eta_1:
            successful_flag=True


        ### CHECKPOINT ###
        # if i % opt['recording_step'] == 0:
        #if n_samples_seen >= opt['recording_step'] * k:
        if True:
            _loss = loss(X, Y, w, opt) # we could also just save the w's and timing and then reconstruct the loss later. (would not be fair compared to scipy though)
            _timing=timing
            timing=(datetime.now() - start).total_seconds()
            print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(
                grad_norm), 'time= ', timing-_timing, 'penalty=', sigma, 'sn=', sn)

            if opt['ARC_subproblem_solver'] in {'sub_exact_exp' ,'sub_lanczos_exp' ,'sub_exact_lin' ,'sub_lanczos_lin' ,'sub_exact_ada','sub_lanczos_ada'}:
                print ('Sample Size Hessian=', opt['ARC_sample_size'])
                print ('Sample Size Gradient=', opt['ARC_sample_size_gradient'])
    
            if opt['x_axis_samples'] == True:
                timings_collector.append(n_samples_seen)
                #loss_file.write(str(n_samples_seen) + '\t' + str(_loss) + '\t'+ str(sn) + '\t' + str(sigma) + '\t' + str(opt['ARC_sample_size']) +'\t' + str(grad_norm)+ '\n')
            else:
                timings_collector.append(timing)
                samples_collector.append(n_samples_seen)

                #loss_file.write(str(timing) + '\t' + str(_loss) + '\t' + str(sn) + '\t' + str(sigma) + '\t' + str(opt['ARC_sample_size']) +'\t'+ str(grad_norm)+  '\n')
            #w_str = ','.join(['%.5f' % num for num in w])
            #param_file.write(w_str + '\n')
            
            #steps_collector.append(sn)
            #samples_collector.append(opt['ARC_sample_size'])
            loss_collector.append(_loss)
            if grad_norm < opt['g_tol']:
                break

            k += 1
        print ('************************')

    #loss_file.close()
    param_file.close()

    return w,timings_collector,loss_collector, samples_collector
    #return w,timings_collector,loss_collector, steps_collector, samples_collector

def solve_ARC_subproblem(grad, sigma, opt, X, Y, w, successful_flag,lambda_k):
    from math import sqrt, ceil

    if ( opt['ARC_subproblem_solver'] == 'cauchy_point'):
        #min m(-a*grad) leads to finding the root of a quadratic polynominal
        Hg=hv(X, Y, w, opt,grad)
        gHg=np.dot(grad,Hg)
        #if gHg>0: <---------- I think this only makes sense in the TR case. Here the penalty should automatically prevent alpha=infty
        a=sigma*np.linalg.norm(grad)**3
        b=gHg
        c=-np.dot(grad,grad)
        (alpha_l,alpha_h)=mitternachtsformel(a,b,c)
        alpha=alpha_h
        #else:
        #   alpha=1 
        s=-alpha*grad
        return (s,0)
    elif( opt['ARC_subproblem_solver'] == 'GD'):
        model_gradient=grad
        grad_norm=np.linalg.norm(grad)
        d=X.shape[1]
        s=np.zeros(d) #<- maybe smart initialization
        i=0
        while True:#<- maybe smart stopping
            #take a step along negative gradient <- maybe linesearch
            eta = backtracking_line_search(X,Y,w,model_gradient,-model_gradient,opt)
            s=s-eta*model_gradient

            #update model gradient
            sn=np.linalg.norm(s)
            Bs=hv(X, Y, w, opt,s)
            model_gradient=grad+Bs+sigma*sn
            print ('mg norm', np.linalg.norm(model_gradient))

            if  np.linalg.norm(model_gradient)< opt['krylov_tol']*grad_norm or i==100:
                print ('GD iterations:'+str(i))
                break  

            i+=1
        return (s,0)
    elif opt['ARC_subproblem_solver'] in {'exact','sub_exact_exp','sub_exact_lin','sub_exact_ada'}:  
        H = hessian(X, Y, w, opt) #would be cool to memoize this. 

        (s, lambda_k) = exact_ARC_suproblem_solver(grad, H, sigma, opt['subproblem_tolerance_exact'],successful_flag,lambda_k,opt)
        return (s,lambda_k)

    elif opt['ARC_subproblem_solver'] in {'lanczos','sub_lanczos_exp','sub_lanczos_lin','sub_lanczos_ada'}:   
        y=grad
        grad_norm=np.linalg.norm(grad)
        gamma_k_next=grad_norm
        #q = []   #just for the moment, this is too memory intensive
        delta=[] 
        gamma=[] # save for cheaper reconstruction of Q

        dimensionality = len(w)
        if dimensionality <= 11000:
            q_list=[]    #For problems with d<10.000 only! Q can fit into memory

        k=0
        T = np.zeros((1, 1))

        while True:
            if gamma_k_next==0: #From T 7.5.16 u_k was the minimizer of m_k. But it was not accepted. Thus we have to be in the hard case.
                H = hessian(X, Y, w, opt)
                (s, lambda_k) = exact_ARC_suproblem_solver(grad,H, sigma, opt['subproblem_tolerance_exact'],successful_flag,lambda_k,opt)
                return (s,lambda_k)

            #a) create g
            e_1=np.zeros(k+1)
            e_1[0]=1.0
            g_lanczos=grad_norm*e_1
            #b) generate H
            gamma_k = gamma_k_next
            gamma.append(gamma_k)
            #q.append(y/gamma_k)

            if not k==0:
                q_old=q
            q=y/gamma_k
            
            if dimensionality<=15001:
                q_list.append(q)    #For problems with d<10.000 only!!!

            Hq=hv(X,Y,w,opt,q) #matrix free            
            delta_k=np.dot(q,Hq)
            delta.append(delta_k)
            T_new = np.zeros((k + 1, k + 1))
            if k==0:
                T[k,k]=delta_k
                y=Hq-delta_k*q
            else:
                T_new[0:k,0:k]=T
                T_new[k, k] = delta_k
                T_new[k - 1, k] = gamma_k
                T_new[k, k - 1] = gamma_k
                T = T_new
                y=Hq-delta_k*q-gamma_k*q_old

            gamma_k_next=np.linalg.norm(y)
            #### Solve Subproblem only in each x-th Krylov space            
            if opt['loss_type'] == 'multiclass_regression':
                if grad_norm<0.0099:
                    if k%100==0 and k>1000 or (k==dimensionality-1):
                        (u,lambda_k) = exact_ARC_suproblem_solver(g_lanczos,T, sigma, opt['subproblem_tolerance_exact'],successful_flag,lambda_k,opt)
                        e_k=np.zeros(k+1)
                        e_k[k]=1.0
                        if linalg.norm(y)*abs(np.dot(u,e_k))< min(opt['krylov_tol'],np.linalg.norm(u)/max(1, sigma))*grad_norm:
                            break
                elif grad_norm<0.5 or (k==dimensionality-1):  #close to the minimizer there is no need to solve the first x% krylov dimensions
                    if k%25==0 and k>300:
                        (u,lambda_k) = exact_ARC_suproblem_solver(g_lanczos,T, sigma, opt['subproblem_tolerance_exact'],successful_flag,lambda_k,opt)
                        e_k=np.zeros(k+1)
                        e_k[k]=1.0
                        if linalg.norm(y)*abs(np.dot(u,e_k))< min(opt['krylov_tol'],np.linalg.norm(u)/max(1, sigma))*grad_norm:
                            break
                else:       
                    if k%opt['solve_each_i-th_krylov_space'] ==0 or (k==dimensionality-1):
                        (u,lambda_k) = exact_ARC_suproblem_solver(g_lanczos,T, sigma, opt['subproblem_tolerance_exact'],successful_flag,lambda_k,opt)
                        e_k=np.zeros(k+1)
                        e_k[k]=1.0
                        if linalg.norm(y)*abs(np.dot(u,e_k))< min(opt['krylov_tol'],np.linalg.norm(u)/max(1, sigma))*grad_norm:
                            break
            else:

                if k %(opt['solve_each_i-th_krylov_space']) ==0 or (k==dimensionality-1) or gamma_k_next==0:
                    (u,lambda_k) = exact_ARC_suproblem_solver(g_lanczos,T, sigma, opt['subproblem_tolerance_exact'],successful_flag,lambda_k,opt)
                    e_k=np.zeros(k+1)
                    e_k[k]=1.0
                    if opt['krylov_stop_as_in_ARC'] == True:
                        if linalg.norm(y)*abs(np.dot(u,e_k))< min(opt['krylov_tol'],np.linalg.norm(u)/max(1, sigma))*grad_norm:
                            break
                    else:
                        if linalg.norm(y)*abs(np.dot(u,e_k)) < opt['krylov_tol']*grad_norm and not np.linalg.norm(u)==0:
                            break     

            if k==dimensionality-1: 
                print ('Krylov dimensionality reach full space!')
                break      
                        
            #print 'subiteration', k,'unorm', np.linalg.norm(u)
            #lambda_k=lambda_k # if exact solver is called in next krylov space, reuse lambda from this space. doesn't seem to help much
            successful_flag=False
            # test for convergence
            


            k=k+1

        print ('number of lanczos iterations= ', k)
        # Recover Q to compute s
        n=np.size(grad)
        
        #Q=np.zeros((n, k + 1))       #<--------- since numpy is ROW MAJOR its better to fill the transpose of Q
        Q=np.zeros((k + 1,n))

        y=grad
        

        for j in range (0,k+1):
            if dimensionality<=11000:
                #Q[:,j]=q_list[j]     
                Q[j,:]=q_list[j]

            else:
                if not j==0:
                    q_re_old=q_re
                q_re=y/gamma[j]
                Q[:,j]=q_re
                Hq=hv(X,Y,w,opt,q_re) #matrix free
                if j==0:
                    y=Hq-delta[j]*q_re
                elif not j==k:
                    y=Hq-delta[j]*q_re-gamma[j]*q_re_old
         ############################ 
    
        #s=np.dot(Q,np.transpose(u))    # <----------------------
        s=np.dot(u,Q)
        del Q
        return (s,lambda_k)
    else: 
        raise ValueError('solver unknown')

def exact_ARC_suproblem_solver(grad,H,sigma, eps_exact,successful_flag,lambda_k,opt):
    s = np.zeros_like(grad)

    #a) EV Bounds
    gershgorin_l=min([H[i, i] - np.sum(np.abs(H[i, :])) + np.abs(H[i, i]) for i in range(len(H))]) 
    gershgorin_u=max([H[i, i] + np.sum(np.abs(H[i, :])) - np.abs(H[i, i]) for i in range(len(H))]) 
    H_ii_min=min(np.diagonal(H))
    H_max_norm=sqrt(H.shape[0]**2)*np.absolute(H).max() # too costly?
    H_fro_norm=np.linalg.norm(H,'fro') # too costly? probably not as each iteration is far more costly because of factorization.

    #b) solve quadratic equation that comes from combining rayleigh coefficients
    (lambda_l1,lambda_u1)=mitternachtsformel(1,gershgorin_l,-np.linalg.norm(grad)*sigma)
    #(lambda_u2,lambda_l2)=mitternachtsformel(1,min(gershgorin_u,H_max_norm,H_fro_norm),-np.linalg.norm(grad)*sigma)
    (lambda_u2,lambda_l2)=mitternachtsformel(1,gershgorin_u,-np.linalg.norm(grad)*sigma)
    
    lambda_lower=max(0,-H_ii_min,lambda_l2)  
    lambda_upper=max(0,lambda_u1)            #0's should not be necessary



    if successful_flag==False and lambda_lower <= lambda_k <= lambda_upper: #reinitialize at previous lambda in case of unscuccesful iterations
        lambda_j=lambda_k
        #lambda_j=np.random.uniform(lambda_lower, lambda_upper)
    else:
        lambda_j=np.random.uniform(lambda_lower, lambda_upper)

    no_of_calls=0 
    #while True:
    for v in range(0,50):
        no_of_calls+=1
        lambda_plus_in_N=False
        lambda_in_N=False

        B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
        
        if lambda_lower==lambda_upper==0 or np.any(grad)==0:
            lambda_in_N=True
        else:
            try: # if this succeeds lambda is in L or G.
                # 1 Factorize B
                L = np.linalg.cholesky(B)
                # 2 Solve LL^Ts=-g
                Li = np.linalg.inv(L)
                s = - np.dot(np.dot(Li.T, Li), grad)
                sn = np.linalg.norm(s)
               
                ## 2.1 Terminate <- maybe more elaborated check possible as Conn L 7.3.5 ??? 
                phi_lambda=1./sn -sigma/lambda_j
                if (abs(phi_lambda)<=eps_exact): #
                    break
                # 3 Solve Lw=s
                w = np.dot(Li, s)
                wn = np.linalg.norm(w)

                
                
                ## Step 1: Lambda in L and thus lambda+ in L
                if phi_lambda < 0: 
                    #print ('lambda: ',lambda_j, ' in L')
                    c_lo,c_hi= mitternachtsformel((wn**2/sn**3),1./sn+(wn**2/sn**3)*lambda_j,1./sn*lambda_j-sigma)
                    lambda_plus=lambda_j+c_hi
                    #lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?

                    lambda_j = lambda_plus
    
                ## Step 2: Lambda in G, hard case possible
                elif phi_lambda>0:
                    #print ('lambda: ',lambda_j, ' in G')
                    #lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?
                    lambda_upper=lambda_j
                    _lo,c_hi= mitternachtsformel((wn**2/sn**3),1./sn+(wn**2/sn**3)*lambda_j,1./sn*lambda_j-sigma)
                    lambda_plus=lambda_j+c_hi
                    ##Step 2a: If lambda_plus positive factorization succeeds: lambda+ in L (right of -lambda_1 and phi(lambda+) always <0) -> hard case impossible
                    if lambda_plus >0:
                        try:
                            #1 Factorize B
                            B_plus = H + lambda_plus*np.eye(H.shape[0], H.shape[1])
                            L = np.linalg.cholesky(B_plus)
                            lambda_j=lambda_plus
                            #print ('lambda+', lambda_plus, 'in L')
                        except np.linalg.LinAlgError: 
                            lambda_plus_in_N=True
                    
                    ##Step 2b/c: else lambda+ in N, hard case possible
                    if lambda_plus <=0 or lambda_plus_in_N==True:
                        #print ('lambda_plus', lambda_plus, 'in N')
                        lambda_lower=max(lambda_lower,lambda_plus) #reset lower safeguard
                        lambda_j=max(sqrt(lambda_lower*lambda_upper),lambda_lower+0.01*(lambda_upper-lambda_lower))  
                        ## Check Hard Case <-might not be necessary
                        #if lambda_upper -1e-4 <= lambda_lower <= lambda_upper +1e-4:
                        lambda_lower=np.float32(lambda_lower)
                        lambda_upper=np.float32(lambda_upper)
                        if lambda_lower==lambda_upper:
                                lambda_j = lambda_lower #should be redundant?
                                ew, ev = linalg.eigh(H, eigvals=(0, 0))
                                d = ev[:, 0]
                                dn = np.linalg.norm(d)
                                #note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk facto may fall. lambda_j-1 should only be digits away!
                                tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-lambda_j**2/sigma**2)
                                s = s + tao_lower * d # both, tao_l and tao_up should give a model minimizer!
                                print ('hard case resolved') 
                                break
                    #else: #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                     #   lambda_in_N = True
                ##Step 3: Lambda in N
            except np.linalg.LinAlgError:
                lambda_in_N = True
        if lambda_in_N == True:
            #print ('lambda: ',lambda_j, ' in N')
            lambda_lower = max(lambda_lower, lambda_j)  # reset lower safeguard
            lambda_j = max(sqrt(lambda_lower * lambda_upper), lambda_lower + 0.01 * (lambda_upper - lambda_lower))  # eq 7.3.1
            #Check Hardcase
            #if (lambda_upper -1e-4 <= lambda_lower <= lambda_upper +1e-4):
            lambda_lower=np.float32(lambda_lower)
            lambda_upper=np.float32(lambda_upper)

            if lambda_lower==lambda_upper:
                lambda_j = lambda_lower #should be redundant?
                ew, ev = linalg.eigh(H, eigvals=(0, 0))
                d = ev[:, 0]
                dn = np.linalg.norm(d)
                if ew >=0: #H is pd and lambda_u=lambda_l=lambda_j=0 (as g=(0,..,0)) So we are done. returns s=(0,..,0)
                    break
                #note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk.fact. may fail. lambda_j-1 should only be digits away!
                sn= np.linalg.norm(s)
                #tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d/dn), sn**2-lambda_j/sigma)
                tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-lambda_j**2/sigma**2)
                s = s + tao_lower * d # both, tao_l and tao_up should give a model minimizer!
                print ('hard case resolved') 
                break 
    #print (no_of_calls,' exact solver iterations')
    return s,lambda_j


###########################################################
# Sub Sampled Newton Methods
###########################################################



# Sub-sampled Newton method + SAGA to compute stochastic gradients
# Erdogdu, Murat A., and Andrea Montanari.
# "Convergence rates of sub-sampled Newton methods." NIPS. 2015.
# TODO: Needs more debugging!!
def sub_sampled_Newton_SAGA(X, Y, w, opt):
    print ('--- Sub-sampled Newton method ---')
    n = X.shape[0]
    d = X.shape[1]
    n_passes = opt['n_passes']
    eta = opt['learning_rate']
    bs = 1

    loss_file = open(opt['log_dir'] + "sub_newton_loss.txt", "w")
    param_file = open(opt['log_dir'] + "sub_newton_param.txt", "w")

    r = opt['sub_newton_r']  # rank Q matrix
    s = opt['sub_newton_s']  # number of samples to compute approximate Hessian
    gamma = log(d) / s

    n_samples_per_step = bs + s
    n_steps = int((n_passes * n) / n_samples_per_step)
    n_samples = 0  # number of samples processed so far

    # Store past gradients in a table
    mem_gradients = {}
    nGradients = 0  # no gradient stored in mem_gradients at initialization
    avg_mg = np.zeros(d)

    # Fill in table
    a = 1.0 / n
    for i in range(n):
        grad = stochastic_gradient(X, Y, w, np.array([i]), opt)
        mem_gradients[i] = grad
        # avg_mg = avg_mg + (grad*a)
        avg_mg = avg_mg + grad
    avg_mg = avg_mg / n
    nGradients = n

    for i in range(n_steps):

        list_idx = np.random.randint(0, high=n, size=1)
        idx = list_idx[0]
        grad = stochastic_gradient(X, Y, w, list_idx, opt)

        # Compute approximate Hessian
        H_s = np.zeros((d, d))
        # indices = np.random.randint(0, high=n, size=s)
        indices = np.random.choice(range(n), s, replace=False)
        for j in range(s):
            H_s = H_s + stochastic_hessian(X, Y, w, indices[j], opt)
        H_s = H_s / s ##do we divide by n in the deterministic case? I don't think so..any difference?

        H = hessian(X, Y, w, opt)
        # TODO, check H-H_s, using H_s seems to yield bad results
        print('|H-H_s| = ', np.linalg.norm(H - H_s))

        n_samples += n_samples_per_step

        # Rank (r+1) truncated SVD
        U, Sigma, VT = randomized_svd(H_s, n_components=r + 1,
                                      n_iter=10,
                                      random_state=None)

        Ur = U[:, range(r)]
        Sigma_r = Sigma[range(r)]
        invSigma_r = np.linalg.inv(np.diag(Sigma_r))

        # # Constuct approximate Hessian matrix
        # # we fill its 0 eigenvalues with the (r +1)-th eigenvalue
        Q = np.eye(d, d) / Sigma[r] + np.dot(Ur, np.dot(invSigma_r - np.eye(r, r) / Sigma[r], np.transpose(
            Ur)))  # uncomment this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if idx in mem_gradients:
            dd = - np.dot(Q, grad - mem_gradients[idx] + avg_mg)  # descent direction
        else:
            dd = - np.dot(Q, grad)  # descent direction

        if i == 0 or n_samples >= opt['recording_step']:
            n_samples = 0
            _loss = loss(X, Y, w, opt)
            print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' eta = ' + str(eta) + ' norm_grad = ' + str(
                np.linalg.norm(grad)))
            print ('Iteration ' + str(i) + ': |H_s^-1 - Q| = ' + str(
                np.linalg.norm(np.linalg.inv(H_s) - Q)) + ' norm(Q) = ' + str(np.linalg.norm(Q)) + ' norm_dd = ' + str(
                np.linalg.norm(dd)))
            loss_file.write(str((i + 1) * n_samples_per_step) + '\t' + str(_loss) + '\n')
            w_str = ','.join(['%.5f' % num for num in w])
            param_file.write(w_str + '\n')

        # eta = backtracking_line_search(X,Y,w,grad,dd,opt)
        w = w + eta * dd

        # Update average gradient
        if idx in mem_gradients:
            delta_grad = grad - mem_gradients[idx]
            a = 1.0 / nGradients
            avg_mg = avg_mg + (delta_grad * a)
        else:
            # Gradient for datapoint idx does not exist yet
            nGradients = nGradients + 1  # increment number of gradients
            a = 1.0 / nGradients
            b = 1.0 - a
            avg_mg = (avg_mg * b) + (grad * a)

        # a = 1.0/n
        # avg_mg_2 = np.zeros(d)
        # for i in range(n):
        #    avg_mg_2 = avg_mg_2 + (mem_gradients[i]*a)
        # print('diff = ' + str(np.linalg.norm(avg_mg_2-avg_mg)))

        # Update memorized gradients
        mem_gradients[idx] = grad

    loss_file.close()
    param_file.close()

    return w
