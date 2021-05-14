from numba import cuda
from numpy.polynomial.hermite import hermgauss
from matplotlib import pyplot as plt
from numpy.linalg import inv
import pandas as pd
import numpy as np
import scipy
import time
from scipy.stats import norm

def Pi(x, beta_0, mu):
    return np.asarray((np.exp(x @ beta_0 + mu) / (1 + np.exp(x @ beta_0 + mu))))

def logsumexp(x):
    c = x.max()
    LSE = c + np.log(np.sum(np.exp(x - c)))
    return np.exp(x - LSE)

def g(x, y, mu, beta_0, tau=1):
    g = sum(y * logsumexp(Pi(x, beta_0, mu)) + (1 - y) * logsumexp(1 - Pi(x, beta_0, mu))) \
    + np.log((np.sqrt(2 * np.pi) * tau)**(-1) * np.exp(-mu**2/(2 * tau**2)))
    return g

def g_b(x, y, mu, beta_0, tau = 1):
    return np.sum(x * y - x * Pi(x, beta_0, mu), axis = 0)

def g_u(x, y, mu, beta_0, tau = 1):
    return sum(y - Pi(x, beta_0, mu)) - mu/tau**2

def g_uu(x, y, mu, beta_0, tau = 1):
    return sum(- np.asarray((np.exp(x @ beta_0 + mu) / (1 + np.exp(x @ beta_0 + mu))**2))) - 1/tau**2

def g_ub(x, y, mu, beta_0, tau = 1):
    return np.sum(- x * np.asarray((np.exp(x @ beta_0 + mu) / (1 + np.exp(x @ beta_0 + mu))**2)), axis = 0)

def g_bb(x, y, mu, beta_0, tau = 1):
    result = 0
    for i in range(len(y)):
        result += -np.asarray(x[i].reshape(x.shape[1],1) @ x[i].reshape(1,x.shape[1])\
        * (np.exp(x[i] @ beta_0 + mu) / (1 + np.exp(x[i] @ beta_0 + mu))**2))
    return result

def g_uuu(x, y, mu, beta_0, tau = 1):
    return sum(- np.asarray((np.exp(x @ beta_0 + mu) * (np.exp(x @ beta_0 + mu) - 1)\
                             / (1 + np.exp(x @ beta_0 + mu))**3)))

def g_uub(x, y, mu, beta_0, tau = 1):
    return np.sum(- x * np.asarray((np.exp(x @ beta_0 + mu) * (np.exp(x @ beta_0 + mu) - 1)\
                             / (1 + np.exp(x @ beta_0 + mu))**3)), axis = 0)

def g_ubb(x, y, mu, beta_0, tau = 1):
    result = 0
    for i in range(len(y)):
        result += -np.asarray(x[i].reshape(x.shape[1],1) @ x[i].reshape(1,x.shape[1])\
                         * (np.exp(x[i] @ beta_0 + mu) * (np.exp(x[i] @ beta_0 + mu) - 1)\
                                       / (1 + np.exp(x[i] @ beta_0 + mu))**3))
    return result

def g_uuuu(x, y, mu, beta_0, tau = 1):
    result = sum(- (np.exp(x @ beta_0 + mu) * (np.exp(x @ beta_0 + mu) - 1)\
                             / (1 + np.exp(x @ beta_0 + mu))**3)\
                 + (3 * np.exp(2 * (x @ beta_0 + mu)) * (np.exp(x @ beta_0 + mu) - 1)\
                             / (1 + np.exp(x @ beta_0 + mu))**4)\
                 - (np.exp(2 * (x @ beta_0 + mu))  / (1 + np.exp(x @ beta_0 + mu))**3)\
                )
    return result

def g_uuub(x, y, mu, beta_0, tau = 1):
    result = np.sum(- x * np.asarray((np.exp(x @ beta_0 + mu) * (np.exp(x @ beta_0 + mu) - 1)\
                             / (1 + np.exp(x @ beta_0 + mu))**3)\
                 + (3 * np.exp(2 * (x @ beta_0 + mu)) * (np.exp(x @ beta_0 + mu) - 1)\
                             / (1 + np.exp(x @ beta_0 + mu))**4)\
                 - (np.exp(2 * (x @ beta_0 + mu))  / (1 + np.exp(x @ beta_0 + mu))**3)), axis = 0)
    return result

def g_uubb(x, y, mu, beta_0, tau = 1):
    result = 0
    for i in range(len(y)):
        result += -np.asarray(x[i].reshape(x.shape[1],1) @ x[i].reshape(1,x.shape[1])\
                             * ((np.exp(x @ beta_0 + mu) * (np.exp(x @ beta_0 + mu) - 1)\
                             / (1 + np.exp(x @ beta_0 + mu))**3)\
                 + (3 * np.exp(2 * (x @ beta_0 + mu)) * (np.exp(x @ beta_0 + mu) - 1)\
                             / (1 + np.exp(x @ beta_0 + mu))**4)\
                 - (np.exp(2 * (x @ beta_0 + mu))  / (1 + np.exp(x @ beta_0 + mu))**3)))
        
    return result

def omega(x, y, mu, beta_0, tau = 1):
    return np.sqrt(-1/g_uu(x, y, mu, beta_0))

def omega_b(x, y, mu, beta_0, tau = 1):
    return 0.5 * omega(x, y, mu, beta_0, tau)**3 * (g_uuu(x, y, mu, beta_0, tau) * mu_b(x, y, mu, beta_0, tau)\
                                                    + g_uub(x, y, mu, beta_0, tau))

def mu_b(x, y, mu, beta_0, tau = 1):
    return omega(x, y, mu, beta_0, tau)**2 * g_ub(x, y, mu, beta_0, tau)

def mu_bb(x, y, mu, beta_0, tau = 1):
    result = omega(x, y, mu, beta_0, tau)**2 * (mu_b(x, y, mu, beta_0, tau).reshape(x.shape[1],1)\
                                              @ mu_b(x, y, mu, beta_0, tau).reshape(1,x.shape[1])\
                                              * g_uuu(x, y, mu, beta_0, tau)\
                                              + 2 * mu_b(x, y, mu, beta_0, tau).reshape(x.shape[1],1)\
                                              @ g_uub(x, y, mu, beta_0, tau).reshape(1,x.shape[1])\
                                              + g_ubb(x, y, mu, beta_0, tau))
    return result

def omega_bb(x, y, mu, beta_0, tau = 1):
    result = 3/4 * omega(x, y, mu, beta_0, tau)**5 * (mu_b(x, y, mu, beta_0, tau) * g_uuu(x, y, mu, beta_0, tau)\
                                                      + g_uub(x, y, mu, beta_0, tau)).reshape(x.shape[1],1)\
     @ (mu_b(x, y, mu, beta_0, tau) * g_uuu(x, y, mu, beta_0, tau)\
        + g_uub(x, y, mu, beta_0, tau)).reshape(1,x.shape[1]) + 1/2 * omega(x, y, mu, beta_0, tau)**3\
     * (mu_bb(x, y, mu, beta_0, tau) * g_uuu(x, y, mu, beta_0, tau)\
        + (mu_b(x, y, mu, beta_0, tau).reshape(x.shape[1],1)\
           @ mu_b(x, y, mu, beta_0, tau).reshape(1,x.shape[1]) * g_uuuu(x, y, mu, beta_0, tau))\
        + mu_b(x, y, mu, beta_0, tau).reshape(x.shape[1],1) @ g_uuub(x, y, mu, beta_0, tau).reshape(1,x.shape[1])\
        + mu_bb(x, y, mu, beta_0, tau)
       )
    return result

def f_k(k, x, y, mu, beta_0, tau = 1):
    [x_k, h_k] = hermgauss(k)
    return h_k * np.exp(g(x, y, mu + np.sqrt(2 * np.pi) * omega(x, y, mu, beta_0, tau) * x_k, beta_0) + x_k**2)

# Plugged in log-sum-exp trick
def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

def f_k(k, x, y, mu, beta_0, tau = 1):
    [x_k, h_k] = hermgauss(k)
#     h_k = np.exp(-x_k**2)
    inside = g(x, y, mu + np.sqrt(2 * np.pi) * omega(x, y, mu, beta_0, tau) * x_k, beta_0) + x_k**2
    result = np.exp(inside - logsumexp(inside))
    return result

def l(k, x, y, mu, beta_0, tau = 1):
    [x_k, h_k] = hermgauss(k)
    return 0.5 * np.log(2 * np.pi) + np.log(omega(x, y, mu, beta_0, tau)) + np.log(sum(f_k(k, x, y, mu, beta_0, tau)))

def f_k_b(k, x, y, mu, beta_0, tau = 1):
    [x_k, h_k] = hermgauss(k)
    result = 0
    for i in range(k):
        result += f_k(k, x, y, mu + np.sqrt(2 * np.pi) * omega(x, y, mu, beta_0, tau) * x_k, beta_0, tau)[i] * g_b(x, y, mu + np.sqrt(2 * np.pi)\
                                                      * omega(x, y, mu, beta_0, tau) * x_k[i], beta_0, tau)
    return result

def f_k_u(k, x, y, mu, beta_0, tau = 1):
    [x_k, h_k] = hermgauss(k)
    return f_k(k, x, y, mu + np.sqrt(2 * np.pi) * omega(x, y, mu, beta_0, tau) * x_k, beta_0, tau) * g_u(x, y, mu + np.sqrt(2 * np.pi) * omega(x, y, mu, beta_0, tau) * x_k, beta_0, tau)

def f_k_w(k, x, y, mu, beta_0, tau = 1):
    [x_k, h_k] = hermgauss(k)
    return f_k(k, x, y, mu + np.sqrt(2 * np.pi) * omega(x, y, mu, beta_0, tau) * x_k, beta_0, tau) * g_u(x, y, mu + np.sqrt(2 * np.pi) * omega(x, y, mu, beta_0, tau) * x_k, beta_0, tau) * np.sqrt(2 * np.pi) * x_k

def l_1(k, x, y, mu, beta_0, tau = 1):
    [x_k, h_k] = hermgauss(k)
    result = omega_b(x, y, mu, beta_0, tau) / omega(x, y, mu, beta_0, tau)
    a = 1 / sum(f_k(k, x, y, mu, beta_0, tau))
    b = 0
    for i in range(k):
        b += f_k_u(k, x, y, mu, beta_0, tau)[i] * mu_b(x, y, mu, beta_0, tau)\
         + f_k_w(k, x, y, mu, beta_0, tau)[i] * omega_b(x, y, mu, beta_0, tau)
    b = b + f_k_b(k, x, y, mu, beta_0, tau)
    result = result + a * b
    return result

def f_k_ub(k, x, y, mu, beta_0, tau = 1):
    [x_k, h_k] = hermgauss(k)
    a = 0
    for i in range(k):
        a += f_k_u(k, x, y, mu, beta_0, tau)[i] * g_ub(x, y, mu, beta_0, tau)
    b = 0
    for i in range(k):
        b += f_k_u(k, x, y, mu, beta_0, tau)[i] * mu_b(x, y, mu, beta_0, tau)\
         + f_k_w(k, x, y, mu, beta_0, tau)[i] * omega_b(x, y, mu, beta_0, tau)
    b = b + f_k_b(k, x, y, mu, beta_0, tau)
    result = b * g_u(x, y, mu, beta_0, tau) + a
    return result

def f_k_wb(k, x, y, mu, beta_0, tau = 1):
    [x_k, h_k] = hermgauss(k)
    a = 0
    for i in range(k):
        a += f_k_u(k, x, y, mu, beta_0, tau)[i] * g_ub(x, y, mu, beta_0, tau) * x_k[i]
    b = 0
    for i in range(k):
        b += (f_k_u(k, x, y, mu, beta_0, tau)[i] * mu_b(x, y, mu, beta_0, tau)\
         + f_k_w(k, x, y, mu, beta_0, tau)[i] * omega_b(x, y, mu, beta_0, tau)\
         + f_k(k, x, y, mu + np.sqrt(2 * np.pi) * omega(x, y, mu, beta_0, tau)\
               * x_k, beta_0, tau)[i] * g_b(x, y, mu + np.sqrt(2 * np.pi)\
                                                      * omega(x, y, mu, beta_0, tau) * x_k[i], beta_0, tau)) * x_k[i]
    result = np.sqrt(2 * np.pi) * (a + b)
    return result

def f_k_bb(k, x, y, mu, beta_0, tau = 1):
    result = f_k_b(k, x, y, mu, beta_0, tau).reshape(len(beta_0),1) @ g_b(x, y, mu, beta_0, tau).reshape(1,len(beta_0))\
    + sum(f_k(k, x, y, mu, beta_0, tau)) * g_bb(x, y, mu, beta_0, tau)
    return result

def part_1(k, x, y, mu, beta_0, tau = 1):
    result = f_k_ub(k, x, y, mu, beta_0, tau).reshape(len(beta_0),1)\
    @ mu_b(x, y, mu, beta_0, tau).reshape(1,len(beta_0))\
    + sum(f_k_u(k, x, y, mu, beta_0, tau)) * mu_bb(x, y, mu, beta_0, tau)\
    + f_k_wb(k, x, y, mu, beta_0, tau).reshape(len(beta_0),1)\
    @ omega_b(x, y, mu, beta_0, tau).reshape(1,len(beta_0))\
    + sum(f_k_w(k, x, y, mu, beta_0, tau)) * omega_bb(x, y, mu, beta_0, tau)\
    + f_k_bb(k, x, y, mu, beta_0, tau)
    return result

def l_2(k, x, y, mu, beta_0, tau = 1):
    a = 1 / sum(f_k(k, x, y, mu, beta_0, tau))
    b = 0
    for i in range(k):
        b += f_k_u(k, x, y, mu, beta_0, tau)[i] * mu_b(x, y, mu, beta_0, tau)\
         + f_k_w(k, x, y, mu, beta_0, tau)[i] * omega_b(x, y, mu, beta_0, tau)
    b = b + f_k_b(k, x, y, mu, beta_0, tau)
    part_2 = a * b
    l2 = omega(x, y, mu, beta_0, tau)**(-2) * (omega_bb(x, y, mu, beta_0, tau) * omega(x, y, mu, beta_0, tau)\
                                               - omega_b(x, y, mu, beta_0, tau).reshape(x.shape[1],1)\
                                               @ omega_b(x, y, mu, beta_0, tau).reshape(1,x.shape[1]))\
     + sum(f_k(k, x, y, mu, beta_0, tau))**(-1) * part_1(k, x, y, mu, beta_0, tau) - part_2.reshape(x.shape[1],1)\
     @ part_2.reshape(1,x.shape[1])
    return l2

def max_mu(x, y, mu, beta_0, tau=1, max_iter=100):
    for step in range(max_iter):
#         print('Step: ', step, '\n')
        mu_new = mu - g_u(x, y, mu, beta_0, tau)/g_uu(x, y, mu, beta_0, tau)
        diff = mu_new - mu
#         print(diff)
        if np.abs(diff) < 10**(-10):
#             print(mu)
            break;
        mu = mu_new
    return mu

def mapping(x):
    return (x - min(x))/(max(x) - min(x))

def loss(x, y, beta, mu):
    return sum((mapping(Pi(x, beta, mu)) - y)**2)

# Definitions for GH
def GH(k, X, y):
    # Added regularization
    pre_los = 10**10
    for lam in np.arange(0, 11, 1):
        try:
    
            # Regression with regularization
            beta = np.repeat(0, 10).reshape(10, 1)
            mu = np.repeat(0.1, len(y))
            tau = 1

            start_time = time.time()
            # print('Initial beta:', beta, "\n")
            for step_mu in range(3):
                for i in range(len(mu)):
                    mu[i] = max_mu(X[i], y[i], mu[i], beta, tau)
                for step in range(100):
                    l1 = 0
                    l2 = 0
                    for i in range(len(mu)):
                        l1 += l_1(k, X[i], y[i], mu[i], beta, tau)
                        l2 += l_2(k, X[i], y[i], mu[i], beta, tau)
                    l1 -= (2 * lam * beta.transpose())[0]
                    l2 -= np.diag(np.repeat(2 * lam, 10))
                    delta = l1 @ inv(l2)
                    new_beta = beta - delta.reshape(10, 1)
                    if max(np.abs(delta)) < 10 ** (-10):
                        break;
                    beta = new_beta
                    if True in np.isnan(beta):
                        break;
                    # print('Step ', step + 1, ':\n')
                    # print('Beta:\n', beta, '\n')
                    # print('Diff:\n', delta, '\n')
                if True in np.isnan(beta):
                    break;
            # print('Beta:\n', beta, '\n')
            # print("--- %s seconds ---" % (time.time() - start_time))
            los = 0
            for i in range(len(X)):
                los += loss(X[i], y[i], beta, mu[i])
            if los < pre_los:
                optimized_beta = beta
                optimized_mu = mu
                optimized_lam = lam
                # reset pre_los
                pre_los = los
        except:
            continue;
    return [optimized_beta, optimized_mu]

def output(X, beta, true_beta):
    X = np.concatenate(X)

    V = np.diagflat(Pi(X, beta, 0) * (1 - Pi(X, beta, 0)))

    SE = np.sqrt(np.diag(inv(np.transpose(X) @ V @ X))).reshape(10,1)

    Z = beta/SE

    P = 2 * norm.cdf(-1 * np.abs(Z))

    CI_025  = beta - 1.959964 * SE
    CI_975  = beta + 1.959964 * SE

    df = pd.DataFrame({'Truth': np.transpose(true_beta)[0], 'Coef': np.transpose(beta)[0], 'Std.Err': np.transpose(SE)[0],
                       'z': np.transpose(Z)[0], 'P-value': np.transpose(P)[0],
                       '[0.025': np.transpose(CI_025)[0], '0.975]': np.transpose(CI_975)[0]},
                      index = var_name)
    return df


##################### Simulation with Penn ############################
import os
truth = np.array([-1.5,0.1,-0.5,-0.3,0.4,-0.2,-0.25,0.35,-0.1,0.5]).reshape(10, 1)
var_name = []
for i in range(10):
    var_name += ['X' + str(i+1)]
k = 2
import sys
# sys.stdout = open('../Simulation_data_GLMM/Result_GH/test.log', 'w')

############### Test ###################

# Setting 1
print('======================\n Here starts Setting 1 \n======================\n\n\n')

file_dir = '../Simulation_data_GLMM/Setting_1/'
file_names = os.listdir(file_dir)
for i in range(len(file_names)):
    try:
        start_time = time.time()
        df = pd.read_csv(file_dir + file_names[i], index_col=0)
        # Load data
        data1 = np.array(df[df['Site_ID'] == 1][var_name])
        data2 = np.array(df[df['Site_ID'] == 2][var_name])
        data = [data1, data2]
        out1 = np.array(df[df['Site_ID'] == 1]['y']).reshape(500,1)
        out2 = np.array(df[df['Site_ID'] == 2]['y']).reshape(500,1)
        out = [out1, out2]
        # Simulations
        [beta, mu] = GH(k, data, out)
        output(data, beta, truth).to_csv('../Simulation_data_GLMM/Result_GH/Setting_1_' +
                                              file_names[i][44:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;

# Setting 2
print('======================\n Here starts Setting 2 \n======================\n\n\n')

file_dir = '../Simulation_data_GLMM/Setting_2/'
file_names = os.listdir(file_dir)
for i in range(len(file_names)):
    try:
        start_time = time.time()
        df = pd.read_csv(file_dir + file_names[i], index_col=0)
        # Load data
        data1 = np.array(df[df['Site_ID'] == 1][var_name])
        data2 = np.array(df[df['Site_ID'] == 2][var_name])
        data = [data1, data2]
        out1 = np.array(df[df['Site_ID'] == 1]['y']).reshape(500,1)
        out2 = np.array(df[df['Site_ID'] == 2]['y']).reshape(500,1)
        out = [out1, out2]
        # Simulations
        [beta, mu] = GH(k, data, out)
        output(data, beta, truth).to_csv('../Simulation_data_GLMM/Result_GH/Setting_2_' +
                                              file_names[i][44:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;

# Setting 3
print('======================\n Here starts Setting 3 \n======================\n\n\n')

file_dir = '../Simulation_data_GLMM/Setting_3/'
file_names = os.listdir(file_dir)
for i in range(len(file_names)):
    try:
        start_time = time.time()
        df = pd.read_csv(file_dir + file_names[i], index_col=0)
#         Load data
        data = []
        for k in range(10):
            globals()['data%s' % str(k+1)] = np.array(df[df['Site_ID'] == k+1][var_name])
            data.append(globals()['data%s' % str(k+1)])
        out = []
        for k in range(10):
            globals()['out%s' % str(k+1)] = np.array(df[df['Site_ID'] == k+1]['y']).reshape(500,1)
            out.append(globals()['out%s' % str(k+1)])
        # Simulations
        [beta, mu] = GH(k, data, out)
        output(data, beta, truth).to_csv('../Simulation_data_GLMM/Result_GH/Setting_3_' +
                                              file_names[i][45:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;

# Setting 4
print('======================\n Here starts Setting 4 \n======================\n\n\n')

file_dir = '../Simulation_data_GLMM/Setting_4/'
file_names = os.listdir(file_dir)
for i in range(len(file_names)):
    try:
        start_time = time.time()
        df = pd.read_csv(file_dir + file_names[i], index_col=0)
#         Load data
        data = []
        for k in range(10):
            globals()['data%s' % str(k+1)] = np.array(df[df['Site_ID'] == k+1][var_name])
            data.append(globals()['data%s' % str(k+1)])
        out = []
        for k in range(10):
            globals()['out%s' % str(k+1)] = np.array(df[df['Site_ID'] == k+1]['y']).reshape(500,1)
            out.append(globals()['out%s' % str(k+1)])
        # Simulations
        [beta, mu] = GH(k, data, out)
        output(data, beta, truth).to_csv('../Simulation_data_GLMM/Result_GH/Setting_4_' +
                                              file_names[i][45:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;

# Setting 5
print('======================\n Here starts Setting 5 \n======================\n\n\n')

file_dir = '../Simulation_data_GLMM/Setting_5/'
file_names = os.listdir(file_dir)
for i in range(len(file_names)):
    try:
        start_time = time.time()
        df = pd.read_csv(file_dir + file_names[i], index_col=0)
        # Load data
        data1 = np.array(df[df['Site_ID'] == 1][var_name])
        data2 = np.array(df[df['Site_ID'] == 2][var_name])
        data = [data1, data2]
        out1 = np.array(df[df['Site_ID'] == 1]['y']).reshape(30,1)
        out2 = np.array(df[df['Site_ID'] == 2]['y']).reshape(30,1)
        out = [out1, out2]
        # Simulations
        [beta, mu] = GH(k, data, out)
        output(data, beta, truth).to_csv('../Simulation_data_GLMM/Result_GH/Setting_5_' +
                                              file_names[i][43:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;

# Setting 6
print('======================\n Here starts Setting 6 \n======================\n\n\n')

file_dir = '../Simulation_data_GLMM/Setting_6/'
file_names = os.listdir(file_dir)
for i in range(len(file_names)):
    try:
        start_time = time.time()
        df = pd.read_csv(file_dir + file_names[i], index_col=0)
        # Load data
        data1 = np.array(df[df['Site_ID'] == 1][var_name])
        data2 = np.array(df[df['Site_ID'] == 2][var_name])
        data = [data1, data2]
        out1 = np.array(df[df['Site_ID'] == 1]['y']).reshape(30,1)
        out2 = np.array(df[df['Site_ID'] == 2]['y']).reshape(30,1)
        out = [out1, out2]
        # Simulations
        [beta, mu] = GH(k, data, out)
        output(data, beta, truth).to_csv('../Simulation_data_GLMM/Result_GH/Setting_6_' +
                                              file_names[i][43:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;

# Setting 7
print('======================\n Here starts Setting 7 \n======================\n\n\n')

file_dir = '../Simulation_data_GLMM/Setting_7/'
file_names = os.listdir(file_dir)
for i in range(len(file_names)):
    try:
        start_time = time.time()
        df = pd.read_csv(file_dir + file_names[i], index_col=0)
#         Load data
        data = []
        for k in range(10):
            globals()['data%s' % str(k+1)] = np.array(df[df['Site_ID'] == k+1][var_name])
            data.append(globals()['data%s' % str(k+1)])
        out = []
        for k in range(10):
            globals()['out%s' % str(k+1)] = np.array(df[df['Site_ID'] == k+1]['y']).reshape(30,1)
            out.append(globals()['out%s' % str(k+1)])
        # Simulations
        [beta, mu] = GH(k, data, out)
        output(data, beta, truth).to_csv('../Simulation_data_GLMM/Result_GH/Setting_7_' +
                                              file_names[i][44:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;

# Setting 8
print('======================\n Here starts Setting 8 \n======================\n\n\n')

file_dir = '../Simulation_data_GLMM/Setting_8/'
file_names = os.listdir(file_dir)
for i in range(len(file_names)):
    try:
        start_time = time.time()
        df = pd.read_csv(file_dir + file_names[i], index_col=0)
#         Load data
        data = []
        for k in range(10):
            globals()['data%s' % str(k+1)] = np.array(df[df['Site_ID'] == k+1][var_name])
            data.append(globals()['data%s' % str(k+1)])
        out = []
        for k in range(10):
            globals()['out%s' % str(k+1)] = np.array(df[df['Site_ID'] == k+1]['y']).reshape(30,1)
            out.append(globals()['out%s' % str(k+1)])
        # Simulations
        [beta, mu] = GH(k, data, out)
        output(data, beta, truth).to_csv('../Simulation_data_GLMM/Result_GH/Setting_8_' +
                                              file_names[i][44:], header = True)
        print('======\n File:\n', file_names[i], '\n Completed!\n')
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        pass;

# sys.stdout.close()



