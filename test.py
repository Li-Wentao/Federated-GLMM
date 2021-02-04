# Loading required packages
from numpy.polynomial.hermite import hermgauss
from numpy.linalg import inv
import pandas as pd
import numpy as np
import scipy
import time
from scipy.stats import norm

def Pi(x, beta_0, mu):
    result = np.asarray((np.exp(x @ beta_0 + mu) / (1 + np.exp(x @ beta_0 + mu))))
    if np.exp(x @ beta_0 + mu).max() == np.inf:
        return np.nan_to_num(result, nan=1)
    else:
        return result

def g(x, y, mu, beta_0, tau=1):
    g = sum(y * np.log(Pi(x, beta_0, mu)) + (1 - y) * np.log(1 - Pi(x, beta_0, mu))) \
        + np.log((np.sqrt(2 * np.pi) * tau) ** (-1) * np.exp(-mu ** 2 / (2 * tau ** 2)))
    return g

def g_b(x, y, mu, beta_0, tau=1):
    return np.sum(x * y - x * Pi(x, beta_0, mu), axis=0)

def g_u(x, y, mu, beta_0, tau=1):
    return sum(y - Pi(x, beta_0, mu)) - mu / tau ** 2

def g_uu(x, y, mu, beta_0, tau=1):
    result = np.nan_to_num(- np.asarray((np.exp(x @ beta_0 + mu) / (1 + np.exp(x @ beta_0 + mu)) ** 2)), nan=0)
    return sum(result) - 1 / tau ** 2

def g_ub(x, y, mu, beta_0, tau=1):
    result = np.nan_to_num(np.asarray((np.exp(x @ beta_0 + mu) / (1 + np.exp(x @ beta_0 + mu)) ** 2)), nan=0)
    return np.sum(- x * result, axis=0)

def g_bb(x, y, mu, beta_0, tau=1):
    result = 0
    for i in range(len(y)):
        result += -np.asarray(x[i].reshape(x.shape[1], 1) @ x[i].reshape(1, x.shape[1]) \
                              * np.nan_to_num((np.exp(x[i] @ beta_0 + mu) / (1 + np.exp(x[i] @ beta_0 + mu)) ** 2),
                                              nan=0))
    return result

def g_uuu(x, y, mu, beta_0, tau=1):
    return sum(- np.nan_to_num(np.asarray((np.exp(x @ beta_0 + mu) * (np.exp(x @ beta_0 + mu) - 1) \
                                           / (1 + np.exp(x @ beta_0 + mu)) ** 3)), nan=0))

def g_uub(x, y, mu, beta_0, tau=1):
    return np.sum(- x * np.nan_to_num(np.asarray((np.exp(x @ beta_0 + mu) * (np.exp(x @ beta_0 + mu) - 1) \
                                                  / (1 + np.exp(x @ beta_0 + mu)) ** 3)), nan=0), axis=0)

def g_ubb(x, y, mu, beta_0, tau=1):
    result = 0
    for i in range(len(y)):
        result += -np.asarray(x[i].reshape(x.shape[1], 1) @ x[i].reshape(1, x.shape[1]) \
                              * np.nan_to_num((np.exp(x[i] @ beta_0 + mu) * (np.exp(x[i] @ beta_0 + mu) - 1) \
                                               / (1 + np.exp(x[i] @ beta_0 + mu)) ** 3), nan=0))
    return result

def g_uuuu(x, y, mu, beta_0, tau=1):
    result = sum(- np.nan_to_num((np.exp(x @ beta_0 + mu) * (np.exp(x @ beta_0 + mu) - 1) \
                                  / (1 + np.exp(x @ beta_0 + mu)) ** 3), nan=0) \
                 + np.nan_to_num((3 * np.exp(2 * (x @ beta_0 + mu)) * (np.exp(x @ beta_0 + mu) - 1) \
                                  / (1 + np.exp(x @ beta_0 + mu)) ** 4), nan=0) \
                 - np.nan_to_num((np.exp(2 * (x @ beta_0 + mu)) / (1 + np.exp(x @ beta_0 + mu)) ** 3), nan=0) \
                 )
    return result

def g_uuub(x, y, mu, beta_0, tau=1):
    result = np.sum(- x * np.asarray(np.nan_to_num((np.exp(x @ beta_0 + mu) * (np.exp(x @ beta_0 + mu) - 1) \
                                                    / (1 + np.exp(x @ beta_0 + mu)) ** 3), nan=0) \
                                     + np.nan_to_num(
        (3 * np.exp(2 * (x @ beta_0 + mu)) * (np.exp(x @ beta_0 + mu) - 1) \
         / (1 + np.exp(x @ beta_0 + mu)) ** 4), nan=0) \
                                     - np.nan_to_num(
        (np.exp(2 * (x @ beta_0 + mu)) / (1 + np.exp(x @ beta_0 + mu)) ** 3), nan=0))
                    , axis=0)
    return result

def g_uubb(x, y, mu, beta_0, tau=1):
    result = 0
    for i in range(len(y)):
        result += -np.asarray(x[i].reshape(x.shape[1], 1) @ x[i].reshape(1, x.shape[1]) \
                              * (np.nan_to_num((np.exp(x @ beta_0 + mu) * (np.exp(x @ beta_0 + mu) - 1) \
                                                / (1 + np.exp(x @ beta_0 + mu)) ** 3), nan=0) \
                                 + np.nan_to_num((3 * np.exp(2 * (x @ beta_0 + mu)) * (np.exp(x @ beta_0 + mu) - 1) \
                                                  / (1 + np.exp(x @ beta_0 + mu)) ** 4), nan=0) \
                                 - np.nan_to_num(
                    (np.exp(2 * (x @ beta_0 + mu)) / (1 + np.exp(x @ beta_0 + mu)) ** 3), nan=0)))

    return result

def omega(x, y, mu, beta_0, tau=1):
    return np.sqrt(-1 / g_uu(x, y, mu, beta_0))

def omega_b(x, y, mu, beta_0, tau=1):
    return 0.5 * omega(x, y, mu, beta_0, tau) ** 3 * (g_uuu(x, y, mu, beta_0, tau) * mu_b(x, y, mu, beta_0, tau) \
                                                      + g_uub(x, y, mu, beta_0, tau))

def mu_b(x, y, mu, beta_0, tau=1):
    return omega(x, y, mu, beta_0, tau) ** 2 * g_ub(x, y, mu, beta_0, tau)

def mu_bb(x, y, mu, beta_0, tau=1):
    result = omega(x, y, mu, beta_0, tau) ** 2 * (mu_b(x, y, mu, beta_0, tau).reshape(x.shape[1], 1) \
                                                  @ mu_b(x, y, mu, beta_0, tau).reshape(1, x.shape[1]) \
                                                  * g_uuu(x, y, mu, beta_0, tau) \
                                                  + 2 * mu_b(x, y, mu, beta_0, tau).reshape(x.shape[1], 1) \
                                                  @ g_uub(x, y, mu, beta_0, tau).reshape(1, x.shape[1]) \
                                                  + g_ubb(x, y, mu, beta_0, tau))
    return result

def omega_bb(x, y, mu, beta_0, tau=1):
    result = 3 / 4 * omega(x, y, mu, beta_0, tau) ** 5 * (mu_b(x, y, mu, beta_0, tau) * g_uuu(x, y, mu, beta_0, tau) \
                                                          + g_uub(x, y, mu, beta_0, tau)).reshape(x.shape[1], 1) \
             @ (mu_b(x, y, mu, beta_0, tau) * g_uuu(x, y, mu, beta_0, tau) \
                + g_uub(x, y, mu, beta_0, tau)).reshape(1, x.shape[1]) + 1 / 2 * omega(x, y, mu, beta_0, tau) ** 3 \
             * (mu_bb(x, y, mu, beta_0, tau) * g_uuu(x, y, mu, beta_0, tau) \
                + (mu_b(x, y, mu, beta_0, tau).reshape(x.shape[1], 1) \
                   @ mu_b(x, y, mu, beta_0, tau).reshape(1, x.shape[1]) * g_uuuu(x, y, mu, beta_0, tau)) \
                + mu_b(x, y, mu, beta_0, tau).reshape(x.shape[1], 1) @ g_uuub(x, y, mu, beta_0, tau).reshape(1,
                                                                                                             x.shape[
                                                                                                                 1]) \
                + mu_bb(x, y, mu, beta_0, tau)
                )
    return result

def l_1(x, y, mu, beta_0, tau=1):
    l1 = omega_b(x, y, mu, beta_0, tau) / omega(x, y, mu, beta_0, tau) \
         + g_u(x, y, mu, beta_0, tau) * mu_b(x, y, mu, beta_0, tau) + g_b(x, y, mu, beta_0, tau)
    return l1

def l_2(x, y, mu, beta_0, tau=1):
    l2 = omega(x, y, mu, beta_0, tau) ** (-2) * (omega_bb(x, y, mu, beta_0, tau) * omega(x, y, mu, beta_0, tau) \
                                                 - omega_b(x, y, mu, beta_0, tau).reshape(x.shape[1], 1) \
                                                 @ omega_b(x, y, mu, beta_0, tau).reshape(1, x.shape[1])) \
         + mu_bb(x, y, mu, beta_0, tau) * g_u(x, y, mu, beta_0, tau) + mu_b(x, y, mu, beta_0, tau).reshape(
        x.shape[1], 1) \
         @ (mu_b(x, y, mu, beta_0, tau) * g_uu(x, y, mu, beta_0, tau) + g_ub(x, y, mu, beta_0, tau)).reshape(1,
                                                                                                             x.shape[
                                                                                                                 1]) \
         + mu_b(x, y, mu, beta_0, tau).reshape(x.shape[1], 1) @ g_ub(x, y, mu, beta_0, tau).reshape(1, x.shape[1]) \
         + g_bb(x, y, mu, beta_0, tau)
    return l2

def max_mu(x, y, mu, beta_0, tau=1, max_iter=100):
    for step in range(max_iter):
        #         print('Step: ', step, '\n')
        mu_new = mu - g_u(x, y, mu, beta_0, tau) / g_uu(x, y, mu, beta_0, tau)
        diff = mu_new - mu
        #         print(diff)
        if np.abs(diff) < 10 ** (-10):
            #             print(mu)
            break;
        mu = mu_new
    return mu

# Definitions for LA
def LA(X, y):

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
                l1 += l_1(X[i], y[i], mu[i], beta, tau)
                l2 += l_2(X[i], y[i], mu[i], beta, tau)
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
    return [beta, mu]

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

# Setting 1
print('======================\n Here starts Setting 1 \n======================\n\n\n')

file_dir = '../Simulation_data_GLMM/Setting_1/'
file_names = os.listdir(file_dir)
for i in range(len(file_names)):
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
    [beta, mu] = LA(data, out)
    output(data, beta, truth).to_csv('../Simulation_data_GLMM/Result/Setting_1_' +
                                          file_names[i][44:], header = True)
    print('======\n File:\n', file_names[i], '\n Completed!\n')
    print("--- %s seconds ---" % (time.time() - start_time))