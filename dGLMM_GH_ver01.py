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

# def g(x, y, mu, beta_0, tau=1):
#     g = sum(y * np.log(Pi(x, beta_0, mu)) + (1 - y) * np.log(1 - Pi(x, beta_0, mu))) \
#         + np.log((np.sqrt(2 * np.pi) * tau) ** (-1) * np.exp(-mu ** 2 / (2 * tau ** 2)))
#     return g

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

class GH:
    def __init__(self, k, X, y):
        """
        A calss to run distributed GLMM
        
        ...
        
        Attributes
        ------------
        k : int
        	The degree of Hermite polynomial
        X : A list of DataFrames
            The data from different sites
        y : A list of arrays
            The binary outcomes from different sites
        beta : An array of params
            Fixed effects coefficents
        lam : Float
            Regularization term
        mu : Float
            The mixed effects coefficients
        
        """
        self.p = X[0].shape[1]
        var_name = []
        for i in range(self.p):
            var_name += ['X' + str(i+1)]
        self.var_name = var_name
        if isinstance(X[0], pd.DataFrame):
            self.var_name = X[0].columns
            self.X = [np.array(data) for data in X]
#         self.y = [np.array(outcome).reshape(len(outcome),1) for outcome in y]
        self.k = k
        self.X = X
        self.y = y
        self.beta = np.repeat(0.1, self.p).reshape(self.p, 1)
        self.lam = np.nan
        self.mu = np.repeat(0.1, len(y))
        self.tau = 1
        self.df = pd.DataFrame

    def fit(self):
        # Added regularization
        pre_score = -10**10
        start_time = time.time()
        for self.lam in np.arange(0, 5, 1):
            # print('Initial self.beta:', self.beta, "\n")
            for step_mu in range(1):
                for i in range(len(self.mu)):                    
                    self.mu[i] = max_mu(self.X[i], self.y[i], self.mu[i], self.beta, self.tau)
                for step in range(100):
#                     score = 0
                    l1 = 0
                    l2 = 0
                    for i in range(len(self.mu)):
#                         score += l(self.X[i], self.y[i], self.mu[i], self.beta, self.tau) - sum(self.lam * (self.beta) **2)
                        l1 += l_1(self.k, self.X[i], self.y[i], self.mu[i], self.beta, self.tau)
                        l2 += l_2(self.k, self.X[i], self.y[i], self.mu[i], self.beta, self.tau)
                    l1 -= (2 * self.lam * self.beta.transpose())[0]
                    l2 -= np.diag(np.repeat(2 * self.lam, self.p))
                    delta = l1 @ inv(l2)
                    new_beta = self.beta - delta.reshape(self.p, 1)
                    if max(np.abs(delta)) < 10 ** (-6):
                        break;
                    if max(np.abs(delta)) > 10 ** (3):
                        break;
                    self.beta = new_beta
                    if True in np.isnan(self.beta):
                        break;
#                     print('Step ', step + 1, ':\n')
#     #                     print('Beta:\n', self.beta, '\n')
#                     print('Diff:\n', delta, '\n')
#                     print('Lam:\n', self.lam, '\n')
#                     print('Score:\n',score,'\n')

            score = 0
            for i in range(len(self.X)):
                score += l(self.k, self.X[i], self.y[i], self.mu[i], self.beta, self.tau) - sum(self.lam * (self.beta) **2)
#             print('Score:\n',score,'\n')
            if score > pre_score:
                optimized_beta = self.beta
                optimized_mu = self.mu
                optimized_lam = self.lam
                # reset
                pre_score = score
                optimized_score = score
    #     except:
    #         continue;
        
        # Returning data
        self.beta = optimized_beta
        self.mu = optimized_mu
        self.lam = optimized_lam
        self.score = optimized_score
        
#         return [optimized_beta, optimized_mu]
#     def output(self):
        #Def
        X = self.X
        beta = self.beta
        var_name = self.var_name
        
        p = X[0].shape[1]

        X = np.concatenate(X)

        V = np.diagflat(Pi(X, beta, 0) * (1 - Pi(X, beta, 0)))

        SE = np.sqrt(np.diag(inv(np.transpose(X) @ V @ X))).reshape(p,1)

        Z = beta/SE

        P = 2 * norm.cdf(-1 * np.abs(Z))

        CI_025  = beta - 1.959964 * SE
        CI_975  = beta + 1.959964 * SE

        self.df = pd.DataFrame({'Coef': np.transpose(beta)[0], 'Std.Err': np.transpose(SE)[0],
                           'z': np.transpose(Z)[0], 'P-value': np.transpose(P)[0],
                           '[0.025': np.transpose(CI_025)[0], '0.975]': np.transpose(CI_975)[0]},
                          index = var_name)
        return self