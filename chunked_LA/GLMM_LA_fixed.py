from numpy.linalg import inv
import pandas as pd
import numpy as np
import time
from scipy.stats import norm
from scipy.linalg import block_diag
def logsumexp(x):
    c = max(x)
    LSE = c + np.log(np.sum(np.exp(x - c)))
    return np.exp(x - LSE)
def Pi(x, y, mu, beta, tau):
    result = np.asarray((np.exp(x @ beta + tau * mu) / (1 + np.exp(x @ beta + tau * mu))))
    # Check if there are exp(0) cases, if true, return \pi = 1 correspondingly
    return np.nan_to_num(result, nan = 1)
def g(x, y, mu, beta, tau):
    g = sum(y * logsumexp(Pi(x, y, mu, beta, tau)) + (1 - y) * logsumexp(1 - Pi(x, y, mu, beta, tau))) \
    + np.nan_to_num(np.log(tau * (np.sqrt(2 * np.pi))**(-1) * np.exp(-mu**2/2)), nan=0)
    return g
def g_u(x, y, mu, beta, tau):
    return tau * sum((y - Pi(x, y, mu, beta, tau))) - mu
def g_b(x, y, mu, beta, tau):
    return np.sum(x * y - x * Pi(x, y, mu, beta, tau), axis = 0)
def g_t(x, y, mu, beta, tau):
    return mu * sum((y - Pi(x, y, mu, beta, tau)))
def g_uu(x, y, mu, beta, tau):
    result = np.nan_to_num(tau * np.asarray((np.exp(x @ beta + tau * mu) /\
                                              (1 + np.exp(x @ beta + tau * mu))**2)), nan = 0)
    return - tau * sum(result) - 1
def g_bb(x, y, mu, beta, tau):
    result = 0
    for i in range(len(y)):
        result += -np.asarray(x[i].reshape(x.shape[1],1) @ x[i].reshape(1,x.shape[1])\
        * np.nan_to_num((np.exp(x[i] @ beta + tau * mu) / (1 + np.exp(x[i] @ beta + tau * tau * mu))**2), nan = 0))
    return result
def g_tt(x, y, mu, beta, tau):
    result = np.nan_to_num(mu * np.asarray((np.exp(x @ beta + tau * mu) /\
                                              (1 + np.exp(x @ beta + tau * mu))**2)), nan = 0)
    return -mu * sum(result)
def g_ub(x, y, mu, beta, tau):
    result = np.nan_to_num(np.asarray((np.exp(x @ beta + tau * mu) /\
                                       (1 + np.exp(x @ beta + tau * mu))**2)), nan = 0)
    return -tau * np.sum(x * result, axis = 0)
def g_ut(x, y, mu, beta, tau):
    result = sum(y - Pi(x, y, mu, beta, tau))\
    - tau * sum(mu * np.nan_to_num(mu * np.asarray((np.exp(x @ beta + tau * mu) /\
                                          (1 + np.exp(x @ beta + tau * mu))**2)), nan = 0))
    return result
def g_bt(x, y, mu, beta, tau):
    result = np.nan_to_num(mu * np.asarray((np.exp(x @ beta + tau * mu) /\
                                          (1 + np.exp(x @ beta + tau * mu))**2)), nan = 0)
    return -np.sum(x * result, axis = 0)
def g_uuu(x, y, mu, beta, tau):
    return -tau * sum(-tau**2 * np.nan_to_num(np.asarray((np.exp(x @ beta + tau * mu) * (np.exp(x @ beta + tau * mu) - 1)\
                             / (1 + np.exp(x @ beta + tau * mu))**3)), nan = 0))
def g_uub(x, y, mu, beta, tau): # add -
    return -tau * np.sum(-tau * x * np.nan_to_num(np.asarray((np.exp(x @ beta + tau * mu) * (np.exp(x @ beta + tau * mu) - 1)\
                             / (1 + np.exp(x @ beta + tau * mu))**3)), nan = 0), axis = 0)
def g_ubb(x, y, mu, beta, tau):
    result = 0
    for i in range(len(y)):
        result += -np.asarray(x[i].reshape(x.shape[1],1) @ x[i].reshape(1,x.shape[1])\
                         * np.nan_to_num((np.exp(x[i] @ beta + tau * mu) * (np.exp(x[i] @ beta + tau * mu) - 1)\
                                       / (1 + np.exp(x[i] @ beta + tau * mu))**3), nan = 0))
    return -tau * result
def g_uut(x, y, mu, beta, tau):
    result = -tau * np.sum(np.nan_to_num((np.exp(x @ beta + tau * mu) * ((1 - tau * mu) * np.exp(x @ beta + tau * mu) + tau * mu + 1))\
    / (1 + np.exp(x @ beta + tau * mu))**3, nan = 0), axis = 0) - np.sum(np.nan_to_num(tau * np.asarray((np.exp(x @ beta + tau * mu) /\
                                              (1 + np.exp(x @ beta + tau * mu))**2)), nan = 0))
    return result
def g_utt(x, y, mu, beta, tau):
    result = -2 * np.sum(np.nan_to_num(mu * np.asarray((np.exp(x @ beta + tau * mu) /\
                                          (1 + np.exp(x @ beta + tau * mu))**2)), nan = 0), axis = 0)\
    - tau * np.sum( np.nan_to_num(-mu**2 * np.exp(x @ beta + tau * mu) * (np.exp(x @ beta + tau * mu) - 1) /\
                   (1 + np.exp(x @ beta + tau * mu))**3, nan = 0) )
    return result
def g_ubt(x, y, mu, beta, tau):
    left = np.nan_to_num(np.asarray((np.exp(x @ beta + tau * mu) /\
                                       (1 + np.exp(x @ beta + tau * mu))**2)), nan = 0)
    left = -np.sum(x * left, axis = 0)
    right = np.nan_to_num(np.asarray((mu * np.exp(x @ beta + tau * mu) * (np.exp(x @ beta + tau * mu) - 1))/\
                                    (np.exp(x @ beta + tau * mu) + 1)**3), nan = 0)
    right = -np.sum(x * right, axis = 0)
    return left + right
def g_uuuu(x, y, mu, beta, tau):
    result = -tau * np.sum( np.nan_to_num(-tau**3 * np.exp(2 * x @ beta + 2 * tau * mu)/(np.exp(x @ beta + tau * mu) + 1)**3\
                           - tau**3 * np.exp(x @ beta + tau * mu) * (np.exp(x @ beta + tau * mu) - 1)/(np.exp(x @ beta + tau * mu) + 1)**3\
                           + 3 * tau**3 * np.exp(2 * x @ beta + 2 * tau * mu) * (np.exp(x @ beta + tau * mu) - 1)/(np.exp(x @ beta + tau * mu) + 1)**4, nan = 0) )
    return result
def g_uuub(x, y, mu, beta, tau):
    result = -tau * np.sum( np.nan_to_num(tau**2 * x * np.exp(x @ beta + tau * mu)\
                           * (-4 * np.exp(x @ beta + tau * mu) + np.exp(2 * x @ beta + 2 * tau * mu) + 1)\
                           /(np.exp(x @ beta + tau * mu) + 1)**4, nan = 0), axis = 0 )
    return result
def g_uubb(x, y, mu, beta, tau):
    result = 0
    for i in range(len(y)):
        result += x[i].reshape(x.shape[1],1) @ x[i].reshape(1,x.shape[1]) * np.nan_to_num(np.exp(x[i] @ beta + tau * mu)\
        * (-4 * np.exp(x[i] @ beta + tau * mu) + np.exp(2 * x[i] @ beta + 2 * tau * mu) + 1)\
        / (np.exp(x[i] @ beta + tau * mu) + 1)**4, nan=0)
        
    result *= tau                     
        
    return -tau * result
def g_uuut(x, y, mu, beta, tau):
    left = -np.sum(-tau**2 * np.nan_to_num(np.asarray((np.exp(x @ beta + tau * mu) * (np.exp(x @ beta + tau * mu) - 1)\
                             / (1 + np.exp(x @ beta + tau * mu))**3)), nan = 0))
    right = -tau * tau * np.sum(np.nan_to_num(np.exp(x @ beta + tau * mu) * (-4 * tau * mu * np.exp(x @ beta + tau * mu)\
                                                        + tau * mu * np.exp(2 * x @ beta + 2 * tau * mu)\
                                                        -2 * np.exp(2 * x @ beta + 2 * tau * mu) + tau * mu + 2)\
           /(np.exp(x @ beta + tau * mu) + 1)**4, nan=0))
    return left + right
def g_uutt(x, y, mu, beta, tau):
    left = -2 * np.sum(np.nan_to_num((np.exp(x @ beta + tau * mu)\
                                  * ((1 - tau * mu) * np.exp(x @ beta + tau * mu) + tau * mu + 1))\
    / (1 + np.exp(x @ beta + tau * mu))**3, nan = 0), axis = 0)
    right = -tau * mu * np.sum(np.nan_to_num(np.exp(x @ beta + tau * mu) * (-4 * tau * mu * np.exp(x @ beta + tau * mu)\
                                                        + tau * mu * np.exp(2 * x @ beta + 2 * tau * mu)\
                                                        -2 * np.exp(2 * x @ beta + 2 * tau * mu) + tau * mu + 2)\
           /(np.exp(x @ beta + tau * mu) + 1)**4, nan=0))
    return left + right
def g_uubt(x, y, mu, beta, tau):
    left = -np.sum(-tau * x * np.nan_to_num(np.asarray((np.exp(x @ beta + tau * mu) * (np.exp(x @ beta + tau * mu) - 1)\
                             / (1 + np.exp(x @ beta + tau * mu))**3)), nan = 0), axis = 0)
    right = -tau * np.sum(x * np.nan_to_num(np.asarray(np.exp(x @ beta + tau * mu) * (-4 * tau * mu * np.exp(x @ beta + tau * mu)\
                                                             + tau * mu * np.exp(2 * x @ beta + 2 * tau * mu)\
                                                             - np.exp(2 * x @ beta + 2 * tau * mu) + tau * mu + 1)\
                          / (np.exp(x @ beta + tau * mu) + 1)**4), nan = 0), axis = 0)
    return left + right
def omega(x, y, mu, beta, tau):
    return np.sqrt(-1/g_uu(x, y, mu, beta, tau))
def mu_b(x, y, mu, beta, tau):
    return omega(x, y, mu, beta, tau)**2 * g_ub(x, y, mu, beta, tau)
def mu_bb(x, y, mu, beta, tau):
    result = omega(x, y, mu, beta, tau)**2 * (mu_b(x, y, mu, beta, tau).reshape(x.shape[1],1)\
                                              @ mu_b(x, y, mu, beta, tau).reshape(1,x.shape[1])\
                                              * g_uuu(x, y, mu, beta, tau)\
                                              + 2 * mu_b(x, y, mu, beta, tau).reshape(x.shape[1],1)\
                                              @ g_uub(x, y, mu, beta, tau).reshape(1,x.shape[1])\
                                              + g_ubb(x, y, mu, beta, tau))
    return result
def omega_b(x, y, mu, beta, tau):
    return 0.5 * omega(x, y, mu, beta, tau)**3 * (g_uuu(x, y, mu, beta, tau) * mu_b(x, y, mu, beta, tau)\
                                                    + g_uub(x, y, mu, beta, tau))
def mu_t(x, y, mu, beta, tau):
    return omega(x, y, mu, beta, tau)**2 * g_ut(x, y, mu, beta, tau)
def mu_tt(x, y, mu, beta, tau):
    result = omega(x, y, mu, beta, tau)**2 * (mu_t(x, y, mu, beta, tau)\
                                              * mu_t(x, y, mu, beta, tau)\
                                              * g_uuu(x, y, mu, beta, tau)\
                                              + 2 * mu_t(x, y, mu, beta, tau)\
                                              * g_uut(x, y, mu, beta, tau)\
                                              + g_utt(x, y, mu, beta, tau))
    return result
def mu_bt(x, y, mu, beta, tau):
    result = omega(x, y, mu, beta, tau)**2 * (mu_b(x, y, mu, beta, tau)\
                                              * mu_t(x, y, mu, beta, tau)\
                                              * g_uuu(x, y, mu, beta, tau)\
                                              + mu_b(x, y, mu, beta, tau)\
                                              * g_uut(x, y, mu, beta, tau)\
                                              + mu_t(x, y, mu, beta, tau)\
                                              * g_uub(x, y, mu, beta, tau)\
                                              + g_ubt(x, y, mu, beta, tau))
    return result
def omega_t(x, y, mu, beta, tau):
    return 0.5 * omega(x, y, mu, beta, tau)**3 * (g_uuu(x, y, mu, beta, tau) * mu_t(x, y, mu, beta, tau)\
                                                    + g_uut(x, y, mu, beta, tau))
def omega_bb(x, y, mu, beta, tau):
    result = 3/4 * omega(x, y, mu, beta, tau)**5 * (mu_b(x, y, mu, beta, tau) * g_uuu(x, y, mu, beta, tau)\
                                                      + g_uub(x, y, mu, beta, tau)).reshape(x.shape[1],1)\
     @ (mu_b(x, y, mu, beta, tau) * g_uuu(x, y, mu, beta, tau)\
        + g_uub(x, y, mu, beta, tau)).reshape(1,x.shape[1]) + 1/2 * omega(x, y, mu, beta, tau)**3\
     * (mu_bb(x, y, mu, beta, tau) * g_uuu(x, y, mu, beta, tau)\
        + (mu_b(x, y, mu, beta, tau).reshape(x.shape[1],1)\
           @ mu_b(x, y, mu, beta, tau).reshape(1,x.shape[1]) * g_uuuu(x, y, mu, beta, tau))\
        + mu_b(x, y, mu, beta, tau).reshape(x.shape[1],1) @ g_uuub(x, y, mu, beta, tau).reshape(1,x.shape[1])\
        + mu_b(x, y, mu, beta, tau).reshape(x.shape[1],1) @ g_uuub(x, y, mu, beta, tau).reshape(1,x.shape[1])\
        + g_uubb(x, y, mu, beta, tau)
       )
    return result
def omega_tt(x, y, mu, beta, tau):
    result = 3/4 * omega(x, y, mu, beta, tau)**5 * (mu_t(x, y, mu, beta, tau) * g_uuu(x, y, mu, beta, tau)\
                                                      + g_uut(x, y, mu, beta, tau))\
     * (mu_t(x, y, mu, beta, tau) * g_uuu(x, y, mu, beta, tau)\
        + g_uut(x, y, mu, beta, tau)) + 1/2 * omega(x, y, mu, beta, tau)**3\
     * (mu_tt(x, y, mu, beta, tau) * g_uuu(x, y, mu, beta, tau)\
        + (mu_t(x, y, mu, beta, tau)\
           * mu_t(x, y, mu, beta, tau) * g_uuuu(x, y, mu, beta, tau))\
        + mu_t(x, y, mu, beta, tau) * g_uuut(x, y, mu, beta, tau)\
        + mu_t(x, y, mu, beta, tau) * g_uuut(x, y, mu, beta, tau)\
        + g_uutt(x, y, mu, beta, tau)
       )
    return result
def omega_bt(x, y, mu, beta, tau):
    result = 3/4 * omega(x, y, mu, beta, tau)**5 * (mu_t(x, y, mu, beta, tau) * g_uuu(x, y, mu, beta, tau)\
                                                      + g_uut(x, y, mu, beta, tau))\
     * (mu_b(x, y, mu, beta, tau) * g_uuu(x, y, mu, beta, tau)\
        + g_uub(x, y, mu, beta, tau)) + 1/2 * omega(x, y, mu, beta, tau)**3\
     * (mu_bt(x, y, mu, beta, tau) * g_uuu(x, y, mu, beta, tau)\
        + (mu_b(x, y, mu, beta, tau)\
           * mu_t(x, y, mu, beta, tau) * g_uuuu(x, y, mu, beta, tau))\
        + mu_b(x, y, mu, beta, tau) * g_uuut(x, y, mu, beta, tau)\
        + mu_t(x, y, mu, beta, tau) * g_uuub(x, y, mu, beta, tau)\
        + g_uubt(x, y, mu, beta, tau)
       )
    return result
def lb_1(x, y, mu, beta, tau):
    l1 = omega_b(x, y, mu, beta, tau) / omega(x, y, mu, beta, tau)\
    + g_u(x, y, mu, beta, tau) * mu_b(x, y, mu, beta, tau) + g_b(x, y, mu, beta, tau)
    return l1
def lt_1(x, y, mu, beta, tau):
    l1 = omega_t(x, y, mu, beta, tau) / omega(x, y, mu, beta, tau)\
    + g_u(x, y, mu, beta, tau) * mu_t(x, y, mu, beta, tau) + g_t(x, y, mu, beta, tau)
    return l1
def lb_2(x, y, mu, beta, tau):
    l2 = omega(x, y, mu, beta, tau)**(-2) * (omega_bb(x, y, mu, beta, tau) * omega(x, y, mu, beta, tau)\
                                               - omega_b(x, y, mu, beta, tau).reshape(x.shape[1],1)\
                                               @ omega_b(x, y, mu, beta, tau).reshape(1,x.shape[1]))\
     + mu_bb(x, y, mu, beta, tau) * g_u(x, y, mu, beta, tau) + mu_b(x, y, mu, beta, tau).reshape(x.shape[1],1)\
     @ (mu_b(x, y, mu, beta, tau) * g_uu(x, y, mu, beta, tau) + g_ub(x, y, mu, beta, tau)).reshape(1,x.shape[1])\
     + mu_b(x, y, mu, beta, tau).reshape(x.shape[1],1) @ g_ub(x, y, mu, beta, tau).reshape(1,x.shape[1])\
     + g_bb(x, y, mu, beta, tau)
    return l2
def lt_2(x, y, mu, beta, tau):
    l2 = omega(x, y, mu, beta, tau)**(-2) * (omega_tt(x, y, mu, beta, tau) * omega(x, y, mu, beta, tau)\
                                               - omega_t(x, y, mu, beta, tau)\
                                               * omega_t(x, y, mu, beta, tau))\
     + mu_tt(x, y, mu, beta, tau) * g_u(x, y, mu, beta, tau) + mu_t(x, y, mu, beta, tau)\
     * (mu_t(x, y, mu, beta, tau) * g_uu(x, y, mu, beta, tau) + g_ut(x, y, mu, beta, tau))\
     + mu_t(x, y, mu, beta, tau) * g_ut(x, y, mu, beta, tau)\
     + g_tt(x, y, mu, beta, tau)
    return l2
def lbt_2(x, y, mu, beta, tau):
    l2 = omega(x, y, mu, beta, tau)**(-2) * (omega_bt(x, y, mu, beta, tau) * omega(x, y, mu, beta, tau)\
                                               - omega_b(x, y, mu, beta, tau)\
                                               * omega_t(x, y, mu, beta, tau))\
     + mu_bt(x, y, mu, beta, tau) * g_u(x, y, mu, beta, tau) + mu_b(x, y, mu, beta, tau)\
     * (mu_t(x, y, mu, beta, tau) * g_uu(x, y, mu, beta, tau) + g_ut(x, y, mu, beta, tau))\
     + mu_t(x, y, mu, beta, tau) * g_ub(x, y, mu, beta, tau)\
     + g_bt(x, y, mu, beta, tau)
    return l2
def max_mu(x, y, mu, beta, tau, max_iter=100):
    for step in range(max_iter):
        mu_new = mu - g_u(x, y, mu, beta, tau)/g_uu(x, y, mu, beta, tau)
        diff = mu_new - mu
        mu = mu_new
        if np.abs(diff) < 10**(-10):
            break;       
    return mu
def l(x, y, mu, beta, tau):
    l = 0.5 * np.log(2 * np.pi) + np.log(omega(x, y, mu, beta, tau)) + g(x, y, mu, beta, tau)
    return l
class LA:
    """
    A calss to run distributed GLMM

    ...

    Attributes
    ------------
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
    tau : Float
        The hyperparameter of the variance of the random efffect

    """
    def __init__(self, X, y):        
        # Initialization
        self.p = X[0].shape[1]      # Number of variables
        self.n = len(y)             # Number of sites
        if isinstance(X[0], pd.DataFrame):
            self.var_name = X[0].columns
            self.X = [np.array(data) for data in X]
            self.y = [np.array(outcome).reshape(len(outcome),1) for outcome in y]
        else:
            var_name = []
            for i in range(self.p):
                var_name += ['X' + str(i+1)]
            self.var_name = var_name
            self.X = X
            self.y = y
        self.beta = np.repeat(0.1, self.p).reshape(self.p, 1)
        self.lam = 0
        self.mu = np.repeat(1, self.n)
        self.tau = 1
#         self.tau = np.repeat(1.0, self.n)
#         self.tau = 0.62909983
        self.df = pd.DataFrame
        self.score = np.nan
        self.predict = np.nan
        self.time = np.nan
        
    def fit(self, lam_it=0, lam_step=1, mu_it=2, theta_it=100):
        # Iteration
        pre_score = -10**10
        for self.lam in np.arange(0, lam_it+lam_step, lam_step):
            print(f'In lambad = {self.lam}')
            for step_mu in range(mu_it):
                print(f'The {step_mu+1} step of mu')
                self.beta = self.beta.reshape(self.p, 1)
                for i in range(self.n):
#                     print('mu:\n', self.mu, '\n')
                    self.mu[i] = max_mu(self.X[i], self.y[i], self.mu[i], self.beta, self.tau)
#                 print('mu:\n', self.mu, '\n')
                for step_theta in range(theta_it):
                    theta = np.append(self.beta, self.tau)
#                     print('theta:\n', theta, '\n')
                    lb1 = 0
                    lb2 = 0
                    lt1 = 0
                    lt2 = 0
                    lbt2 = 0
#                     lt1 = []
#                     lt2 = []
                    for i in range(self.n):
                        lb1 += lb_1(self.X[i], self.y[i], self.mu[i], self.beta, self.tau)
                        lb2 += lb_2(self.X[i], self.y[i], self.mu[i], self.beta, self.tau)
                        lt1 += lt_1(self.X[i], self.y[i], self.mu[i], self.beta, self.tau)
                        lt2 += lt_2(self.X[i], self.y[i], self.mu[i], self.beta, self.tau)
                        lbt2 += lbt_2(self.X[i], self.y[i], self.mu[i], self.beta, self.tau)
#                         lt1 = np.append(lt1, [lt_1(self.X[i], self.y[i], self.mu[i], self.beta, self.tau[i])])
#                         lt2 = np.append(lt2, [lt_2(self.X[i], self.y[i], self.mu[i], self.beta, self.tau[i])])
#                     print('diff of tau:\n',lt1/lt2,'\n')
                    lb1 -= (2 * self.lam * self.beta.transpose())[0]
                    lt1 -= 2 * self.lam * self.tau
                    lb2 -= np.diag(np.repeat(2 * self.lam, self.p)) + 0.00001
                    lt2 = np.diag(lt2 - 2 * self.lam)#np.diag(lt2)
                    lt2 -= 0.00001
                    L1 = np.append(lb1, lt1)
#                     L2 = block_diag(lb2, lt2)
                    L2 = np.block([
                        [lb2, lbt2.reshape(self.p,1)],
                        [lbt2.reshape(1,self.p), lt2]
                    ])
                    delta = L1 @ inv(L2)
                    new_theta = theta - delta#.reshape(self.p+self.n, 1)
                    if (max(np.abs(delta[:-1])) < 10 **(-2)) and (delta[-1]<10**(-2)):
                        self.beta = new_theta[:self.p]
                        self.tau = new_theta[self.p:]
                        converge = True
                        print('Done with iteration')
                        break;
                    if max(np.abs(delta)) > 10 **(2):
                        converge = False
                        print('Get out of iteration (delta > 10^(2)')
                        break;
                    if True in np.isnan(new_theta[:self.p]):
                        print('Error: NaN beta, rested to 0')
                        # Reset beta and tau
                        self.beta = np.repeat(0, self.p).reshape(self.p, 1)
                        break;
                    if True in np.isnan(new_theta[self.p:]):
                        print('Error: NaN tau, rested to 1.0')
                        # Reset beta and tau
                        converge = False
                        self.tau = 1.0
                        break;
                    if True in np.isnan(self.mu):
                        print('Error: NaN mu, rested to 0')
                        # Reset beta and tau
                        converge = False
                        self.mu = np.repeat(0.1, self.n)
                        break;                    
#                     if new_theta[-1] < 0:
# #                         print('Error: negative tau detectived')
#                         break;
                    self.beta = new_theta[:self.p].reshape(self.p, 1)
                    self.tau = new_theta[self.p:]
                    if step_theta == theta_it-1:
                        print('Error: Did not converged, reset all\n')
                        self.beta = np.repeat(0, self.p).reshape(self.p, 1)
                        self.mu = np.repeat(1, self.n)
                        self.tau = 1
                        converge = False
                        break;
#                     print('Step ', step_theta + 1, ':\n')
#                     print('delta:', delta, '\n')
#                     print('Beta:\n', self.beta, '\n')
#                     print('Diff:\n', delta, '\n')
#                     print('Lam:\n', self.lam, '\n')
#                     print('Score:\n',score,'\n')
#                     print('tau:', self.tau, '\n')
#                     print('mu:\n', self.mu, '\n')
                if not converge:
                    break;
#                 else:
#                     continue
#                 break;
                    
#                     print('l:\n',l(self.X[i], self.y[i], self.mu[i], self.beta, self.tau) - sum(self.lam * (self.beta) **2),'\n')

            score = 0
            predict = []
            self.beta = new_theta[:self.p].reshape(self.p, 1)
            for i in range(self.n):
#                     print('Lam:\n', self.lam, '\n')
#                     print('tau:', self.tau, '\n')
#                     print('Beta:\n', self.beta, '\n')
                score += l(self.X[i], self.y[i], self.mu[i], self.beta, self.tau) - sum(self.lam * (self.beta) **2)
                predict += [Pi(self.X[i], self.y[i], self.mu[i], self.beta, self.tau)]
            print('score:\n', score, '\n')
            if True in np.isnan(score):
                print('Error: NaN score')
                break;
            if (np.mean(score) > pre_score) and converge:
                optimized_beta = self.beta
                optimized_mu = self.mu
                optimized_lam = self.lam
                optimized_tau = self.tau
                # reset
                pre_score = score
                optimized_score = score
#                 self.tau = np.repeat(1.0, self.n)

        # Returning data
        optimized_beta[0] = optimized_beta[0] + np.mean(optimized_tau * optimized_mu)
        self.beta = optimized_beta
        self.mu = optimized_mu
        self.lam = optimized_lam
        self.tau = optimized_tau
        self.score = optimized_score
        self.predict = np.concatenate(predict)


        X = np.concatenate(self.X)
        
        y = np.concatenate(self.y)

        V = np.diagflat(self.predict * (1 - self.predict) + 0.0001)

        SE = np.sqrt(np.diag(inv(np.transpose(X) @ V @ X))).reshape(self.p,1)

        Z = self.beta/SE

        P = 2 * norm.cdf(-1 * np.abs(Z))

        CI_025  = self.beta - 1.959964 * SE
        CI_975  = self.beta + 1.959964 * SE

        self.df = pd.DataFrame({'Coef': np.transpose(self.beta)[0], 'Std.Err': np.transpose(SE)[0],
                           'z': np.transpose(Z)[0], 'P-value': np.transpose(P)[0],
                           '[0.025': np.transpose(CI_025)[0], '0.975]': np.transpose(CI_975)[0]},
                          index = self.var_name)
        
        return self