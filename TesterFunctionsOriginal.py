#!usr/bin/env python

import os, sys
import numpy as np
import scipy.stats as spstat
import math
from math import *
import scipy.optimize as spopt

global mu_test
mu_test=1
#from contur.TestingFunctions import covariance_matrix as cv

# mu_test --> Null hypothesis
# mu  --> Signal strength
# mu_hat --> Maximum likelihood value of mu


def Min_function(x, n_obs, b_obs, s_obs, db, ds):
    d_lnL_db = 0.0
    d_lnL_ds = 0.0
    if fabs((x[0] + x[1])) > 0.0:
        d_lnL_db += n_obs / (x[0] + x[1]) - 1
        d_lnL_ds += n_obs / (x[0] + x[1]) - 1
    if fabs(db) >= 1e-5:
        d_lnL_db += (b_obs - x[0]) / db**2
    if x[1] > 0.0:
        d_lnL_ds += ds / x[1]
    if s_obs > 0.0:
        d_lnL_ds -= ds/s_obs
    return [d_lnL_db,d_lnL_ds]


def Min_find(n_obs, b_obs, s_obs, db, ds):
    return spopt.root(Min_function, [b_obs, s_obs], args=(n_obs, b_obs, s_obs, db, ds))



def ML_mu_hat(n_obs,b_in,s_in):
    if fabs(s_in) <= 1E-5:
      return 0
    else:
      return (n_obs - b_in)/s_in


def Covar_Matrix(b_count,s_count,db_count, ds_count):
    Var_matrix_inv=np.zeros([(len(b_count)+len(s_count)+1),(len(b_count)+len(s_count)+1)])

    for i in range(0, len(b_count)+len(s_count)):

        if i < (len(b_count)):
            res = Min_find(b_count[i], b_count[i], s_count[i], db_count[i], ds_count[i]).x
        else:
            res = Min_find(b_count[i-len(b_count)], b_count[i-len(b_count)], s_count[i-len(b_count)], db_count[i-len(b_count)], ds_count[i-len(b_count)]).x
        b_hat_hat = res[0]
        s_hat_hat = res[1]

        if i < len(b_count):

            ##mu mu
            Var_matrix_inv[0,0] += s_hat_hat**2/(mu_test*s_hat_hat+b_hat_hat)
            ##mu b
            Var_matrix_inv[i+1,0]=Var_matrix_inv[0,i+1] = s_hat_hat/(mu_test*s_hat_hat+b_hat_hat)
            ##b b
            if db_count[i]**2 > 0.0:
                Var_matrix_inv[i+1,i+1]=1/(mu_test*s_hat_hat+b_hat_hat) + 1/db_count[i]**2
            else:
                Var_matrix_inv[i+1,i+1]=1/(mu_test*s_hat_hat+b_hat_hat)
        if i>=(len(b_count)):
            ##mu s
            Var_matrix_inv[i+1,0]=Var_matrix_inv[0,i+1] = (mu_test*s_hat_hat)/(mu_test*s_hat_hat+b_hat_hat)
            ##s s
            if s_hat_hat >0.0:
                Var_matrix_inv[i+1,i+1]=(mu_test**2)/(mu_test*s_hat_hat+b_hat_hat) + ds_count[i-len(b_count)]/(s_hat_hat**2)

            else:
                Var_matrix_inv[i+1,i+1]=(mu_test**2)/(mu_test*s_hat_hat+b_hat_hat)
        if i < len(s_count):
            ## b s
            Var_matrix_inv[len(b_count)+1+i,i+1] = Var_matrix_inv[i+1,len(b_count)+1+i] = mu_test/(mu_test*s_hat_hat+b_hat_hat)

#    print 'inv matrix '+str(Var_matrix_inv[0,0])
#    print 'det '+str(np.linalg.det(Var_matrix_inv))

    if np.linalg.det(Var_matrix_inv) == 0:
        Var_matrix = np.zeros([(len(b_count)+1),(len(b_count)+1)])
  #Invert and return it
    else:
        Var_matrix = np.linalg.inv(Var_matrix_inv)

#    print 'matrix '+str(Var_matrix[0,0])

    return Var_matrix

def confLevel(sigCount, bgCount, bgErr, sgErr,mu_test=1):

    mu_hat = 0
    # mu_hat= ML_mu_hat(n_obs,bgCount,sigCount)

    varMat= Covar_Matrix(bgCount,sigCount,bgErr, sgErr)[0,0]

    if varMat <=0:
        return 0
    else:
        q_mu=0
        p_val=0
        q_mu = (mu_test-mu_hat)**2/(varMat)
        if 0 < q_mu <= (mu_test**2)/(varMat):
            p_val=2.0*spstat.norm.sf(np.sqrt(q_mu))
        elif q_mu > (mu_test**2)/(varMat):
            p_val=2.0*spstat.norm.sf( (q_mu + (mu_test**2/varMat))/(2*mu_test/(np.sqrt(varMat))) )

    return float('%10.6f' % float(1-p_val))
