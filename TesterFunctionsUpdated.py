
import os, sys
import numpy as np
import scipy.stats as spstat
import math
from math import *
import scipy.optimize as spopt

# global mu_test
# mu_test=1
# This global value of mu_test is commented out, as it corresponds to the initial assumption where data = SM expectation
# mu_test is defined for the complete signal and null hypothesis signal, as discussed below.

# mu_test --> Null hypothesis signal strength
# mu  --> Signal strength
# mu_hat --> Maximum likelihood value of mu


def Min_function(x, n_obs, b_obs, s_obs, db, ds):
    # Defining the functional form of the vector function Min_function[0], corresponding to the derivative
    # dlnL/db. Min_function[1] corresponds to the derivative dlnL/ds, taking the argument
    # x = [b\_hat\_hat, s\_hat\_hat]. The maximum likelihood value of the signal strength mu, mu_hat
    # is passed and included within the functional form of the first derivatives. Currently set to a value of 1.
    # The use of 'if' statements mitigates the rapid expansion of any terms with a denominator close to zero.
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

# This function is responsible for running the root finding process on the previously defined Min_function.
# The necessary arguments are passed through, defined by args = (n_obs, b_obs, s_obs, db, ds).
def Min_find(n_obs, b_obs, s_obs, db, ds):
    return spopt.root(Min_function, [b_obs, s_obs], args=(n_obs, b_obs, s_obs, db, ds))


# Computes the maximum likelihood estimate, mu_hat, for the signal strength parameter,'mu'.
def ML_mu_hat(n_obs,b_in,s_in):
    if fabs(s_in) <= 1E-5:
      return 0
    else:
      return (n_obs - b_in)/s_in

# This Covar_Matrix function is responsible for constructing the inverse covariance matrix from the expected values of the second derivatives
# of the log-likelihood function for a given signal. Extracting the matrix element 'V_mu mu' gives the variance on the maximum likelihood parameter.
# A matrix of zeros is first defined, with exception handling added to account for signal events not being equal to the
# background ie 's' not equal to 'b' The function is called to simultaneously to minimise the values of 's' and 'b' for a specified maximimum
# likelihood value 'mu'. If this minimum is not found, the input values 's' and 'b' are returned.
# The original implementation incorporated a cancellation of the 'n' term in the numerator and 'mu_s+b' term in the denominator due to the
# assumption of equivalence between the data and Standard Model expectations. An updated implementation using simulated background counts would have
# these two event counts separated, and thus the second derivatives have been rewritten here to retain the observed event count, n_obs, together with the
# signal + background term in the denominator. As discussed below, n_obs may have to be replaced by a new variable n_count.
# An additional argument, mu_test, has been added. This facilitates defining the two values of mu_test (0 and 1) corresponding to separate
# event counts for 'n' and 'b'.
def Covar_Matrix(b_count,s_count,db_count, ds_count, mu_test):

    Var_matrix_inv=np.zeros([(len(b_count)+len(s_count)+1),(len(b_count)+len(s_count)+1)])

    # n_obs = bgCount
    # As outlined in the confLevel comments, the value of bgCount will be taken from simulation
    # data via a separate YODA file

    for i in range(0, len(b_count) + len(s_count)):
        if i < len(b_count):
            # mu mu
            Var_matrix_inv[0,0] += -n_obs*s_hat_hat**2/(mu_test*s_hat_hat + b_hat_hat)**2
            # mu b = b mu
            Var_matrix_inv[i+1,0]=Var_matrix_inv[0,i+1] = -n_obs*s/(mu_test*s_hat_hat + b_hat_hat)**2
            # bb. dbcount[i] is error on background count (sigma_b) as in the CONTUR reference paper
            if db_count[i]**2 > 0.0: #Error > 0.0
                Var_matrix_inv[i+1,i+1] = -n_obs/(mu_test*s_hat_hat + b_hat_hat)**2 - 1/db_count[i]**2
            else:
                Var_matrix_inv[i+1,i+1] = -n_obs/(mu_test*s_hat_hat + b_hat_hat)**2

        if i >= (len(b_count)):
            # mu s = s mu
            Var_matrix_inv[i+1,0]=Var_matrix_inv[0,i+1] = n_obs*b/(mu_test*s_hat_hat + b_hat_hat)**2 - 1
            # ss
            # k/s^2 is stored in sigError - this is the ratio of the Monte Carlo luminosity values to the data luminosity values.
            if s_hat_hat > 0.0:
                Var_matrix_inv[i+1,i+1] = -n_obs*mu_test**2/(mu_test*s_hat_hat + b_hat_hat)**2 - sgErr #k/s^2
            else:
                Var_matrix_inv[i+1,i+1] = -n_obs*mu_test**2/(mu_test*s_hat_hat + b_hat_hat)**2

            if i < len(s_count):
                # bs = sb
                Var_matrix_inv[len(b_count)+1+i,i+1] = Var_matrix_inv[i+1,len(b_count)+1+i] = -n_obs*mu_test/(mu_test*s_hat_hat + b_hat_hat)**2

        # print 'inv matrix '+str(Var_matrix_inv[0,0])
        # print 'det '+str(np.linalg.det(Var_matrix_inv))

        if np.linalg.det(Var_matrix_inv) == 0:
            Var_matrix = np.zeros([(len(b_count)+1), (len(b_count)+1)])
        else:
            Var_matrix = np.linalg.inv(Var_matrix_inv) # Inverting V^-1 to give V
        # print 'matrix '+str(Var)matrix[0,0])

        return Var_matrix



# This function computes the confidence level for a given set of results in accordance with the CLs method
# outlined in the thesis.
# A constant factor of 2 arises due to following CLs relation: CL_s = 1-(p_s+b)/1-p_b. 1/1-p_b = 1/1-0.5 = 2
# Evaluation of the p-value in the null hypothesis requires running the confLevel algorithm a second time, but using the null signal hypothesis,
# where mu_test = 0. The probability value (p-value) in the null signal hypothesis must now be computed separately, that is,
# the value of 1 - p_b must be calculated, instead of using a factor of 2 throughout.
def confLevel(sigCount, bgCount, bgErr, sgErr, mu_test):

    # bgCount =  # Under the original assumption, n = b, bgCount is set equal to n_obs
    # Future implementations should seek to add functionality here to lookup the value of bgCount from data via an SQL database,
    # containing a YODA file with Standard Model Monte Carlo simulations. This would then differentiate bgCount from n_obs (potential to remove n_obs discussed below)

    # mu_hat = 0
    # This is commented out as it is a consequence of the original assumption that data = SM expectation, setting mu_hat as the null hypothesis.

    mu_hat = ML_mu_hat(n_obs,bgCount,sigCount)

# Calculation of b_hat_hat and s_hat_hat --> Originally found in Covar_Matrix, this computation only needs to be performed once, here, in confLevel.
    for i in range(0, len(b_count) + len(s_count)):
        if i < (len(b_count)):
            res = Min_find(b_count[i], b_count[i], s_count[i], db_count[i], ds_count[i]).x
        else:
            res = Min_find(b_count[i-len(b_count)], b_count[i-len(b_count)], s_count[i-len(b_count)], db_count[i-len(b_count)], ds_count[i-len(b_count)]).x
            break
        b_hat_hat = res[0]
        s_hat_hat = res[1]

# Two separate instances where varMat is called --> First for mu_test = 1 (complete signal hypothesis) and second for the null hypothesis at mu_test = 0
# varMat1 corresponds to using the complete signal hypothesis, varMat2 to using the null hypothesis
# Future work should focus on including an additional argument, n_count when calling Covar_Matrix to compute varMat1 and 2.
# As discussed in the thesis, it is not clear how the inclusion of n_obs in the updated second derivative forms
# should relate to the new n_count argument. The SQL datbase mechanism will ensure that bgCount is defined independently,
# however future work should focus on verifying whether n_obs should be removed from the code entirely and replaced
# with n_count.
# The factor of two, originally included for the n = b assumption in the test statistic calculation, has been removed.
# The two 'for' loops calculate the test statistic values q_mu_1 (for mu_test = 0) and q_mu_2 (for mu_test = 1)

    varMat1 = Covar_Matrix(bgCount,sigCount,bgErr, sgErr, 0)[0,0]

    if varMat1 <=0:
        return 0
    else:
        q_mu_1 = 0
        p_val_1 = 0
        q_mu_1 = (mu_test-mu_hat)/(varMat1)
        if 0 < q_mu_1 <= (mu_test)/(varMat1):
            p_val_1 = spstat.norm.sf(np.sqrt(q_mu_1))
        elif q_mu_1 > (mu_test)/(varMat):
            p_val_l = spstat.norm.sf( (q_mu_1 + (mu_test/varMat1))/(mu_test/(np.sqrt(varMat1))))
            return float('%10.6f' % float(1-p_val_1))

    varMat2 = Covar_Matrix(bgCount, sigCount, bgErr, sgErr, 1)[0,0]

    if varMat2 <=0:
        return 0
    else:
        q_mu_2 = 0
        p_val_2 = 0
        q_mu_2 = (mu_test - mu_hat)/(varMat2)
        if 0 < q_mu_2 <= (mu_test)/(varMat2):
            p_val_2 = spstat.norm.sf(np.sqrt(q_mu_2))
        elif q_mu_2 > (mu_test)/(varMat):
            p_val_2 = spstat.norm.sf((q_mu_2 + (mu_test/varMat2))/(mu_test/(np.sqrt(varMat2))))

            return float('%10.6f' % float(1-p_val_2))
