"""
Method for sampling from a multivariate Gaussian using the conjugate gradient method.
Included here to get a back-of-the-envelope sense of how much slower this method will be over classical Cholesky decomposition based sampling.
"""

from scipy import linalg
import numpy as np
from numpy import random
import math
import logging

# TODO could wrap these in a class so that we can precompute and retain the Cholesky factor of the LD portion.
# We would have to change the signatures of the functions somewhat to take in the LD / Psi separately though instead of combined as an unscaled precision matrix

def sample_mvn_alg_3_4(unscaled_Q, beta_hat, sigma2, N, max_iterations=None, error=None, preconditioned=False):
    # we will see a lot of squeezes because the base PRScs implementation typically works with 2Darrays with dimension 1 instead of 1Darrays
    max_iterations = len(unscaled_Q) if max_iterations is None else max_iterations
    sigma2 = np.squeeze(sigma2) # sigma2 sometimes comes in a 2darray
    beta_hat = np.squeeze(beta_hat)
    scaled_Q = (N / sigma2)*unscaled_Q
    Q_sample = math.sqrt(N / sigma2)*cholesky_sample(0, unscaled_Q) # or however else you want to sample it
    Q_sample = np.squeeze(Q_sample)
    eta = Q_sample + ((N / sigma2)*beta_hat)
    if not preconditioned:
        theta, n_iterations = cg_solve(scaled_Q, eta, max_iterations=max_iterations, error=error)
        # logging.info("Nonprecon MVN call on blocksize %d has condition number %f, iterations %d" % (len(unscaled_Q), condition_number(unscaled_Q), n_iterations))
        # where n_iterations is the number of iterations actually performed
        return theta, n_iterations
    preconditioner = linalg.inv(np.diag(np.diagonal(unscaled_Q).copy()))
    theta, n_iterations = preconditioned_cg_solve(scaled_Q, eta, preconditioner, max_iterations=max_iterations, error=error)
    # logging.info("Precond MVN call on blocksize %d has condition number %f, iterations %d" % (len(unscaled_Q), condition_number(preconditioner @ unscaled_Q), n_iterations))
    return theta, n_iterations

def cholesky_sample(mu, A):
    # Cholesky sample from the MVN with mean mu and covariance matrix A
    C = linalg.cholesky(A, lower=True) # I prefer the lower triangular convention
    z = random.randn(len(A), 1)
    return mu + (C @ z)

def cg_solve(A, b, max_iterations, error=None, x=None):
    """Approximates a solution x to A@x = b. Algorithm adapted from algorithm B2 in J. Shewchuk's "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain" """
    # the number of iterations one can perform tops off at the matrix dimension, which results in the exact answer. Alg is not well defined if we go over
    n = A.shape[0]
    if max_iterations > n or error is None:
        max_iterations = n
        error = 0
    if x is None:
        x = np.zeros(n)
    r = b - (A @ x)
    d = r
    delta_new = np.inner(r, r)

    max_error = delta_new * error * error
    n_iterations = 0
    
    for i in range(max_iterations):
        if delta_new <= max_error:
            break
        q = A @ d
        alpha = delta_new / np.inner(d, q)
        x = x + (alpha * d)
        if i % 50 == 0:
            r = b - (A @ x)
        else:
            r = r - (alpha * q)
        delta_old = delta_new
        delta_new = np.inner(r, r)
        beta = delta_new / delta_old
        d = r + (beta * d)
        n_iterations += 1
    return x, n_iterations

def preconditioned_cg_solve(A, b, M_inv, max_iterations, error=None, x=None):
    """Approximates a solution x to M^-1Ax = M^-1bx with starting value `x` and `max_iterations` of the Conjugate Gradient method. Adapted from algorithm B2 in J. Shewchuk's "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"""
    n = A.shape[0]
    if max_iterations > n or error is None:
        max_iterations = n
        error = 0
    if x is None:
        x = np.zeros(n)
    r = b - (A @ x)
    d = M_inv @ r
    delta_new = np.inner(r, d)

    max_error = delta_new * error * error
    n_iterations = 0
    
    for i in range(max_iterations):
        if delta_new <= max_error:
            break
        q = A @ d
        alpha = delta_new / np.inner(d, q)
        x = x + (alpha * d)
        if i % 50 == 0:
            r = b - (A @ x)
        else:
            r = r - (alpha * q)
        s = M_inv @ r
        delta_old = delta_new
        delta_new = np.inner(r, s)
        beta = delta_new / delta_old
        d = s + (beta * d)
        n_iterations += 1
    return x, n_iterations

def condition_number(A):
    """Calculates the conjugate gradient method condition number of a real symmetric matrix A."""
    eigvals = linalg.eigvalsh(A)
    tmp_eigvals = np.sort(np.absolute(eigvals))
    return tmp_eigvals[-1] / tmp_eigvals[0]
