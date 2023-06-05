"""
Method for sampling from a multivariate Gaussian using the conjugate gradient method.
Included here to get a back-of-the-envelope sense of how much slower this method will be over classical Cholesky decomposition based sampling.
"""

from scipy import linalg
import numpy as np
from numpy import random
import math


def sample_mvn_alg_3_4(unscaled_Q, beta_hat, sigma2, N, n_iterations=None):
    n_iterations = len(unscaled_Q) if n_iterations is None else n_iterations
    scaled_Q = (N / sigma2)*unscaled_Q
    Q_sample = math.sqrt(N / sigma2)*cholesky_sample(0, unscaled_Q) # or however else you want to sample it
    eta = Q_sample + ((N / sigma2)*beta_hat)
    theta = cg_solve(scaled_Q, eta, n_iterations=n_iterations)
    return theta

def cholesky_sample(mu, A):
    # Cholesky sample from the MVN with mean mu and covariance matrix A
    C = linalg.cholesky(A, lower=True) # I prefer the lower triangular convention
    z = random.randn(len(A), 1)
    w = linalg.solve_triangular(C, z, lower=True, trans='T')
    return mu + w

def cg_solve(A, b, n_iterations, x=None):
    """Approximates a solution x to A@x = b. Algorithm adapted from algorithm B2 in J. Shewchuk's "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain" """
    # the number of iterations one can perform tops off at the matrix dimension, which results in the exact answer. Alg is not well defined if we go over
    n = A.shape[0]
    if n_iterations > n:
        n_iterations = n
    if x is None:
        x = np.zeros(n)
    r = b - (A @ x)
    d = r
    delta_new = np.inner(r, r)
    
    # print('initialization')
    # print('r is', r)
    # print('r is', r)
    # print('delta is', delta_new)
    for i in range(n_iterations):
        # print(f' === iteration {i} ===')
        q = A @ d
        # print('q is', q)
        alpha = delta_new / np.inner(d, q)
        # print('alpha is', alpha)
        x = x + (alpha * d)
        # print('x is', x)
        if i % 50 == 0:
            r = b - (A @ x)
        else:
            r = r - (alpha * q)
        # print('r is', r)
        delta_old = delta_new
        delta_new = np.inner(r, r)
        # print('new delta is', delta_new)
        beta = delta_new / delta_old
        # print('beta is', beta)
        d = r + (beta * d)
        # print('d is', d)
    return x

def preconditioned_cg_solve(A, b, M_inv, n_iterations, x=None):
    """Approximates a solution x to M^-1Ax = M^-1bx with starting value `x` and `n_iterations` of the Conjugate Gradient method. Adapted from algorithm B2 in J. Shewchuk's "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"""
    n = A.shape[0]
    if n_iterations > n:
        n_iterations = n
    if x is None:
        x = np.zeros(n)
    r = b - (A @ x)
    d = M_inv @ r
    delta_new = np.inner(r, d)
    
    # print('initialization')
    # print('r is', r)
    # print('r is', r)
    # print('delta is', delta_new)
    for i in range(n_iterations):
        # print(f' === iteration {i} ===')
        q = A @ d
        # print('q is', q)
        alpha = delta_new / np.inner(d, q)
        # print('alpha is', alpha)
        x = x + (alpha * d)
        # print('x is', x)
        if i % 50 == 0:
            r = b - (A @ x)
        else:
            r = r - (alpha * q)
        # print('r is', r)
        s = M_inv @ r
        # print('s is', s)
        delta_old = delta_new
        delta_new = np.inner(r, s)
        # print('new delta is', delta_new)
        beta = delta_new / delta_old
        # print('beta is', beta)
        d = s + (beta * d)
        # print('d is', d)
    return x
